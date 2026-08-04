"""Bounded, caller-configured Codex process boundary.

Domain workflows depend only on :class:`ModelRunner`.  Process composition is
kept here so workflows never need ambient environment, executable discovery,
or subprocess access.
"""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import secrets
import stat
import subprocess
import unicodedata
from time import monotonic
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from career_agent_workbench.workflow_diagnostics import (
    DiagnosticEvent,
    FailureCategory,
    WorkflowStage,
    attempt_event,
    emit_diagnostic,
)

MAX_LOGICAL_VALUE_BYTES = 256
MAX_PROMPT_BYTES = 524_288
MAX_RESPONSE_BYTES = 1_048_576
MAX_TIMEOUT_SECONDS = 1_800.0
MAX_ATTEMPTS = 4
MAX_BASE_ARGV_PARTS = 16
MAX_COMMAND_PARTS = 32
MAX_ARGV_PART_BYTES = 4_096
MAX_ENVIRONMENT_ENTRIES = 64
MAX_ENVIRONMENT_BYTES = 32_768
MAX_OUTPUT_FILES = 1
MIN_PROCESS_RETURN_CODE = -(2**31)
MAX_PROCESS_RETURN_CODE = 2**31 - 1
_MAX_CLEANUP_ENTRIES = 64
_MAX_CLEANUP_DEPTH = 4
_OUTPUT_NAME = "last-message.json"
_MODEL_METADATA_VERSION = 1
_PATH_TYPE = type(Path())
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_RESERVED_BASE_ARGUMENTS = frozenset(
    {
        "--ask-for-approval",
        "--approval",
        "--sandbox",
        "--cd",
        "-C",
        "--output-last-message",
        "--model",
        "-m",
        "-c",
        "-",
        "--",
    }
)
_ALLOWED_EXTRA_BASE_ARGUMENTS = frozenset({"--skip-git-repo-check"})


class CodexRunnerError(Exception):
    """Base class for stable, content-free runner failures."""

    __slots__ = ()


class CodexConfigurationError(CodexRunnerError):
    """Raised when explicit runner configuration is unsafe or invalid."""

    __slots__ = ()


class CodexExecutionError(CodexRunnerError):
    """Raised when an attempted model process does not complete successfully."""

    __slots__ = ()


class CodexTimeoutError(CodexRunnerError):
    """Raised when all permitted attempts time out."""

    __slots__ = ()


class CodexOutputError(CodexRunnerError):
    """Raised when the output-last-message artifact is absent or unsafe."""

    __slots__ = ()


class CodexCleanupError(CodexRunnerError):
    """Raised when the runner cannot prove its owned child was removed."""

    __slots__ = ()


class CodexCancellationError(CodexRunnerError):
    """Raised after an interrupted attempt has completed safe cleanup."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class CodexModelConfig:
    """Logical model policy, independent of any executable configuration."""

    model: str
    reasoning_effort: str
    workflow: str = "workflow"
    profile: str | None = None

    def __post_init__(self) -> None:
        _validate_logical_value(self.model, allow_empty=True)
        _validate_logical_value(self.reasoning_effort, allow_empty=True)
        _validate_logical_value(self.workflow, allow_empty=False)
        if self.profile is not None:
            _validate_logical_value(self.profile, allow_empty=False)

    def __repr__(self) -> str:
        return (
            "CodexModelConfig("
            "model=configured, reasoning_effort=configured, "
            "workflow=configured, "
            f"profile_configured={self.profile is not None})"
        )


def resolve_codex_model_config(
    *,
    default_model: str,
    default_reasoning_effort: str,
    workflow_model_override: str | None = None,
    workflow_reasoning_effort_override: str | None = None,
    workflow: str = "workflow",
    profile: str | None = None,
) -> CodexModelConfig:
    """Resolve model and effort independently with presence-aware overrides."""

    model = (
        workflow_model_override
        if workflow_model_override is not None
        else default_model
    )
    effort = (
        workflow_reasoning_effort_override
        if workflow_reasoning_effort_override is not None
        else default_reasoning_effort
    )
    return CodexModelConfig(
        model=model,
        reasoning_effort=effort,
        workflow=workflow,
        profile=profile,
    )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One bounded model request whose content is hidden from ``repr``."""

    prompt: str = field(repr=False)
    config: CodexModelConfig = field(repr=False)
    timeout_seconds: float = 300.0
    max_attempts: int = 1
    max_response_bytes: int = MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        _validate_content_text(
            self.prompt,
            maximum_bytes=MAX_PROMPT_BYTES,
            allow_empty=False,
        )
        if type(self.config) is not CodexModelConfig:
            raise CodexConfigurationError("Invalid model request configuration.")
        _validate_timeout(self.timeout_seconds)
        if (
            type(self.max_attempts) is not int
            or not 1 <= self.max_attempts <= MAX_ATTEMPTS
        ):
            raise CodexConfigurationError("Invalid model request attempts.")
        if (
            type(self.max_response_bytes) is not int
            or not 1 <= self.max_response_bytes <= MAX_RESPONSE_BYTES
        ):
            raise CodexConfigurationError("Invalid model response limit.")

    def __repr__(self) -> str:
        return (
            "ModelRequest(prompt=hidden, config=hidden, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_attempts={self.max_attempts!r}, "
            f"max_response_bytes={self.max_response_bytes!r})"
        )


@dataclass(frozen=True, slots=True)
class ModelResult:
    """One bounded response plus immutable public-safe logical metadata."""

    response: str = field(repr=False)
    model_metadata: Mapping[str, str | int] = field(repr=False)

    def __post_init__(self) -> None:
        _validate_content_text(
            self.response,
            maximum_bytes=MAX_RESPONSE_BYTES,
            allow_empty=False,
        )
        metadata = _validated_model_metadata(self.model_metadata)
        object.__setattr__(self, "model_metadata", MappingProxyType(metadata))

    @property
    def text(self) -> str:
        """Return the response for callers that use text-oriented naming."""

        return self.response

    def __repr__(self) -> str:
        return "ModelResult(response=hidden, model_metadata=hidden)"


class ModelRunner(Protocol):
    """The only model capability required by P09 domain workflows."""

    def run(self, request: ModelRequest, /) -> ModelResult:
        """Run one fully configured request."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Minimal inert result returned by an injected process executor."""

    returncode: int

    def __post_init__(self) -> None:
        if (
            type(self.returncode) is not int
            or not MIN_PROCESS_RETURN_CODE <= self.returncode <= MAX_PROCESS_RETURN_CODE
        ):
            raise CodexExecutionError("Codex process returned an invalid result.")


class ProcessExecutor(Protocol):
    """Explicit low-level process capability used by ``CodexProcessRunner``."""

    def run(
        self,
        command: tuple[str, ...],
        *,
        input: str,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
        stdout: int,
        stderr: int,
        check: bool,
        shell: bool,
        encoding: str,
        errors: str,
    ) -> ProcessResult:
        """Execute one argv-only process invocation."""


@dataclass(frozen=True, slots=True)
class SubprocessExecutor:
    """Explicit production executor; tests and workflows may inject a fake."""

    def __repr__(self) -> str:
        return "SubprocessExecutor(configured=True)"

    def run(
        self,
        command: tuple[str, ...],
        *,
        input: str,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
        stdout: int,
        stderr: int,
        check: bool,
        shell: bool,
        encoding: str,
        errors: str,
    ) -> ProcessResult:
        completed = subprocess.run(
            command,
            input=input,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdout=stdout,
            stderr=stderr,
            check=check,
            shell=shell,
            encoding=encoding,
            errors=errors,
        )
        return ProcessResult(returncode=completed.returncode)


@dataclass(frozen=True, slots=True)
class CodexProcessConfig:
    """Explicit process inputs, with all path and environment content hidden."""

    executable: Path
    argv: tuple[str, ...]
    working_directory: Path
    tmp_dir: Path
    child_environment: Mapping[str, str] = field(repr=False)
    _executable_identity: tuple[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _working_directory_identity: tuple[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _tmp_dir_identity: tuple[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        executable = _validate_path_value(self.executable)
        working_directory = _validate_path_value(self.working_directory)
        tmp_dir = _validate_path_value(self.tmp_dir)
        _validate_executable(executable)
        _validate_directory(working_directory)
        _validate_directory(tmp_dir)
        argv = _validated_base_argv(self.argv)
        environment = _validated_environment(self.child_environment)
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "working_directory", working_directory)
        object.__setattr__(self, "tmp_dir", tmp_dir)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(
            self,
            "child_environment",
            MappingProxyType(environment),
        )
        object.__setattr__(
            self,
            "_executable_identity",
            _required_path_identity(executable),
        )
        object.__setattr__(
            self,
            "_working_directory_identity",
            _required_path_identity(working_directory),
        )
        object.__setattr__(
            self,
            "_tmp_dir_identity",
            _required_path_identity(tmp_dir),
        )

    def __repr__(self) -> str:
        return (
            "CodexProcessConfig("
            "executable=configured, argv=hidden, "
            "working_directory=configured, tmp_dir=configured, "
            "child_environment=hidden)"
        )


@dataclass(slots=True, repr=False)
class _OwnedChild:
    root_fd: int
    child_fd: int
    name: str
    path: Path
    root_identity: tuple[int, int]
    child_identity: tuple[int, int]


class CodexProcessRunner:
    """Run Codex with fixed safety flags and caller-supplied capabilities."""

    __slots__ = ("_config", "_executor")

    def __init__(
        self,
        *,
        config: CodexProcessConfig,
        executor: ProcessExecutor,
    ) -> None:
        if type(config) is not CodexProcessConfig:
            raise CodexConfigurationError("Invalid Codex process configuration.")
        if executor is None:
            raise CodexConfigurationError("A process executor is required.")
        self._config = _snapshot_process_config(config)
        self._executor = executor

    def __repr__(self) -> str:
        return "CodexProcessRunner(configured=True)"

    def run(self, request: ModelRequest, /) -> ModelResult:
        initial = _snapshot_model_request(request)
        diagnostic_stage = _diagnostic_stage(initial.config.workflow)

        for attempt in range(1, initial.max_attempts + 1):
            current = _snapshot_model_request(request)
            if not _request_snapshots_match(initial, current):
                raise CodexConfigurationError("Invalid model request.")
            process = _snapshot_process_config(self._config)
            started = monotonic()
            if diagnostic_stage is not None:
                emit_diagnostic(
                    attempt_event(
                        event=DiagnosticEvent.ATTEMPT_START,
                        stage=diagnostic_stage,
                        attempt=attempt,
                        total_attempts=initial.max_attempts,
                    )
                )
            outcome, failure, retryable = self._run_attempt(
                current,
                process,
                attempt,
            )
            elapsed = monotonic() - started
            if diagnostic_stage is not None:
                emit_diagnostic(
                    attempt_event(
                        event=DiagnosticEvent.ATTEMPT_ELAPSED,
                        stage=diagnostic_stage,
                        attempt=attempt,
                        total_attempts=initial.max_attempts,
                        elapsed_seconds=elapsed,
                    )
                )
            if outcome is not None:
                if diagnostic_stage is not None:
                    emit_diagnostic(
                        attempt_event(
                            event=DiagnosticEvent.ATTEMPT_COMPLETION,
                            stage=diagnostic_stage,
                            attempt=attempt,
                            total_attempts=initial.max_attempts,
                        )
                    )
                return outcome
            if failure is None:
                raise CodexExecutionError("Codex execution failed.")
            if retryable and attempt < initial.max_attempts:
                if diagnostic_stage is not None:
                    emit_diagnostic(
                        attempt_event(
                            event=DiagnosticEvent.TIMEOUT,
                            stage=diagnostic_stage,
                            attempt=attempt,
                            total_attempts=initial.max_attempts,
                            category=FailureCategory.TIMEOUT,
                        )
                    )
                    emit_diagnostic(
                        attempt_event(
                            event=DiagnosticEvent.RETRY_DECISION,
                            stage=diagnostic_stage,
                            attempt=attempt,
                            total_attempts=initial.max_attempts,
                            retry=True,
                            category=FailureCategory.TIMEOUT,
                        )
                    )
                continue
            if diagnostic_stage is not None:
                category = _diagnostic_failure_category(failure)
                if category is FailureCategory.TIMEOUT:
                    emit_diagnostic(
                        attempt_event(
                            event=DiagnosticEvent.TIMEOUT,
                            stage=diagnostic_stage,
                            attempt=attempt,
                            total_attempts=initial.max_attempts,
                            category=category,
                        )
                    )
                emit_diagnostic(
                    attempt_event(
                        event=DiagnosticEvent.RETRY_DECISION,
                        stage=diagnostic_stage,
                        attempt=attempt,
                        total_attempts=initial.max_attempts,
                        retry=False,
                        category=category,
                    )
                )
                emit_diagnostic(
                    attempt_event(
                        event=DiagnosticEvent.FAILURE,
                        stage=diagnostic_stage,
                        attempt=attempt,
                        total_attempts=initial.max_attempts,
                        category=category,
                    )
                )
            raise failure
        raise CodexExecutionError("Codex execution failed.")

    def _run_attempt(
        self,
        request: ModelRequest,
        process: CodexProcessConfig,
        attempt: int,
    ) -> tuple[ModelResult | None, CodexRunnerError | None, bool]:
        owned, creation_failure = _create_owned_child(
            process.tmp_dir,
            expected_root_identity=process._tmp_dir_identity,
        )
        if creation_failure is not None:
            return None, creation_failure, False
        if owned is None:
            return None, CodexExecutionError("Codex execution failed."), False

        outcome: ModelResult | None = None
        failure: CodexRunnerError | None = None
        caught: BaseException | None = None
        try:
            command = _build_command(process, request.config, owned.path)
            process_result = self._executor.run(
                command,
                input=request.prompt,
                cwd=process.working_directory,
                env=process.child_environment,
                timeout=float(request.timeout_seconds),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
                encoding="utf-8",
                errors="strict",
            )
            returncode = _validated_process_returncode(process_result)
            response: str | None = None
            if returncode != 0:
                failure = CodexExecutionError("Codex execution failed.")
            else:
                response, failure = _read_owned_output(
                    owned,
                    request.max_response_bytes,
                )
            if failure is None:
                if response is None:
                    failure = CodexOutputError("Codex output is invalid.")
                else:
                    metadata: dict[str, str | int] = {
                        "workflow": request.config.workflow,
                        "model": request.config.model,
                        "reasoning_effort": request.config.reasoning_effort,
                        "attempt": attempt,
                        "timestamp": _timestamp(),
                        "version": _MODEL_METADATA_VERSION,
                    }
                    if request.config.profile is not None:
                        metadata["profile"] = request.config.profile
                    outcome = ModelResult(
                        response=response,
                        model_metadata=metadata,
                    )
        except BaseException as exc:  # noqa: BLE001 - cleanup covers cancellation
            caught = exc

        cleanup_failure = _cleanup_owned_child_safely(owned)
        if cleanup_failure is not None:
            return None, cleanup_failure, False

        if caught is not None:
            return _classified_attempt_failure(caught)
        if failure is not None:
            return (
                None,
                failure,
                False,
            )
        if outcome is None:
            return None, CodexExecutionError("Codex execution failed."), False
        return outcome, None, False


def _diagnostic_stage(workflow: str) -> WorkflowStage | None:
    return {
        "manual": WorkflowStage.MANUAL,
        "manual_pass": WorkflowStage.MANUAL,
        "highlight": WorkflowStage.HIGHLIGHT,
        "highlighting": WorkflowStage.HIGHLIGHT,
    }.get(workflow)


def _diagnostic_failure_category(error: CodexRunnerError) -> FailureCategory:
    if type(error) is CodexTimeoutError:
        return FailureCategory.TIMEOUT
    if type(error) is CodexConfigurationError:
        return FailureCategory.CONFIG
    if type(error) is CodexOutputError:
        return FailureCategory.OUTPUT
    if type(error) in {CodexCleanupError, CodexCancellationError}:
        return FailureCategory.POLICY
    return FailureCategory.PROCESS


def _snapshot_model_config(value: object) -> CodexModelConfig:
    """Copy one exact inert model configuration without trusting construction."""

    if type(value) is not CodexModelConfig:
        raise CodexConfigurationError("Invalid logical model configuration.")
    missing = False
    model: object = None
    reasoning_effort: object = None
    workflow: object = None
    profile: object = None
    try:
        model = object.__getattribute__(value, "model")
        reasoning_effort = object.__getattribute__(value, "reasoning_effort")
        workflow = object.__getattribute__(value, "workflow")
        profile = object.__getattribute__(value, "profile")
    except (AttributeError, TypeError):
        missing = True
    if (
        missing
        or type(model) is not str
        or type(reasoning_effort) is not str
        or type(workflow) is not str
        or profile is not None
        and type(profile) is not str
    ):
        raise CodexConfigurationError("Invalid logical model configuration.")

    snapshot: CodexModelConfig | None = None
    invalid = False
    try:
        snapshot = CodexModelConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            workflow=workflow,
            profile=profile,
        )
    except BaseException:  # noqa: BLE001 - normalize forged exact objects
        invalid = True
    if invalid or snapshot is None:
        raise CodexConfigurationError("Invalid logical model configuration.")
    return snapshot


def _snapshot_model_request(value: object) -> ModelRequest:
    """Copy and revalidate every exact request field before child creation."""

    if type(value) is not ModelRequest:
        raise CodexConfigurationError("Invalid model request.")
    missing = False
    prompt: object = None
    config: object = None
    timeout_seconds: object = None
    max_attempts: object = None
    max_response_bytes: object = None
    try:
        prompt = object.__getattribute__(value, "prompt")
        config = object.__getattribute__(value, "config")
        timeout_seconds = object.__getattribute__(value, "timeout_seconds")
        max_attempts = object.__getattribute__(value, "max_attempts")
        max_response_bytes = object.__getattribute__(value, "max_response_bytes")
    except (AttributeError, TypeError):
        missing = True
    if (
        missing
        or type(prompt) is not str
        or type(config) is not CodexModelConfig
        or type(timeout_seconds) not in {int, float}
        or type(max_attempts) is not int
        or type(max_response_bytes) is not int
    ):
        raise CodexConfigurationError("Invalid model request.")

    config_snapshot = _snapshot_model_config(config)
    snapshot: ModelRequest | None = None
    invalid = False
    try:
        snapshot = ModelRequest(
            prompt=prompt,
            config=config_snapshot,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            max_response_bytes=max_response_bytes,
        )
    except BaseException:  # noqa: BLE001 - normalize forged exact objects
        invalid = True
    if invalid or snapshot is None:
        raise CodexConfigurationError("Invalid model request.")
    return snapshot


def _request_snapshots_match(
    initial: ModelRequest,
    current: ModelRequest,
) -> bool:
    """Compare only trusted, freshly constructed request snapshots."""

    return bool(
        initial.prompt == current.prompt
        and initial.timeout_seconds == current.timeout_seconds
        and initial.max_attempts == current.max_attempts
        and initial.max_response_bytes == current.max_response_bytes
        and initial.config.model == current.config.model
        and initial.config.reasoning_effort == current.config.reasoning_effort
        and initial.config.workflow == current.config.workflow
        and initial.config.profile == current.config.profile
    )


def _exact_path_identity(value: object) -> tuple[int, int] | None:
    if (
        type(value) is not tuple
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        return None
    return value[0], value[1]


def _snapshot_process_config(value: object) -> CodexProcessConfig:
    """Copy canonical process inputs and re-prove all stored path identities."""

    if type(value) is not CodexProcessConfig:
        raise CodexConfigurationError("Invalid Codex process configuration.")
    missing = False
    executable: object = None
    argv: object = None
    working_directory: object = None
    tmp_dir: object = None
    child_environment: object = None
    executable_identity: object = None
    working_identity: object = None
    tmp_identity: object = None
    try:
        executable = object.__getattribute__(value, "executable")
        argv = object.__getattribute__(value, "argv")
        working_directory = object.__getattribute__(value, "working_directory")
        tmp_dir = object.__getattribute__(value, "tmp_dir")
        child_environment = object.__getattribute__(value, "child_environment")
        executable_identity = object.__getattribute__(
            value,
            "_executable_identity",
        )
        working_identity = object.__getattribute__(
            value,
            "_working_directory_identity",
        )
        tmp_identity = object.__getattribute__(value, "_tmp_dir_identity")
    except (AttributeError, TypeError):
        missing = True

    executable_identity_snapshot = _exact_path_identity(executable_identity)
    working_identity_snapshot = _exact_path_identity(working_identity)
    tmp_identity_snapshot = _exact_path_identity(tmp_identity)
    if (
        missing
        or type(executable) is not _PATH_TYPE
        or type(argv) is not tuple
        or not 1 <= len(argv) <= MAX_BASE_ARGV_PARTS
        or any(type(part) is not str for part in argv)
        or type(working_directory) is not _PATH_TYPE
        or type(tmp_dir) is not _PATH_TYPE
        or type(child_environment) is not MappingProxyType
        or executable_identity_snapshot is None
        or working_identity_snapshot is None
        or tmp_identity_snapshot is None
    ):
        raise CodexConfigurationError("Invalid Codex process configuration.")

    environment: dict[str, str] = {}
    environment_invalid = False
    try:
        referents = gc.get_referents(child_environment)
        if (
            len(referents) != 1
            or type(referents[0]) is not dict
            or len(referents[0]) > MAX_ENVIRONMENT_ENTRIES
        ):
            environment_invalid = True
        else:
            source = referents[0]
            for key, item in dict.items(source):
                if type(key) is not str or type(item) is not str:
                    environment_invalid = True
                    break
                environment[key] = item
    except BaseException:  # noqa: BLE001 - reject active/concurrently changed maps
        environment_invalid = True
    if environment_invalid:
        raise CodexConfigurationError("Invalid Codex process configuration.")

    snapshot: CodexProcessConfig | None = None
    invalid = False
    try:
        snapshot = CodexProcessConfig(
            executable=executable,
            argv=argv,
            working_directory=working_directory,
            tmp_dir=tmp_dir,
            child_environment=environment,
        )
    except BaseException:  # noqa: BLE001 - normalize malformed exact paths
        invalid = True
    if (
        invalid
        or snapshot is None
        or snapshot._executable_identity != executable_identity_snapshot
        or snapshot._working_directory_identity != working_identity_snapshot
        or snapshot._tmp_dir_identity != tmp_identity_snapshot
    ):
        raise CodexConfigurationError("Invalid Codex process configuration.")
    return snapshot


def _validated_process_returncode(value: object) -> int:
    if type(value) is not ProcessResult:
        raise CodexExecutionError("Codex process returned an invalid result.")
    missing = False
    returncode: object = None
    try:
        returncode = object.__getattribute__(value, "returncode")
    except (AttributeError, TypeError):
        missing = True
    if (
        missing
        or type(returncode) is not int
        or not MIN_PROCESS_RETURN_CODE <= returncode <= MAX_PROCESS_RETURN_CODE
    ):
        raise CodexExecutionError("Codex process returned an invalid result.")
    return returncode


def _classified_attempt_failure(
    caught: BaseException,
) -> tuple[None, CodexRunnerError, bool]:
    caught_type = type(caught)
    if caught_type is subprocess.TimeoutExpired:
        return None, CodexTimeoutError("Codex execution timed out."), True
    if caught_type in {TimeoutError, CodexTimeoutError}:
        return None, CodexTimeoutError("Codex execution timed out."), False
    if caught_type in {
        asyncio.CancelledError,
        KeyboardInterrupt,
        CodexCancellationError,
    }:
        return None, CodexCancellationError("Codex execution cancelled."), False
    if caught_type is CodexConfigurationError:
        return None, CodexConfigurationError("Invalid Codex configuration."), False
    if caught_type is CodexOutputError:
        return None, CodexOutputError("Codex output is invalid."), False
    return None, CodexExecutionError("Codex execution failed."), False


def _validate_logical_value(value: object, *, allow_empty: bool) -> None:
    if type(value) is not str:
        raise CodexConfigurationError("Invalid logical model configuration.")
    _validate_safe_scalar_text(
        value,
        maximum_bytes=MAX_LOGICAL_VALUE_BYTES,
        allow_empty=allow_empty,
    )


def _validate_safe_scalar_text(
    value: str,
    *,
    maximum_bytes: int,
    allow_empty: bool,
) -> None:
    encoded = _encoded(value)
    if (
        encoded is None
        or len(encoded) > maximum_bytes
        or not allow_empty
        and not value
        or any(_is_control(character) for character in value)
    ):
        raise CodexConfigurationError("Invalid bounded string value.")


def _validate_content_text(
    value: object,
    *,
    maximum_bytes: int,
    allow_empty: bool,
) -> None:
    if type(value) is not str:
        raise CodexConfigurationError("Invalid bounded content.")
    encoded = _encoded(value)
    if (
        encoded is None
        or len(encoded) > maximum_bytes
        or not allow_empty
        and not value.strip()
        or any(
            _is_control(character) and character not in "\t\n\r" for character in value
        )
    ):
        raise CodexConfigurationError("Invalid bounded content.")


def _encoded(value: str) -> bytes | None:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeError:
        return None


def _is_control(value: str) -> bool:
    return unicodedata.category(value) == "Cc"


def _validate_timeout(value: object) -> None:
    if type(value) is int:
        valid = 0 < value <= MAX_TIMEOUT_SECONDS
    elif type(value) is float:
        valid = math.isfinite(value) and 0 < value <= MAX_TIMEOUT_SECONDS
    else:
        valid = False
    if not valid:
        raise CodexConfigurationError("Invalid model request timeout.")


def _validated_model_metadata(
    value: object,
) -> dict[str, str | int]:
    if type(value) is not dict:
        raise CodexConfigurationError("Invalid model result metadata.")
    expected = {
        "workflow",
        "model",
        "reasoning_effort",
        "attempt",
        "timestamp",
        "version",
    }
    raw_keys = tuple(value.keys())
    if any(type(key) is not str for key in raw_keys):
        raise CodexConfigurationError("Invalid model result metadata.")
    keys = set(raw_keys)
    if "profile" in value:
        expected.add("profile")
    if keys != expected:
        raise CodexConfigurationError("Invalid model result metadata.")
    copied: dict[str, str | int] = {}
    for key in ("workflow", "model", "reasoning_effort", "timestamp"):
        item = value[key]
        if type(item) is not str:
            raise CodexConfigurationError("Invalid model result metadata.")
        _validate_safe_scalar_text(
            item,
            maximum_bytes=MAX_LOGICAL_VALUE_BYTES,
            allow_empty=key in {"model", "reasoning_effort"},
        )
        copied[key] = item
    if "profile" in value:
        profile = value["profile"]
        if type(profile) is not str:
            raise CodexConfigurationError("Invalid model result metadata.")
        _validate_safe_scalar_text(
            profile,
            maximum_bytes=MAX_LOGICAL_VALUE_BYTES,
            allow_empty=False,
        )
        copied["profile"] = profile
    attempt = value["attempt"]
    version = value["version"]
    if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise CodexConfigurationError("Invalid model result metadata.")
    if type(version) is not int or version != _MODEL_METADATA_VERSION:
        raise CodexConfigurationError("Invalid model result metadata.")
    copied["attempt"] = attempt
    copied["version"] = version
    return copied


def _validate_path_value(value: object) -> Path:
    if type(value) is str:
        raw = value
        if not raw or len(raw.encode("utf-8", errors="ignore")) > MAX_ARGV_PART_BYTES:
            raise CodexConfigurationError("Invalid process path configuration.")
        if any(_is_control(character) for character in raw):
            raise CodexConfigurationError("Invalid process path configuration.")
        path = Path(raw)
    elif type(value) is _PATH_TYPE:
        path = value
    else:
        raise CodexConfigurationError("Invalid process path configuration.")
    raw = os.fspath(path)
    encoded = _encoded(raw)
    if (
        encoded is None
        or not raw
        or len(encoded) > MAX_ARGV_PART_BYTES
        or any(_is_control(character) for character in raw)
        or not path.is_absolute()
        or path != Path(os.path.normpath(raw))
    ):
        raise CodexConfigurationError("Invalid process path configuration.")
    return path


def _validate_executable(path: Path) -> None:
    info = _safe_lstat(path)
    if (
        info is None
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or not info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    ):
        raise CodexConfigurationError("Invalid Codex executable configuration.")


def _validate_directory(path: Path) -> None:
    info = _safe_lstat(path)
    if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise CodexConfigurationError("Invalid process directory configuration.")


def _safe_lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except (OSError, ValueError):
        return None


def _required_path_identity(path: Path) -> tuple[int, int]:
    info = _safe_lstat(path)
    if info is None:
        raise CodexConfigurationError("Configured process path is unavailable.")
    return info.st_dev, info.st_ino


def _validated_base_argv(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= MAX_BASE_ARGV_PARTS:
        raise CodexConfigurationError("Invalid Codex argv configuration.")
    validated: list[str] = []
    for index, part in enumerate(value):
        if type(part) is not str:
            raise CodexConfigurationError("Invalid Codex argv configuration.")
        _validate_safe_scalar_text(
            part,
            maximum_bytes=MAX_ARGV_PART_BYTES,
            allow_empty=False,
        )
        if (
            index == 0
            and part != "exec"
            or index > 0
            and (
                part in _RESERVED_BASE_ARGUMENTS
                or part not in _ALLOWED_EXTRA_BASE_ARGUMENTS
            )
        ):
            raise CodexConfigurationError("Invalid Codex argv configuration.")
        validated.append(part)
    return tuple(validated)


def _validated_environment(value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise CodexConfigurationError("Invalid child environment configuration.")
    if len(value) > MAX_ENVIRONMENT_ENTRIES:
        raise CodexConfigurationError("Invalid child environment configuration.")
    copied: dict[str, str] = {}
    total = 0
    for key, item in value.items():
        if type(key) is not str or type(item) is not str:
            raise CodexConfigurationError("Invalid child environment configuration.")
        if not key or "=" in key:
            raise CodexConfigurationError("Invalid child environment configuration.")
        _validate_safe_scalar_text(key, maximum_bytes=256, allow_empty=False)
        _validate_safe_scalar_text(
            item,
            maximum_bytes=4_096,
            allow_empty=True,
        )
        total += len(key.encode("utf-8")) + len(item.encode("utf-8"))
        copied[key] = item
    if total > MAX_ENVIRONMENT_BYTES:
        raise CodexConfigurationError("Invalid child environment configuration.")
    return copied


def _build_command(
    process: CodexProcessConfig,
    model: CodexModelConfig,
    child_path: Path,
) -> tuple[str, ...]:
    parts = [
        os.fspath(process.executable),
        *process.argv,
        "--ask-for-approval",
        "never",
        "--sandbox",
        "read-only",
        "--cd",
        os.fspath(process.working_directory),
        "--output-last-message",
        os.fspath(child_path / _OUTPUT_NAME),
    ]
    if model.model:
        parts.extend(("--model", model.model))
    if model.reasoning_effort:
        parts.extend(
            (
                "-c",
                f"model_reasoning_effort={json.dumps(model.reasoning_effort)}",
            )
        )
    parts.append("-")
    if len(parts) > MAX_COMMAND_PARTS:
        raise CodexConfigurationError("Codex command exceeds its bounded shape.")
    for part in parts:
        if type(part) is not str:
            raise CodexConfigurationError("Codex command is invalid.")
        _validate_safe_scalar_text(
            part,
            maximum_bytes=MAX_ARGV_PART_BYTES,
            allow_empty=False,
        )
    return tuple(parts)


def _create_owned_child(
    root: Path,
    *,
    expected_root_identity: tuple[int, int],
) -> tuple[_OwnedChild | None, CodexRunnerError | None]:
    root_fd: int | None = None
    child_fd: int | None = None
    child_name: str | None = None
    child_identity: tuple[int, int] | None = None
    child_created = False
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(root, flags)
        root_info = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or (root_info.st_dev, root_info.st_ino) != expected_root_identity
        ):
            raise OSError
        for _ in range(16):
            candidate = f".codex-run-{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, 0o700, dir_fd=root_fd)
            except FileExistsError:
                continue
            child_name = candidate
            child_created = True
            created_info = os.stat(
                child_name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(created_info.st_mode):
                raise OSError
            child_identity = (created_info.st_dev, created_info.st_ino)
            break
        if child_name is None:
            raise OSError
        child_fd = os.open(child_name, flags, dir_fd=root_fd)
        child_info = os.fstat(child_fd)
        if (
            not stat.S_ISDIR(child_info.st_mode)
            or (child_info.st_dev, child_info.st_ino) != child_identity
        ):
            raise OSError
        child_path = root / child_name
        return (
            _OwnedChild(
                root_fd=root_fd,
                child_fd=child_fd,
                name=child_name,
                path=child_path,
                root_identity=(root_info.st_dev, root_info.st_ino),
                child_identity=child_identity,
            ),
            None,
        )
    except OSError:
        if child_fd is not None:
            _close_fd(child_fd)
        cleanup_failed = False
        if root_fd is not None:
            if child_name is not None and child_identity is not None:
                try:
                    visible = os.stat(
                        child_name,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                    if (
                        visible.st_dev,
                        visible.st_ino,
                    ) == child_identity and stat.S_ISDIR(visible.st_mode):
                        os.rmdir(child_name, dir_fd=root_fd)
                    else:
                        cleanup_failed = True
                except OSError:
                    cleanup_failed = True
            _close_fd(root_fd)
        if child_created and child_identity is None:
            cleanup_failed = True
        if cleanup_failed:
            return None, CodexCleanupError("Codex temporary workspace cleanup failed.")
        return None, CodexExecutionError("Codex temporary workspace creation failed.")


def _read_owned_output(
    owned: _OwnedChild,
    maximum_bytes: int,
) -> tuple[str | None, CodexRunnerError | None]:
    if not _visible_owned_identity_matches(owned):
        return None, CodexOutputError("Codex output is invalid.")
    try:
        names = os.listdir(owned.child_fd)
    except OSError:
        return None, CodexOutputError("Codex output is invalid.")
    if len(names) > MAX_OUTPUT_FILES or names != [_OUTPUT_NAME]:
        return None, CodexOutputError("Codex output is invalid.")

    output_fd: int | None = None
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        output_fd = os.open(_OUTPUT_NAME, flags, dir_fd=owned.child_fd)
        info = os.fstat(output_fd)
        linked = os.stat(
            _OUTPUT_NAME,
            dir_fd=owned.child_fd,
            follow_symlinks=False,
        )
        identity = (info.st_dev, info.st_ino)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (linked.st_dev, linked.st_ino) != identity
            or not stat.S_ISREG(linked.st_mode)
            or not 0 < info.st_size <= maximum_bytes
        ):
            raise OSError
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(output_fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or len(raw) > maximum_bytes:
            raise OSError
        linked = os.stat(
            _OUTPUT_NAME,
            dir_fd=owned.child_fd,
            follow_symlinks=False,
        )
        if (
            (linked.st_dev, linked.st_ino) != identity
            or not stat.S_ISREG(linked.st_mode)
            or linked.st_nlink != 1
        ):
            raise OSError
        response = raw.decode("utf-8", errors="strict")
        _validate_content_text(
            response,
            maximum_bytes=maximum_bytes,
            allow_empty=False,
        )
        return response, None
    except (CodexRunnerError, OSError, UnicodeError):
        return None, CodexOutputError("Codex output is invalid.")
    finally:
        if output_fd is not None:
            _close_fd(output_fd)


def _visible_owned_identity_matches(owned: _OwnedChild) -> bool:
    root_info = _safe_lstat(owned.path.parent)
    child_info = _safe_lstat(owned.path)
    return bool(
        root_info is not None
        and child_info is not None
        and (root_info.st_dev, root_info.st_ino) == owned.root_identity
        and (child_info.st_dev, child_info.st_ino) == owned.child_identity
        and stat.S_ISDIR(child_info.st_mode)
    )


def _cleanup_owned_child(owned: _OwnedChild) -> CodexCleanupError | None:
    cleanup_ok = True
    try:
        linked = os.stat(
            owned.name,
            dir_fd=owned.root_fd,
            follow_symlinks=False,
        )
        if (
            linked.st_dev,
            linked.st_ino,
        ) != owned.child_identity or not _purge_directory_fd(
            owned.child_fd,
            depth=0,
            remaining=[_MAX_CLEANUP_ENTRIES],
        ):
            cleanup_ok = False
    except OSError:
        cleanup_ok = False

    _close_fd(owned.child_fd)
    if cleanup_ok:
        try:
            linked = os.stat(
                owned.name,
                dir_fd=owned.root_fd,
                follow_symlinks=False,
            )
            if (linked.st_dev, linked.st_ino) != owned.child_identity:
                cleanup_ok = False
            else:
                os.rmdir(owned.name, dir_fd=owned.root_fd)
        except OSError:
            cleanup_ok = False
    _close_fd(owned.root_fd)
    if cleanup_ok:
        return None
    return CodexCleanupError("Codex temporary workspace cleanup failed.")


def _cleanup_owned_child_safely(
    owned: _OwnedChild,
) -> CodexCleanupError | None:
    """Normalize every cleanup path and make cleanup failure dominant."""

    result: object = None
    cleanup_raised = False
    try:
        result = _cleanup_owned_child(owned)
    except BaseException:  # noqa: BLE001 - cleanup must dominate all failures
        cleanup_raised = True
    if cleanup_raised:
        for descriptor in (owned.child_fd, owned.root_fd):
            try:
                os.close(descriptor)
            except BaseException:  # noqa: BLE001 - best-effort descriptor close
                pass
        return CodexCleanupError("Codex temporary workspace cleanup failed.")
    if result is None:
        return None
    return CodexCleanupError("Codex temporary workspace cleanup failed.")


def _purge_directory_fd(
    directory_fd: int,
    *,
    depth: int,
    remaining: list[int],
) -> bool:
    if depth > _MAX_CLEANUP_DEPTH:
        return False
    try:
        names = os.listdir(directory_fd)
    except OSError:
        return False
    if len(names) > remaining[0]:
        return False
    remaining[0] -= len(names)
    for name in names:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                nested_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    nested_info = os.fstat(nested_fd)
                    if (nested_info.st_dev, nested_info.st_ino) != (
                        info.st_dev,
                        info.st_ino,
                    ):
                        return False
                    if not _purge_directory_fd(
                        nested_fd,
                        depth=depth + 1,
                        remaining=remaining,
                    ):
                        return False
                finally:
                    _close_fd(nested_fd)
                os.rmdir(name, dir_fd=directory_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)
        except OSError:
            return False
    return True


def _close_fd(value: int) -> None:
    try:
        os.close(value)
    except OSError:
        pass


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "CodexCancellationError",
    "CodexCleanupError",
    "CodexConfigurationError",
    "CodexExecutionError",
    "CodexModelConfig",
    "CodexOutputError",
    "CodexProcessConfig",
    "CodexProcessRunner",
    "CodexRunnerError",
    "CodexTimeoutError",
    "ModelRequest",
    "ModelResult",
    "ModelRunner",
    "ProcessExecutor",
    "ProcessResult",
    "SubprocessExecutor",
    "resolve_codex_model_config",
]
