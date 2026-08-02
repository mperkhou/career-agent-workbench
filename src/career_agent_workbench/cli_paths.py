"""Shared post-parse runtime and process composition for command-line tools."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

from career_agent_workbench import __version__
from career_agent_workbench.codex_cli import (
    CodexProcessConfig,
    CodexProcessRunner,
    ModelRequest,
    ModelResult,
    ModelRunner,
    SubprocessExecutor,
)
from career_agent_workbench.config import (
    RuntimeConfig,
    RuntimeOverrides,
    WorkspaceMember,
    WorkspacePaths,
    load_runtime_config,
)


class CliConfigurationError(Exception):
    """Stable, content-free command composition failure."""

    __slots__ = ()


_PATH_ARGUMENTS: Mapping[str, tuple[tuple[str, ...], str]] = {
    "workspace": (("--workspace",), "Workspace root override."),
    "database": (("--database",), "Tracker database override."),
    "output_dir": (("--output-dir",), "Output directory override."),
    "profile_dir": (("--profile-dir",), "Profile directory override."),
    "master_resume": (("--master-resume",), "Master resume YAML override."),
    "master_resume_text": (
        ("--master-resume-text",),
        "Master resume source-text override.",
    ),
    "blacklist": (
        ("--blacklist-path", "--blacklist"),
        "Company blacklist override.",
    ),
    "tmp_dir": (("--tmp-dir",), "Temporary directory override."),
    "download_dir": (("--download-dir",), "Private download directory override."),
}


def add_runtime_path_arguments(
    parser: argparse.ArgumentParser,
    *members: str,
) -> None:
    """Add requested state-path flags with explicit ``None`` defaults."""

    if not any("--version" in action.option_strings for action in parser._actions):
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
        )
    for member in members:
        try:
            option_strings, help_text = _PATH_ARGUMENTS[member]
        except KeyError:
            raise CliConfigurationError(
                "Command path configuration is invalid."
            ) from None
        parser.add_argument(
            *option_strings,
            dest=member,
            type=Path,
            default=None,
            help=help_text,
        )


def runtime_overrides_from_namespace(
    args: argparse.Namespace,
    *,
    database: Path | None = None,
    master_resume: Path | None = None,
    setting_overrides: Mapping[str, object] | None = None,
) -> RuntimeOverrides:
    """Copy only declared parser values into the central override object."""

    values: dict[str, object] = {}
    for field_name in (
        "workspace",
        "profile_dir",
        "master_resume",
        "master_resume_text",
        "output_dir",
        "database",
        "blacklist",
        "tmp_dir",
        "download_dir",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            values[field_name] = value
    if database is not None:
        values["database"] = database
    if master_resume is not None:
        values["master_resume"] = master_resume
    for key, value in (setting_overrides or {}).items():
        if value is not None:
            values[key] = value
    try:
        return RuntimeOverrides(**values)
    except TypeError:
        raise CliConfigurationError("Command configuration is invalid.") from None


def load_command_config(
    args: argparse.Namespace,
    *,
    required: Iterable[WorkspaceMember | str] = (),
    database: Path | None = None,
    master_resume: Path | None = None,
    setting_overrides: Mapping[str, object] | None = None,
) -> RuntimeConfig:
    """Load central runtime configuration exactly once after argument parsing."""

    try:
        config = load_runtime_config(
            overrides=runtime_overrides_from_namespace(
                args,
                database=database,
                master_resume=master_resume,
                setting_overrides=setting_overrides,
            )
        )
        for member in required:
            config.paths.require(member)
        return config
    except CliConfigurationError:
        raise
    except Exception:
        raise CliConfigurationError("Command configuration is invalid.") from None


def seed_job_derived_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    """Return the two explicitly bounded cutoff composition derivations."""

    database = getattr(args, "database", None)
    output_dir = getattr(args, "output_dir", None)
    if database is None and output_dir is not None:
        database = output_dir / "tracking" / "applications.sqlite3"

    master_resume = getattr(args, "master_resume", None)
    profile_dir = getattr(args, "profile_dir", None)
    master_resume_name = getattr(args, "master_resume_name", None)
    if master_resume is None and profile_dir is not None:
        if (
            type(master_resume_name) is not str
            or not master_resume_name
            or master_resume_name in {".", ".."}
            or "/" in master_resume_name
            or "\\" in master_resume_name
        ):
            raise CliConfigurationError("Command configuration is invalid.")
        name = Path(master_resume_name)
        if (
            name.is_absolute()
            or len(name.parts) != 1
            or name.name != master_resume_name
            or name.name in {"", ".", ".."}
        ):
            raise CliConfigurationError("Command configuration is invalid.")
        master_resume = profile_dir / name
    return database, master_resume


def reject_compatibility_path(value: Path | None) -> None:
    """Reject legacy artifact directories that could retain model traffic."""

    if value is not None:
        raise CliConfigurationError("Compatibility artifact output is disabled.")


def resolve_private_workspace_path(
    paths: WorkspacePaths,
    value: Path,
    *,
    must_exist: bool = False,
    directory: bool = False,
) -> Path:
    """Resolve one explicit input/output beneath the configured private root."""

    try:
        root = paths.require(WorkspaceMember.ROOT)
        if type(value) is not type(Path()) or not root.is_absolute():
            raise ValueError
        candidate = value if value.is_absolute() else root / value
        if root.is_symlink():
            raise ValueError
        lexical_root = Path(os.path.abspath(root))
        lexical_candidate = Path(os.path.abspath(candidate))
        relative = lexical_candidate.relative_to(lexical_root)
        current = lexical_root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError
        selected_root = root.resolve(strict=True)
        selected = candidate.resolve(strict=False)
        selected.relative_to(selected_root)
        if selected == selected_root:
            raise ValueError
        module_root = Path(__file__).resolve(strict=False).parents[2]
        if (module_root / "pyproject.toml").is_file():
            try:
                selected.relative_to(module_root)
            except ValueError:
                pass
            else:
                raise ValueError
        if must_exist and (not selected.is_file() or selected.is_symlink()):
            raise ValueError
        if directory and selected.exists() and not selected.is_dir():
            raise ValueError
        if not directory and not must_exist and selected.exists() and selected.is_dir():
            raise ValueError
        return selected
    except Exception:  # noqa: BLE001 - sanitize caller-owned path details
        raise CliConfigurationError("Private workspace path is invalid.") from None


def build_codex_runner(
    *,
    command: str,
    working_directory: Path,
    tmp_dir: Path,
) -> CodexProcessRunner:
    """Create the bounded production runner after configuration resolution."""

    try:
        parts = shlex.split(command)
    except ValueError:
        raise CliConfigurationError("Codex command configuration is invalid.") from None
    if not parts:
        raise CliConfigurationError("Codex command configuration is invalid.")
    executable = Path(parts[0])
    if not executable.is_absolute():
        located = shutil.which(parts[0])
        if located is None:
            raise CliConfigurationError("Codex command configuration is invalid.")
        executable = Path(located)
    try:
        executable = executable.resolve(strict=True)
    except OSError:
        raise CliConfigurationError("Codex command configuration is invalid.") from None
    extra = tuple(parts[1:])
    if extra not in {(), ("exec",), ("exec", "--skip-git-repo-check")}:
        raise CliConfigurationError("Codex command configuration is invalid.")
    argv = extra if extra else ("exec",)
    child_environment = {
        key: value
        for key in ("HOME", "PATH", "LANG", "LC_ALL", "TERM", "CODEX_HOME")
        if (value := os.environ.get(key)) is not None
    }
    try:
        return CodexProcessRunner(
            config=CodexProcessConfig(
                executable=executable,
                argv=argv,
                working_directory=working_directory,
                tmp_dir=tmp_dir,
                child_environment=child_environment,
            ),
            executor=SubprocessExecutor(),
        )
    except Exception:
        raise CliConfigurationError("Codex command configuration is invalid.") from None


class _RequestPolicyRunner:
    __slots__ = ("_attempts", "_runner", "_timeout")

    def __init__(
        self,
        runner: ModelRunner,
        *,
        timeout_seconds: float,
        retry_count: int,
    ) -> None:
        self._runner = runner
        self._timeout = timeout_seconds
        self._attempts = retry_count + 1

    def run(self, request: ModelRequest, /) -> ModelResult:
        return self._runner.run(
            ModelRequest(
                prompt=request.prompt,
                config=request.config,
                timeout_seconds=self._timeout,
                max_attempts=self._attempts,
                max_response_bytes=request.max_response_bytes,
            )
        )


def with_model_request_policy(
    runner: ModelRunner,
    *,
    timeout_seconds: float,
    retry_count: int,
) -> ModelRunner:
    """Apply bounded CLI timeout/retry options without changing domain code."""

    if not 0 < timeout_seconds <= 1_800 or not 0 <= retry_count <= 3:
        raise CliConfigurationError("Model execution configuration is invalid.")
    return _RequestPolicyRunner(
        runner,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
    )


__all__ = [
    "CliConfigurationError",
    "add_runtime_path_arguments",
    "build_codex_runner",
    "load_command_config",
    "reject_compatibility_path",
    "resolve_private_workspace_path",
    "runtime_overrides_from_namespace",
    "seed_job_derived_paths",
    "with_model_request_policy",
]
