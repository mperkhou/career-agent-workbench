"""Content-free diagnostics for bounded resume workflows."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from enum import StrEnum
from typing import TextIO

INVOCATION_SOURCE_ENV = "CAREER_AGENT_WORKBENCH_INVOCATION_SOURCE"
MAX_DIAGNOSTIC_BYTES = 4_096


class ConfigurationSource(StrEnum):
    """Closed provenance vocabulary for effective workflow settings."""

    CLI = "cli"
    MAKE = "make"
    PROCESS = "process"
    PRIVATE_DOTENV = "private_dotenv"
    DEFAULT = "default"


class WorkflowStage(StrEnum):
    """Closed public stage labels used by resume workflow diagnostics."""

    V1_CORE = "v1_core"
    V1_JOD = "v1_jod"
    V1_EXPERIENCE = "v1_experience"
    V2_CRITIQUE = "v2_critique"
    MANUAL = "manual"
    HIGHLIGHT = "highlight"


class DiagnosticEvent(StrEnum):
    """Closed event vocabulary accepted by stderr and status persistence."""

    CONFIGURATION = "configuration"
    ATTEMPT_START = "attempt_start"
    ATTEMPT_ELAPSED = "attempt_elapsed"
    ATTEMPT_COMPLETION = "attempt_completion"
    TIMEOUT = "timeout"
    RETRY_DECISION = "retry_decision"
    FAILURE = "failure"


class FailureCategory(StrEnum):
    """Content-free workflow failure families."""

    TIMEOUT = "timeout"
    MODEL = "model"
    PARSE = "parse"
    POLICY = "policy"
    RENDER = "render"
    ATS = "ats"
    STATE = "state"
    ARTIFACT = "artifact"
    CONFIG = "config"
    PROCESS = "process"
    OUTPUT = "output"
    LOCAL_IO = "local_io"
    UNEXPECTED = "unexpected"


def invocation_argument_source(
    environ: Mapping[str, str] | None = None,
) -> ConfigurationSource:
    """Return the safe source label for an explicitly supplied CLI argument."""

    values = os.environ if environ is None else environ
    return (
        ConfigurationSource.MAKE
        if values.get(INVOCATION_SOURCE_ENV) == ConfigurationSource.MAKE.value
        else ConfigurationSource.CLI
    )


def configuration_event(
    *,
    stage: WorkflowStage,
    model: str,
    effort: str,
    timeout_seconds: float,
    retry_count: int,
    sources: Mapping[str, str | ConfigurationSource],
    workspace_configured: bool,
    profile: str | None = None,
) -> dict[str, object]:
    """Build one validated, content-free effective-configuration event."""

    if type(model) is not str or not model or not _safe_label(model):
        raise ValueError("Workflow diagnostic configuration is invalid.")
    rendered_effort = effort if effort else "inherit"
    if not _safe_label(rendered_effort):
        raise ValueError("Workflow diagnostic configuration is invalid.")
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or type(retry_count) is not int
        or not 0 <= retry_count <= 3
        or type(workspace_configured) is not bool
    ):
        raise ValueError("Workflow diagnostic configuration is invalid.")
    expected_sources = {"model", "effort", "timeout", "retry_count"}
    if set(sources) != expected_sources:
        raise ValueError("Workflow diagnostic configuration is invalid.")
    normalized_sources: dict[str, str] = {}
    for key in sorted(expected_sources):
        try:
            normalized_sources[key] = ConfigurationSource(sources[key]).value
        except (TypeError, ValueError):
            raise ValueError("Workflow diagnostic configuration is invalid.") from None
    event: dict[str, object] = {
        "event": DiagnosticEvent.CONFIGURATION.value,
        "stage": stage.value,
        "model": model,
        "effort": rendered_effort,
        "timeout_seconds": float(timeout_seconds),
        "retry_count": retry_count,
        "total_attempts": retry_count + 1,
        "sources": normalized_sources,
        "workspace_configured": workspace_configured,
    }
    if profile is not None:
        if not _safe_label(profile):
            raise ValueError("Workflow diagnostic configuration is invalid.")
        event["profile"] = profile
    return event


def attempt_event(
    *,
    event: DiagnosticEvent,
    stage: WorkflowStage,
    attempt: int,
    total_attempts: int,
    elapsed_seconds: float | None = None,
    retry: bool | None = None,
    category: FailureCategory | None = None,
) -> dict[str, object]:
    """Build one bounded attempt lifecycle event without content values."""

    if event is DiagnosticEvent.CONFIGURATION:
        raise ValueError("Workflow diagnostic event is invalid.")
    if (
        type(attempt) is not int
        or type(total_attempts) is not int
        or not 1 <= attempt <= total_attempts <= 4
    ):
        raise ValueError("Workflow diagnostic event is invalid.")
    result: dict[str, object] = {
        "event": event.value,
        "stage": stage.value,
        "attempt": attempt,
        "total_attempts": total_attempts,
    }
    if elapsed_seconds is not None:
        if (
            type(elapsed_seconds) not in {int, float}
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0
        ):
            raise ValueError("Workflow diagnostic event is invalid.")
        result["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    if retry is not None:
        if type(retry) is not bool:
            raise ValueError("Workflow diagnostic event is invalid.")
        result["retry"] = retry
    if category is not None:
        result["category"] = category.value
    return result


def emit_diagnostic(
    event: Mapping[str, object], *, stream: TextIO | None = None
) -> None:
    """Write one bounded JSON line to stderr while preserving stdout APIs."""

    encoded = json.dumps(dict(event), sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("Workflow diagnostic event is invalid.")
    target = sys.stderr if stream is None else stream
    target.write(f"{encoded}\n")
    target.flush()


def sanitized_diagnostic_event(value: object) -> dict[str, object] | None:
    """Return a defensive copy of one valid emitted event, else ``None``."""

    if not isinstance(value, Mapping):
        return None
    try:
        event = DiagnosticEvent(value.get("event"))
        stage = WorkflowStage(value.get("stage"))
    except (TypeError, ValueError):
        return None
    if event is DiagnosticEvent.CONFIGURATION:
        expected = {
            "event",
            "stage",
            "model",
            "effort",
            "timeout_seconds",
            "retry_count",
            "total_attempts",
            "sources",
            "workspace_configured",
        }
        if "profile" in value:
            expected.add("profile")
        if set(value) != expected:
            return None
        try:
            normalized = configuration_event(
                stage=stage,
                model=value["model"],
                effort=("" if value["effort"] == "inherit" else value["effort"]),
                timeout_seconds=value["timeout_seconds"],
                retry_count=value["retry_count"],
                sources=value["sources"],
                workspace_configured=value["workspace_configured"],
                profile=value.get("profile"),
            )
        except (TypeError, ValueError):
            return None
        return (
            normalized
            if normalized.get("total_attempts") == value["total_attempts"]
            else None
        )

    allowed = {"event", "stage", "attempt", "total_attempts"}
    if "elapsed_seconds" in value:
        allowed.add("elapsed_seconds")
    if "retry" in value:
        allowed.add("retry")
    if "category" in value:
        allowed.add("category")
    if set(value) != allowed:
        return None
    try:
        normalized = attempt_event(
            event=event,
            stage=stage,
            attempt=value["attempt"],
            total_attempts=value["total_attempts"],
            elapsed_seconds=value.get("elapsed_seconds"),
            retry=value.get("retry"),
            category=(
                None
                if value.get("category") is None
                else FailureCategory(value["category"])
            ),
        )
    except (TypeError, ValueError):
        return None
    return normalized


def _safe_label(value: object) -> bool:
    return bool(
        type(value) is str
        and value
        and len(value.encode("utf-8", errors="ignore")) <= 256
        and all(character.isprintable() for character in value)
        and not value.startswith(("/", "."))
        and ".." not in value
        and "://" not in value
        and "\\" not in value
    )


__all__ = [
    "ConfigurationSource",
    "DiagnosticEvent",
    "FailureCategory",
    "INVOCATION_SOURCE_ENV",
    "WorkflowStage",
    "attempt_event",
    "configuration_event",
    "emit_diagnostic",
    "invocation_argument_source",
    "sanitized_diagnostic_event",
]
