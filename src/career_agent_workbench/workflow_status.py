"""Bounded persistence for sanitized local workflow status events."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

from career_agent_workbench.errors import ModelFailureSubtype
from career_agent_workbench.workflow_diagnostics import sanitized_diagnostic_event

MAX_STATUS_RUNS = 8
MAX_STATUS_EVENTS = 160
MAX_STATUS_FILE_BYTES = 262_144
STATUS_DIRECTORY_NAME = "workflow-status"
_RUN_FILE_SUFFIX = ".json"
_LOCAL_EVENTS = frozenset(
    {
        "queued",
        "stage_start",
        "stage_completion",
        "stage_failure",
        "workflow_completion",
        "workflow_failure",
        "partial_failure",
        "seed_empty",
        "restart_recovered",
    }
)
_LOCAL_STAGES = frozenset({"workflow", "seed", "v1", "v2", "manual", "highlight"})
_LOCAL_CATEGORIES = frozenset(
    {"none", "config", "process", "timeout", "workflow", "unexpected"}
)


class WorkflowStatusError(Exception):
    """Raised when sanitized status persistence cannot be proven safe."""

    __slots__ = ()


class WorkflowStatusStore:
    """Atomic, no-follow store for a bounded set of sanitized run records."""

    __slots__ = ("_directory", "_identity")

    def __init__(self, tmp_dir: Path) -> None:
        self._directory = _prepare_status_directory(tmp_dir)
        self._identity = _path_identity(self._directory)
        directory_fd = self._open_directory()
        try:
            if self._clean_temporary_files_locked(directory_fd):
                os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def save(self, run: Mapping[str, object]) -> None:
        """Validate and atomically persist one complete run snapshot."""

        normalized = _normalized_run(run)
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_STATUS_FILE_BYTES:
            raise WorkflowStatusError("Workflow status record is too large.")
        run_id = str(normalized["id"])
        directory_fd = self._open_directory()
        temporary_name = f".{run_id}.{secrets.token_hex(8)}.tmp"
        file_fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            file_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
            os.fchmod(file_fd, 0o600)
            _write_all(file_fd, encoded)
            os.fsync(file_fd)
            os.close(file_fd)
            file_fd = None
            target_name = f"{run_id}{_RUN_FILE_SUFFIX}"
            _validate_existing_target(directory_fd, target_name)
            os.replace(
                temporary_name,
                target_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
            self._trim_locked(directory_fd)
            os.fsync(directory_fd)
        except WorkflowStatusError:
            raise
        except OSError:
            raise WorkflowStatusError(
                "Workflow status could not be persisted."
            ) from None
        finally:
            if file_fd is not None:
                try:
                    os.close(file_fd)
                except OSError:
                    pass
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
            os.close(directory_fd)

    def load(self) -> list[dict[str, object]]:
        """Return valid retained runs, ignoring malformed or replaced files."""

        directory_fd = self._open_directory()
        try:
            runs = [
                run
                for name in _run_names(directory_fd)
                if (run := _read_run(directory_fd, name)) is not None
            ]
        finally:
            os.close(directory_fd)
        runs.sort(key=lambda item: str(item["started_at"]), reverse=True)
        return runs[:MAX_STATUS_RUNS]

    def metadata(self, run_id: str) -> dict[str, object]:
        """Expose availability/count metadata without revealing a path."""

        if not _valid_run_id(run_id):
            return {"log_available": False, "log_event_count": 0}
        directory_fd = self._open_directory()
        try:
            run = _read_run(directory_fd, f"{run_id}{_RUN_FILE_SUFFIX}")
        finally:
            os.close(directory_fd)
        events = run.get("events", []) if run is not None else []
        return {
            "log_available": run is not None,
            "log_event_count": len(events) if isinstance(events, list) else 0,
        }

    def _open_directory(self) -> int:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._directory, flags)
            metadata = os.fstat(descriptor)
        except OSError:
            raise WorkflowStatusError(
                "Workflow status directory is unavailable."
            ) from None
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._identity
        ):
            os.close(descriptor)
            raise WorkflowStatusError("Workflow status directory is unavailable.")
        return descriptor

    def _trim_locked(self, directory_fd: int) -> None:
        retained: list[tuple[str, str]] = []
        for name in _run_names(directory_fd):
            run = _read_run(directory_fd, name)
            if run is not None:
                retained.append((str(run["started_at"]), name))
        retained.sort(reverse=True)
        for _started_at, name in retained[MAX_STATUS_RUNS:]:
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    os.unlink(name, dir_fd=directory_fd)
            except OSError:
                continue

    def _clean_temporary_files_locked(self, directory_fd: int) -> bool:
        removed = False
        try:
            names = os.listdir(directory_fd)
        except OSError:
            return False
        for name in names:
            if not _valid_temporary_name(name):
                continue
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    os.unlink(name, dir_fd=directory_fd)
                    removed = True
            except OSError:
                continue
        return removed


def local_status_event(
    event: str,
    *,
    timestamp: str,
    stage: str = "workflow",
    category: str = "none",
    count: int | None = None,
) -> dict[str, object]:
    """Build one closed, content-free Flask status event."""

    candidate: dict[str, object] = {
        "event": event,
        "timestamp": timestamp,
        "stage": stage,
        "category": category,
    }
    if count is not None:
        candidate["count"] = count
    normalized = _normalized_local_event(candidate)
    if normalized is None:
        raise WorkflowStatusError("Workflow status event is invalid.")
    return normalized


def normalized_status_event(value: object) -> dict[str, object] | None:
    """Return one safe local or child diagnostic event, else ``None``."""

    local = _normalized_local_event(value)
    if local is not None:
        return local
    if not isinstance(value, Mapping):
        return None
    timestamp = value.get("timestamp")
    if timestamp is None:
        return sanitized_diagnostic_event(value)
    if not _safe_text(timestamp, maximum=40):
        return None
    diagnostic = sanitized_diagnostic_event(
        {key: item for key, item in value.items() if key != "timestamp"}
    )
    if diagnostic is None:
        return None
    return {**diagnostic, "timestamp": timestamp}


def captured_diagnostic_event(
    value: object,
    *,
    timestamp: str,
) -> dict[str, object] | None:
    """Capture one valid child diagnostic with a local receipt timestamp."""

    if not _safe_text(timestamp, maximum=40):
        return None
    diagnostic = sanitized_diagnostic_event(value)
    if diagnostic is None:
        return None
    return {**diagnostic, "timestamp": timestamp}


def status_event_message(event: Mapping[str, object]) -> str:
    """Render one sanitized event into a compact local-UI message."""

    timestamp = str(event.get("timestamp") or "")
    prefix = timestamp[11:19] if len(timestamp) >= 19 else "--:--:--"
    kind = str(event.get("event") or "")
    stage = str(event.get("stage") or "workflow").replace("_", " ")
    attempt = event.get("attempt")
    total = event.get("total_attempts")
    category = str(event.get("category") or "")
    subtype = _failure_subtype_label(event)
    if kind == "configuration":
        return (
            f"{prefix} {stage} configured: {event.get('model')}, "
            f"effort {event.get('effort')}, {event.get('timeout_seconds')}s, "
            f"{event.get('total_attempts')} attempt(s)."
        )
    if kind == "attempt_start":
        return f"{prefix} {stage} attempt {attempt}/{total} started."
    if kind == "attempt_completion":
        return f"{prefix} {stage} attempt {attempt}/{total} completed."
    if kind == "attempt_elapsed":
        return f"{prefix} {stage} attempt {attempt}/{total} elapsed {event.get('elapsed_seconds')}s."
    if kind == "timeout":
        suffix = f" ({subtype})" if subtype is not None else ""
        return f"{prefix} {stage} attempt {attempt}/{total} timed out{suffix}."
    if kind == "retry_decision":
        decision = "retrying" if event.get("retry") is True else "not retrying"
        suffix = f" ({subtype})" if subtype is not None else ""
        return f"{prefix} {stage} attempt {attempt}/{total}: {decision}{suffix}."
    if kind == "failure":
        detail = category or "unexpected"
        if subtype is not None:
            detail = f"{detail}; {subtype}"
        return f"{prefix} {stage} failed ({detail})."
    if kind == "queued":
        return f"{prefix} Background action queued."
    if kind == "stage_start":
        return f"{prefix} {stage} stage started for {event.get('count', 0)} job(s)."
    if kind == "stage_completion":
        return f"{prefix} {stage} stage completed."
    if kind == "stage_failure":
        return f"{prefix} {stage} stage failed ({category})."
    if kind == "partial_failure":
        return (
            f"{prefix} Workflow completed with {event.get('count', 0)} failed stage(s)."
        )
    if kind == "seed_empty":
        return f"{prefix} Seed completed without new jobs."
    if kind == "restart_recovered":
        return f"{prefix} Interrupted run recovered after restart."
    if kind == "workflow_completion":
        return f"{prefix} Background action completed."
    return f"{prefix} Background action failed ({category or 'unexpected'})."


def _failure_subtype_label(event: Mapping[str, object]) -> str | None:
    value = event.get("failure_subtype")
    if value is None:
        return None
    try:
        subtype = ModelFailureSubtype(value)
    except (TypeError, ValueError):
        return None
    return subtype.value.replace("_", " ")


def _prepare_status_directory(tmp_dir: Path) -> Path:
    if type(tmp_dir) is not type(Path()) or not tmp_dir.is_absolute():
        raise WorkflowStatusError("Workflow status directory is invalid.")
    try:
        if not tmp_dir.exists():
            tmp_dir.mkdir(mode=0o700)
        tmp_metadata = tmp_dir.lstat()
        if not stat.S_ISDIR(tmp_metadata.st_mode) or tmp_dir.is_symlink():
            raise OSError
        directory = tmp_dir / STATUS_DIRECTORY_NAME
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink():
            raise OSError
        os.chmod(directory, 0o700, follow_symlinks=False)
        return directory
    except OSError:
        raise WorkflowStatusError("Workflow status directory is invalid.") from None


def _path_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError:
        raise WorkflowStatusError("Workflow status directory is unavailable.") from None
    return metadata.st_dev, metadata.st_ino


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError
        view = view[written:]


def _validate_existing_target(directory_fd: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise WorkflowStatusError("Workflow status target is invalid.") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise WorkflowStatusError("Workflow status target is invalid.")


def _run_names(directory_fd: int) -> list[str]:
    try:
        names = os.listdir(directory_fd)
    except OSError:
        return []
    return sorted(
        name
        for name in names
        if name.endswith(_RUN_FILE_SUFFIX)
        and _valid_run_id(name.removesuffix(_RUN_FILE_SUFFIX))
    )


def _read_run(directory_fd: int, name: str) -> dict[str, object] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_STATUS_FILE_BYTES
        ):
            return None
        data = bytearray()
        while len(data) <= MAX_STATUS_FILE_BYTES:
            chunk = os.read(
                descriptor, min(65_536, MAX_STATUS_FILE_BYTES + 1 - len(data))
            )
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAX_STATUS_FILE_BYTES:
            return None
        return _normalized_run(json.loads(bytes(data)))
    except (OSError, ValueError, TypeError, UnicodeError, WorkflowStatusError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _normalized_run(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "id",
        "title",
        "status",
        "started_at",
        "finished_at",
        "return_code",
        "events",
    }:
        raise WorkflowStatusError("Workflow status record is invalid.")
    run_id = value["id"]
    title = value["title"]
    status_value = value["status"]
    started_at = value["started_at"]
    finished_at = value["finished_at"]
    return_code = value["return_code"]
    events_value = value["events"]
    if (
        not _valid_run_id(run_id)
        or not _safe_text(title, maximum=160)
        or status_value not in {"running", "completed", "failed"}
        or not _safe_text(started_at, maximum=40)
        or finished_at is not None
        and not _safe_text(finished_at, maximum=40)
        or return_code is not None
        and (type(return_code) is not int or not -255 <= return_code <= 255)
        or not isinstance(events_value, list)
    ):
        raise WorkflowStatusError("Workflow status record is invalid.")
    events: list[dict[str, object]] = []
    for item in events_value[-MAX_STATUS_EVENTS:]:
        normalized = normalized_status_event(item)
        if normalized is None:
            raise WorkflowStatusError("Workflow status record is invalid.")
        events.append(normalized)
    return {
        "id": run_id,
        "title": title,
        "status": status_value,
        "started_at": started_at,
        "finished_at": finished_at,
        "return_code": return_code,
        "events": events,
    }


def _normalized_local_event(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = {"event", "timestamp", "stage", "category", "count"}
    if not set(value).issubset(allowed) or not {
        "event",
        "timestamp",
        "stage",
        "category",
    }.issubset(value):
        return None
    event = value["event"]
    timestamp = value["timestamp"]
    stage = value["stage"]
    category = value["category"]
    count = value.get("count")
    if (
        event not in _LOCAL_EVENTS
        or stage not in _LOCAL_STAGES
        or category not in _LOCAL_CATEGORIES
        or not _safe_text(timestamp, maximum=40)
        or count is not None
        and (type(count) is not int or not 0 <= count <= 10_000)
    ):
        return None
    result: dict[str, object] = {
        "event": event,
        "timestamp": timestamp,
        "stage": stage,
        "category": category,
    }
    if count is not None:
        result["count"] = count
    return result


def _valid_run_id(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_temporary_name(value: object) -> bool:
    if (
        type(value) is not str
        or not value.startswith(".")
        or not value.endswith(".tmp")
    ):
        return False
    parts = value[1:-4].split(".")
    return bool(
        len(parts) == 2
        and _valid_run_id(parts[0])
        and len(parts[1]) == 16
        and all(character in "0123456789abcdef" for character in parts[1])
    )


def _safe_text(value: object, *, maximum: int) -> bool:
    return bool(
        type(value) is str
        and value
        and len(value.encode("utf-8", errors="ignore")) <= maximum
        and all(character.isprintable() for character in value)
        and "\\" not in value
        and "://" not in value
    )


__all__ = [
    "MAX_STATUS_EVENTS",
    "MAX_STATUS_RUNS",
    "WorkflowStatusError",
    "WorkflowStatusStore",
    "captured_diagnostic_event",
    "local_status_event",
    "normalized_status_event",
    "status_event_message",
]
