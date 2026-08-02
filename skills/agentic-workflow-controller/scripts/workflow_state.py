#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = SKILL_DIR / "assets"
TRACKER_TEMPLATE = ASSETS_DIR / "workflow-tracker.template.json"
PLAN_TEMPLATE = ASSETS_DIR / "workflow-plan.template.md"
ARTIFACT_MANIFEST_SCHEMA = ASSETS_DIR / "artifact-manifest.schema.json"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "tmp" / "agentic-workflows"
DEFAULT_EXTERNAL_ARTIFACT_ROOT = (
    Path(os.environ.get("TMPDIR", "/tmp")) / "career-agent-workbench-agentic"
)

REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "schema_version": int,
    "workflow_id": str,
    "release_version": str,
    "objective": str,
    "state": str,
    "current_step": (str, type(None)),
    "planned_steps": list,
    "completed_steps": list,
    "gates": list,
    "pause_conditions": list,
    "validation": list,
    "branch": (str, type(None)),
    "pull_request": (str, int, type(None)),
    "artifacts": list,
    "created_at": str,
    "updated_at": str,
}
OPTIONAL_KEYS: dict[str, type | tuple[type, ...]] = {
    "target_version": (str, type(None)),
    "plan_path": (str, type(None)),
    "plan_revision": (int, type(None)),
    "plan_digest": (str, type(None)),
    "bootstrap_commit": (str, type(None)),
    "attempted_steps": list,
    "artifact_manifests": list,
    "evidence_routes": list,
}
VALID_STATES = {"running", "paused", "complete", "blocked"}
VALID_ROUTE_STATUSES = {"running", "complete", "failed"}
VALID_ROUTE_EXECUTION_MODES = {"subagent", "local_fallback"}
VALID_MANIFEST_STORAGE_ROOTS = {"repo", "external"}
SECRET_KEY_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "credential",
    "cookie",
    "api_key",
    "apikey",
    "private_key",
    "client_secret",
    "access_key",
)
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"(?i)(?:api[_-]?key|token|password|secret|cookie|credential)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class WorkflowStateError(RuntimeError):
    """Raised when workflow state cannot be updated safely."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _replace_placeholders(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_placeholders(item, replacements)
            for key, item in value.items()
        }
    return value


def _runtime_dir(workflow_id: str, runtime_root: Path) -> Path:
    return runtime_root / workflow_id


def _tracker_path(workflow_id: str, runtime_root: Path) -> Path:
    return _runtime_dir(workflow_id, runtime_root) / "tracker.json"


def _runtime_plan_path(workflow_id: str, runtime_root: Path) -> Path:
    return _runtime_dir(workflow_id, runtime_root) / "plan.md"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise WorkflowStateError(f"tracker not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowStateError(f"tracker is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkflowStateError(f"{path} must contain a JSON object.")
    return data


@contextmanager
def _tracker_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise WorkflowStateError(f"tracker lock already exists: {lock_path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise WorkflowStateError(f"path must stay inside repository: {value}") from exc
    return resolved


def _resolve_runtime_path(tracker_path: Path, value: str | Path) -> Path:
    base = tracker_path.parent.resolve()
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise WorkflowStateError(
            f"runtime path must stay inside {base}: {value}"
        ) from exc
    return resolved


def _resolve_recorded_path(tracker_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_candidate = (PROJECT_ROOT / path).resolve()
    if repo_candidate.exists() or str(path).startswith("tmp/agentic-workflows/"):
        return repo_candidate
    return _resolve_runtime_path(tracker_path, path)


def _assert_no_secrets(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if any(fragment in normalized for fragment in SECRET_KEY_FRAGMENTS):
                raise WorkflowStateError(
                    f"secret-like key is not allowed at {location}.{key}"
                )
            _assert_no_secrets(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{location}[{index}]")
        return
    if isinstance(value, str):
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                raise WorkflowStateError(
                    f"secret-like value is not allowed at {location}"
                )


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("target_version", normalized.get("release_version"))
    normalized.setdefault("plan_path", None)
    normalized.setdefault("plan_revision", None)
    normalized.setdefault("plan_digest", None)
    normalized.setdefault("bootstrap_commit", None)
    normalized.setdefault("attempted_steps", [])
    normalized.setdefault("artifact_manifests", [])
    normalized.setdefault("evidence_routes", [])
    return normalized


def _validate_routes(payload: dict[str, Any], errors: list[str]) -> None:
    for index, route in enumerate(payload.get("evidence_routes") or []):
        if not isinstance(route, dict):
            errors.append(f"evidence_routes[{index}] must be an object")
            continue
        for key in ("route_id", "step_id", "title", "status", "execution_mode"):
            if not route.get(key):
                errors.append(f"evidence_routes[{index}].{key} is required")
        if route.get("status") not in VALID_ROUTE_STATUSES:
            errors.append(
                f"evidence_routes[{index}].status must be one of "
                f"{sorted(VALID_ROUTE_STATUSES)}"
            )
        if route.get("execution_mode") not in VALID_ROUTE_EXECUTION_MODES:
            errors.append(
                f"evidence_routes[{index}].execution_mode must be one of "
                f"{sorted(VALID_ROUTE_EXECUTION_MODES)}"
            )


def _validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected_type in REQUIRED_KEYS.items():
        if key not in payload:
            errors.append(f"missing required key: {key}")
            continue
        if not isinstance(payload[key], expected_type):
            errors.append(
                f"{key} must be {expected_type}, got {type(payload[key]).__name__}"
            )
    for key, expected_type in OPTIONAL_KEYS.items():
        if key not in payload:
            continue
        if not isinstance(payload[key], expected_type):
            errors.append(
                f"{key} must be {expected_type}, got {type(payload[key]).__name__}"
            )
    extra_keys = sorted(set(payload) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    for key in extra_keys:
        errors.append(f"unexpected key: {key}")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("state") not in VALID_STATES:
        errors.append(f"state must be one of {sorted(VALID_STATES)}")
    if (
        payload.get("target_version")
        and payload.get("release_version")
        and payload["target_version"] != payload["release_version"]
    ):
        errors.append("target_version must match release_version when both are present")
    completed_ids = [
        step.get("step_id")
        for step in payload.get("completed_steps") or []
        if isinstance(step, dict)
    ]
    if len(completed_ids) != len(set(completed_ids)):
        errors.append("completed step ids must be unique")
    _validate_routes(payload, errors)
    return errors


def _validate_plan_binding(payload: dict[str, Any]) -> list[str]:
    if not payload.get("plan_path"):
        return []
    errors: list[str] = []
    try:
        plan_path = _resolve_repo_path(payload["plan_path"])
    except WorkflowStateError as exc:
        return [str(exc)]
    if not plan_path.exists():
        return [f"bound plan does not exist: {payload['plan_path']}"]
    current_digest = _sha256_file(plan_path)
    if payload.get("plan_digest") and payload["plan_digest"] != current_digest:
        errors.append("bound plan digest does not match current plan file")
    if not payload.get("plan_revision"):
        errors.append("plan_revision is required when plan_path is set")
    return errors


def _load_valid_tracker(path: Path, *, check_binding: bool = True) -> dict[str, Any]:
    payload = _normalize_payload(_read_json(path))
    errors = _validate_payload(payload)
    if check_binding:
        errors.extend(_validate_plan_binding(payload))
    if errors:
        raise WorkflowStateError("; ".join(errors))
    return payload


def _save_tracker(path: Path, payload: dict[str, Any]) -> None:
    payload = _normalize_payload(payload)
    payload["updated_at"] = _now()
    _assert_no_secrets(payload)
    errors = _validate_payload(payload)
    if errors:
        raise WorkflowStateError("; ".join(errors))
    with _tracker_lock(path):
        _write_json_atomic(path, payload)


def _routes_dir(tracker_path: Path) -> Path:
    return tracker_path.parent / "routes"


def _route_artifact_path(tracker_path: Path, route_id: str) -> Path:
    return _routes_dir(tracker_path) / f"{route_id}.md"


def _route_prompt_path(tracker_path: Path, route_id: str) -> Path:
    return _routes_dir(tracker_path) / f"{route_id}.prompt.md"


def _route_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    routes = payload.setdefault("evidence_routes", [])
    if not isinstance(routes, list):
        raise WorkflowStateError("evidence_routes must be a list")
    return routes


def _find_route(payload: dict[str, Any], route_id: str) -> dict[str, Any]:
    for route in _route_list(payload):
        if route.get("route_id") == route_id:
            return route
    raise WorkflowStateError(f"evidence route not found: {route_id}")


def _write_optional_text(path: Path, text: str | None) -> None:
    if text is None:
        return
    _assert_no_secrets({"text": text}, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plan_pointer_text(tracker: dict[str, Any]) -> str:
    if tracker.get("plan_path"):
        return (
            "# Runtime Plan Pointer\n\n"
            "The committed canonical workflow plan is the source of truth.\n\n"
            f"- Plan: `{tracker['plan_path']}`\n"
            f"- Revision: `{tracker.get('plan_revision')}`\n"
            f"- Digest: `{tracker.get('plan_digest')}`\n"
            f"- Bootstrap commit: `{tracker.get('bootstrap_commit') or ''}`\n\n"
            "Do not edit this runtime file as a competing plan. Update the "
            "committed plan and rebind the tracker when G-gate amendments change "
            "future work.\n"
        )
    return (
        "# Runtime Plan Pointer\n\n"
        "No committed canonical plan is bound. This file is retained only for "
        "legacy helper compatibility and must not become a second source of truth.\n"
    )


def _apply_plan_binding(
    payload: dict[str, Any],
    *,
    plan_path: Path | None,
    plan_revision: int | None,
    bootstrap_commit: str | None = None,
) -> None:
    if plan_path is None:
        return
    repo_plan_path = _resolve_repo_path(plan_path)
    if not repo_plan_path.exists():
        raise WorkflowStateError(f"plan_path does not exist: {plan_path}")
    payload["plan_path"] = _relative_to_project(repo_plan_path)
    payload["plan_digest"] = _sha256_file(repo_plan_path)
    payload["plan_revision"] = plan_revision
    if bootstrap_commit is not None:
        payload["bootstrap_commit"] = bootstrap_commit


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    routes = payload.get("evidence_routes") or []
    return {
        "workflow_id": payload["workflow_id"],
        "release_version": payload["release_version"],
        "target_version": payload.get("target_version"),
        "state": payload["state"],
        "current_step": payload["current_step"],
        "branch": payload.get("branch"),
        "plan_path": payload.get("plan_path"),
        "plan_revision": payload.get("plan_revision"),
        "completed_steps": len(payload["completed_steps"]),
        "attempted_steps": len(payload.get("attempted_steps") or []),
        "gates": len(payload["gates"]),
        "evidence_routes": len(routes),
        "artifact_manifests": len(payload.get("artifact_manifests") or []),
        "pause_conditions": len(payload["pause_conditions"]),
        "updated_at": payload["updated_at"],
    }


def init_workflow(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = args.runtime_root.resolve()
    workflow_dir = _runtime_dir(args.workflow_id, runtime_root)
    tracker_path = _tracker_path(args.workflow_id, runtime_root)
    runtime_plan_path = _runtime_plan_path(args.workflow_id, runtime_root)
    if workflow_dir.exists() and not args.force:
        raise WorkflowStateError(
            f"{workflow_dir} already exists; use --force only when intentionally replacing it."
        )
    if workflow_dir.exists():
        shutil.rmtree(workflow_dir)
    workflow_dir.mkdir(parents=True, exist_ok=True)

    created_at = _now()
    replacements = {
        "__WORKFLOW_ID__": args.workflow_id,
        "__VERSION__": args.version,
        "__OBJECTIVE__": args.objective,
        "__CREATED_AT__": created_at,
    }
    template = _read_json(TRACKER_TEMPLATE)
    tracker = _replace_placeholders(template, replacements)
    tracker["current_step"] = args.current_step
    tracker["release_version"] = args.version
    tracker["target_version"] = args.version
    tracker["branch"] = args.branch
    tracker["bootstrap_commit"] = args.bootstrap_commit
    _apply_plan_binding(
        tracker,
        plan_path=args.plan_path,
        plan_revision=args.plan_revision,
        bootstrap_commit=args.bootstrap_commit,
    )
    _save_tracker(tracker_path, tracker)
    runtime_plan_path.write_text(_plan_pointer_text(tracker), encoding="utf-8")

    return {
        "workflow_id": args.workflow_id,
        "tracker": str(tracker_path),
        "plan": str(runtime_plan_path),
        "canonical_plan": tracker.get("plan_path"),
        "plan_revision": tracker.get("plan_revision"),
        "plan_digest": tracker.get("plan_digest"),
        "state": tracker["state"],
        "current_step": tracker["current_step"],
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    skip_binding_check = getattr(args, "skip_binding_check", False)
    payload = _load_valid_tracker(args.tracker, check_binding=not skip_binding_check)
    return _summary(payload)


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    skip_binding_check = getattr(args, "skip_binding_check", False)
    payload = _load_valid_tracker(args.tracker, check_binding=not skip_binding_check)
    return {
        **_summary(payload),
        "bootstrap_commit": payload.get("bootstrap_commit"),
        "plan_digest": payload.get("plan_digest"),
        "completed_step_ids": [
            step.get("step_id") for step in payload.get("completed_steps", [])
        ],
        "attempted_step_ids": [
            step.get("step_id") for step in payload.get("attempted_steps", [])
        ],
    }


def set_current(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    payload["current_step"] = args.step_id
    if payload["state"] == "paused":
        payload["state"] = "running"
    _save_tracker(args.tracker, payload)
    return status(args)


def begin(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    if any(step.get("step_id") == args.step_id for step in payload["completed_steps"]):
        raise WorkflowStateError(
            f"step is already complete and immutable: {args.step_id}"
        )
    payload.setdefault("attempted_steps", []).append(
        {
            "step_id": args.step_id,
            "attempted_at": _now(),
            "note": args.note,
            "evidence": args.evidence or [],
        }
    )
    payload["current_step"] = args.step_id
    if payload["state"] == "paused":
        payload["state"] = "running"
    _save_tracker(args.tracker, payload)
    return status(args)


def complete(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    if any(step.get("step_id") == args.step_id for step in payload["completed_steps"]):
        raise WorkflowStateError(
            f"step is already complete and immutable: {args.step_id}"
        )
    payload["completed_steps"].append(
        {
            "step_id": args.step_id,
            "completed_at": _now(),
            "note": args.note,
            "evidence": args.evidence or [],
        }
    )
    payload["current_step"] = args.next_step
    payload["state"] = "running"
    _save_tracker(args.tracker, payload)
    return status(args)


def gate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    payload["gates"].append(
        {
            "gate_id": args.gate_id,
            "assessed_at": _now(),
            "decision": args.decision,
            "note": args.note,
            "amendments": args.amendment or [],
        }
    )
    payload["current_step"] = args.next_step
    payload["state"] = "paused" if args.decision == "pause" else "running"
    _save_tracker(args.tracker, payload)
    return status(args)


def pause(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    payload["pause_conditions"].append(
        {
            "condition_id": args.condition_id,
            "paused_at": _now(),
            "reason": args.reason,
            "needed": args.needed,
        }
    )
    payload["state"] = "paused"
    _save_tracker(args.tracker, payload)
    return status(args)


def resume(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    if payload["state"] == "paused":
        payload["state"] = "running"
    if args.next_step:
        payload["current_step"] = args.next_step
    _save_tracker(args.tracker, payload)
    return status(args)


def rebind_plan(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker, check_binding=False)
    plan_path = args.plan_path or payload.get("plan_path")
    if not plan_path:
        raise WorkflowStateError("plan_path is required for plan rebind")
    next_revision = args.plan_revision
    if next_revision is None:
        next_revision = int(payload.get("plan_revision") or 0) + 1
    _apply_plan_binding(
        payload,
        plan_path=Path(plan_path),
        plan_revision=next_revision,
        bootstrap_commit=args.bootstrap_commit
        if args.bootstrap_commit is not None
        else payload.get("bootstrap_commit"),
    )
    payload["validation"].append(
        {
            "recorded_at": _now(),
            "kind": "plan_rebind",
            "note": args.note,
            "plan_revision": payload["plan_revision"],
            "plan_digest": payload["plan_digest"],
        }
    )
    _save_tracker(args.tracker, payload)
    (_runtime_plan_path(payload["workflow_id"], args.tracker.parent.parent)).write_text(
        _plan_pointer_text(payload), encoding="utf-8"
    )
    return status(args)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(
        args.tracker, check_binding=not args.skip_binding_check
    )
    return {
        "workflow_id": payload["workflow_id"],
        "valid": True,
        "state": payload["state"],
        "current_step": payload["current_step"],
        "plan_path": payload.get("plan_path"),
        "plan_revision": payload.get("plan_revision"),
    }


def _validate_artifact_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version": int,
        "manifest_id": str,
        "workflow_id": str,
        "step_id": str,
        "run_id": str,
        "created_at": str,
        "storage_root": str,
        "artifacts": list,
    }
    for key, expected_type in required.items():
        if key not in manifest:
            errors.append(f"missing required key: {key}")
        elif not isinstance(manifest[key], expected_type):
            errors.append(
                f"{key} must be {expected_type}, got {type(manifest[key]).__name__}"
            )
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("storage_root") not in VALID_MANIFEST_STORAGE_ROOTS:
        errors.append(
            f"storage_root must be one of {sorted(VALID_MANIFEST_STORAGE_ROOTS)}"
        )
    storage_root = manifest.get("storage_root")
    for index, artifact in enumerate(manifest.get("artifacts") or []):
        if not isinstance(artifact, dict):
            errors.append(f"artifacts[{index}] must be an object")
            continue
        for key in ("artifact_id", "path", "kind", "summary", "sha256"):
            if not artifact.get(key):
                errors.append(f"artifacts[{index}].{key} is required")
        if artifact.get("content"):
            errors.append(f"artifacts[{index}] must not embed raw content")
        artifact_path = artifact.get("path")
        if isinstance(artifact_path, str) and artifact_path:
            path = Path(artifact_path)
            if storage_root == "repo":
                try:
                    _resolve_repo_path(path)
                except WorkflowStateError as exc:
                    errors.append(f"artifacts[{index}].path {exc}")
            if storage_root == "external":
                resolved = (
                    path.resolve()
                    if path.is_absolute()
                    else (DEFAULT_EXTERNAL_ARTIFACT_ROOT / path).resolve()
                )
                try:
                    resolved.relative_to(PROJECT_ROOT.resolve())
                except ValueError:
                    pass
                else:
                    errors.append(
                        f"artifacts[{index}].path external evidence must stay outside repo"
                    )
    return errors


def manifest_add(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    manifest_path = _resolve_runtime_path(args.tracker, args.manifest_path)
    manifest = _read_json(manifest_path)
    _assert_no_secrets(manifest, str(manifest_path))
    errors = _validate_artifact_manifest(manifest)
    if errors:
        raise WorkflowStateError("; ".join(errors))
    payload.setdefault("artifact_manifests", []).append(
        {
            "path": _relative_to_project(manifest_path),
            "manifest_id": manifest["manifest_id"],
            "step_id": manifest["step_id"],
            "storage_root": manifest["storage_root"],
            "recorded_at": _now(),
        }
    )
    _save_tracker(args.tracker, payload)
    return status(args)


def route_start(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    routes = _route_list(payload)
    if any(route.get("route_id") == args.route_id for route in routes):
        raise WorkflowStateError(f"evidence route already exists: {args.route_id}")
    if args.execution_mode not in VALID_ROUTE_EXECUTION_MODES:
        raise WorkflowStateError(
            f"execution_mode must be one of {sorted(VALID_ROUTE_EXECUTION_MODES)}"
        )

    routes_directory = _routes_dir(args.tracker)
    routes_directory.mkdir(parents=True, exist_ok=True)
    prompt_path = (
        _resolve_runtime_path(args.tracker, args.prompt_path)
        if args.prompt_path
        else _route_prompt_path(args.tracker, args.route_id)
    )
    artifact_path = (
        _resolve_runtime_path(args.tracker, args.artifact_path)
        if args.artifact_path
        else _route_artifact_path(args.tracker, args.route_id)
    )
    _write_optional_text(prompt_path, args.prompt_text)
    created_at = _now()
    routes.append(
        {
            "route_id": args.route_id,
            "step_id": args.step_id,
            "title": args.title,
            "status": "running",
            "execution_mode": args.execution_mode,
            "prompt_path": _relative_to_project(prompt_path),
            "prompt_summary": args.prompt_summary,
            "artifact_path": _relative_to_project(artifact_path),
            "summary": "",
            "findings": [],
            "recommendation": "",
            "created_at": created_at,
            "updated_at": created_at,
        }
    )
    _save_tracker(args.tracker, payload)
    return status(args)


def route_complete(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    route = _find_route(payload, args.route_id)
    artifact_path = (
        _resolve_runtime_path(args.tracker, args.artifact_path)
        if args.artifact_path
        else _resolve_recorded_path(args.tracker, route["artifact_path"])
    )
    _write_optional_text(artifact_path, args.artifact_text)
    route.update(
        {
            "status": "complete",
            "summary": args.summary,
            "findings": args.finding or [],
            "recommendation": args.recommendation,
            "artifact_path": _relative_to_project(artifact_path),
            "updated_at": _now(),
        }
    )
    _save_tracker(args.tracker, payload)
    return status(args)


def route_fail(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_valid_tracker(args.tracker)
    route = _find_route(payload, args.route_id)
    artifact_path = (
        _resolve_runtime_path(args.tracker, args.artifact_path)
        if args.artifact_path
        else _resolve_recorded_path(args.tracker, route["artifact_path"])
    )
    _write_optional_text(artifact_path, args.artifact_text)
    route.update(
        {
            "status": "failed",
            "summary": args.summary,
            "findings": args.finding or [],
            "recommendation": args.recommendation,
            "error": args.error,
            "artifact_path": _relative_to_project(artifact_path),
            "updated_at": _now(),
        }
    )
    _save_tracker(args.tracker, payload)
    return status(args)


def _tracker_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tracker", type=Path, help="Path to runtime tracker.json.")


def _skip_binding_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-binding-check",
        action="store_true",
        help="Validate tracker shape without requiring the committed plan digest to match.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage agentic workflow runtime state."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create runtime state from templates."
    )
    init_parser.add_argument("workflow_id")
    init_parser.add_argument("--version", required=True)
    init_parser.add_argument("--objective", required=True)
    init_parser.add_argument("--current-step")
    init_parser.add_argument("--branch")
    init_parser.add_argument("--plan-path", type=Path)
    init_parser.add_argument("--plan-revision", type=int, default=1)
    init_parser.add_argument("--bootstrap-commit")
    init_parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
        help="Directory that contains ignored runtime workflow folders.",
    )
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=init_workflow)

    status_parser = subparsers.add_parser("status", help="Print tracker status.")
    _tracker_arg(status_parser)
    _skip_binding_arg(status_parser)
    status_parser.set_defaults(func=status)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print tracker binding details."
    )
    _tracker_arg(inspect_parser)
    _skip_binding_arg(inspect_parser)
    inspect_parser.set_defaults(func=inspect)

    set_parser = subparsers.add_parser("set-current", help="Set current P/G step.")
    _tracker_arg(set_parser)
    set_parser.add_argument("step_id")
    set_parser.set_defaults(func=set_current)

    begin_parser = subparsers.add_parser("begin", help="Record a step attempt.")
    _tracker_arg(begin_parser)
    begin_parser.add_argument("step_id")
    begin_parser.add_argument("--note", default="")
    begin_parser.add_argument("--evidence", action="append")
    begin_parser.set_defaults(func=begin)

    complete_parser = subparsers.add_parser("complete", help="Mark a P step complete.")
    _tracker_arg(complete_parser)
    complete_parser.add_argument("step_id")
    complete_parser.add_argument("--next-step")
    complete_parser.add_argument("--note", default="")
    complete_parser.add_argument("--evidence", action="append")
    complete_parser.set_defaults(func=complete)

    gate_parser = subparsers.add_parser("gate", help="Record a G reassessment gate.")
    _tracker_arg(gate_parser)
    gate_parser.add_argument("gate_id")
    gate_parser.add_argument(
        "--decision",
        choices=("continue", "amend", "pause"),
        required=True,
    )
    gate_parser.add_argument("--next-step")
    gate_parser.add_argument("--note", default="")
    gate_parser.add_argument("--amendment", action="append")
    gate_parser.set_defaults(func=gate)

    pause_parser = subparsers.add_parser(
        "pause", help="Pause for user or external input."
    )
    _tracker_arg(pause_parser)
    pause_parser.add_argument("--condition-id", required=True)
    pause_parser.add_argument("--reason", required=True)
    pause_parser.add_argument("--needed", default="")
    pause_parser.set_defaults(func=pause)

    resume_parser = subparsers.add_parser("resume", help="Resume a paused workflow.")
    _tracker_arg(resume_parser)
    resume_parser.add_argument("--next-step")
    resume_parser.set_defaults(func=resume)

    validate_parser = subparsers.add_parser("validate", help="Validate tracker shape.")
    _tracker_arg(validate_parser)
    _skip_binding_arg(validate_parser)
    validate_parser.set_defaults(func=validate)

    rebind_parser = subparsers.add_parser(
        "rebind-plan",
        help="Update tracker binding after an approved committed plan amendment.",
    )
    _tracker_arg(rebind_parser)
    rebind_parser.add_argument("--plan-path", type=Path)
    rebind_parser.add_argument("--plan-revision", type=int)
    rebind_parser.add_argument("--bootstrap-commit")
    rebind_parser.add_argument("--note", default="")
    rebind_parser.set_defaults(func=rebind_plan)

    manifest_parser = subparsers.add_parser(
        "manifest-add",
        help="Validate and record a small sanitized artifact manifest.",
    )
    _tracker_arg(manifest_parser)
    manifest_parser.add_argument("manifest_path", type=Path)
    manifest_parser.set_defaults(func=manifest_add)

    route_start_parser = subparsers.add_parser(
        "route-start",
        help="Record a read-only evidence route as running.",
    )
    _tracker_arg(route_start_parser)
    route_start_parser.add_argument("route_id")
    route_start_parser.add_argument("--step-id", required=True)
    route_start_parser.add_argument("--title", required=True)
    route_start_parser.add_argument(
        "--execution-mode",
        choices=sorted(VALID_ROUTE_EXECUTION_MODES),
        default="local_fallback",
    )
    route_start_parser.add_argument("--prompt-path", type=Path)
    route_start_parser.add_argument("--prompt-summary", default="")
    route_start_parser.add_argument("--prompt-text")
    route_start_parser.add_argument("--artifact-path", type=Path)
    route_start_parser.set_defaults(func=route_start)

    route_complete_parser = subparsers.add_parser(
        "route-complete",
        help="Mark an evidence route complete with findings.",
    )
    _tracker_arg(route_complete_parser)
    route_complete_parser.add_argument("route_id")
    route_complete_parser.add_argument("--summary", default="")
    route_complete_parser.add_argument("--finding", action="append")
    route_complete_parser.add_argument("--recommendation", default="")
    route_complete_parser.add_argument("--artifact-path", type=Path)
    route_complete_parser.add_argument("--artifact-text")
    route_complete_parser.set_defaults(func=route_complete)

    route_fail_parser = subparsers.add_parser(
        "route-fail",
        help="Mark an evidence route failed with findings.",
    )
    _tracker_arg(route_fail_parser)
    route_fail_parser.add_argument("route_id")
    route_fail_parser.add_argument("--summary", default="")
    route_fail_parser.add_argument("--finding", action="append")
    route_fail_parser.add_argument("--recommendation", default="")
    route_fail_parser.add_argument("--error", required=True)
    route_fail_parser.add_argument("--artifact-path", type=Path)
    route_fail_parser.add_argument("--artifact-text")
    route_fail_parser.set_defaults(func=route_fail)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except WorkflowStateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
