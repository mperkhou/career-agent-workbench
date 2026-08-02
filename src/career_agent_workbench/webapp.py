"""Small local Flask adapter over the configured application workspace."""

from __future__ import annotations

import argparse
import subprocess
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request

from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import RuntimeConfig, WorkspaceMember, WorkspacePaths
from career_agent_workbench.webapp_tracker import (
    TRACKER_DIRECTIONS,
    TRACKER_SORTS,
    TRACKER_STATUSES,
    TrackerView,
    TrackerViewError,
    tracker_applications,
    tracker_counts,
)


CommandExecutor = Callable[[Sequence[str]], int]

ACTION_TARGETS = {
    "regenerate-draft-resumes": "Regenerate draft resumes",
    "regenerate-resumes": "Regenerate and refine resumes",
    "refine-draft-resumes": "Refine draft resumes",
    "regenerate-aro-objects": "Regenerate application resume objects",
    "sync-draft-to-aro": "Sync drafts to application resume objects",
    "highlight-draft-resumes": "Highlight draft resumes",
    "manual-pass-resumes": "Run manual resume pass",
}
_PATH_ASSIGNMENTS = (
    ("root", "WORKSPACE"),
    ("database", "DATABASE"),
    ("output_dir", "OUTPUT_DIR"),
    ("profile_dir", "PROFILE_DIR"),
    ("master_resume", "MASTER_RESUME"),
    ("master_resume_text", "MASTER_RESUME_TEXT"),
    ("blacklist", "BLACKLIST"),
    ("tmp_dir", "TMP_DIR"),
)
_MAX_ACTIONS = 32
_EXTENSION_KEY = "career_agent_workbench"


@dataclass(slots=True)
class _ActionRecord:
    action_id: str
    target_label: str
    status: str
    queued_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    message: str = "Action queued."


@dataclass(slots=True)
class _ActionRegistry:
    actions: dict[str, _ActionRecord] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.Lock)


def _port(value: str) -> int:
    try:
        selected = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.") from None
    if not 1 <= selected <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free local web command parser."""

    parser = argparse.ArgumentParser(
        description="Launch the local Career Agent Workbench tracker."
    )
    add_runtime_path_arguments(
        parser,
        "workspace",
        "database",
        "output_dir",
        "profile_dir",
        "master_resume",
        "master_resume_text",
        "blacklist",
        "tmp_dir",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    return parser


def _default_command_executor(argv: Sequence[str]) -> int:
    completed = subprocess.run(
        list(argv),
        shell=False,
        check=False,
    )
    return completed.returncode


def _bound_project_root(project_root: Path | None) -> Path | None:
    candidate = (
        project_root
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        resolved = candidate.expanduser().resolve(strict=False)
        if not (resolved / "Makefile").is_file():
            return None
    except (OSError, RuntimeError):
        return None
    return resolved


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _action_snapshots(registry: _ActionRegistry) -> list[dict[str, object]]:
    with registry.lock:
        records = tuple(reversed(tuple(registry.actions.values())))
        return [
            {
                "id": record.action_id,
                "target": record.target_label,
                "status": record.status,
                "queued_at": record.queued_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "return_code": record.return_code,
                "message": record.message,
            }
            for record in records
        ]


def _create_action(registry: _ActionRegistry, *, target: str) -> _ActionRecord:
    record = _ActionRecord(
        action_id=uuid.uuid4().hex,
        target_label=ACTION_TARGETS[target],
        status="queued",
        queued_at=_timestamp(),
    )
    with registry.lock:
        registry.actions[record.action_id] = record
        while len(registry.actions) > _MAX_ACTIONS:
            registry.actions.pop(next(iter(registry.actions)))
    return record


def _run_action(
    registry: _ActionRegistry,
    action_id: str,
    executor: CommandExecutor,
    argv: tuple[str, ...],
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.status = "running"
        record.started_at = _timestamp()
        record.message = "Action running."
    try:
        return_code = executor(argv)
        if type(return_code) is not int:
            return_code = 1
    except Exception:  # noqa: BLE001 - background failures stay content-free.
        return_code = 1
    with registry.lock:
        record = registry.actions[action_id]
        record.return_code = return_code
        record.finished_at = _timestamp()
        if return_code == 0:
            record.status = "completed"
            record.message = "Action completed."
        else:
            record.status = "failed"
            record.message = "Action failed."


def _make_argv(
    *,
    project_root: Path,
    target: str,
    job_ids: tuple[str, ...],
    paths: WorkspacePaths,
) -> tuple[str, ...]:
    argv = [
        "make",
        "-C",
        str(project_root),
        target,
        f"JOB_IDS={' '.join(job_ids)}",
    ]
    for member, assignment in _PATH_ASSIGNMENTS:
        value = getattr(paths, member)
        if value is not None:
            argv.append(f"{assignment}={value}")
    return tuple(argv)


def _selected_job_ids(values: Sequence[str]) -> tuple[str, ...] | None:
    if not values:
        return None
    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        if type(value) is not str:
            return None
        job_id = value.strip()
        if not job_id or job_id in seen:
            return None
        seen.add(job_id)
        selected.append(job_id)
    return tuple(selected)


def _tracker_view(*, form: bool = False) -> TrackerView:
    return TrackerView.parse(
        request.form if form else request.args, prefix="view_" if form else ""
    )


def _date_applied(value: str) -> str | None:
    selected = value.strip()
    if not selected:
        return None
    try:
        parsed = date.fromisoformat(selected)
    except ValueError:
        raise ValueError from None
    if parsed.isoformat() != selected:
        raise ValueError
    return selected


def create_app(
    runtime: RuntimeConfig,
    *,
    command_executor: CommandExecutor | None = None,
    project_root: Path | None = None,
) -> Flask:
    """Create one app from an already-resolved immutable runtime."""

    paths = runtime.paths
    paths.require(WorkspaceMember.DATABASE)
    paths.require(WorkspaceMember.OUTPUT_DIR)
    store = ApplicationStateStore(paths)
    store.assert_workspace_binding(paths)
    store.initialize()

    app = Flask(__name__)
    extension = {
        "paths": paths,
        "store": store,
        "executor": command_executor or _default_command_executor,
        "project_root": _bound_project_root(project_root),
        "actions": _ActionRegistry(),
    }
    app.extensions[_EXTENSION_KEY] = extension

    @app.get("/")
    def index():
        try:
            view = _tracker_view()
            applications = tracker_applications(store, view)
        except TrackerViewError:
            return "Application scope is invalid.", 400
        except Exception:  # noqa: BLE001 - keep store failures content-free.
            return "Tracker data is unavailable.", 503
        return render_template(
            "webapp/index.html",
            applications=applications,
            view=view,
            counts=tracker_counts(applications),
            tracker_statuses=TRACKER_STATUSES,
            tracker_sorts=TRACKER_SORTS,
            tracker_directions=TRACKER_DIRECTIONS,
            targets=ACTION_TARGETS,
            actions=_action_snapshots(extension["actions"]),
        )

    @app.post("/applications/<job_id>")
    def update_application(job_id: str):
        try:
            view = _tracker_view(form=True)
            if "applied_to" not in request.form or "date_applied" not in request.form:
                raise ValueError
            if "notes" not in request.form:
                raise ValueError
            store.update_application_status(
                job_id,
                applied_to=request.form["applied_to"],
                date_applied=_date_applied(request.form["date_applied"]),
                notes=request.form["notes"],
            )
        except Exception:  # noqa: BLE001 - mutation errors stay content-free.
            return "Application update is invalid.", 400
        return redirect(view.index_url)

    def _bulk_mutation(operation: str):
        try:
            view = _tracker_view(form=True)
            job_ids = _selected_job_ids(request.form.getlist("job_id"))
            if job_ids is None:
                raise ValueError
            if operation == "archive":
                store.archive(job_ids)
            elif operation == "unarchive":
                store.unarchive(job_ids)
            elif (
                operation == "delete" and request.form.get("confirm_delete") == "delete"
            ):
                store.delete(job_ids)
            else:
                raise ValueError
        except Exception:  # noqa: BLE001 - mutation errors stay content-free.
            return "Application mutation is invalid.", 400
        return redirect(view.index_url)

    @app.post("/applications/archive")
    def archive_applications():
        return _bulk_mutation("archive")

    @app.post("/applications/unarchive")
    def unarchive_applications():
        return _bulk_mutation("unarchive")

    @app.post("/applications/delete")
    def delete_applications():
        return _bulk_mutation("delete")

    @app.post("/actions/run")
    def run_action():
        target = request.form.get("target", "")
        job_ids = _selected_job_ids(request.form.getlist("job_id"))
        if target not in ACTION_TARGETS or job_ids is None:
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        try:
            records = store.fetch_job_records(job_ids)
        except Exception:  # noqa: BLE001 - keep identifier failures content-free.
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        if tuple(record.job_id for record in records) != job_ids:
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        bound_root = extension["project_root"]
        if bound_root is None or not (bound_root / "Makefile").is_file():
            return jsonify(message="Action is unavailable.", status="rejected"), 503

        argv = _make_argv(
            project_root=bound_root,
            target=target,
            job_ids=job_ids,
            paths=paths,
        )
        registry = extension["actions"]
        action = _create_action(registry, target=target)
        thread = threading.Thread(
            target=_run_action,
            args=(registry, action.action_id, extension["executor"], argv),
            daemon=True,
        )
        thread.start()
        return jsonify(action_id=action.action_id, status="accepted"), 202

    @app.get("/actions/status")
    def action_status():
        return jsonify(actions=_action_snapshots(extension["actions"]))

    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration once and run the local Flask adapter."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        runtime = load_command_config(
            args,
            required=(WorkspaceMember.DATABASE, WorkspaceMember.OUTPUT_DIR),
        )
    except CliConfigurationError as exc:
        parser.error(str(exc))
    try:
        app = create_app(runtime, project_root=args.project_root)
    except Exception:  # noqa: BLE001 - keep startup failures content-free.
        parser.error("Web application configuration is invalid.")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
