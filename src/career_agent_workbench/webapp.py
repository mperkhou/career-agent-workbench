"""Small local Flask adapter over the configured application workspace."""

from __future__ import annotations

import argparse
import subprocess
import threading
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request

from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import RuntimeConfig, WorkspaceMember
from career_agent_workbench.webapp_actions import (
    ACTION_OPTIONS,
    ATS_ACTION,
    ActionRegistry,
    CommandExecutor,
    CommandStage,
    action_snapshots,
    build_action_stages,
    build_seed_argv,
    create_action,
    run_ats_action,
    run_command_action,
)
from career_agent_workbench.webapp_ingestion import (
    GenericHtmlFetcher,
    LinkedInDetailsFetcher,
    WebIngestionError,
    fetch_generic_html,
    fetch_linkedin_details,
    ingest_generic_urls,
    ingest_linkedin_urls,
    parse_job_url_batch,
)
from career_agent_workbench.webapp_tracker import (
    TRACKER_DIRECTIONS,
    TRACKER_SORTS,
    TRACKER_STATUSES,
    TrackerView,
    TrackerViewError,
    tracker_applications,
    tracker_counts,
)

_EXTENSION_KEY = "career_agent_workbench"


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


def _seed_request() -> tuple[str, str, int, int, int]:
    location = str(request.form.get("location") or "").strip()
    date_posted = str(request.form.get("date_posted") or "")
    if (
        not location
        or len(location) > 256
        or any(ord(character) < 32 for character in location)
        or date_posted not in {"any_time", "past_24_hours", "past_week", "past_month"}
    ):
        raise ValueError
    return (
        location,
        date_posted,
        _bounded_form_integer("limit_per_query", upper=100),
        _bounded_form_integer("max_queries", upper=100),
        _bounded_form_integer("max_jobs", upper=50),
    )


def _bounded_form_integer(name: str, *, upper: int) -> int:
    value = request.form.get(name)
    if type(value) is not str or not value.isascii() or not value.isdigit():
        raise ValueError
    selected = int(value)
    if not 1 <= selected <= upper:
        raise ValueError
    return selected


def create_app(
    runtime: RuntimeConfig,
    *,
    command_executor: CommandExecutor | None = None,
    project_root: Path | None = None,
    linkedin_details_fetcher: LinkedInDetailsFetcher | None = None,
    generic_html_fetcher: GenericHtmlFetcher | None = None,
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
        "actions": ActionRegistry(),
    }
    app.extensions[_EXTENSION_KEY] = extension
    linkedin_fetcher = linkedin_details_fetcher or (
        lambda url: fetch_linkedin_details(
            url,
            user_agent=runtime.settings.user_agent,
            timeout_seconds=runtime.settings.timeout_seconds,
        )
    )
    generic_fetcher = generic_html_fetcher or (
        lambda url: fetch_generic_html(
            url,
            user_agent=runtime.settings.user_agent,
            timeout_seconds=runtime.settings.timeout_seconds,
        )
    )

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
            targets={
                **ACTION_OPTIONS,
                ATS_ACTION: "Recalculate selected ATS",
            },
            actions=action_snapshots(extension["actions"]),
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

    @app.get("/applications/add")
    def add_applications():
        try:
            view = _tracker_view()
        except TrackerViewError:
            return "Add view is invalid.", 400
        return render_template("webapp/add.html", view=view)

    @app.post("/applications/add/seed")
    def seed_applications():
        try:
            view = _tracker_view(form=True)
            location, date_posted, limit_per_query, max_queries, max_jobs = (
                _seed_request()
            )
            bound_root = extension["project_root"]
            if bound_root is None or not (bound_root / "Makefile").is_file():
                return jsonify(
                    message="Seed action is unavailable.", status="rejected"
                ), 503
            argv = build_seed_argv(
                project_root=bound_root,
                location=location,
                date_posted=date_posted,
                limit_per_query=limit_per_query,
                max_queries=max_queries,
                max_jobs=max_jobs,
                paths=paths,
            )
            registry = extension["actions"]
            action = create_action(
                registry,
                label="Seed and match jobs",
                total_stages=1,
            )
            thread = threading.Thread(
                target=run_command_action,
                args=(
                    registry,
                    action.action_id,
                    extension["executor"],
                    (CommandStage(label="Seed and match jobs", argv=argv),),
                ),
                daemon=True,
            )
            thread.start()
        except Exception:  # noqa: BLE001 - request failures stay content-free.
            return jsonify(message="Seed request is invalid.", status="rejected"), 400
        return jsonify(
            action_id=action.action_id,
            refresh_url=view.index_url,
            status="accepted",
        ), 202

    def _ingestion_response(*, linkedin: bool):
        try:
            _tracker_view(form=True)
            field = "linkedin_urls" if linkedin else "other_urls"
            batch = parse_job_url_batch(request.form.get(field), linkedin=linkedin)
            result = (
                ingest_linkedin_urls(
                    store=store,
                    batch=batch,
                    fetcher=linkedin_fetcher,
                )
                if linkedin
                else ingest_generic_urls(
                    store=store,
                    batch=batch,
                    html_fetcher=generic_fetcher,
                )
            )
        except WebIngestionError:
            return jsonify(message="URL ingestion is invalid.", status="rejected"), 400
        except Exception:  # noqa: BLE001 - request failures stay content-free.
            return jsonify(message="URL ingestion failed.", status="rejected"), 400
        status = "accepted" if result.failed == 0 else "partial"
        response_code = 200 if result.accepted else 422
        return (
            jsonify(
                accepted=result.accepted,
                created=result.created,
                failed=result.failed,
                message="URL ingestion completed.",
                refreshed=result.refreshed,
                status=status if result.accepted else "rejected",
            ),
            response_code,
        )

    @app.post("/applications/add/linkedin")
    def add_linkedin_applications():
        return _ingestion_response(linkedin=True)

    @app.post("/applications/add/other")
    def add_generic_applications():
        return _ingestion_response(linkedin=False)

    @app.post("/actions/run")
    def run_action():
        try:
            view = _tracker_view(form=True)
        except TrackerViewError:
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        target = request.form.get("target", "")
        job_ids = _selected_job_ids(request.form.getlist("job_id"))
        highlight_value = request.form.get("highlight")
        if (
            target not in {*ACTION_OPTIONS, ATS_ACTION}
            or highlight_value not in {None, "1"}
            or job_ids is None
        ):
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        try:
            records = store.fetch_job_records(job_ids)
        except Exception:  # noqa: BLE001 - keep identifier failures content-free.
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        if tuple(record.job_id for record in records) != job_ids:
            return jsonify(message="Action request is invalid.", status="rejected"), 400
        registry = extension["actions"]
        if target == ATS_ACTION:
            if highlight_value is not None:
                return jsonify(
                    message="Action request is invalid.", status="rejected"
                ), 400
            action = create_action(
                registry,
                label="Recalculate selected ATS",
                total_stages=1,
            )
            thread = threading.Thread(
                target=run_ats_action,
                args=(registry, action.action_id, store, job_ids),
                daemon=True,
            )
        else:
            bound_root = extension["project_root"]
            if bound_root is None or not (bound_root / "Makefile").is_file():
                return jsonify(message="Action is unavailable.", status="rejected"), 503
            try:
                stages = build_action_stages(
                    project_root=bound_root,
                    workflow=target,
                    job_ids=job_ids,
                    paths=paths,
                    manual_profile=request.form.get("manual_pass_profile", "regular"),
                    highlight=highlight_value == "1",
                )
            except Exception:  # noqa: BLE001 - invalid policy stays content-free.
                return jsonify(
                    message="Action request is invalid.", status="rejected"
                ), 400
            action = create_action(
                registry,
                label=ACTION_OPTIONS[target],
                total_stages=len(stages),
            )
            thread = threading.Thread(
                target=run_command_action,
                args=(
                    registry,
                    action.action_id,
                    extension["executor"],
                    stages,
                ),
                daemon=True,
            )
        thread.start()
        return jsonify(
            action_id=action.action_id,
            refresh_url=view.index_url,
            status="accepted",
        ), 202

    @app.get("/actions/status")
    def action_status():
        return jsonify(actions=action_snapshots(extension["actions"]))

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
