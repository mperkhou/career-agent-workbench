"""Small local Flask adapter over the configured application workspace."""

from __future__ import annotations

import argparse
import subprocess
import threading
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, Response, jsonify, redirect, render_template, request

from career_agent_workbench.application_state import (
    MAX_QUERY_RESULTS,
    ApplicationStateConflictError,
    ApplicationStateStore,
    workflow_revision_from_token,
    workflow_revision_token,
)
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import RuntimeConfig, WorkspaceMember
from career_agent_workbench.cover_letter_rendering import (
    MAX_COVER_LETTER_HTML_CHARS,
    blank_cover_letter,
    render_cover_letter,
)
from career_agent_workbench.webapp_actions import (
    ACTION_OPTIONS,
    ATS_ACTION,
    ActionRegistry,
    CommandExecution,
    CommandExecutor,
    CommandStage,
    action_snapshots,
    build_action_stages,
    build_ingestion_stages,
    build_seed_argv,
    create_action,
    create_retry_action,
    dismiss_action,
    parse_workflow_composition,
    run_ats_action,
    run_command_action,
    run_seed_action,
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
from career_agent_workbench.webapp_artifacts import (
    StoredArtifact,
    copy_artifact_to_downloads,
    cover_letter_artifact,
    selected_resume_artifact,
    variant_resume_artifact,
    variant_review,
)
from career_agent_workbench.webapp_editors import (
    AtsCalculator,
    ResumeHtmlRenderer,
    ResumePdfRenderer,
    active_resume_target,
    calculate_optional_ats,
    compare_jod,
    render_resume_edit,
    resume_yaml_text,
)
from career_agent_workbench.webapp_resume_fields import (
    apply_resume_field_payload,
    resume_field_capabilities,
    resume_field_model,
    resume_field_payload_text,
)
from career_agent_workbench.webapp_tracker import (
    TRACKER_DIRECTIONS,
    TRACKER_SORTS,
    TRACKER_STATUSES,
    TrackerView,
    TrackerViewError,
    tracker_applications,
    tracker_counts,
    tracker_rows,
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
        "download_dir",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    return parser


def _default_command_executor(argv: Sequence[str]) -> CommandExecution:
    completed = subprocess.run(
        list(argv),
        shell=False,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return CommandExecution(
        return_code=completed.returncode,
        output=completed.stdout,
    )


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
        if len(selected) > 50:
            return None
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


def _checkbox(name: str) -> bool:
    value = request.form.get(name)
    if value not in {None, "1"}:
        raise ValueError
    return value == "1"


def _workflow_composition():
    return parse_workflow_composition(
        run_v1=_checkbox("run_v1"),
        run_v2=_checkbox("run_v2"),
        run_manual=_checkbox("run_manual"),
        run_highlight=_checkbox("run_highlight"),
        manual_profile=request.form.get("manual_pass_profile", "regular"),
    )


def _bounded_form_integer(name: str, *, upper: int) -> int:
    value = request.form.get(name)
    if type(value) is not str or not value.isascii() or not value.isdigit():
        raise ValueError
    selected = int(value)
    if not 1 <= selected <= upper:
        raise ValueError
    return selected


def _artifact_response(
    artifact: StoredArtifact,
    *,
    attachment: bool,
) -> Response:
    disposition = "attachment" if attachment else "inline"
    return Response(
        artifact.content,
        content_type=artifact.mime_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{artifact.filename}"'
        },
    )


def create_app(
    runtime: RuntimeConfig,
    *,
    command_executor: CommandExecutor | None = None,
    project_root: Path | None = None,
    linkedin_details_fetcher: LinkedInDetailsFetcher | None = None,
    generic_html_fetcher: GenericHtmlFetcher | None = None,
    resume_html_renderer: ResumeHtmlRenderer | None = None,
    resume_pdf_renderer: ResumePdfRenderer | None = None,
    ats_calculator: AtsCalculator | None = None,
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

    def _render_active_resume(yaml_text: str, snapshot):
        options = {}
        if resume_html_renderer is not None:
            options["html_renderer"] = resume_html_renderer
        if resume_pdf_renderer is not None:
            options["pdf_renderer"] = resume_pdf_renderer
        if ats_calculator is not None:
            options["ats_calculator"] = ats_calculator
        return render_resume_edit(
            yaml_text,
            prompt_jod=snapshot.application.prompt_job_description,
            source_jod=snapshot.application.job_description,
            **options,
        )

    @app.get("/")
    def index():
        try:
            view = _tracker_view()
            applications = tracker_applications(store, view)
            rows = tracker_rows(paths.require(WorkspaceMember.DATABASE), applications)
        except TrackerViewError:
            return "Application scope is invalid.", 400
        except Exception:  # noqa: BLE001 - keep store failures content-free.
            return "Tracker data is unavailable.", 503
        return render_template(
            "webapp/index.html",
            rows=rows,
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
        return render_template(
            "webapp/add.html",
            view=view,
            actions=action_snapshots(extension["actions"]),
        )

    @app.post("/applications/add/seed")
    def seed_applications():
        try:
            view = _tracker_view(form=True)
            composition = _workflow_composition()
            location, date_posted, limit_per_query, max_queries, max_jobs = (
                _seed_request()
            )
            bound_root = extension["project_root"]
            if bound_root is None or not (bound_root / "Makefile").is_file():
                return jsonify(
                    message="Seed action is unavailable.", status="rejected"
                ), 503
            existing_job_ids = tuple(
                record.job_id for record in store.list_applications("all")
            )
            if (
                composition.selected
                and len(existing_job_ids) + max_jobs > MAX_QUERY_RESULTS
            ):
                return jsonify(
                    message="Seed workflow is unavailable.", status="rejected"
                ), 409
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
                target=run_seed_action,
                args=(
                    registry,
                    action.action_id,
                    extension["executor"],
                    CommandStage(label="Seed and match jobs", argv=argv),
                    store,
                    existing_job_ids,
                    bound_root,
                    paths,
                    composition,
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
            view = _tracker_view(form=True)
            composition = _workflow_composition()
            bound_root = extension["project_root"]
            if composition.selected and (
                bound_root is None or not (bound_root / "Makefile").is_file()
            ):
                return jsonify(
                    message="URL workflow is unavailable.", status="rejected"
                ), 503
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
        except (WebIngestionError, ValueError):
            return jsonify(message="URL ingestion is invalid.", status="rejected"), 400
        except Exception:  # noqa: BLE001 - request failures stay content-free.
            return jsonify(message="URL ingestion failed.", status="rejected"), 400
        status = "accepted" if result.failed == 0 else "partial"
        response_code = 200 if result.accepted else 422
        action = None
        if result.accepted and composition.selected:
            assert bound_root is not None
            stages = build_ingestion_stages(
                project_root=bound_root,
                job_ids=result.job_ids,
                paths=paths,
                composition=composition,
            )
            registry = extension["actions"]
            action = create_action(
                registry,
                label="Ingested job workflow",
                total_stages=len(stages),
                job_ids=result.job_ids,
                stages=stages,
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
            response_code = 202
        payload = {
            "accepted": result.accepted,
            "created": result.created,
            "failed": result.failed,
            "message": "URL ingestion completed.",
            "refreshed": result.refreshed,
            "status": status if result.accepted else "rejected",
        }
        if action is not None:
            payload["action_id"] = action.action_id
            payload["refresh_url"] = view.index_url
        return jsonify(payload), response_code

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
                job_ids=job_ids,
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
                job_ids=job_ids,
                stages=stages,
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

    @app.post("/actions/<action_id>/retry")
    def retry_action(action_id: str):
        repeat_value = request.form.get("repeat_completed")
        if repeat_value not in {None, "1"}:
            return jsonify(message="Retry request is invalid.", status="rejected"), 400
        registry = extension["actions"]
        try:
            action = create_retry_action(
                registry,
                action_id,
                repeat_completed=repeat_value == "1",
            )
        except Exception:  # noqa: BLE001 - retry failures stay content-free.
            return jsonify(message="Retry request is invalid.", status="rejected"), 400
        thread = threading.Thread(
            target=run_command_action,
            args=(
                registry,
                action.action_id,
                extension["executor"],
                action.stages,
            ),
            daemon=True,
        )
        thread.start()
        return jsonify(action_id=action.action_id, status="accepted"), 202

    @app.post("/actions/<action_id>/dismiss")
    def dismiss_action_status(action_id: str):
        if not dismiss_action(extension["actions"], action_id):
            return jsonify(
                message="Dismiss request is invalid.", status="rejected"
            ), 400
        return jsonify(status="dismissed")

    def _selected_artifact(job_id: str, kind: str, *, attachment: bool):
        try:
            artifact = selected_resume_artifact(store, job_id, kind)
        except Exception:  # noqa: BLE001 - missing artifacts stay content-free.
            return "Artifact was not found.", 404
        return _artifact_response(artifact, attachment=attachment)

    def _variant_artifact(
        job_id: str,
        variant_key: str,
        kind: str,
        *,
        attachment: bool,
    ):
        try:
            artifact = variant_resume_artifact(store, job_id, variant_key, kind)
        except Exception:  # noqa: BLE001 - exact-target failures are generic.
            return "Artifact was not found.", 404
        return _artifact_response(artifact, attachment=attachment)

    @app.get("/resumes/<job_id>")
    def selected_resume_pdf(job_id: str):
        return _selected_artifact(job_id, "pdf", attachment=False)

    @app.get("/resumes/<job_id>/download")
    def download_selected_resume_pdf(job_id: str):
        return _selected_artifact(job_id, "pdf", attachment=True)

    @app.get("/resume-html/<job_id>")
    def selected_resume_html(job_id: str):
        return _selected_artifact(job_id, "html", attachment=False)

    @app.get("/resume-html/<job_id>/download")
    def download_selected_resume_html(job_id: str):
        return _selected_artifact(job_id, "html", attachment=True)

    @app.get("/resumes/<job_id>/variants/<variant_key>")
    def variant_resume_pdf(job_id: str, variant_key: str):
        return _variant_artifact(job_id, variant_key, "pdf", attachment=False)

    @app.get("/resumes/<job_id>/variants/<variant_key>/download")
    def download_variant_resume_pdf(job_id: str, variant_key: str):
        return _variant_artifact(job_id, variant_key, "pdf", attachment=True)

    @app.get("/resume-html/<job_id>/variants/<variant_key>")
    def variant_resume_html(job_id: str, variant_key: str):
        return _variant_artifact(job_id, variant_key, "html", attachment=False)

    @app.get("/resume-html/<job_id>/variants/<variant_key>/download")
    def download_variant_resume_html(job_id: str, variant_key: str):
        return _variant_artifact(job_id, variant_key, "html", attachment=True)

    @app.get("/resumes/<job_id>/variants")
    def review_resume_variants(job_id: str):
        try:
            view = _tracker_view()
            snapshot = store.get_workflow_snapshot(job_id)
            comparisons = variant_review(snapshot)
        except TrackerViewError:
            return "Variant view is invalid.", 400
        except Exception:  # noqa: BLE001 - state failures stay content-free.
            return "Application was not found.", 404
        return render_template(
            "webapp/variant_review.html",
            application=snapshot.application,
            comparisons=comparisons,
            view=view,
            view_query=urlencode(view.query_items),
        )

    @app.post("/resumes/<job_id>/variants/<variant_key>/use")
    def use_resume_variant(job_id: str, variant_key: str):
        try:
            view = _tracker_view(form=True)
            store.select_resume_variant(job_id, variant_key)
        except Exception:  # noqa: BLE001 - mutation failures stay content-free.
            return "Resume selection is invalid.", 400
        return redirect(f"/resumes/{job_id}/variants?{urlencode(view.query_items)}")

    @app.post("/resumes/<job_id>/variants/reset")
    def reset_resume_variant(job_id: str):
        try:
            view = _tracker_view(form=True)
            store.reset_resume_variant_selection(job_id)
        except Exception:  # noqa: BLE001 - mutation failures stay content-free.
            return "Resume selection is invalid.", 400
        return redirect(f"/resumes/{job_id}/variants?{urlencode(view.query_items)}")

    @app.post("/resumes/<job_id>/copy-to-downloads")
    def copy_selected_resume(job_id: str):
        try:
            view = _tracker_view(form=True)
            artifact = selected_resume_artifact(store, job_id, "pdf")
            copy_artifact_to_downloads(paths, artifact)
        except Exception:  # noqa: BLE001 - private copy failures stay generic.
            return "Resume copy could not be completed.", 400
        return redirect(view.index_url)

    @app.get("/applications/<job_id>/jod")
    def edit_jod(job_id: str):
        try:
            view = _tracker_view()
            application = store.get_application(job_id)
            comparison = compare_jod(
                application.job_description,
                application.prompt_job_description,
            )
            result = request.args.get("result", "")
            if result not in {"", "refreshed", "skipped"}:
                raise ValueError
        except TrackerViewError:
            return "JOD view is invalid.", 400
        except ValueError:
            return "JOD view is invalid.", 400
        except Exception:  # noqa: BLE001 - state failures stay content-free.
            return "Application was not found.", 404
        return render_template(
            "webapp/jod.html",
            application=application,
            comparison=comparison,
            result=result,
            view=view,
        )

    @app.post("/applications/<job_id>/jod")
    def save_jod(job_id: str):
        try:
            view = _tracker_view(form=True)
            if "source_text" not in request.form or "prompt_text" not in request.form:
                raise ValueError
            source_text = request.form["source_text"]
            prompt_text = request.form["prompt_text"]
            application = store.get_application(job_id)
            ats = None
            if application.resume_pdf is not None:
                options = {}
                if ats_calculator is not None:
                    options["ats_calculator"] = ats_calculator
                ats = calculate_optional_ats(
                    application.resume_pdf,
                    prompt_jod=prompt_text,
                    source_jod=source_text,
                    **options,
                )
            store.store_jod(
                job_id,
                source_text=source_text,
                prompt_text=prompt_text,
                ats=ats,
            )
        except Exception:  # noqa: BLE001 - mutation failures stay content-free.
            return "JOD update is invalid.", 400
        result = "refreshed" if ats is not None else "skipped"
        return redirect(
            f"/applications/{job_id}/jod?{urlencode((*view.query_items, ('result', result)))}"
        )

    @app.get("/resumes/<job_id>/edit")
    def edit_resume(job_id: str):
        try:
            view = _tracker_view()
            snapshot = store.get_workflow_snapshot(job_id)
            if snapshot.active_resume_yaml is None:
                raise LookupError
            result = request.args.get("result", "")
            if result not in {"", "saved", "reverted", "synced"}:
                raise ValueError
        except TrackerViewError:
            return "Resume editor view is invalid.", 400
        except ValueError:
            return "Resume editor view is invalid.", 400
        except Exception:  # noqa: BLE001 - state failures stay content-free.
            return "Resume was not found.", 404
        target = active_resume_target(snapshot)
        can_revert = (
            snapshot.application.application_resume_backup is not None
            and snapshot.application.application_resume_backup_target == target
        )
        if snapshot.application.application_resume is None:
            return "Resume was not found.", 404
        fields = resume_field_model(snapshot.application.application_resume)
        return render_template(
            "webapp/resume_edit.html",
            application=snapshot.application,
            yaml_text=snapshot.active_resume_yaml,
            revision=workflow_revision_token(snapshot.edit_revision),
            target=target,
            can_revert=can_revert,
            resume_fields=fields,
            resume_fields_json=resume_field_payload_text(
                snapshot.application.application_resume
            ),
            result=result,
            view=view,
        )

    @app.get("/resumes/structured-fields")
    def structured_resume_fields():
        return jsonify(resume_field_capabilities())

    def _resume_mutation(job_id: str, operation: str):
        try:
            view = _tracker_view(form=True)
            revision = workflow_revision_from_token(request.form.get("revision"))
            snapshot = store.get_workflow_snapshot(job_id)
            if snapshot.edit_revision != revision:
                raise ApplicationStateConflictError
            if operation == "save":
                if (
                    "structured_payload" in request.form
                    and "yaml_text" not in request.form
                ):
                    if snapshot.application.application_resume is None:
                        raise ValueError
                    mapping = apply_resume_field_payload(
                        snapshot.application.application_resume,
                        request.form["structured_payload"],
                    )
                    yaml_text = resume_yaml_text(mapping)
                elif (
                    "yaml_text" in request.form
                    and "structured_payload" not in request.form
                ):
                    yaml_text = request.form["yaml_text"]
                else:
                    raise ValueError
                rendered = _render_active_resume(yaml_text, snapshot)
                store.store_active_resume_if_revision(
                    job_id,
                    yaml_text=rendered.yaml_text,
                    resume_html=rendered.html,
                    resume_pdf=rendered.pdf,
                    ats=rendered.ats,
                    expected_revision=revision,
                    backup_current=True,
                )
                result = "saved"
            elif operation == "sync":
                if snapshot.active_resume_yaml is None:
                    raise ValueError
                rendered = _render_active_resume(snapshot.active_resume_yaml, snapshot)
                store.store_active_resume_if_revision(
                    job_id,
                    yaml_text=snapshot.active_resume_yaml,
                    resume_html=rendered.html,
                    resume_pdf=rendered.pdf,
                    ats=rendered.ats,
                    expected_revision=revision,
                    backup_current=False,
                )
                result = "synced"
            elif operation == "revert":
                backup = snapshot.application.application_resume_backup
                if backup is None:
                    raise ValueError
                rendered = _render_active_resume(resume_yaml_text(backup), snapshot)
                store.revert_active_resume_if_revision(
                    job_id,
                    resume_html=rendered.html,
                    resume_pdf=rendered.pdf,
                    ats=rendered.ats,
                    expected_revision=revision,
                )
                result = "reverted"
            else:
                raise ValueError
        except ApplicationStateConflictError:
            return "Resume changed; reload before editing.", 409
        except Exception:  # noqa: BLE001 - editor failures stay content-free.
            return "Resume update is invalid.", 400
        return redirect(
            f"/resumes/{job_id}/edit?{urlencode((*view.query_items, ('result', result)))}"
        )

    @app.post("/resumes/<job_id>/edit")
    def save_resume(job_id: str):
        return _resume_mutation(job_id, "save")

    @app.post("/resumes/<job_id>/edit/revert")
    def revert_resume(job_id: str):
        return _resume_mutation(job_id, "revert")

    @app.post("/resumes/<job_id>/edit/sync")
    def sync_resume(job_id: str):
        return _resume_mutation(job_id, "sync")

    @app.get("/applications/<job_id>/cover-letter")
    def edit_cover_letter(job_id: str):
        try:
            view = _tracker_view()
            application = store.get_application(job_id)
            value = application.cover_letter or blank_cover_letter()
            body_html = value.get("body_html", "")
            if (
                type(body_html) is not str
                or len(body_html) > MAX_COVER_LETTER_HTML_CHARS
            ):
                raise ValueError
            result = request.args.get("result", "")
            if result not in {"", "saved"}:
                raise ValueError
        except TrackerViewError:
            return "Cover letter view is invalid.", 400
        except ValueError:
            return "Cover letter data is invalid.", 400
        except Exception:  # noqa: BLE001 - state failures stay content-free.
            return "Application was not found.", 404
        return render_template(
            "webapp/cover_letter_edit.html",
            application=application,
            body_html=body_html,
            result=result,
            view=view,
        )

    @app.post("/applications/<job_id>/cover-letter")
    def save_cover_letter(job_id: str):
        try:
            view = _tracker_view(form=True)
            if "body_html" not in request.form:
                raise ValueError
            rendered = render_cover_letter(request.form["body_html"])
            store.store_clo(
                job_id,
                value=rendered.value,
                pdf_content=rendered.pdf,
            )
        except Exception:  # noqa: BLE001 - mutations and content stay private.
            return "Cover letter update is invalid.", 400
        return redirect(
            f"/applications/{job_id}/cover-letter?{urlencode((*view.query_items, ('result', 'saved')))}"
        )

    def _cover_letter_response(job_id: str, *, attachment: bool):
        try:
            artifact = cover_letter_artifact(store, job_id)
        except Exception:  # noqa: BLE001 - missing artifacts stay content-free.
            return "Artifact was not found.", 404
        return _artifact_response(artifact, attachment=attachment)

    @app.get("/cover-letters/<job_id>")
    def cover_letter_pdf(job_id: str):
        return _cover_letter_response(job_id, attachment=False)

    @app.get("/cover-letters/<job_id>/download")
    def download_cover_letter_pdf(job_id: str):
        return _cover_letter_response(job_id, attachment=True)

    @app.post("/cover-letters/<job_id>/copy-to-downloads")
    def copy_cover_letter(job_id: str):
        try:
            view = _tracker_view(form=True)
            artifact = cover_letter_artifact(store, job_id)
            copy_artifact_to_downloads(paths, artifact)
        except Exception:  # noqa: BLE001 - private copy failures stay generic.
            return "Cover letter copy could not be completed.", 400
        return redirect(view.index_url)

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
