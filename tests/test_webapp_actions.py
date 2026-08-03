from __future__ import annotations

import sqlite3
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from reportlab.pdfgen import canvas

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths
from career_agent_workbench.webapp_actions import (
    ACTION_TARGETS,
    ActionRegistry,
    action_snapshots,
    build_action_stages,
    build_ingestion_stages,
    create_action,
    parse_workflow_composition,
)

ROOT = Path(__file__).resolve().parents[1]


class _InlineThread:
    def __init__(self, *, target, args, daemon: bool) -> None:
        assert daemon is True
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def _paths(tmp_path: Path, name: str = "workspace") -> WorkspacePaths:
    workspace = tmp_path / name
    return WorkspacePaths(
        root=workspace,
        database=workspace / "state" / "applications.sqlite3",
        output_dir=workspace / "artifacts",
        profile_dir=workspace / "profile",
        master_resume=workspace / "profile" / "MASTER-RESUME.yml",
        master_resume_text=workspace / "profile" / "MASTER-RESUME.txt",
        blacklist=workspace / "profile" / "blacklist.txt",
        tmp_dir=workspace / "tmp",
    )


def _runtime(paths: WorkspacePaths) -> RuntimeConfig:
    return RuntimeConfig(paths=paths, settings=Settings(), env_file=None)


def _view_form() -> dict[str, str]:
    return {
        "view_q": "fictional",
        "view_status": "all",
        "view_scope": "all",
        "view_sort": "company",
        "view_direction": "asc",
    }


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    return project


def _seed(app, *job_ids: str) -> None:
    store = app.extensions["career_agent_workbench"]["store"]
    for job_id in job_ids:
        store.upsert_application(
            ApplicationMetadata(
                job_id=job_id,
                company="Example Systems",
                job_title="Fictional Engineer",
                job_url=f"https://example.com/jobs/{job_id}",
                source="synthetic",
            )
        )


def _pdf_bytes(label: str) -> bytes:
    output = BytesIO()
    document = canvas.Canvas(output)
    lines = (
        "Fictional Candidate",
        "fictional@example.com",
        "Professional Summary",
        f"{label} Python automation and reliable API engineering",
        "Core Technical Skills",
        "Python, testing, observability, automation",
        "Professional Experience",
        "Built fictional services with documented operational reviews.",
        "Education",
        "Example University",
    )
    for index, line in enumerate(lines):
        document.drawString(72, 760 - index * 28, line)
    document.save()
    return output.getvalue()


def _seed_ats_candidate(
    app,
    job_id: str,
    *,
    prompt: str,
    with_pdf: bool = True,
) -> None:
    store = app.extensions["career_agent_workbench"]["store"]
    source = (
        "Responsibilities: build fictional Python APIs with automation, testing, "
        "observability, reliable operations, and documented reviews."
    )
    store.seed_application(
        ApplicationMetadata(
            job_id=job_id,
            company="Example Systems",
            job_title="Fictional Engineer",
            job_url=f"https://example.com/jobs/{job_id}",
            source="synthetic",
        ),
        source_text=source,
        prompt_text=prompt,
    )
    if with_pdf:
        store.upsert_resume_variant(
            job_id,
            ResumeVariantWrite(
                variant_key="v1",
                variant_label="Draft v1",
                source="synthetic",
                application_resume_yaml="name: Fictional Candidate\n",
                resume_html="<p>Fictional resume</p>",
                resume_pdf=_pdf_bytes(job_id),
                ats=AtsFields(score=1, missing_terms="stale"),
            ),
        )


def test_exact_make_allowlist_composites_force_manual_profile_and_highlight(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project = _project(tmp_path)
    for target in ACTION_TARGETS:
        stages = build_action_stages(
            project_root=project,
            workflow=target,
            job_ids=("job-a",),
            paths=paths,
            manual_profile="regular",
            highlight=False,
        )
        assert len(stages) == 1
        assert stages[0].argv[3] == target
        assert set(item.argv[3] for item in stages) <= set(ACTION_TARGETS)
        if target in {"regenerate-draft-resumes", "regenerate-resumes"}:
            assert "FIRST_DRAFT_FORCE=1" in stages[0].argv
        if target == "manual-pass-resumes":
            assert "MANUAL_PASS_PROFILE=regular" in stages[0].argv

    v1_v2_manual = build_action_stages(
        project_root=project,
        workflow="v1-v2-manual",
        job_ids=("job-a", "job-b"),
        paths=paths,
        manual_profile="premium",
        highlight=True,
    )
    assert [stage.argv[3] for stage in v1_v2_manual] == [
        "regenerate-draft-resumes",
        "refine-draft-resumes",
        "manual-pass-resumes",
        "highlight-draft-resumes",
    ]
    assert "FIRST_DRAFT_FORCE=1" in v1_v2_manual[0].argv
    assert "FIRST_DRAFT_FORCE=1" not in v1_v2_manual[1].argv
    assert "MANUAL_PASS_PROFILE=premium" in v1_v2_manual[2].argv
    assert "HIGHLIGHT_RESUME_VARIANT=manual" in v1_v2_manual[3].argv
    assert all("JOB_IDS=job-a job-b" in stage.argv for stage in v1_v2_manual)

    v1_highlight = build_action_stages(
        project_root=project,
        workflow="regenerate-draft-resumes",
        job_ids=("job-a",),
        paths=paths,
        manual_profile="regular",
        highlight=True,
    )
    assert "HIGHLIGHT_RESUME_VARIANT=v1" in v1_highlight[-1].argv
    v2_highlight = build_action_stages(
        project_root=project,
        workflow="v1-v2",
        job_ids=("job-a",),
        paths=paths,
        manual_profile="regular",
        highlight=True,
    )
    assert "HIGHLIGHT_RESUME_VARIANT=v2" in v2_highlight[-1].argv

    for workflow, profile, highlight in (
        ("unsupported", "regular", False),
        ("manual-pass-resumes", "unsupported", False),
        ("refine-draft-resumes", "regular", True),
        ("highlight-draft-resumes", "regular", True),
    ):
        with pytest.raises(Exception):
            build_action_stages(
                project_root=project,
                workflow=workflow,
                job_ids=("job-a",),
                paths=paths,
                manual_profile=profile,
                highlight=highlight,
            )


def test_composite_action_failure_is_bounded_and_hides_details(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project = _project(tmp_path)
    commands: list[tuple[str, ...]] = []

    def executor(argv) -> int:
        commands.append(tuple(argv))
        if len(commands) == 2:
            raise RuntimeError("synthetic-private-command-detail")
        return 0

    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(
        _runtime(paths),
        project_root=project,
        command_executor=executor,
    )
    _seed(app, "job-a")
    response = app.test_client().post(
        "/actions/run",
        data={
            **_view_form(),
            "target": "v1-v2-manual",
            "manual_pass_profile": "regular",
            "highlight": "1",
            "job_id": "job-a",
        },
    )
    assert response.status_code == 202
    assert response.get_json()["refresh_url"].endswith(
        "/?q=fictional&status=all&scope=all&sort=company&direction=asc"
    )
    assert [command[3] for command in commands] == [
        "regenerate-draft-resumes",
        "refine-draft-resumes",
    ]
    status = app.test_client().get("/actions/status").get_json()["actions"][0]
    assert status["status"] == "failed"
    assert status["completed_stages"] == 2
    assert status["total_stages"] == 4
    assert status["current_stage"] == "Refine draft resumes"
    assert status["successful_steps"] == 1
    assert status["failed_steps"] == 1
    assert status["skipped_steps"] == 2
    assert status["failed_work"] == [
        {
            "job_id": "job-a",
            "stage": "Refine draft resumes",
            "stage_index": 2,
        }
    ]
    assert status["message"] == "Action failed for every selected job."
    assert "synthetic-private-command-detail" not in repr(status)
    assert all("Running" not in message for message in status["messages"])


def test_ingestion_composition_dependencies_and_selected_stages_are_exact(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project = _project(tmp_path)
    for options in (
        {"run_v1": False, "run_v2": True, "run_manual": False, "run_highlight": False},
        {"run_v1": True, "run_v2": False, "run_manual": True, "run_highlight": False},
        {"run_v1": False, "run_v2": False, "run_manual": False, "run_highlight": True},
    ):
        with pytest.raises(ValueError, match="Workflow composition is invalid"):
            parse_workflow_composition(**options, manual_profile="regular")

    composition = parse_workflow_composition(
        run_v1=True,
        run_v2=True,
        run_manual=True,
        run_highlight=True,
        manual_profile="premium",
    )
    stages = build_ingestion_stages(
        project_root=project,
        job_ids=("job-a", "job-b"),
        paths=paths,
        composition=composition,
    )
    assert [stage.argv[3] for stage in stages] == [
        "regenerate-draft-resumes",
        "refine-draft-resumes",
        "manual-pass-resumes",
        "highlight-draft-resumes",
    ]
    assert "MANUAL_PASS_PROFILE=premium" in stages[2].argv
    assert "HIGHLIGHT_RESUME_VARIANT=manual" in stages[3].argv


def test_partial_survivor_retry_resumes_failed_stage_without_repeating_survivor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project = _project(tmp_path)
    commands: list[tuple[str, ...]] = []
    failed_once = False

    def executor(argv) -> int:
        nonlocal failed_once
        command = tuple(argv)
        commands.append(command)
        if (
            command[3] == "refine-draft-resumes"
            and "JOB_IDS=job-a" in command
            and not failed_once
        ):
            failed_once = True
            raise RuntimeError("synthetic-sensitive-stage-detail")
        return 0

    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(
        _runtime(paths),
        project_root=project,
        command_executor=executor,
    )
    _seed(app, "job-a", "job-b")
    client = app.test_client()
    response = client.post(
        "/actions/run",
        data={
            "target": "v1-v2-manual",
            "manual_pass_profile": "regular",
            "highlight": "1",
            "job_id": ["job-a", "job-b"],
        },
    )
    assert response.status_code == 202
    original_id = response.get_json()["action_id"]
    original = next(
        item
        for item in client.get("/actions/status").get_json()["actions"]
        if item["id"] == original_id
    )
    assert original["status"] == "partial"
    assert original["completed_jobs"] == 1
    assert original["total_jobs"] == 2
    assert original["retryable"] is True
    assert original["progress_current"] == original["progress_total"] == 8
    assert len(original["messages"]) == 8
    assert "synthetic-sensitive-stage-detail" not in repr(original)
    assert [command[3:5] for command in commands] == [
        ("regenerate-draft-resumes", "JOB_IDS=job-a"),
        ("regenerate-draft-resumes", "JOB_IDS=job-b"),
        ("refine-draft-resumes", "JOB_IDS=job-a"),
        ("refine-draft-resumes", "JOB_IDS=job-b"),
        ("manual-pass-resumes", "JOB_IDS=job-b"),
        ("highlight-draft-resumes", "JOB_IDS=job-b"),
    ]

    retry = client.post(f"/actions/{original_id}/retry")
    assert retry.status_code == 202
    retry_id = retry.get_json()["action_id"]
    retried_commands = commands[6:]
    assert [command[3:5] for command in retried_commands] == [
        ("refine-draft-resumes", "JOB_IDS=job-a"),
        ("manual-pass-resumes", "JOB_IDS=job-a"),
        ("highlight-draft-resumes", "JOB_IDS=job-a"),
    ]
    assert not any("JOB_IDS=job-b" in command for command in retried_commands)
    snapshots = client.get("/actions/status").get_json()["actions"]
    retried = next(item for item in snapshots if item["id"] == retry_id)
    assert retried["status"] == "completed"
    assert retried["retry_of"] == original_id
    assert retried["progress_current"] == retried["progress_total"] == 3
    assert client.post(f"/actions/{retry_id}/dismiss").status_code == 200
    assert all(
        item["id"] != retry_id
        for item in client.get("/actions/status").get_json()["actions"]
    )


def test_explicit_ats_recalculation_updates_selected_projection_and_skips_safely(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(_runtime(paths), project_root=tmp_path / "no-project")
    store = app.extensions["career_agent_workbench"]["store"]
    prompt_description = (
        "Required Python automation, API reliability, observability, and testing."
    )
    _seed_ats_candidate(app, "prompt-job", prompt=prompt_description)
    _seed_ats_candidate(app, "source-fallback-job", prompt="")
    store.select_resume_variant("source-fallback-job", "v1")
    _seed_ats_candidate(app, "no-pdf-job", prompt=prompt_description, with_pdf=False)
    _seed(app, "no-jod-job")
    store.upsert_resume_variant(
        "no-jod-job",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Draft v1",
            source="synthetic",
            application_resume_yaml="name: Fictional Candidate\n",
            resume_pdf=_pdf_bytes("no-jod"),
        ),
    )
    for job_id in ("prompt-job", "source-fallback-job"):
        store.store_clo(
            job_id,
            value={"letter": "Synthetic cover letter"},
            pdf_content=b"synthetic-cover-pdf",
        )
        store.update_application_status(
            job_id,
            applied_to="Yes",
            date_applied="2035-02-04",
            notes="Synthetic operator note",
        )
    store.archive(["prompt-job", "source-fallback-job"])
    with sqlite3.connect(paths.database) as connection:
        connection.execute(
            """
            UPDATE applications SET job_description = NULL
            WHERE job_id = ?
            """,
            ("prompt-job",),
        )
        connection.execute(
            """
            UPDATE applications SET prompt_job_description = NULL
            WHERE job_id = ?
            """,
            ("source-fallback-job",),
        )
    before = {
        job_id: store.get_application(job_id)
        for job_id in ("prompt-job", "source-fallback-job")
    }
    variants_before = {
        job_id: store.list_resume_variants(job_id)
        for job_id in ("prompt-job", "source-fallback-job")
    }
    skipped_before = {
        job_id: store.get_application(job_id) for job_id in ("no-pdf-job", "no-jod-job")
    }
    assert before["prompt-job"].job_description is None
    assert before["prompt-job"].prompt_job_description == prompt_description
    assert before["prompt-job"].resume_variant_selection_mode == "auto"
    assert before["source-fallback-job"].prompt_job_description is None
    assert before["source-fallback-job"].job_description is not None
    assert before["source-fallback-job"].resume_variant_selection_mode == "manual"

    response = app.test_client().post(
        "/actions/run",
        data={
            **_view_form(),
            "target": "recalculate-ats",
            "job_id": [
                "prompt-job",
                "source-fallback-job",
                "no-pdf-job",
                "no-jod-job",
            ],
        },
    )
    assert response.status_code == 202
    status = app.test_client().get("/actions/status").get_json()["actions"][0]
    assert status["status"] == "completed"
    assert status["updated_count"] == 2
    assert status["skipped_count"] == 2
    assert status["message"] == "ATS recalculation completed: 2 updated, 2 skipped."

    for job_id in ("prompt-job", "source-fallback-job"):
        after = store.get_application(job_id)
        original = before[job_id]
        assert after.ats.score is not None
        assert after.ats.score != 1
        assert after.selected_resume_variant == original.selected_resume_variant
        assert (
            after.resume_variant_selection_mode
            == original.resume_variant_selection_mode
        )
        assert after.job_description == original.job_description
        assert after.prompt_job_description == original.prompt_job_description
        assert (
            replace(
                after,
                ats=original.ats,
                selected_variant=original.selected_variant,
                updated_at=original.updated_at,
            )
            == original
        )
        variant_after = store.list_resume_variants(job_id)[0]
        variant_before = variants_before[job_id][0]
        assert (
            replace(
                variant_after,
                ats=variant_before.ats,
                ats_diagnostics=variant_before.ats_diagnostics,
                updated_at=variant_before.updated_at,
            )
            == variant_before
        )
    assert store.get_application("no-pdf-job") == skipped_before["no-pdf-job"]
    assert store.get_application("no-jod-job") == skipped_before["no-jod-job"]


def test_sync_selected_draft_flask_action_completes_bounded_operation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(_runtime(paths), project_root=ROOT)
    store = app.extensions["career_agent_workbench"]["store"]
    paths.output_dir.mkdir(parents=True)
    store.seed_application(
        ApplicationMetadata(
            job_id="fictional-sync",
            company="Example Systems",
            job_title="Synthetic Engineer",
            job_url="https://example.com/jobs/fictional-sync",
            source="synthetic",
        ),
        source_text="Required: Python, Flask, testing, and observability.",
        prompt_text="Responsibilities: Build synthetic Python services.",
    )
    resume = yaml.safe_load(
        (ROOT / "examples/demo-workspace/profile/MASTER-RESUME.yml").read_text(
            encoding="utf-8"
        )
    )
    store.upsert_resume_variant(
        "fictional-sync",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Synthetic selected draft",
            source="synthetic",
            application_resume_yaml=yaml.safe_dump(resume, sort_keys=False),
            ats=AtsFields(score=1, missing_terms="stale"),
        ),
    )
    before = store.select_resume_variant("fictional-sync", "v1")
    before_revision = store.get_workflow_snapshot("fictional-sync").revision

    response = app.test_client().post(
        "/actions/run",
        data={"target": "sync-draft-to-aro", "job_id": "fictional-sync"},
    )
    assert response.status_code == 202
    action_id = response.get_json()["action_id"]
    action = next(
        item
        for item in app.test_client().get("/actions/status").get_json()["actions"]
        if item["id"] == action_id
    )
    assert action["status"] == "completed"
    assert action["progress_current"] == action["progress_total"] == 1
    after = store.get_application("fictional-sync")
    assert after.selected_resume_variant == before.selected_resume_variant == "v1"
    assert after.resume_variant_selection_mode == "manual"
    assert after.ats.score is not None and after.ats.score != 1
    assert store.get_workflow_snapshot("fictional-sync").revision != before_revision


def test_registry_history_is_bounded_and_app_registries_stay_isolated(
    tmp_path: Path,
) -> None:
    registry = ActionRegistry()
    for index in range(40):
        create_action(registry, label=f"Fictional action {index}", total_stages=1)
    snapshots = action_snapshots(registry)
    assert len(snapshots) == 32
    assert snapshots[0]["target"] == "Fictional action 39"
    assert snapshots[-1]["target"] == "Fictional action 8"

    first = webapp.create_app(
        _runtime(_paths(tmp_path, "first")),
        project_root=tmp_path / "no-project",
    )
    second = webapp.create_app(
        _runtime(_paths(tmp_path, "second")),
        project_root=tmp_path / "no-project",
    )
    first_registry = first.extensions["career_agent_workbench"]["actions"]
    create_action(first_registry, label="First app only", total_stages=1)
    assert len(first.test_client().get("/actions/status").get_json()["actions"]) == 1
    assert second.test_client().get("/actions/status").get_json() == {"actions": []}


def test_packaged_ui_fetches_forms_polls_nonterminal_progress_and_refreshes(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    app = webapp.create_app(_runtime(_paths(tmp_path)), project_root=project)
    _seed(app, "job-a")
    page = app.test_client().get("/").get_data(as_text=True)
    add_page = app.test_client().get("/applications/add").get_data(as_text=True)
    script = project.parent.parent if False else Path(webapp.__file__).parent
    script_text = (script / "static" / "webapp" / "app.js").read_text(encoding="utf-8")
    assert 'data-action-form="workflow"' in page
    assert "Recalculate selected ATS" in page
    assert "Highlight after valid workflow" in page
    assert "/static/webapp/app.js" in page
    assert "/static/webapp/app.js" in add_page
    assert "fetch(form.action" in script_text
    assert 'fetch("/actions/status"' in script_text
    assert 'new Set(["completed", "failed"])' in script_text
    assert 'terminalStates.add("partial")' in script_text
    assert "if (terminalStates.has(action.status))" in script_text
    assert "pollIntervalMilliseconds = 1500" in script_text
    assert "window.location.assign(refreshUrl" in script_text
    assert 'event.submitter?.hasAttribute("formaction")' in script_text
    assert 'document.querySelector("#action-progress-bar")' in script_text
    assert 'document.querySelector("#action-collapse")' in script_text
    assert 'document.querySelector("#action-retry")' in script_text
    assert 'document.querySelector("#action-dismiss")' in script_text
    assert "renderPreferredAction()" in script_text
    assert "action.current_job_id" in script_text
