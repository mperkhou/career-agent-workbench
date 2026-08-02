from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
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
    create_action,
)


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


def test_composite_action_runs_in_order_stops_first_failure_and_hides_details(
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
    assert status["completed_stages"] == 1
    assert status["total_stages"] == 4
    assert status["current_stage"] == "Refine draft resumes"
    assert status["message"] == "Stage 2 of 4 failed."
    assert "synthetic-private-command-detail" not in repr(status)
    assert all("Running" not in message for message in status["messages"])


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
    before = {
        job_id: store.get_application(job_id)
        for job_id in ("prompt-job", "source-fallback-job")
    }
    variants_before = {
        job_id: store.list_resume_variants(job_id)
        for job_id in ("prompt-job", "source-fallback-job")
    }
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
        assert after.application_resume == original.application_resume
        assert after.resume_html == original.resume_html
        assert after.resume_pdf == original.resume_pdf
        variant_after = store.list_resume_variants(job_id)[0]
        variant_before = variants_before[job_id][0]
        assert variant_after.application_resume == variant_before.application_resume
        assert variant_after.resume_html == variant_before.resume_html
        assert variant_after.resume_pdf == variant_before.resume_pdf
        assert variant_after.parent_variant_key == variant_before.parent_variant_key
    assert store.get_application("no-pdf-job").ats.score is None
    assert store.get_application("no-jod-job").ats.score is None


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
    assert "if (terminalStates.has(action.status))" in script_text
    assert "pollIntervalMilliseconds = 1500" in script_text
    assert "window.location.assign(refreshUrl" in script_text
    assert 'event.submitter?.hasAttribute("formaction")' in script_text
