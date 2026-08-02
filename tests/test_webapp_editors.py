from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
    ResumeVariantWrite,
    workflow_revision_token,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths


def _diagnostics(score: int = 88):
    proxy = SimpleNamespace(
        overall_score=score,
        parsing_score=91,
        keyword_match_score=82,
        semantic_match_score=84,
        formatting_risk="Low",
        missing_high_value_terms=("fictional term",),
    )
    components = SimpleNamespace(
        overall_score=score,
        parsing_score=91,
        keyword_match_score=82,
        semantic_match_score=84,
        formatting_score=100,
        formatting_risk="Low",
    )
    return SimpleNamespace(
        score=proxy,
        component_scores=components,
        matched_terms=(),
        unmatched_weighted_terms=(),
    )


def _app(tmp_path: Path):
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths(
        root=workspace,
        database=workspace / "state" / "applications.sqlite3",
        output_dir=workspace / "output",
        download_dir=workspace / "downloads",
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    ats_calls: list[tuple[bytes, str]] = []

    def html_renderer(*, resume):
        return f"<p>{resume.get('name', 'missing')}</p>"

    def pdf_renderer(html: str):
        return b"pdf:" + html.encode("utf-8")

    def ats_calculator(*, resume_pdf: bytes, job_description: str):
        ats_calls.append((resume_pdf, job_description))
        return _diagnostics()

    app = webapp.create_app(
        RuntimeConfig(paths=paths, settings=Settings(), env_file=None),
        project_root=project,
        resume_html_renderer=html_renderer,
        resume_pdf_renderer=pdf_renderer,
        ats_calculator=ats_calculator,
    )
    store = app.extensions["career_agent_workbench"]["store"]
    store.upsert_application(
        ApplicationMetadata(
            job_id="fictional-job",
            company="Example Systems",
            job_title="Synthetic Engineer",
            job_url="https://example.com/jobs/fictional-job",
            source="synthetic",
        )
    )
    store.store_jod(
        "fictional-job",
        source_text="Source responsibility\nRemoved detail",
        prompt_text="Source responsibility",
    )
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="First draft",
            source="synthetic",
            application_resume_yaml="name: First\nskills:\n  - Python\n",
            resume_html="<p>first-old</p>",
            resume_pdf=b"pdf-first-old",
            ats=AtsFields(score=70),
            evidence_packet={"grounded": True},
        ),
    )
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="v2",
            variant_label="Second draft",
            source="synthetic",
            parent_variant_key="v1",
            application_resume_yaml="name: Second\nskills:\n  - Python\n",
            resume_html="<p>second-old</p>",
            resume_pdf=b"pdf-second-old",
            ats=AtsFields(score=80),
            evidence_packet={"grounded": True},
        ),
    )
    return app, store, ats_calls


def test_jod_view_save_refresh_and_skip_are_document_safe(tmp_path: Path) -> None:
    app, store, ats_calls = _app(tmp_path)
    client = app.test_client()
    before = store.get_workflow_snapshot("fictional-job")
    page = client.get("/applications/fictional-job/jod")
    assert page.status_code == 200
    rendered = page.get_data(as_text=True)
    assert "Removed detail" in rendered
    assert "data-jod-diff" in rendered
    assert store.get_workflow_snapshot("fictional-job") == before

    response = client.post(
        "/applications/fictional-job/jod",
        data={
            "source_text": "  Exact source JOD\n",
            "prompt_text": "Exact prompt JOD\n",
        },
    )
    assert response.status_code == 302
    assert "result=refreshed" in response.headers["Location"]
    refreshed = store.get_application("fictional-job")
    assert refreshed.job_description == "  Exact source JOD\n"
    assert refreshed.prompt_job_description == "Exact prompt JOD\n"
    assert ats_calls[-1] == (b"pdf-second-old", "Exact prompt JOD")
    assert refreshed.ats.score == 88
    assert (
        refreshed.selected_resume_variant == before.application.selected_resume_variant
    )
    assert refreshed.resume_variant_selection_mode == "auto"
    assert refreshed.application_resume == before.application.application_resume
    assert refreshed.resume_html == before.application.resume_html
    assert refreshed.resume_pdf == before.application.resume_pdf
    assert store.get_resume_variant("fictional-job", "v1") == before.variants[0]
    selected_after_refresh = store.get_resume_variant("fictional-job", "v2")
    assert selected_after_refresh.parent_variant_key == "v1"
    assert (
        selected_after_refresh.application_resume
        == before.variants[1].application_resume
    )
    assert selected_after_refresh.resume_html == before.variants[1].resume_html
    assert selected_after_refresh.resume_pdf == before.variants[1].resume_pdf

    old_ats = refreshed.ats
    skipped = client.post(
        "/applications/fictional-job/jod",
        data={"source_text": "", "prompt_text": ""},
    )
    assert skipped.status_code == 302
    assert "result=skipped" in skipped.headers["Location"]
    after_skip = store.get_application("fictional-job")
    assert after_skip.job_description == ""
    assert after_skip.prompt_job_description == ""
    assert after_skip.ats == old_ats
    assert len(ats_calls) == 1


def test_resume_save_sync_revert_and_conflict_are_active_target_scoped(
    tmp_path: Path,
) -> None:
    app, store, ats_calls = _app(tmp_path)
    client = app.test_client()
    initial = store.get_workflow_snapshot("fictional-job")
    token = workflow_revision_token(initial.edit_revision)
    page = client.get("/resumes/fictional-job/edit")
    assert page.status_code == 200
    assert "Target: <strong>v2</strong>" in page.get_data(as_text=True)
    assert store.get_workflow_snapshot("fictional-job") == initial

    saved_response = client.post(
        "/resumes/fictional-job/edit",
        data={
            "revision": token,
            "yaml_text": "name: Edited\nskills:\n  - Python\n  - Testing\n",
        },
    )
    assert saved_response.status_code == 302
    saved = store.get_workflow_snapshot("fictional-job")
    assert saved.application.selected_resume_variant == "v2"
    assert saved.application.resume_variant_selection_mode == "auto"
    assert saved.application.application_resume == {
        "name": "Edited",
        "skills": ("Python", "Testing"),
    }
    assert saved.application.resume_html == "<p>Edited</p>"
    assert saved.application.resume_pdf == b"pdf:<p>Edited</p>"
    assert saved.application.ats.score == 88
    assert saved.application.application_resume_backup == (
        initial.application.application_resume
    )
    assert saved.application.application_resume_backup_target == "v2"
    assert saved.variants[0] == initial.variants[0]
    assert (
        saved.variants[1].parent_variant_key == initial.variants[1].parent_variant_key
    )
    assert saved.variants[1].evidence_packet == initial.variants[1].evidence_packet
    assert ats_calls[-1][1] == "Source responsibility"

    sync_token = workflow_revision_token(saved.edit_revision)
    sync = client.post(
        "/resumes/fictional-job/edit/sync",
        data={"revision": sync_token},
    )
    assert sync.status_code == 302
    synced = store.get_workflow_snapshot("fictional-job")
    assert synced.active_resume_yaml == saved.active_resume_yaml
    assert synced.application.application_resume_backup == (
        saved.application.application_resume_backup
    )
    assert synced.application.application_resume_backup_target == "v2"

    revert = client.post(
        "/resumes/fictional-job/edit/revert",
        data={"revision": workflow_revision_token(synced.edit_revision)},
    )
    assert revert.status_code == 302
    reverted = store.get_workflow_snapshot("fictional-job")
    assert reverted.application.application_resume == (
        initial.application.application_resume
    )
    assert reverted.application.application_resume_backup == (
        saved.application.application_resume
    )
    assert reverted.application.selected_resume_variant == "v2"

    stale_token = workflow_revision_token(reverted.edit_revision)
    store.select_resume_variant("fictional-job", "v1")
    before_conflict = store.get_workflow_snapshot("fictional-job")
    conflict = client.post(
        "/resumes/fictional-job/edit",
        data={"revision": stale_token, "yaml_text": "name: Rejected\n"},
    )
    assert conflict.status_code == 409
    assert store.get_workflow_snapshot("fictional-job") == before_conflict


def test_resume_editor_rejects_invalid_yaml_without_partial_write(
    tmp_path: Path,
) -> None:
    app, store, _ats_calls = _app(tmp_path)
    client = app.test_client()
    before = store.get_workflow_snapshot("fictional-job")
    response = client.post(
        "/resumes/fictional-job/edit",
        data={
            "revision": workflow_revision_token(before.edit_revision),
            "yaml_text": "- not\n- a\n- mapping\n",
        },
    )
    assert response.status_code == 400
    assert store.get_workflow_snapshot("fictional-job") == before
