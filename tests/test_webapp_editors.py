from __future__ import annotations

import json
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
from career_agent_workbench.webapp_resume_fields import resume_field_model


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
    calls_before_conflict = len(ats_calls)
    conflict = client.post(
        "/resumes/fictional-job/edit",
        data={"revision": stale_token, "yaml_text": "name: Rejected\n"},
    )
    assert conflict.status_code == 409
    assert store.get_workflow_snapshot("fictional-job") == before_conflict
    assert len(ats_calls) == calls_before_conflict


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


def test_structured_resume_save_is_lossless_revision_bound_and_ats_after_validation(
    tmp_path: Path,
) -> None:
    app, store, ats_calls = _app(tmp_path)
    client = app.test_client()
    source_yaml = """
header_top:
  line_1_name_header_text: Jules Example
  contact_items: [jules@example.test, Portfolio]
  links:
    - label: Portfolio
      url: https://portfolio.example.test
      unknown_link: preserve
professional_summary:
  paragraph: Builds synthetic systems.
professional_experience:
  jobs:
    - order: '01'
      line_1:
        company_name_text: Example Systems
        position_name_text: Engineer
        position_dates_text: 2024-Present
        unknown_line: preserve
      bullet_points:
        - text: Built bounded tools.
          evidence_ids: [example-1]
unknown_section:
  preserve: [one, two]
""".lstrip()
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="v2",
            variant_label="Second draft",
            source="synthetic",
            parent_variant_key="v1",
            application_resume_yaml=source_yaml,
            resume_html="<p>before</p>",
            resume_pdf=b"pdf-before",
        ),
    )
    before = store.get_workflow_snapshot("fictional-job")
    page = client.get("/resumes/fictional-job/edit")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Save structured fields and render" in html
    assert "Advanced YAML escape hatch" in html
    assert 'target="_blank" rel="noopener noreferrer">HTML preview</a>' in html
    assert 'target="_blank" rel="noopener noreferrer">PDF preview</a>' in html
    assert store.get_workflow_snapshot("fictional-job") == before
    schema = client.get("/resumes/structured-fields")
    assert schema.status_code == 200
    assert schema.get_json()["lossless_unknown_values"] is True
    assert store.get_workflow_snapshot("fictional-job") == before

    payload = resume_field_model(before.application.application_resume)
    payload["professional_experience"]["jobs"][0]["role"] = "Senior Engineer"
    payload["header_top"]["contact_items"] = list(
        reversed(payload["header_top"]["contact_items"])
    )
    calls_before = len(ats_calls)
    response = client.post(
        "/resumes/fictional-job/edit",
        data={
            "revision": workflow_revision_token(before.edit_revision),
            "structured_payload": json.dumps(payload),
        },
    )

    assert response.status_code == 302
    saved = store.get_workflow_snapshot("fictional-job")
    assert saved.application.selected_resume_variant == "v2"
    assert saved.application.resume_variant_selection_mode == "auto"
    assert saved.application.application_resume["unknown_section"] == {
        "preserve": ("one", "two")
    }
    job = saved.application.application_resume["professional_experience"]["jobs"][0]
    assert job["line_1"]["position_name_text"] == "Senior Engineer"
    assert job["line_1"]["unknown_line"] == "preserve"
    assert job["bullet_points"][0]["evidence_ids"] == ("example-1",)
    assert saved.application.application_resume_backup == (
        before.application.application_resume
    )
    assert len(ats_calls) == calls_before + 1

    invalid = resume_field_model(saved.application.application_resume)
    invalid["header_top"]["contact_items"].append(
        invalid["header_top"]["contact_items"][0]
    )
    before_invalid = store.get_workflow_snapshot("fictional-job")
    invalid_calls = len(ats_calls)
    rejected = client.post(
        "/resumes/fictional-job/edit",
        data={
            "revision": workflow_revision_token(before_invalid.edit_revision),
            "structured_payload": json.dumps(invalid),
        },
    )
    assert rejected.status_code == 400
    assert store.get_workflow_snapshot("fictional-job") == before_invalid
    assert len(ats_calls) == invalid_calls
