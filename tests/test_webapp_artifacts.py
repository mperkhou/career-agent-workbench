from __future__ import annotations

from pathlib import Path

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths
from career_agent_workbench.webapp_artifacts import variant_review


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
    app = webapp.create_app(
        RuntimeConfig(paths=paths, settings=Settings(), env_file=None),
        project_root=project,
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
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="First draft",
            source="synthetic",
            application_resume_yaml="name: Fictional One\nsummary: first\n",
            resume_html="<p>fictional-v1</p>",
            resume_pdf=b"%PDF-fictional-v1",
            ats=AtsFields(score=71, missing_terms="one"),
        ),
    )
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="v2",
            variant_label="Second draft",
            source="synthetic",
            parent_variant_key="v1",
            application_resume_yaml="name: Fictional Two\nsummary: second\n",
            resume_html="<p>fictional-v2</p>",
            resume_pdf=b"%PDF-fictional-v2",
            ats=AtsFields(score=82, missing_terms="two"),
            evidence_packet={"private_value": "synthetic-evidence-secret"},
            external_critique={"raw": "synthetic-provider-secret"},
            critique={
                "proposed_changes": [
                    {"change_id": "change-1", "text": "synthetic-private-change"}
                ],
                "accepted_change_ids": ["change-1"],
            },
            validation={
                "rejected_changes": [{"reason": "synthetic-private-reason"}],
                "is_valid": True,
            },
            model_metadata={
                "model": "synthetic-private-model",
                "review_state": "awaiting_user_review",
            },
        ),
    )
    return app, store, paths


def test_selected_and_exact_variant_artifact_routes(tmp_path: Path) -> None:
    app, store, _paths = _app(tmp_path)
    client = app.test_client()

    selected_html = client.get("/resume-html/fictional-job")
    selected_pdf = client.get("/resumes/fictional-job")
    assert selected_html.status_code == selected_pdf.status_code == 200
    assert selected_html.content_type == "text/html; charset=utf-8"
    assert selected_html.data == b"<p>fictional-v2</p>"
    assert selected_pdf.content_type == "application/pdf"
    assert selected_pdf.data == b"%PDF-fictional-v2"
    assert selected_pdf.headers["Content-Disposition"].startswith("inline;")

    exact = client.get("/resumes/fictional-job/variants/v1/download")
    assert exact.status_code == 200
    assert exact.data == b"%PDF-fictional-v1"
    assert exact.headers["Content-Disposition"].startswith("attachment;")
    html_download = client.get("/resume-html/fictional-job/variants/v1/download")
    assert html_download.data == b"<p>fictional-v1</p>"
    assert html_download.headers["Content-Disposition"].startswith("attachment;")

    before = store.get_workflow_snapshot("fictional-job")
    review = client.get("/resumes/fictional-job/variants")
    assert review.status_code == 200
    page = review.get_data(as_text=True)
    assert 'data-variant-key="v1"' in page
    assert 'data-variant-key="v2"' in page
    assert "summary" in page
    assert store.get_workflow_snapshot("fictional-job") == before

    assert client.get("/resumes/fictional-job/variants/manual").status_code == 404
    assert client.get("/resume-html/missing-job").status_code == 404


def test_variant_review_uses_the_declared_parent(tmp_path: Path) -> None:
    app, store, _paths = _app(tmp_path)
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="manual",
            variant_label="Manual pass",
            source="synthetic",
            parent_variant_key="v1",
            application_resume_yaml=(
                "name: Fictional One\nsummary: first\nmanual_only: true\n"
            ),
            resume_html="<p>fictional-manual</p>",
            resume_pdf=b"%PDF-fictional-manual",
            ats=AtsFields(score=86, missing_terms=""),
        ),
    )

    comparisons = variant_review(store.get_workflow_snapshot("fictional-job"))
    manual = next(
        item for item in comparisons if item["variant"].variant_key == "manual"
    )
    assert manual["parent"] == "v1"
    assert manual["aro_comparison"] == {
        "added_count": 1,
        "removed_count": 0,
        "changed_count": 0,
        "added_fields": ("manual_only",),
        "removed_fields": (),
        "changed_fields": (),
        "truncated": False,
    }
    assert manual["ats_comparison"]["metrics"]["overall"] == {
        "parent": 71,
        "current": 86,
        "delta": 15,
    }

    page = app.test_client().get("/resumes/fictional-job/variants")
    assert page.status_code == 200
    manual_section = page.get_data(as_text=True).split(
        'data-variant-key="manual"', maxsplit=1
    )[1]
    assert "Declared parent: v1" in manual_section
    assert "Added 1, removed 0," in manual_section
    assert "manual_only" in manual_section
    assert "Fictional One" not in manual_section
    assert "Fictional Two" not in manual_section


def test_variant_review_evidence_is_bounded_and_missing_optional_is_explicit(
    tmp_path: Path,
) -> None:
    app, store, _paths = _app(tmp_path)
    comparisons = variant_review(store.get_workflow_snapshot("fictional-job"))
    baseline = next(item for item in comparisons if item["variant"].variant_key == "v1")
    refined = next(item for item in comparisons if item["variant"].variant_key == "v2")
    assert baseline["parent"] is None
    assert baseline["aro_comparison"] is None
    assert baseline["ats_comparison"] is None
    assert baseline["evidence"] == {
        "evidence_packet": "not recorded",
        "external_critique": "not recorded",
        "critique": "not recorded",
        "validation": "not recorded",
        "accepted_count": None,
        "rejected_count": None,
    }
    assert refined["parent"] == "v1"
    assert refined["evidence"]["accepted_count"] == 1
    assert refined["evidence"]["rejected_count"] == 1
    assert refined["review_metadata"]["validation_outcome"] is True
    assert refined["review_metadata"]["review_state"] == "awaiting_user_review"

    page = app.test_client().get("/resumes/fictional-job/variants")
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    assert "No parent comparison applies to this baseline variant." in text
    assert "Accepted evidence</dt><dd>1" in text
    assert "Rejected evidence</dt><dd>1" in text
    for forbidden in (
        "synthetic-evidence-secret",
        "synthetic-provider-secret",
        "synthetic-private-change",
        "synthetic-private-reason",
        "synthetic-private-model",
    ):
        assert forbidden not in text


def test_selection_reset_and_tracker_workbench_links(tmp_path: Path) -> None:
    app, store, _paths = _app(tmp_path)
    client = app.test_client()
    selected = client.post("/resumes/fictional-job/variants/v1/use")
    assert selected.status_code == 302
    pinned = store.get_application("fictional-job")
    assert pinned.selected_resume_variant == "v1"
    assert pinned.resume_variant_selection_mode == "manual"
    assert pinned.resume_pdf == b"%PDF-fictional-v1"

    reset = client.post("/resumes/fictional-job/variants/reset")
    assert reset.status_code == 302
    automatic = store.get_application("fictional-job")
    assert automatic.selected_resume_variant == "v2"
    assert automatic.resume_variant_selection_mode == "auto"
    assert automatic.resume_pdf == b"%PDF-fictional-v2"

    page = client.get("/").get_data(as_text=True)
    for route in (
        "/applications/fictional-job/jod",
        "/resumes/fictional-job/variants",
        "/resumes/fictional-job/edit",
        "/resume-html/fictional-job",
        "/resumes/fictional-job",
        "/applications/fictional-job/cover-letter",
    ):
        assert route in page


def test_resume_editor_exposes_copy_form_with_preserved_view_state(
    tmp_path: Path,
) -> None:
    app, _store, _paths = _app(tmp_path)
    response = app.test_client().get(
        "/resumes/fictional-job/edit"
        "?q=example&status=all&scope=all&sort=company&direction=desc"
    )

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    action = 'action="/resumes/fictional-job/copy-to-downloads"'
    copy_form = page.split(action, maxsplit=1)[1].split("</form>", maxsplit=1)[0]
    assert (
        'method="post"'
        in page.split(action, maxsplit=1)[0].rsplit("<form", maxsplit=1)[1]
    )
    for name, value in (
        ("view_q", "example"),
        ("view_status", "all"),
        ("view_scope", "all"),
        ("view_sort", "company"),
        ("view_direction", "desc"),
    ):
        assert f'name="{name}" value="{value}"' in copy_form


def test_explicit_copy_is_atomic_bounded_and_does_not_clean_up(tmp_path: Path) -> None:
    app, store, paths = _app(tmp_path)
    client = app.test_client()
    downloads = paths.download_dir
    assert downloads is not None
    downloads.mkdir(parents=True)
    unrelated = downloads / "operator-note.txt"
    unrelated.write_text("keep", encoding="utf-8")
    target = downloads / store.get_application("fictional-job").resume_pdf_filename
    target.write_bytes(b"old")

    response = client.post("/resumes/fictional-job/copy-to-downloads")
    assert response.status_code == 302
    assert target.read_bytes() == b"%PDF-fictional-v2"
    assert unrelated.read_text(encoding="utf-8") == "keep"

    store.archive(["fictional-job"])
    assert target.read_bytes() == b"%PDF-fictional-v2"
    assert unrelated.read_text(encoding="utf-8") == "keep"

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    target.unlink()
    target.symlink_to(outside)
    rejected = client.post("/resumes/fictional-job/copy-to-downloads")
    assert rejected.status_code == 400
    assert outside.read_bytes() == b"outside"
