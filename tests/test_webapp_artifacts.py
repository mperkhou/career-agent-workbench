from __future__ import annotations

from pathlib import Path

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths


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
