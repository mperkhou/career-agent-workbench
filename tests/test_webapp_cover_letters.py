from __future__ import annotations

from pathlib import Path

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
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
    store.store_jod(
        "fictional-job",
        source_text="Source JOD",
        prompt_text="Prompt JOD",
    )
    store.store_aro("fictional-job", yaml_text="name: Fictional\n")
    store.store_application_artifacts(
        "fictional-job",
        resume_html="<p>resume</p>",
        resume_pdf=b"resume-pdf",
        ats=AtsFields(score=73),
    )
    store.update_application_status(
        "fictional-job", applied_to="No", notes="Keep this note"
    )
    return app, store, paths


def test_blank_edit_save_sanitize_and_artifact_responses_are_isolated(
    tmp_path: Path,
) -> None:
    app, store, _paths = _app(tmp_path)
    client = app.test_client()
    before = store.get_application("fictional-job")

    blank = client.get("/applications/fictional-job/cover-letter")
    assert blank.status_code == 200
    assert 'name="body_html"' in blank.get_data(as_text=True)
    assert store.get_application("fictional-job") == before

    saved = client.post(
        "/applications/fictional-job/cover-letter",
        data={
            "body_html": (
                "<!--removed--><p class='drop'><b>Hello</b> Example</p>"
                "<script>privateMarker()</script>"
                "<div><a href='https://example.com/public'>Details</a> "
                "<a href='javascript:privateMarker()'>Unsafe</a></div>"
            )
        },
    )
    assert saved.status_code == 302
    after = store.get_application("fictional-job")
    assert after.cover_letter is not None
    body_html = after.cover_letter["body_html"]
    assert "privateMarker" not in body_html
    assert "class=" not in body_html
    assert "javascript:" not in body_html
    assert 'href="https://example.com/public"' in body_html
    assert after.cover_letter["body_text"] == "Hello Example\nDetails Unsafe"
    assert after.cover_letter_pdf is not None
    assert after.cover_letter_pdf.startswith(b"%PDF-")

    for field in (
        "job_description",
        "prompt_job_description",
        "application_resume",
        "application_resume_backup",
        "application_resume_backup_target",
        "selected_resume_variant",
        "resume_variant_selection_mode",
        "resume_html",
        "resume_pdf",
        "ats",
        "applied_to",
        "notes",
        "archived_at",
    ):
        assert getattr(after, field) == getattr(before, field)

    inline = client.get("/cover-letters/fictional-job")
    download = client.get("/cover-letters/fictional-job/download")
    assert inline.status_code == download.status_code == 200
    assert inline.content_type == download.content_type == "application/pdf"
    assert inline.data == download.data == after.cover_letter_pdf
    assert inline.headers["Content-Disposition"].startswith("inline;")
    assert download.headers["Content-Disposition"].startswith("attachment;")
    assert client.get("/cover-letters/missing-job").status_code == 404


def test_cover_letter_copy_is_explicit_and_bounded(tmp_path: Path) -> None:
    app, store, paths = _app(tmp_path)
    client = app.test_client()
    client.post(
        "/applications/fictional-job/cover-letter",
        data={"body_html": "<p>Fictional letter</p>"},
    )
    application = store.get_application("fictional-job")
    response = client.post("/cover-letters/fictional-job/copy-to-downloads")
    assert response.status_code == 302
    assert paths.download_dir is not None
    copied = paths.download_dir / application.cover_letter_filename
    assert copied.read_bytes() == application.cover_letter_pdf


def test_cover_letter_render_failure_does_not_partially_mutate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app, store, _paths = _app(tmp_path)
    client = app.test_client()
    before = store.get_application("fictional-job")

    def fail(_value):
        raise RuntimeError("private-render-marker")

    monkeypatch.setattr(webapp, "render_cover_letter", fail)
    response = client.post(
        "/applications/fictional-job/cover-letter",
        data={"body_html": "<p>Rejected</p>"},
    )
    assert response.status_code == 400
    assert "private-render-marker" not in response.get_data(as_text=True)
    assert store.get_application("fictional-job") == before
