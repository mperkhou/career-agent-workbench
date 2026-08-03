from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths
from career_agent_workbench.webapp_tracker import (
    TRACKER_SORTS,
    TrackerView,
    tracker_applications,
    tracker_rows,
)


def _paths(tmp_path: Path) -> WorkspacePaths:
    workspace = tmp_path / "workspace"
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


def _app(tmp_path: Path):
    paths = _paths(tmp_path)
    runtime = RuntimeConfig(paths=paths, settings=Settings(), env_file=None)
    return webapp.create_app(runtime, project_root=tmp_path / "no-project")


def _seed(
    app,
    job_id: str,
    *,
    company: str,
    title: str,
    source: str = "synthetic",
    matched: str | None = None,
    posted: str | None = None,
    experience: str | None = None,
):
    store = app.extensions["career_agent_workbench"]["store"]
    return store.upsert_application(
        ApplicationMetadata(
            job_id=job_id,
            company=company,
            job_title=title,
            job_url=f"https://example.com/jobs/{job_id}",
            source=source,
            date_matched=matched,
            date_posted=posted,
            experience_level=experience,
        )
    )


def _view_form(**overrides: str) -> dict[str, str]:
    values = {
        "view_q": "",
        "view_status": "all",
        "view_scope": "active",
        "view_sort": "updated",
        "view_direction": "desc",
    }
    values.update(overrides)
    return values


def test_tracker_filters_searches_sorts_and_rebuilds_local_view(tmp_path: Path) -> None:
    app = _app(tmp_path)
    store = app.extensions["career_agent_workbench"]["store"]
    _seed(
        app,
        "job-b",
        company="Beta Systems",
        title="Platform Engineer",
        source="fictional-board",
        matched="2026-02-02",
        posted="2026-02-01",
        experience="Senior",
    )
    _seed(
        app,
        "job-a",
        company="Alpha Systems",
        title="Reliability Engineer",
        matched="2026-01-02",
        posted="2026-01-01",
    )
    _seed(
        app,
        "job-aa",
        company="Alpha Systems",
        title="Reliability Engineer",
    )
    _seed(app, "job-c", company="Archived Co", title="Builder")
    store.update_application_status(
        "job-b",
        applied_to="Yes",
        date_applied="2026-02-03",
        notes="needle operator note",
    )
    store.store_application_artifacts(
        "job-b",
        resume_pdf=b"fictional-pdf",
        ats=AtsFields(score=84, formatting_risk="Low"),
    )
    store.store_clo("job-b", value={"body": "fictional"})
    store.archive(("job-c",))

    client = app.test_client()
    response = client.get(
        "/",
        query_string={
            "q": "needle",
            "status": "Yes",
            "scope": "all",
            "sort": "company",
            "direction": "asc",
        },
    )
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "job-b" in page
    assert "job-a" not in page
    assert "job-c" not in page
    assert "fictional-board" in page
    assert "2026-02-03" in page
    assert "Senior" in page
    assert "Selection: none" in page
    assert "(auto)" in page
    assert "ATS: 84" in page
    assert "CLO: yes" in page
    assert 'target="_blank" rel="noopener noreferrer"' in page
    assert (
        'href="/applications/add?q=needle&amp;status=Yes&amp;scope=all&amp;'
        'sort=company&amp;direction=asc"'
    ) in page

    company_view = TrackerView(scope="all", sort="company", direction="asc")
    assert [item.job_id for item in tracker_applications(store, company_view)] == [
        "job-a",
        "job-aa",
        "job-c",
        "job-b",
    ]
    for selected_sort in TRACKER_SORTS:
        ordered = tracker_applications(
            store,
            TrackerView(scope="all", sort=selected_sort, direction="asc"),
        )
        assert {item.job_id for item in ordered} == {
            "job-a",
            "job-aa",
            "job-b",
            "job-c",
        }

    for query in (
        {"status": "invalid"},
        {"scope": "invalid"},
        {"sort": "invalid"},
        {"direction": "invalid"},
        {"q": "bad\nquery"},
    ):
        assert client.get("/", query_string=query).status_code == 400


def test_tracker_rows_expose_dense_status_badges_timestamps_and_safe_targets(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    store = app.extensions["career_agent_workbench"]["store"]
    _seed(
        app,
        "dense-job",
        company="Example Systems " + "X" * 200,
        title="Synthetic Engineer " + "Y" * 200,
    )
    store.upsert_resume_variant(
        "dense-job",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Draft v1",
            source="synthetic",
            application_resume_yaml="name: Fictional\n",
            resume_html="<p>Fictional</p>",
            resume_pdf=b"pdf-v1",
            ats=AtsFields(score=70, updated_at="2026-08-02T12:00:00+00:00"),
        ),
    )
    store.upsert_resume_variant(
        "dense-job",
        ResumeVariantWrite(
            variant_key="manual",
            variant_label="Manual draft",
            source="synthetic",
            parent_variant_key="v1",
            application_resume_yaml="name: Fictional Manual\n",
            resume_html="<p>Fictional manual</p>",
            resume_pdf=b"pdf-manual",
            ats=AtsFields(score=82, updated_at="2026-08-02T13:00:00+00:00"),
        ),
    )
    store.store_clo(
        "dense-job",
        value={"body_html": "<p>Synthetic letter</p>"},
        pdf_content=b"pdf-clo",
    )
    store.update_application_status(
        "dense-job",
        applied_to="Accepted for interview",
        notes="bounded " + "note " * 100,
    )

    before = store.get_workflow_snapshot("dense-job")
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert store.get_workflow_snapshot("dense-job") == before
    soup = BeautifulSoup(response.get_data(as_text=True), "html.parser")
    row = soup.select_one('tr[data-application-status="interview"]')
    assert row is not None
    assert "status-interview" in row.get("class", [])
    badges = {
        badge.get_text(strip=True): badge for badge in row.select(".variant-badge")
    }
    assert set(badges) == {"v1", "v2", "manual"}
    assert "available" in badges["v1"].get("class", [])
    assert "selected" in badges["manual"].get("class", [])
    assert "available" not in badges["v2"].get("class", [])
    assert "2026-08-02T13:00:00+00:00" in row.get_text(" ", strip=True)
    assert len(row.select(".bounded-value")) >= 4

    expected_targets = (
        "/resume-html/dense-job",
        "/resumes/dense-job",
        "/cover-letters/dense-job",
    )
    for href in expected_targets:
        link = row.find("a", href=href)
        assert link is not None
        assert link.get("target") == "_blank"
        assert set(link.get("rel", [])) == {"noopener", "noreferrer"}
    for href in (
        "/resumes/dense-job/download",
        "/cover-letters/dense-job/download",
    ):
        assert row.find("a", href=href).get("target") is None
    edit_link = row.find("a", href=lambda value: bool(value and "/edit?" in value))
    assert edit_link is not None
    assert edit_link.get("target") is None

    rows = tracker_rows(
        _paths(tmp_path).database,
        tracker_applications(store, TrackerView()),
    )
    assert rows[0].status_key == "interview"
    assert rows[0].variant_keys == ("v1", "manual")


def test_gets_are_read_only_and_lifecycle_mutations_preserve_resume_state(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    store = app.extensions["career_agent_workbench"]["store"]
    _seed(app, "job-a", company="Alpha", title="Engineer")
    store.seed_application(
        ApplicationMetadata(
            job_id="job-a",
            company="Alpha",
            job_title="Engineer",
            job_url="https://example.com/jobs/job-a",
            source="synthetic",
        ),
        source_text="A sufficiently bounded fictional source description.",
        prompt_text="Fictional prompt description.",
    )
    store.upsert_resume_variant(
        "job-a",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Draft v1",
            source="synthetic",
            application_resume_yaml="name: Fictional Candidate\n",
            resume_html="<p>Fictional resume</p>",
            resume_pdf=b"fictional-resume-pdf",
            ats=AtsFields(score=71),
        ),
    )
    before = store.get_application("job-a")
    before_variants = store.list_resume_variants("job-a")

    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/applications/add").status_code == 200
    assert client.get("/actions/status").status_code == 200
    assert store.get_application("job-a") == before
    assert store.list_resume_variants("job-a") == before_variants

    view = _view_form(
        view_q="Alpha",
        view_scope="all",
        view_sort="company",
        view_direction="asc",
    )
    response = client.post(
        "/applications/job-a",
        data={
            **view,
            "applied_to": "Accepted for interview",
            "date_applied": "2026-08-02",
            "notes": "Synthetic follow-up",
        },
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith(
        "/?q=Alpha&status=all&scope=all&sort=company&direction=asc"
    )
    updated = store.get_application("job-a")
    assert updated.applied_to == "Accepted for interview"
    assert updated.date_applied == "2026-08-02"
    assert updated.notes == "Synthetic follow-up"
    assert updated.selected_resume_variant == before.selected_resume_variant
    assert updated.resume_variant_selection_mode == before.resume_variant_selection_mode
    assert updated.application_resume == before.application_resume
    assert updated.resume_html == before.resume_html
    assert updated.resume_pdf == before.resume_pdf
    assert store.list_resume_variants("job-a") == before_variants

    archived = client.post(
        "/applications/archive",
        data={**view, "job_id": "job-a"},
    )
    assert archived.status_code == 302
    assert store.get_application("job-a").archived_at is not None
    assert store.list_resume_variants("job-a") == before_variants
    restored = client.post(
        "/applications/unarchive",
        data={**view, "job_id": "job-a"},
    )
    assert restored.status_code == 302
    assert store.get_application("job-a").archived_at is None
    assert store.list_resume_variants("job-a") == before_variants


def test_lifecycle_rejects_partial_or_unconfirmed_mutations(tmp_path: Path) -> None:
    app = _app(tmp_path)
    store = app.extensions["career_agent_workbench"]["store"]
    _seed(app, "job-a", company="Alpha", title="Engineer")
    _seed(app, "job-b", company="Beta", title="Engineer")
    client = app.test_client()
    view = _view_form()

    invalid_updates = (
        {**view, "applied_to": "invalid", "date_applied": "", "notes": "x"},
        {**view, "applied_to": "Yes", "date_applied": "08/02/2026", "notes": "x"},
        {**view, "applied_to": "Yes", "date_applied": ""},
    )
    for payload in invalid_updates:
        assert client.post("/applications/job-a", data=payload).status_code == 400
    assert store.get_application("job-a").applied_to == "No"

    partial = client.post(
        "/applications/archive",
        data={**view, "job_id": ["job-a", "missing-job"]},
    )
    assert partial.status_code == 400
    assert store.get_application("job-a").archived_at is None

    unconfirmed = client.post(
        "/applications/delete",
        data={**view, "job_id": "job-a", "confirm_delete": "no"},
    )
    assert unconfirmed.status_code == 400
    assert store.get_application("job-a").job_id == "job-a"

    confirmed = client.post(
        "/applications/delete",
        data={**view, "job_id": "job-a", "confirm_delete": "delete"},
    )
    assert confirmed.status_code == 302
    assert [item.job_id for item in store.list_applications("all")] == ["job-b"]


@pytest.mark.parametrize("route", ["/applications/archive", "/applications/unarchive"])
def test_bulk_lifecycle_requires_identifiers(tmp_path: Path, route: str) -> None:
    app = _app(tmp_path)
    assert app.test_client().post(route, data=_view_form()).status_code == 400
