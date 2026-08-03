from __future__ import annotations

from pathlib import Path

import pytest

from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateStore,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.generic_job_scraper import generic_job_id
from career_agent_workbench.models import JobDetails
from career_agent_workbench.webapp_ingestion import (
    MAX_INGESTION_URLS,
    WebIngestionError,
    ingest_generic_urls,
    ingest_linkedin_urls,
    parse_job_url_batch,
)


def _store(tmp_path: Path) -> ApplicationStateStore:
    workspace = tmp_path / "workspace"
    store = ApplicationStateStore(
        WorkspacePaths(
            database=workspace / "state" / "applications.sqlite3",
            output_dir=workspace / "artifacts",
        )
    )
    store.initialize()
    return store


def _details(
    job_id: str,
    *,
    url: str,
    company: str = "Example Systems",
    title: str = "Fictional Platform Engineer",
    source: str = "linkedin_public",
) -> JobDetails:
    return JobDetails(
        job_id=job_id,
        title=title,
        company=company,
        listed_at="2042-04-10T12:00:00Z",
        job_url=url,
        source=source,
        description=(
            "Responsibilities: build fictional public platform services with Python, "
            "reliable APIs, careful testing, automation, and clear operational reviews. "
            "Qualifications: evidence-based engineering experience."
        ),
        seniority_level="mid_senior",
    )


def _generic_html(*, title: str = "Fictional Reliability Engineer") -> str:
    description = (
        "Design and operate fictional public services with Python, reliable APIs, "
        "automated testing, observability, documentation, and collaborative reviews. "
        "This synthetic posting contains enough bounded detail for the parser."
    )
    return f"""
    <html><head><script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "JobPosting",
      "title": "{title}",
      "description": "{description}",
      "datePosted": "2042-04-10",
      "hiringOrganization": {{"@type": "Organization", "name": "Example Harbor"}}
    }}
    </script></head><body><main>{description}</main></body></html>
    """


def test_linkedin_ingestion_deduplicates_and_keeps_failures_count_only(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    calls: list[str] = []

    def fetcher(url: str) -> JobDetails:
        calls.append(url)
        job_id = url.rstrip("/").rsplit("/", 1)[-1]
        if job_id == "456":
            raise RuntimeError("synthetic hidden provider detail")
        return _details(job_id, url=url)

    batch = parse_job_url_batch(
        "\n".join(
            (
                "https://www.linkedin.com/jobs/view/123/?trk=one",
                "https://www.linkedin.com/jobs/view/123",
                "https://www.linkedin.com/jobs/view/456",
                "https://example.com/not-linkedin",
            )
        ),
        linkedin=True,
    )
    result = ingest_linkedin_urls(store=store, batch=batch, fetcher=fetcher)

    assert (result.created, result.refreshed, result.failed, result.accepted) == (
        1,
        0,
        2,
        1,
    )
    assert calls == [
        "https://www.linkedin.com/jobs/view/123",
        "https://www.linkedin.com/jobs/view/456",
    ]
    record = store.get_application("123")
    assert record.company == "Example Systems"
    assert record.source == "linkedin_public"
    assert record.job_description is not None
    assert record.prompt_job_description is not None
    assert "provider detail" not in repr(result)


def test_existing_ingestion_refresh_preserves_reviewed_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.seed_application(
        ApplicationMetadata(
            job_id="123",
            company="Original Example Systems",
            job_title="Original Fictional Role",
            job_url="https://www.linkedin.com/jobs/view/123",
            source="synthetic",
        ),
        source_text="Original fictional source description.",
        prompt_text="Original fictional prompt description.",
    )
    store.update_application_status(
        "123",
        applied_to="Yes",
        date_applied="2042-04-11",
        notes="Keep this fictional note",
    )
    store.upsert_resume_variant(
        "123",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Draft v1",
            source="synthetic",
            application_resume_yaml="name: Fictional Candidate\n",
            resume_html="<p>Keep this fictional resume</p>",
            resume_pdf=b"keep-this-fictional-pdf",
            ats=AtsFields(score=73),
        ),
    )
    store.select_resume_variant("123", "v1")
    store.store_clo("123", value={"body": "Keep this fictional CLO"})
    store.archive(("123",))
    before = store.get_application("123")
    variants_before = store.list_resume_variants("123")
    batch = parse_job_url_batch(
        "https://www.linkedin.com/jobs/view/123",
        linkedin=True,
    )

    result = ingest_linkedin_urls(
        store=store,
        batch=batch,
        fetcher=lambda url: _details(
            "123",
            url=url,
            company="Refreshed Example Systems",
            title="Refreshed Fictional Role",
        ),
    )

    after = store.get_application("123")
    assert result.refreshed == 1
    assert after.company == "Refreshed Example Systems"
    assert after.job_title == "Refreshed Fictional Role"
    assert after.job_description != before.job_description
    assert after.prompt_job_description != before.prompt_job_description
    assert after.applied_to == before.applied_to
    assert after.date_applied == before.date_applied
    assert after.notes == before.notes
    assert after.archived_at == before.archived_at
    assert after.selected_resume_variant == before.selected_resume_variant
    assert after.resume_variant_selection_mode == before.resume_variant_selection_mode
    assert after.application_resume == before.application_resume
    assert after.resume_html == before.resume_html
    assert after.resume_pdf == before.resume_pdf
    assert after.ats == before.ats
    assert after.cover_letter == before.cover_letter
    assert store.list_resume_variants("123") == variants_before


def test_generic_ingestion_uses_bounded_parser_and_reports_partial_result(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    calls: list[str] = []

    def fetcher(url: str) -> str:
        calls.append(url)
        if "failed" in url:
            raise RuntimeError("synthetic hidden transport detail")
        return _generic_html()

    batch = parse_job_url_batch(
        "\n".join(
            (
                "https://jobs.example.com/opening?utm_source=test",
                "https://jobs.example.com/opening",
                "https://jobs.example.com/failed",
            )
        ),
        linkedin=False,
    )
    result = ingest_generic_urls(store=store, batch=batch, html_fetcher=fetcher)

    assert (result.created, result.refreshed, result.failed) == (1, 0, 1)
    assert calls == [
        "https://jobs.example.com/opening",
        "https://jobs.example.com/failed",
    ]
    record = store.get_application(generic_job_id("https://jobs.example.com/opening"))
    assert record.company == "Example Harbor"
    assert record.source == "generic_url"
    assert record.job_description is not None
    assert record.prompt_job_description is not None
    assert "transport detail" not in repr(result)


def test_ingestion_input_is_bounded() -> None:
    with pytest.raises(WebIngestionError):
        parse_job_url_batch("", linkedin=False)
    with pytest.raises(WebIngestionError):
        parse_job_url_batch(
            "\n".join(
                f"https://jobs.example.com/{index}"
                for index in range(MAX_INGESTION_URLS + 1)
            ),
            linkedin=False,
        )
