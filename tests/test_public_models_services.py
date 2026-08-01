from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.models import (
    JobDetails,
    JobPosting,
    JobRawPayload,
    JobSearchQuery,
)
from career_agent_workbench.services import JobSearchService


def _posting(job_id: str) -> JobPosting:
    return JobPosting(
        job_id=job_id,
        title=f"Synthetic Role {job_id}",
        company="Example Harbor Systems",
        job_url=f"https://jobs.example.invalid/postings/{job_id}",
    )


class _SyntheticProvider:
    name = "synthetic_public"

    def __init__(self) -> None:
        self.query: JobSearchQuery | None = None
        self.detail_input: str | None = None
        self.raw_input: str | None = None

    async def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        self.query = query
        return [_posting("900000001"), _posting("900000002"), _posting("900000003")]

    async def get_job_details(self, job_id_or_url: str) -> JobDetails:
        self.detail_input = job_id_or_url
        return JobDetails(**_posting("900000004").model_dump())

    async def get_job_raw_payload(self, job_id_or_url: str) -> JobRawPayload:
        self.raw_input = job_id_or_url
        parsed = JobDetails(**_posting("900000005").model_dump())
        payload = "<p>Synthetic</p>"
        return JobRawPayload(
            job_id="900000005",
            detail_url="https://jobs.example.invalid/postings/900000005",
            status_code=200,
            content_type="text/html",
            payload_chars=len(payload),
            payload=payload,
            parsed=parsed,
        )


def test_models_strip_enforce_literals_bounds_and_strict_extras() -> None:
    query = JobSearchQuery(
        keywords="  platform engineer  ",
        location="  Example City  ",
        date_posted="past_week",
        job_type="full_time",
        workplace_type="remote",
        experience_level="associate",
        sort_by="recent",
        page=10_000,
        exclude_job_ids={" 900000001 "},
    )
    assert query.keywords == "platform engineer"
    assert query.location == "Example City"
    assert query.exclude_job_ids == {"900000001"}

    invalid_cases = (
        {"keywords": "", "location": "Example City"},
        {"keywords": "x" * 257, "location": "Example City"},
        {"keywords": "role", "location": "x" * 257},
        {"keywords": "role", "location": "city", "page": 10_001},
        {"keywords": "role", "location": "city", "job_type": "unknown"},
        {"keywords": "role", "location": "city", "unexpected": True},
        {
            "keywords": "role",
            "location": "city",
            "exclude_job_ids": {str(index) for index in range(501)},
        },
    )
    for values in invalid_cases:
        with pytest.raises(ValidationError):
            JobSearchQuery(**values)


def test_models_enforce_safe_urls_display_description_and_raw_payload() -> None:
    marker = "SENSITIVE-URL-MARKER"
    for value in (
        f"https://user:{marker}@jobs.example.invalid/posting",
        "ftp://jobs.example.invalid/posting",
        "https:///missing-host",
    ):
        with pytest.raises(ValidationError) as captured:
            JobPosting(job_id="900000010", title="Role", job_url=value)
        assert marker not in str(captured.value)

    with pytest.raises(ValidationError):
        JobPosting(job_id="x" * 129, title="Role")
    with pytest.raises(ValidationError):
        JobPosting(job_id="900000010", title="x" * 1_025)
    with pytest.raises(ValidationError):
        JobDetails(
            job_id="900000010",
            title="Role",
            description="x" * 500_001,
        )

    parsed = JobDetails(job_id="900000011", title="Synthetic Role")
    payload = "<p>Synthetic</p>"
    raw = JobRawPayload(
        job_id="900000011",
        detail_url="https://jobs.example.invalid/postings/900000011",
        status_code=200,
        payload_chars=len(payload),
        payload=payload,
        parsed=parsed,
    )
    assert raw.payload_chars == len(raw.payload)
    assert raw.payload not in repr(raw)
    with pytest.raises(ValidationError):
        JobRawPayload(
            job_id="900000011",
            detail_url="https://jobs.example.invalid/postings/900000011",
            status_code=200,
            payload_chars=17,
            payload="<p>Synthetic</p>",
            parsed=parsed,
        )
    with pytest.raises(ValidationError):
        JobRawPayload(
            job_id="900000011",
            detail_url="https://jobs.example.invalid/postings/900000011",
            status_code=200,
            payload_chars=2_000_001,
            payload="x" * 2_000_001,
            parsed=parsed,
        )


def test_service_caps_excludes_counts_and_delegates() -> None:
    provider = _SyntheticProvider()
    service = JobSearchService(provider=provider, max_results=2)
    query = JobSearchQuery(
        keywords="platform",
        location="Example City",
        limit=50,
        exclude_job_ids={"900000001"},
    )

    result = asyncio.run(service.search(query))
    assert provider.query is not None
    assert provider.query.limit == 2
    assert result.provider == "synthetic_public"
    assert result.count == 2
    assert [job.job_id for job in result.jobs] == ["900000002", "900000003"]

    details = asyncio.run(service.get_details("synthetic-detail-input"))
    raw = asyncio.run(service.get_raw_payload("synthetic-raw-input"))
    assert details.job_id == "900000004"
    assert raw.job_id == "900000005"
    assert provider.detail_input == "synthetic-detail-input"
    assert provider.raw_input == "synthetic-raw-input"
    for prohibited in ("submit", "apply", "contact"):
        assert not hasattr(service, prohibited)


def test_all_new_modules_import_off_root_without_env_or_side_effect(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).parents[1] / "src"
    modules = [
        "career_agent_workbench.api_client",
        "career_agent_workbench.errors",
        "career_agent_workbench.generic_job_scraper",
        "career_agent_workbench.jod",
        "career_agent_workbench.llm",
        "career_agent_workbench.models",
        "career_agent_workbench.models.jobs",
        "career_agent_workbench.ollama",
        "career_agent_workbench.providers",
        "career_agent_workbench.providers.base",
        "career_agent_workbench.providers.linkedin_public",
        "career_agent_workbench.query_optimizer",
        "career_agent_workbench.services",
        "career_agent_workbench.services.job_search",
    ]
    script = (
        "import importlib,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        f"[importlib.import_module(name) for name in {modules!r}];"
        "from career_agent_workbench.config import WorkspacePaths;"
        "assert WorkspacePaths().root is None"
    )
    before = set(tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        env={},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert set(tmp_path.iterdir()) == before
    assert WorkspacePaths().root is None
    assert ".env" not in os.listdir(tmp_path)
