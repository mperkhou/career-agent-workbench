"""Bounded public job-search orchestration."""

from __future__ import annotations

from career_agent_workbench.models import (
    JobDetails,
    JobRawPayload,
    JobSearchQuery,
    JobSearchResult,
)
from career_agent_workbench.providers.base import JobProvider


class JobSearchService:
    """Apply caller-provided limits around one explicit provider."""

    def __init__(self, *, provider: JobProvider, max_results: int) -> None:
        self._provider = provider
        self._max_results = min(max(1, int(max_results)), 100)

    async def search(self, query: JobSearchQuery) -> JobSearchResult:
        effective = query.model_copy(
            update={"limit": min(query.limit, self._max_results)}
        )
        jobs = await self._provider.search_jobs(effective)
        excluded = effective.exclude_job_ids
        selected = [job for job in jobs if job.job_id not in excluded][
            : effective.limit
        ]
        return JobSearchResult(
            query=effective,
            count=len(selected),
            jobs=selected,
            provider=self._provider.name,
        )

    async def get_details(self, job_id_or_url: str) -> JobDetails:
        return await self._provider.get_job_details(job_id_or_url)

    async def get_raw_payload(self, job_id_or_url: str) -> JobRawPayload:
        return await self._provider.get_job_raw_payload(job_id_or_url)
