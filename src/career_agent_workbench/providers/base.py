"""Provider interface for public job research."""

from __future__ import annotations

from abc import ABC, abstractmethod

from career_agent_workbench.models import (
    JobDetails,
    JobPosting,
    JobRawPayload,
    JobSearchQuery,
)


class JobProvider(ABC):
    """Abstract source of public job data."""

    name: str

    @abstractmethod
    async def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        """Return normalized public postings for a query."""

    @abstractmethod
    async def get_job_details(self, job_id_or_url: str) -> JobDetails:
        """Return normalized public detail data."""

    @abstractmethod
    async def get_job_raw_payload(self, job_id_or_url: str) -> JobRawPayload:
        """Return bounded raw public detail data with its normalized parse."""
