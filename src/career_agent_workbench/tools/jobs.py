"""Workspace-independent public job tools."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, Protocol

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from career_agent_workbench.errors import CareerAgentWorkbenchError
from career_agent_workbench.models import (
    DatePosted,
    ExperienceLevel,
    JobDetails,
    JobRawPayload,
    JobSearchQuery,
    JobSearchResult,
    JobType,
    SortBy,
    WorkplaceType,
)

_PUBLIC_TOOL_ERROR = "Public job tool failed."


class PublicJobService(Protocol):
    """Narrow public capability used by the three MCP tools."""

    def search(self, query: JobSearchQuery) -> Awaitable[JobSearchResult]: ...

    def get_details(self, job_id_or_url: str) -> Awaitable[JobDetails]: ...

    def get_raw_payload(self, job_id_or_url: str) -> Awaitable[JobRawPayload]: ...


def register_job_tools(server: FastMCP, service: PublicJobService) -> None:
    """Register three bounded public tools against one injected service."""

    @server.tool(
        name="search_linkedin_jobs",
        description="Search bounded guest-accessible public LinkedIn job data.",
    )
    async def search_linkedin_jobs(
        keywords: Annotated[str, Field(min_length=1, max_length=256)],
        location: Annotated[str, Field(min_length=1, max_length=256)],
        date_posted: DatePosted = "any_time",
        job_type: JobType | None = None,
        workplace_type: WorkplaceType | None = None,
        experience_level: ExperienceLevel | None = None,
        sort_by: SortBy = "recent",
        distance: Annotated[int | None, Field(ge=0, le=100)] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
        page: Annotated[int, Field(ge=0, le=10_000)] = 0,
        exclude_job_ids: Annotated[
            list[Annotated[str, Field(min_length=1, max_length=128)]],
            Field(max_length=500),
        ]
        | None = None,
    ) -> dict[str, object]:
        try:
            query = JobSearchQuery(
                keywords=keywords,
                location=location,
                date_posted=date_posted,
                job_type=job_type,
                workplace_type=workplace_type,
                experience_level=experience_level,
                sort_by=sort_by,
                distance=distance,
                limit=limit,
                page=page,
                exclude_job_ids=set(exclude_job_ids or ()),
            )
            return (await service.search(query)).model_dump(mode="json")
        except CareerAgentWorkbenchError as exc:
            return {"error": str(exc)}
        except Exception:  # noqa: BLE001 - public tool errors are content-free
            return {"error": _PUBLIC_TOOL_ERROR}

    @server.tool(
        name="get_linkedin_job_details",
        description="Get normalized details for one public LinkedIn job.",
    )
    async def get_linkedin_job_details(
        job_id_or_url: Annotated[str, Field(min_length=1, max_length=4_096)],
    ) -> dict[str, object]:
        try:
            return (await service.get_details(job_id_or_url)).model_dump(mode="json")
        except CareerAgentWorkbenchError as exc:
            return {"error": str(exc)}
        except Exception:  # noqa: BLE001 - public tool errors are content-free
            return {"error": _PUBLIC_TOOL_ERROR}

    @server.tool(
        name="get_linkedin_job_raw_payload",
        description="Get one bounded in-memory public LinkedIn detail payload.",
    )
    async def get_linkedin_job_raw_payload(
        job_id_or_url: Annotated[str, Field(min_length=1, max_length=4_096)],
    ) -> dict[str, object]:
        try:
            return (await service.get_raw_payload(job_id_or_url)).model_dump(
                mode="json"
            )
        except CareerAgentWorkbenchError as exc:
            return {"error": str(exc)}
        except Exception:  # noqa: BLE001 - public tool errors are content-free
            return {"error": _PUBLIC_TOOL_ERROR}


__all__ = ["PublicJobService", "register_job_tools"]
