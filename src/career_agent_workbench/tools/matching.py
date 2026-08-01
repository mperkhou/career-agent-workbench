"""One lazy configured matching tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from career_agent_workbench.config import RuntimeConfig, WorkspaceMember
from career_agent_workbench.models import DatePosted
from career_agent_workbench.tools.jobs import PublicJobService
from career_agent_workbench.workflows.matching import (
    MatchingBounds,
    MatchingWorkflowResult,
    run_matching_from_runtime,
)

_CONFIGURATION_ERROR = "Matching tool configuration is invalid."
_MATCHING_ERROR = "Matching tool failed."

MatchingRunner = Callable[..., Awaitable[MatchingWorkflowResult]]


def register_matching_tools(
    server: FastMCP,
    runtime: RuntimeConfig,
    service: PublicJobService,
    *,
    runner: MatchingRunner = run_matching_from_runtime,
) -> None:
    """Register matching with one captured runtime and public service."""

    @server.tool(
        name="find_matching_linkedin_jobs",
        description="Run configured local matching over public LinkedIn job data.",
    )
    async def find_matching_linkedin_jobs(
        location: Annotated[str, Field(min_length=1, max_length=256)] = (
            "United States"
        ),
        date_posted: DatePosted = "past_week",
        limit_per_query: Annotated[int, Field(ge=1, le=100)] = 10,
        max_queries: Annotated[int, Field(ge=1, le=20)] = 6,
        max_jobs: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict[str, object]:
        try:
            runtime.paths.require(WorkspaceMember.MASTER_RESUME)
            runtime.paths.require(WorkspaceMember.OUTPUT_DIR)
            runtime.paths.require(WorkspaceMember.DATABASE)
        except Exception:
            return {"error": _CONFIGURATION_ERROR}

        try:
            bounds = MatchingBounds(
                location=location,
                date_posted=date_posted,
                workplace_types=("remote", "hybrid"),
                experience_levels=("associate", "mid_senior", "director"),
                job_types=("full_time", "contract"),
                limit_per_query=limit_per_query,
                max_queries=max_queries,
                max_jobs=max_jobs,
            )
            result = await runner(runtime, bounds=bounds, service=service)
            return {
                "queries_planned": result.queries_planned,
                "queries_searched": result.queries_searched,
                "jobs_seen": result.jobs_seen,
                "jobs_seeded": result.jobs_seeded,
            }
        except Exception:  # noqa: BLE001 - matching tool errors are content-free
            return {"error": _MATCHING_ERROR}


__all__ = ["MatchingRunner", "register_matching_tools"]
