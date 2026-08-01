"""Small stdio FastMCP server for public jobs and configured matching."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from career_agent_workbench.config import RuntimeConfig, load_runtime_config
from career_agent_workbench.providers import LinkedInPublicJobsProvider
from career_agent_workbench.services import JobSearchService
from career_agent_workbench.tools import register_job_tools, register_matching_tools
from career_agent_workbench.tools.jobs import PublicJobService
from career_agent_workbench.tools.matching import MatchingRunner

_INSTRUCTIONS = (
    "Search guest-accessible public job data and run explicitly configured local "
    "matching. This server does not authenticate to LinkedIn or submit applications."
)


def create_server(
    runtime: RuntimeConfig | None = None,
    *,
    public_service: PublicJobService | None = None,
    matching_runner: MatchingRunner | None = None,
) -> FastMCP:
    """Create one server from one resolved runtime configuration."""

    resolved = runtime if runtime is not None else load_runtime_config()
    service = public_service
    if service is None:
        provider = LinkedInPublicJobsProvider(
            user_agent=resolved.settings.user_agent,
            timeout_seconds=resolved.settings.timeout_seconds,
        )
        service = JobSearchService(
            provider=provider,
            max_results=resolved.settings.max_results,
        )

    server = FastMCP("Career Agent Workbench", instructions=_INSTRUCTIONS)
    register_job_tools(server, service)
    if matching_runner is None:
        register_matching_tools(server, resolved, service)
    else:
        register_matching_tools(server, resolved, service, runner=matching_runner)
    return server


def main() -> None:
    """Run the normal FastMCP stdio transport."""

    create_server().run()


__all__ = ["create_server", "main"]


if __name__ == "__main__":
    main()
