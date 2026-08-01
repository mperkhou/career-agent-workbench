"""FastMCP tool registration for Career Agent Workbench."""

from career_agent_workbench.tools.jobs import register_job_tools
from career_agent_workbench.tools.matching import register_matching_tools

__all__ = ["register_job_tools", "register_matching_tools"]
