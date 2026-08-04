"""Settings-driven language-model client selection."""

from __future__ import annotations

import httpx

from career_agent_workbench.api_client import ApiLlmClient
from career_agent_workbench.config import Settings
from career_agent_workbench.errors import WorkflowError
from career_agent_workbench.ollama import OllamaClient


def build_llm_client(
    settings: Settings,
    *,
    api_model: str | None = None,
    timeout_seconds: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ApiLlmClient | OllamaClient:
    """Construct the configured client without making a request."""
    provider = settings.llm_provider.casefold().strip()
    if provider == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=(
                settings.ollama_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            ),
            transport=transport,
        )
    if provider != "api":
        raise WorkflowError(
            "Unsupported LLM provider. Set CAREER_AGENT_WORKBENCH_LLM_PROVIDER "
            "to 'api' or 'ollama'."
        ) from None
    if not settings.llm_api_key:
        raise WorkflowError(
            "CAREER_AGENT_WORKBENCH_LLM_API_KEY is required when "
            "CAREER_AGENT_WORKBENCH_LLM_PROVIDER=api."
        ) from None
    return ApiLlmClient(
        base_url=settings.llm_api_base_url,
        model=api_model or settings.llm_api_model,
        api_key=settings.llm_api_key,
        timeout_seconds=(
            settings.llm_api_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        ),
        transport=transport,
    )


def llm_settings_label(settings: Settings) -> str:
    """Return only the normalized provider and configured model."""
    provider = settings.llm_provider.casefold().strip()
    model = (
        settings.ollama_model.strip()
        if provider == "ollama"
        else settings.llm_api_model.strip()
    )
    return f"{provider}:{model}"


__all__ = ["build_llm_client", "llm_settings_label"]
