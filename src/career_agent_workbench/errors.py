"""Stable, public-safe exception types."""


class CareerAgentWorkbenchError(Exception):
    """Base class for expected workbench failures."""

    __slots__ = ()


class ProviderError(CareerAgentWorkbenchError):
    """Raised when a public job provider cannot complete an operation."""

    __slots__ = ()


class JobNotFoundError(ProviderError):
    """Raised when a public job cannot be found."""

    __slots__ = ()


class WorkflowError(CareerAgentWorkbenchError):
    """Raised when a career-workflow component is not configured."""

    __slots__ = ()


class OllamaError(CareerAgentWorkbenchError):
    """Raised when a local or remote Ollama operation fails."""

    __slots__ = ()


class LlmError(CareerAgentWorkbenchError):
    """Raised when an API-backed language-model operation fails."""

    __slots__ = ()
