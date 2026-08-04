"""Stable, public-safe exception types."""

from __future__ import annotations

import math
from enum import StrEnum


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


class ModelTimeoutError(CareerAgentWorkbenchError):
    """Marker base for provider timeouts eligible for workflow retry."""

    __slots__ = ()


class ModelFailureSubtype(StrEnum):
    """Closed, content-free provider/model failure vocabulary."""

    TIMEOUT = "timeout"
    TRANSIENT_HTTP = "transient_http"
    PERMANENT_HTTP = "permanent_http"
    TRANSPORT_READ = "transport_read"
    TRANSPORT_PROTOCOL = "transport_protocol"
    TRANSPORT_CONNECT = "transport_connect"
    EMPTY_COMPLETION = "empty_completion"
    MALFORMED_ENVELOPE = "malformed_envelope"
    INVALID_GENERATION_JSON = "invalid_generation_json"
    RESPONSE_TOO_LARGE = "response_too_large"
    UNEXPECTED_MODEL = "unexpected_model"


TRANSIENT_MODEL_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_MODEL_FAILURE_SUBTYPES = frozenset(
    {
        ModelFailureSubtype.TRANSIENT_HTTP,
        ModelFailureSubtype.TRANSPORT_READ,
        ModelFailureSubtype.TRANSPORT_PROTOCOL,
        ModelFailureSubtype.TRANSPORT_CONNECT,
        ModelFailureSubtype.EMPTY_COMPLETION,
    }
)
NON_RETRYABLE_MODEL_FAILURE_SUBTYPES = frozenset(
    {
        ModelFailureSubtype.PERMANENT_HTTP,
        ModelFailureSubtype.MALFORMED_ENVELOPE,
        ModelFailureSubtype.INVALID_GENERATION_JSON,
        ModelFailureSubtype.RESPONSE_TOO_LARGE,
        ModelFailureSubtype.UNEXPECTED_MODEL,
    }
)


class TypedModelError(LlmError):
    """Sanitized model failure carrying only validated closed metadata."""

    __slots__ = ("_http_status", "_retry_after_seconds", "_subtype")

    def __init__(
        self,
        *,
        subtype: ModelFailureSubtype,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        if not isinstance(subtype, ModelFailureSubtype):
            raise ValueError("Model failure metadata is invalid.")
        if subtype is ModelFailureSubtype.TIMEOUT:
            if not isinstance(self, ModelTimeoutError):
                raise ValueError("Model failure metadata is invalid.")
        elif isinstance(self, ModelTimeoutError):
            raise ValueError("Model failure metadata is invalid.")
        elif isinstance(self, RetryableModelError):
            if subtype not in RETRYABLE_MODEL_FAILURE_SUBTYPES:
                raise ValueError("Model failure metadata is invalid.")
        elif isinstance(self, NonRetryableModelError):
            if subtype not in NON_RETRYABLE_MODEL_FAILURE_SUBTYPES:
                raise ValueError("Model failure metadata is invalid.")
        else:
            raise ValueError("Model failure metadata is invalid.")
        http_subtype = subtype in {
            ModelFailureSubtype.TRANSIENT_HTTP,
            ModelFailureSubtype.PERMANENT_HTTP,
        }
        if http_subtype != (http_status is not None):
            raise ValueError("Model failure metadata is invalid.")
        if http_status is not None and (
            type(http_status) is not int or not 300 <= http_status <= 599
        ):
            raise ValueError("Model failure metadata is invalid.")
        if subtype is ModelFailureSubtype.TRANSIENT_HTTP and (
            http_status not in TRANSIENT_MODEL_HTTP_STATUSES
        ):
            raise ValueError("Model failure metadata is invalid.")
        if subtype is ModelFailureSubtype.PERMANENT_HTTP and (
            http_status in TRANSIENT_MODEL_HTTP_STATUSES
        ):
            raise ValueError("Model failure metadata is invalid.")
        if (
            retry_after_seconds is not None
            and subtype is not ModelFailureSubtype.TRANSIENT_HTTP
        ):
            raise ValueError("Model failure metadata is invalid.")
        if retry_after_seconds is not None and (
            type(retry_after_seconds) not in {int, float}
            or not math.isfinite(float(retry_after_seconds))
            or not 0 <= float(retry_after_seconds) <= 120
        ):
            raise ValueError("Model failure metadata is invalid.")
        message = {
            ModelFailureSubtype.TIMEOUT: "Language-model request timed out.",
            ModelFailureSubtype.TRANSIENT_HTTP: "Language-model request failed transiently.",
            ModelFailureSubtype.PERMANENT_HTTP: "Language-model request failed permanently.",
            ModelFailureSubtype.TRANSPORT_READ: "Language-model response was interrupted.",
            ModelFailureSubtype.TRANSPORT_PROTOCOL: "Language-model protocol was interrupted.",
            ModelFailureSubtype.TRANSPORT_CONNECT: "Language-model connection failed.",
            ModelFailureSubtype.EMPTY_COMPLETION: "Language-model generation was empty.",
            ModelFailureSubtype.MALFORMED_ENVELOPE: "Language-model response envelope was invalid.",
            ModelFailureSubtype.INVALID_GENERATION_JSON: "Language-model generation JSON was invalid.",
            ModelFailureSubtype.RESPONSE_TOO_LARGE: "Language-model response exceeded the size limit.",
            ModelFailureSubtype.UNEXPECTED_MODEL: "Language-model operation failed.",
        }[subtype]
        if http_status is not None:
            message = f"{message.removesuffix('.')} with HTTP status {http_status}."
        super().__init__(message)
        self._subtype = subtype
        self._http_status = http_status
        self._retry_after_seconds = (
            None if retry_after_seconds is None else float(retry_after_seconds)
        )

    @property
    def subtype(self) -> ModelFailureSubtype:
        return self._subtype

    @property
    def http_status(self) -> int | None:
        return self._http_status

    @property
    def retry_after_seconds(self) -> float | None:
        return self._retry_after_seconds

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(subtype={self.subtype.value!r}, "
            f"http_status={self.http_status!r}, retry_after_seconds="
            f"{self.retry_after_seconds!r})"
        )


class RetryableModelError(TypedModelError):
    """Typed transient failure eligible for the workflow retry budget."""

    __slots__ = ()


class NonRetryableModelError(TypedModelError):
    """Typed permanent failure that must not consume a workflow retry."""

    __slots__ = ()


class LlmTimeoutError(RetryableModelError, ModelTimeoutError):
    """Raised when an API-backed language-model request times out."""

    __slots__ = ()

    def __init__(self, _message: str | None = None) -> None:
        super().__init__(subtype=ModelFailureSubtype.TIMEOUT)


class OllamaTimeoutError(OllamaError, ModelTimeoutError):
    """Raised when an Ollama generation request times out."""

    __slots__ = ()


def model_failure_subtype(error: BaseException) -> ModelFailureSubtype:
    """Classify one model-boundary failure without inspecting its message."""

    if isinstance(error, TypedModelError):
        return error.subtype
    if isinstance(error, ModelTimeoutError):
        return ModelFailureSubtype.TIMEOUT
    return ModelFailureSubtype.UNEXPECTED_MODEL


def retryable_model_failure(error: BaseException) -> bool:
    """Return whether one typed failure may consume the outer retry budget."""

    return isinstance(error, (RetryableModelError, ModelTimeoutError))


__all__ = [
    "CareerAgentWorkbenchError",
    "JobNotFoundError",
    "LlmError",
    "LlmTimeoutError",
    "ModelFailureSubtype",
    "ModelTimeoutError",
    "NonRetryableModelError",
    "NON_RETRYABLE_MODEL_FAILURE_SUBTYPES",
    "OllamaError",
    "OllamaTimeoutError",
    "ProviderError",
    "RetryableModelError",
    "RETRYABLE_MODEL_FAILURE_SUBTYPES",
    "TRANSIENT_MODEL_HTTP_STATUSES",
    "TypedModelError",
    "WorkflowError",
    "model_failure_subtype",
    "retryable_model_failure",
]
