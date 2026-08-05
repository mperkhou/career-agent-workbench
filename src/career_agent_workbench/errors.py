"""Stable, public-safe exception types."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    EMBEDDED_TRANSIENT = "embedded_transient"
    EMBEDDED_PERMANENT = "embedded_permanent"
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
        ModelFailureSubtype.EMBEDDED_TRANSIENT,
    }
)
NON_RETRYABLE_MODEL_FAILURE_SUBTYPES = frozenset(
    {
        ModelFailureSubtype.PERMANENT_HTTP,
        ModelFailureSubtype.EMBEDDED_PERMANENT,
        ModelFailureSubtype.MALFORMED_ENVELOPE,
        ModelFailureSubtype.INVALID_GENERATION_JSON,
        ModelFailureSubtype.RESPONSE_TOO_LARGE,
        ModelFailureSubtype.UNEXPECTED_MODEL,
    }
)


class ModelResponseErrorPresence(StrEnum):
    """Closed location vocabulary for an API response error object."""

    UNAVAILABLE = "unavailable"
    NONE = "none"
    TOP_LEVEL = "top_level"
    CHOICE = "choice"
    BOTH = "both"


class ModelResponseErrorType(StrEnum):
    """Closed, provider-neutral response error classification."""

    RATE_LIMIT = "rate_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    OVERLOADED = "overloaded"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    SERVER = "server"
    UNMAPPED = "unmapped"
    PERMANENT_REQUEST = "permanent_request"
    MISSING = "missing"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class ModelResponseFinishReason(StrEnum):
    """Closed completion finish-state vocabulary."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    FUNCTION_CALL = "function_call"
    ERROR = "error"
    MISSING = "missing"
    NULL = "null"
    OTHER = "other"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class ModelResponseContentState(StrEnum):
    """Closed completion-content shape vocabulary."""

    UNAVAILABLE = "unavailable"
    MISSING = "missing"
    NULL = "null"
    INVALID_TYPE = "invalid_type"
    EMPTY = "empty"
    PRESENT = "present"


TRANSIENT_MODEL_RESPONSE_ERROR_TYPES = frozenset(
    {
        ModelResponseErrorType.RATE_LIMIT,
        ModelResponseErrorType.PROVIDER_UNAVAILABLE,
        ModelResponseErrorType.OVERLOADED,
        ModelResponseErrorType.UPSTREAM_TIMEOUT,
        ModelResponseErrorType.SERVER,
        ModelResponseErrorType.UNMAPPED,
    }
)

TRANSIENT_MODEL_RESPONSE_CODE_PAIRS = frozenset(
    {
        (ModelResponseErrorType.RATE_LIMIT, 429),
        (ModelResponseErrorType.PROVIDER_UNAVAILABLE, 409),
        (ModelResponseErrorType.PROVIDER_UNAVAILABLE, 425),
        (ModelResponseErrorType.PROVIDER_UNAVAILABLE, 502),
        (ModelResponseErrorType.OVERLOADED, 503),
        (ModelResponseErrorType.UPSTREAM_TIMEOUT, 408),
        (ModelResponseErrorType.UPSTREAM_TIMEOUT, 504),
        (ModelResponseErrorType.SERVER, 500),
        (ModelResponseErrorType.UNMAPPED, 500),
    }
)


@dataclass(frozen=True, slots=True)
class ModelResponseSummary:
    """Immutable, content-free summary of one bounded API response."""

    http_status: int | None
    error_presence: ModelResponseErrorPresence
    error_code: int | None
    error_type: ModelResponseErrorType
    finish_reason: ModelResponseFinishReason
    choices_count: int | None
    content_state: ModelResponseContentState

    def __post_init__(self) -> None:
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise ValueError("Model response metadata is invalid.")
        if not isinstance(self.error_presence, ModelResponseErrorPresence):
            raise ValueError("Model response metadata is invalid.")
        if self.error_code is not None and (
            type(self.error_code) is not int or not 0 <= self.error_code <= 999_999
        ):
            raise ValueError("Model response metadata is invalid.")
        if not isinstance(self.error_type, ModelResponseErrorType):
            raise ValueError("Model response metadata is invalid.")
        if not isinstance(self.finish_reason, ModelResponseFinishReason):
            raise ValueError("Model response metadata is invalid.")
        if self.choices_count is not None and (
            type(self.choices_count) is not int or not 0 <= self.choices_count <= 10_000
        ):
            raise ValueError("Model response metadata is invalid.")
        if not isinstance(self.content_state, ModelResponseContentState):
            raise ValueError("Model response metadata is invalid.")
        if self.error_presence is ModelResponseErrorPresence.UNAVAILABLE:
            if (
                self.error_code is not None
                or self.error_type is not ModelResponseErrorType.UNAVAILABLE
            ):
                raise ValueError("Model response metadata is invalid.")
        elif self.error_presence is ModelResponseErrorPresence.NONE:
            if (
                self.error_code is not None
                or self.error_type is not ModelResponseErrorType.MISSING
            ):
                raise ValueError("Model response metadata is invalid.")
        elif self.error_type in {
            ModelResponseErrorType.MISSING,
            ModelResponseErrorType.UNAVAILABLE,
        }:
            raise ValueError("Model response metadata is invalid.")
        if (
            self.error_type in TRANSIENT_MODEL_RESPONSE_ERROR_TYPES
            and self.error_code is not None
            and self.error_code not in TRANSIENT_MODEL_HTTP_STATUSES
        ):
            raise ValueError("Model response metadata is invalid.")
        if (
            self.error_type is ModelResponseErrorType.PERMANENT_REQUEST
            and self.error_code is not None
            and (
                self.error_code in TRANSIENT_MODEL_HTTP_STATUSES
                or not 400 <= self.error_code <= 599
            )
        ):
            raise ValueError("Model response metadata is invalid.")

    def as_dict(self) -> dict[str, object]:
        """Return the exact closed JSON-safe representation."""

        return {
            "http_status": self.http_status,
            "error_presence": self.error_presence.value,
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "finish_reason": self.finish_reason.value,
            "choices_count": self.choices_count,
            "content_state": self.content_state.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ModelResponseSummary:
        """Validate one exact JSON mapping without accepting extra fields."""

        expected = {
            "http_status",
            "error_presence",
            "error_code",
            "error_type",
            "finish_reason",
            "choices_count",
            "content_state",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Model response metadata is invalid.")
        try:
            return cls(
                http_status=value["http_status"],
                error_presence=ModelResponseErrorPresence(value["error_presence"]),
                error_code=value["error_code"],
                error_type=ModelResponseErrorType(value["error_type"]),
                finish_reason=ModelResponseFinishReason(value["finish_reason"]),
                choices_count=value["choices_count"],
                content_state=ModelResponseContentState(value["content_state"]),
            )
        except (TypeError, ValueError):
            raise ValueError("Model response metadata is invalid.") from None


class TypedModelError(LlmError):
    """Sanitized model failure carrying only validated closed metadata."""

    __slots__ = (
        "_http_status",
        "_response_summary",
        "_retry_after_seconds",
        "_subtype",
    )

    def __init__(
        self,
        *,
        subtype: ModelFailureSubtype,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
        response_summary: ModelResponseSummary | None = None,
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
        if retry_after_seconds is not None and subtype not in {
            ModelFailureSubtype.TRANSIENT_HTTP,
            ModelFailureSubtype.EMBEDDED_TRANSIENT,
        }:
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
            ModelFailureSubtype.EMBEDDED_TRANSIENT: "Language-model provider reported a transient failure.",
            ModelFailureSubtype.EMBEDDED_PERMANENT: "Language-model provider rejected the request.",
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
        if response_summary is not None and not isinstance(
            response_summary, ModelResponseSummary
        ):
            raise ValueError("Model failure metadata is invalid.")
        if (
            response_summary is not None
            and http_status is not None
            and response_summary.http_status != http_status
        ):
            raise ValueError("Model failure metadata is invalid.")
        if response_summary is not None and subtype in {
            ModelFailureSubtype.EMBEDDED_TRANSIENT,
            ModelFailureSubtype.EMBEDDED_PERMANENT,
        }:
            if response_summary.error_presence in {
                ModelResponseErrorPresence.NONE,
                ModelResponseErrorPresence.UNAVAILABLE,
            }:
                raise ValueError("Model failure metadata is invalid.")
            if (
                subtype is ModelFailureSubtype.EMBEDDED_TRANSIENT
                and response_summary.error_type
                not in TRANSIENT_MODEL_RESPONSE_ERROR_TYPES
            ):
                raise ValueError("Model failure metadata is invalid.")
            if (
                subtype is ModelFailureSubtype.EMBEDDED_PERMANENT
                and response_summary.error_type
                is not ModelResponseErrorType.PERMANENT_REQUEST
            ):
                raise ValueError("Model failure metadata is invalid.")
        self._response_summary = response_summary

    @property
    def subtype(self) -> ModelFailureSubtype:
        return self._subtype

    @property
    def http_status(self) -> int | None:
        return self._http_status

    @property
    def retry_after_seconds(self) -> float | None:
        return self._retry_after_seconds

    @property
    def response_summary(self) -> ModelResponseSummary | None:
        return self._response_summary

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(subtype={self.subtype.value!r}, "
            f"http_status={self.http_status!r}, retry_after_seconds="
            f"{self.retry_after_seconds!r}, response_summary="
            f"{self.response_summary!r})"
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
    "ModelResponseContentState",
    "ModelResponseErrorPresence",
    "ModelResponseErrorType",
    "ModelResponseFinishReason",
    "ModelResponseSummary",
    "ModelTimeoutError",
    "NonRetryableModelError",
    "NON_RETRYABLE_MODEL_FAILURE_SUBTYPES",
    "OllamaError",
    "OllamaTimeoutError",
    "ProviderError",
    "RetryableModelError",
    "RETRYABLE_MODEL_FAILURE_SUBTYPES",
    "TRANSIENT_MODEL_HTTP_STATUSES",
    "TRANSIENT_MODEL_RESPONSE_CODE_PAIRS",
    "TRANSIENT_MODEL_RESPONSE_ERROR_TYPES",
    "TypedModelError",
    "WorkflowError",
    "model_failure_subtype",
    "retryable_model_failure",
]
