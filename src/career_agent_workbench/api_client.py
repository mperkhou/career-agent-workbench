"""Bounded client for an OpenAI-compatible public API."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from career_agent_workbench.errors import (
    LlmError,
    LlmTimeoutError,
    ModelFailureSubtype,
    ModelResponseContentState,
    ModelResponseErrorPresence,
    ModelResponseErrorType,
    ModelResponseFinishReason,
    ModelResponseSummary,
    NonRetryableModelError,
    RetryableModelError,
    TRANSIENT_MODEL_HTTP_STATUSES,
    TRANSIENT_MODEL_RESPONSE_CODE_PAIRS,
    TypedModelError,
)

TRANSIENT_HTTP_STATUSES = TRANSIENT_MODEL_HTTP_STATUSES
MAX_RESPONSE_BYTES = 2_000_000
MAX_RETRY_AFTER_SECONDS = 120.0
MAX_BACKOFF_SECONDS = 60.0
MAX_CHOICES_COUNT = 10_000
MAX_EMBEDDED_ERROR_CODE = 999_999

_PACKAGE_HTTP_REQUEST = contextvars.ContextVar(
    "career_agent_workbench_package_http_request",
    default=False,
)
_PACKAGE_HTTP_LOGGERS = (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)

_MISSING = object()
_TRANSIENT_ERROR_TYPES = {
    "rate_limit": ModelResponseErrorType.RATE_LIMIT,
    "rate_limit_error": ModelResponseErrorType.RATE_LIMIT,
    "rate_limit_exceeded": ModelResponseErrorType.RATE_LIMIT,
    "rate_limited": ModelResponseErrorType.RATE_LIMIT,
    "provider_unavailable": ModelResponseErrorType.PROVIDER_UNAVAILABLE,
    "provider_unavailable_error": ModelResponseErrorType.PROVIDER_UNAVAILABLE,
    "service_unavailable": ModelResponseErrorType.PROVIDER_UNAVAILABLE,
    "unavailable": ModelResponseErrorType.PROVIDER_UNAVAILABLE,
    "overload": ModelResponseErrorType.OVERLOADED,
    "overloaded": ModelResponseErrorType.OVERLOADED,
    "overloaded_error": ModelResponseErrorType.OVERLOADED,
    "provider_overloaded": ModelResponseErrorType.OVERLOADED,
    "timeout": ModelResponseErrorType.UPSTREAM_TIMEOUT,
    "upstream_timeout": ModelResponseErrorType.UPSTREAM_TIMEOUT,
    "gateway_timeout": ModelResponseErrorType.UPSTREAM_TIMEOUT,
    "server_error": ModelResponseErrorType.SERVER,
    "internal_server_error": ModelResponseErrorType.SERVER,
    "server": ModelResponseErrorType.SERVER,
    "unmapped": ModelResponseErrorType.UNMAPPED,
}
_PERMANENT_ERROR_TYPES = frozenset(
    {
        "authentication",
        "authentication_error",
        "bad_request",
        "context_length_exceeded",
        "context_window_exceeded",
        "content_policy_violation",
        "forbidden",
        "image_format_not_supported",
        "image_download_failed",
        "image_not_found",
        "image_too_large",
        "image_too_small",
        "image_url_fetch_failed",
        "image_url_invalid",
        "image_url_not_supported",
        "image_url_required",
        "invalid_image",
        "invalid_prompt",
        "invalid_request",
        "invalid_request_error",
        "max_tokens_exceeded",
        "not_found",
        "payload_too_large",
        "payment_required",
        "permission_denied",
        "permission_error",
        "policy_error",
        "policy_violation",
        "precondition_failed",
        "refusal",
        "string_too_long",
        "token_limit_exceeded",
        "unauthorized",
        "unprocessable",
        "unsupported_image_format",
    }
)


class _ResponseTooLarge(Exception):
    """Signal a response-size violation without retaining response data."""


class _EmbeddedDisposition(StrEnum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class _EmbeddedError:
    disposition: _EmbeddedDisposition
    code: int | None
    error_type: ModelResponseErrorType


class _PackageHttpLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        del record
        return not _PACKAGE_HTTP_REQUEST.get()


async def _bounded_http_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_response_bytes: int,
    **kwargs: Any,
) -> tuple[httpx.Response, bytes]:
    """Issue one scoped request and close its bounded streamed response."""
    log_filter = _PackageHttpLogFilter()
    token = _PACKAGE_HTTP_REQUEST.set(True)
    filtered_loggers: list[logging.Logger] = []
    try:
        for logger_name in _PACKAGE_HTTP_LOGGERS:
            logger = logging.getLogger(logger_name)
            logger.addFilter(log_filter)
            filtered_loggers.append(logger)
        async with client.stream(method, url, **kwargs) as response:
            if not 200 <= response.status_code < 300:
                return response, b""
            declared_size = _content_length(response)
            if declared_size is not None and declared_size > max_response_bytes:
                raise _ResponseTooLarge
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > max_response_bytes:
                    raise _ResponseTooLarge
                body.extend(chunk)
            return response, bytes(body)
    finally:
        for logger in reversed(filtered_loggers):
            logger.removeFilter(log_filter)
        _PACKAGE_HTTP_REQUEST.reset(token)


def _content_length(response: httpx.Response) -> int | None:
    raw_value = response.headers.get("content-length")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError, OverflowError):
        value = -1
    return value if value >= 0 else None


def _load_json(value: str | bytes) -> tuple[bool, object, bool]:
    """Return JSON success, parsed value, and bounded-recursion failure."""
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except RecursionError:
        recursion_error = True
    except (ValueError, UnicodeDecodeError):
        recursion_error = False
    else:
        return True, parsed, False
    return False, None, recursion_error


def _raw_decode_json(
    decoder: json.JSONDecoder,
    value: str,
) -> tuple[bool, object, bool]:
    try:
        parsed, _ = decoder.raw_decode(value)
    except RecursionError:
        recursion_error = True
    except ValueError:
        recursion_error = False
    else:
        return True, parsed, False
    return False, None, recursion_error


def _reject_json_constant(_value: str) -> None:
    raise ValueError("Non-standard JSON constants are not accepted.")


def _validated_api_base(value: str) -> str:
    parts = None
    hostname = None
    port = None
    try:
        parts = urlsplit(str(value).strip())
        hostname = parts.hostname
        port = parts.port
    except (TypeError, ValueError):
        pass
    if (
        parts is None
        or parts.scheme.casefold() != "https"
        or not hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port is not None
        and not 0 < port <= 65_535
    ):
        raise LlmError("API endpoint configuration is invalid.")
    path = parts.path.rstrip("/")
    return urlunsplit(("https", parts.netloc, path, "", ""))


def _bounded_retry_attempts(value: object) -> int:
    parsed = None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        pass
    if parsed is None:
        raise LlmError("API retry configuration is invalid.")
    return min(max(parsed, 1), 5)


def _bounded_backoff(value: object) -> float:
    parsed = None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        pass
    if parsed is None or not math.isfinite(parsed) or parsed < 0:
        raise LlmError("API retry configuration is invalid.")
    return parsed


class ApiLlmClient:
    """Own a clean, bounded API client with finite transient retries."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_attempts: int = 4,
        retry_backoff_seconds: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._base_url = _validated_api_base(base_url)
        self._model = str(model).strip()
        self._api_key = str(api_key)
        self._retry_attempts = _bounded_retry_attempts(retry_attempts)
        self._retry_backoff_seconds = _bounded_backoff(retry_backoff_seconds)
        self._sleep = sleep
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=timeout_seconds,
                transport=transport,
                trust_env=False,
                follow_redirects=False,
            )
        except (TypeError, ValueError):
            pass
        if client is None:
            raise LlmError("API client configuration is invalid.")
        self._client = client

    def __repr__(self) -> str:
        return "ApiLlmClient(configured=True)"

    @property
    def model(self) -> str:
        return self._model

    async def generate_text(self, prompt: str) -> str:
        return await self._generate(prompt, json_response=False)

    async def generate_json(self, prompt: str) -> Mapping[str, Any]:
        response = await self._generate(prompt, json_response=True)
        return _parse_json_object(response)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _generate(self, prompt: str, *, json_response: bool) -> str:
        del json_response  # API compatibility is prompt-driven.
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mperkhou/career-agent-workbench",
            "X-Title": "career-agent-workbench",
        }

        completion_error: TypedModelError = RetryableModelError(
            subtype=ModelFailureSubtype.EMPTY_COMPLETION
        )
        for attempt in range(1, self._retry_attempts + 1):
            request_error: TypedModelError | None = None
            response: httpx.Response | None = None
            try:
                response, body = await _bounded_http_request(
                    self._client,
                    "POST",
                    f"{self._base_url}/chat/completions",
                    max_response_bytes=MAX_RESPONSE_BYTES,
                    json=payload,
                    headers=headers,
                )
            except _ResponseTooLarge:
                request_error = NonRetryableModelError(
                    subtype=ModelFailureSubtype.RESPONSE_TOO_LARGE
                )
            except httpx.TimeoutException:
                request_error = LlmTimeoutError()
            except httpx.ConnectError:
                request_error = RetryableModelError(
                    subtype=ModelFailureSubtype.TRANSPORT_CONNECT
                )
            except httpx.ReadError:
                request_error = RetryableModelError(
                    subtype=ModelFailureSubtype.TRANSPORT_READ
                )
            except httpx.RemoteProtocolError:
                request_error = RetryableModelError(
                    subtype=ModelFailureSubtype.TRANSPORT_PROTOCOL
                )
            except httpx.ProtocolError:
                request_error = NonRetryableModelError(
                    subtype=ModelFailureSubtype.UNEXPECTED_MODEL
                )
            except httpx.HTTPError:
                request_error = NonRetryableModelError(
                    subtype=ModelFailureSubtype.UNEXPECTED_MODEL
                )
            except Exception:
                request_error = NonRetryableModelError(
                    subtype=ModelFailureSubtype.UNEXPECTED_MODEL
                )
            if request_error is not None:
                raise request_error

            assert response is not None
            if not 200 <= response.status_code < 300:
                retry_after = _bounded_retry_after(response)
                response_summary = _unavailable_response_summary(
                    http_status=response.status_code
                )
                if response.status_code in TRANSIENT_HTTP_STATUSES:
                    status_error: TypedModelError = RetryableModelError(
                        subtype=ModelFailureSubtype.TRANSIENT_HTTP,
                        http_status=response.status_code,
                        retry_after_seconds=retry_after,
                        response_summary=response_summary,
                    )
                else:
                    status_error = NonRetryableModelError(
                        subtype=ModelFailureSubtype.PERMANENT_HTTP,
                        http_status=response.status_code,
                        response_summary=response_summary,
                    )
                if (
                    isinstance(status_error, RetryableModelError)
                    and attempt < self._retry_attempts
                ):
                    await self._sleep(
                        _typed_retry_delay_seconds(
                            status_error,
                            attempt,
                            self._retry_backoff_seconds,
                        )
                    )
                    continue
                raise status_error

            data = _response_json(body, http_status=response.status_code)
            text, completion_error = _completion_text(
                data,
                http_status=response.status_code,
                retry_after_seconds=_bounded_retry_after(response),
            )
            if completion_error is None:
                return text
            if not isinstance(completion_error, RetryableModelError):
                raise completion_error
            if attempt < self._retry_attempts:
                await self._sleep(
                    _typed_retry_delay_seconds(
                        completion_error,
                        attempt,
                        self._retry_backoff_seconds,
                    )
                )

        raise completion_error


def _response_json(body: bytes, *, http_status: int = 200) -> Mapping[str, Any]:
    loaded, value, _ = _load_json(body)
    if not loaded:
        raise NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE,
            response_summary=_unavailable_response_summary(http_status=http_status),
        )
    if not isinstance(value, Mapping):
        raise NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE,
            response_summary=_unavailable_response_summary(http_status=http_status),
        )
    return value


def _completion_text(
    data: Mapping[str, Any],
    *,
    http_status: int = 200,
    retry_after_seconds: float | None = None,
) -> tuple[str, TypedModelError | None]:
    return _analyze_completion(
        data,
        http_status=http_status,
        retry_after_seconds=retry_after_seconds,
    )


def _analyze_completion(
    data: Mapping[str, Any],
    *,
    http_status: int,
    retry_after_seconds: float | None,
) -> tuple[str, TypedModelError | None]:
    top_value = data.get("error", _MISSING)
    top_present = top_value is not _MISSING and top_value is not None
    top_error = _embedded_error(top_value) if top_present else None

    choices_value = data.get("choices", _MISSING)
    choices_count: int | None = None
    choice: Mapping[str, Any] | None = None
    choice_shape_invalid = False
    if choices_value is _MISSING or choices_value is None:
        pass
    elif not isinstance(choices_value, list) or len(choices_value) > MAX_CHOICES_COUNT:
        choice_shape_invalid = True
    else:
        choices_count = len(choices_value)
        if choices_value:
            first_choice = choices_value[0]
            if isinstance(first_choice, Mapping):
                choice = first_choice
            else:
                choice_shape_invalid = True

    choice_error = None
    choice_present = False
    finish_reason = ModelResponseFinishReason.UNAVAILABLE
    content_state = ModelResponseContentState.UNAVAILABLE
    normalized_text = ""
    if choice is not None:
        choice_error_value = choice.get("error", _MISSING)
        choice_present = (
            choice_error_value is not _MISSING and choice_error_value is not None
        )
        if choice_present:
            choice_error = _embedded_error(choice_error_value)
        finish_reason = _normalized_finish_reason(choice)
        content_state, normalized_text = _normalized_content(choice)

    presence = _error_presence(top_present=top_present, choice_present=choice_present)
    embedded = _combined_embedded_error(top_error, choice_error)
    if choice_present and finish_reason is not ModelResponseFinishReason.ERROR:
        embedded = _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=None if embedded is None else embedded.code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    structural_malformed = (
        choice_shape_invalid
        or finish_reason is ModelResponseFinishReason.INVALID
        or content_state is ModelResponseContentState.INVALID_TYPE
    )
    summary = ModelResponseSummary(
        http_status=http_status,
        error_presence=presence,
        error_code=None if embedded is None else embedded.code,
        error_type=(
            ModelResponseErrorType.MISSING if embedded is None else embedded.error_type
        ),
        finish_reason=finish_reason,
        choices_count=choices_count,
        content_state=content_state,
    )

    if structural_malformed:
        return "", NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE,
            response_summary=summary,
        )
    if embedded is not None:
        if embedded.disposition is _EmbeddedDisposition.MALFORMED:
            return "", NonRetryableModelError(
                subtype=ModelFailureSubtype.MALFORMED_ENVELOPE,
                response_summary=summary,
            )
        if embedded.disposition is _EmbeddedDisposition.TRANSIENT:
            return "", RetryableModelError(
                subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
                retry_after_seconds=retry_after_seconds,
                response_summary=summary,
            )
        return "", NonRetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_PERMANENT,
            response_summary=summary,
        )

    if (
        finish_reason is ModelResponseFinishReason.ERROR
        or choices_count in {None, 0}
        or content_state
        in {
            ModelResponseContentState.UNAVAILABLE,
            ModelResponseContentState.MISSING,
            ModelResponseContentState.NULL,
            ModelResponseContentState.EMPTY,
        }
    ):
        return "", RetryableModelError(
            subtype=ModelFailureSubtype.EMPTY_COMPLETION,
            response_summary=summary,
        )
    return normalized_text, None


def _embedded_error(value: object) -> _EmbeddedError:
    if not isinstance(value, Mapping):
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=None,
            error_type=ModelResponseErrorType.UNKNOWN,
        )

    code_value = value.get("code", _MISSING)
    code: int | None
    if code_value is _MISSING or code_value is None:
        code = None
        code_disposition = None
    elif type(code_value) is int and 0 <= code_value <= MAX_EMBEDDED_ERROR_CODE:
        code = code_value
        code_disposition = _embedded_code_disposition(code)
    else:
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=None,
            error_type=ModelResponseErrorType.UNKNOWN,
        )

    direct_type = _normalized_embedded_error_type(value.get("type", _MISSING))
    metadata_value = value.get("metadata", _MISSING)
    if metadata_value is _MISSING or metadata_value is None:
        metadata_type = None
    elif isinstance(metadata_value, Mapping):
        metadata_type = _normalized_embedded_error_type(
            metadata_value.get("error_type", _MISSING)
        )
    else:
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    if direct_type is _EmbeddedDisposition.MALFORMED or (
        metadata_type is _EmbeddedDisposition.MALFORMED
    ):
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    direct_normalized, direct_disposition = direct_type
    if metadata_type is None:
        metadata_normalized = ModelResponseErrorType.MISSING
        metadata_disposition = None
    else:
        metadata_normalized, metadata_disposition = metadata_type
    if (
        direct_disposition is not None
        and metadata_disposition is not None
        and (
            direct_normalized is not metadata_normalized
            or direct_disposition is not metadata_disposition
        )
    ):
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    normalized_type = (
        direct_normalized if direct_disposition is not None else metadata_normalized
    )
    type_disposition = direct_disposition or metadata_disposition

    if code_disposition is _EmbeddedDisposition.MALFORMED:
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    if code_disposition is None and type_disposition is None:
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    if (
        code_disposition is not None
        and type_disposition is not None
        and code_disposition is not type_disposition
    ):
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    if (
        code is not None
        and type_disposition is _EmbeddedDisposition.TRANSIENT
        and (normalized_type, code) not in TRANSIENT_MODEL_RESPONSE_CODE_PAIRS
    ):
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    disposition = code_disposition or type_disposition
    assert disposition is not None
    if normalized_type is ModelResponseErrorType.MISSING:
        normalized_type = _error_type_from_code(code)
    return _EmbeddedError(
        disposition=disposition,
        code=code,
        error_type=normalized_type,
    )


def _normalized_embedded_error_type(
    value: object,
) -> (
    tuple[ModelResponseErrorType, _EmbeddedDisposition]
    | tuple[ModelResponseErrorType, None]
    | _EmbeddedDisposition
):
    if value is _MISSING or value is None:
        return ModelResponseErrorType.MISSING, None
    if type(value) is not str or len(value) > 128:
        return _EmbeddedDisposition.MALFORMED
    safe_type = value.strip().casefold()
    normalized = _TRANSIENT_ERROR_TYPES.get(safe_type)
    if normalized is not None:
        return normalized, _EmbeddedDisposition.TRANSIENT
    if safe_type in _PERMANENT_ERROR_TYPES:
        return ModelResponseErrorType.PERMANENT_REQUEST, _EmbeddedDisposition.PERMANENT
    return _EmbeddedDisposition.MALFORMED


def _combined_embedded_error(
    top_error: _EmbeddedError | None,
    choice_error: _EmbeddedError | None,
) -> _EmbeddedError | None:
    errors = tuple(item for item in (top_error, choice_error) if item is not None)
    if not errors:
        return None
    if any(item.disposition is _EmbeddedDisposition.MALFORMED for item in errors):
        first = errors[0]
        return _EmbeddedError(
            disposition=_EmbeddedDisposition.MALFORMED,
            code=first.code,
            error_type=ModelResponseErrorType.UNKNOWN,
        )
    first = errors[0]
    if len(errors) == 2:
        second = errors[1]
        contradictory = (
            first.disposition is not second.disposition
            or first.code is not None
            and second.code is not None
            and first.code != second.code
            or first.error_type is not second.error_type
        )
        if contradictory:
            return _EmbeddedError(
                disposition=_EmbeddedDisposition.MALFORMED,
                code=first.code if first.code == second.code else None,
                error_type=ModelResponseErrorType.UNKNOWN,
            )
        return _EmbeddedError(
            disposition=first.disposition,
            code=first.code if first.code is not None else second.code,
            error_type=first.error_type,
        )
    return first


def _embedded_code_disposition(code: int) -> _EmbeddedDisposition:
    if code in TRANSIENT_HTTP_STATUSES:
        return _EmbeddedDisposition.TRANSIENT
    if 400 <= code <= 599:
        return _EmbeddedDisposition.PERMANENT
    return _EmbeddedDisposition.MALFORMED


def _error_type_from_code(code: int | None) -> ModelResponseErrorType:
    if code == 429:
        return ModelResponseErrorType.RATE_LIMIT
    if code in {408, 504}:
        return ModelResponseErrorType.UPSTREAM_TIMEOUT
    if code in {409, 425, 502}:
        return ModelResponseErrorType.PROVIDER_UNAVAILABLE
    if code == 503:
        return ModelResponseErrorType.OVERLOADED
    if code == 500:
        return ModelResponseErrorType.SERVER
    return ModelResponseErrorType.PERMANENT_REQUEST


def _error_presence(
    *,
    top_present: bool,
    choice_present: bool,
) -> ModelResponseErrorPresence:
    if top_present and choice_present:
        return ModelResponseErrorPresence.BOTH
    if top_present:
        return ModelResponseErrorPresence.TOP_LEVEL
    if choice_present:
        return ModelResponseErrorPresence.CHOICE
    return ModelResponseErrorPresence.NONE


def _normalized_finish_reason(
    choice: Mapping[str, Any],
) -> ModelResponseFinishReason:
    value = choice.get("finish_reason", _MISSING)
    if value is _MISSING:
        return ModelResponseFinishReason.MISSING
    if value is None:
        return ModelResponseFinishReason.NULL
    if type(value) is not str:
        return ModelResponseFinishReason.INVALID
    return {
        "stop": ModelResponseFinishReason.STOP,
        "length": ModelResponseFinishReason.LENGTH,
        "content_filter": ModelResponseFinishReason.CONTENT_FILTER,
        "tool_calls": ModelResponseFinishReason.TOOL_CALLS,
        "function_call": ModelResponseFinishReason.FUNCTION_CALL,
        "error": ModelResponseFinishReason.ERROR,
    }.get(value.strip().casefold(), ModelResponseFinishReason.OTHER)


def _normalized_content(
    choice: Mapping[str, Any],
) -> tuple[ModelResponseContentState, str]:
    message = choice.get("message", _MISSING)
    if message is _MISSING:
        return ModelResponseContentState.MISSING, ""
    if message is None:
        return ModelResponseContentState.NULL, ""
    if not isinstance(message, Mapping):
        return ModelResponseContentState.INVALID_TYPE, ""
    content = message.get("content", _MISSING)
    if content is _MISSING:
        return ModelResponseContentState.MISSING, ""
    if content is None:
        return ModelResponseContentState.NULL, ""
    if type(content) is not str:
        return ModelResponseContentState.INVALID_TYPE, ""
    normalized = _strip_thinking(content.strip())
    if not normalized:
        return ModelResponseContentState.EMPTY, ""
    return ModelResponseContentState.PRESENT, normalized


def _unavailable_response_summary(*, http_status: int) -> ModelResponseSummary:
    return ModelResponseSummary(
        http_status=http_status,
        error_presence=ModelResponseErrorPresence.UNAVAILABLE,
        error_code=None,
        error_type=ModelResponseErrorType.UNAVAILABLE,
        finish_reason=ModelResponseFinishReason.UNAVAILABLE,
        choices_count=None,
        content_state=ModelResponseContentState.UNAVAILABLE,
    )


def _bounded_retry_after(response: httpx.Response) -> float | None:
    retry_after = response.headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        parsed = float(retry_after)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return min(parsed, MAX_RETRY_AFTER_SECONDS)


def _typed_retry_delay_seconds(
    error: RetryableModelError,
    attempt: int,
    base_delay_seconds: float,
) -> float:
    if error.retry_after_seconds is not None:
        return error.retry_after_seconds
    exponent = max(0, int(attempt) - 1)
    return min(base_delay_seconds * (2**exponent), MAX_BACKOFF_SECONDS)


def _retry_delay_seconds(
    response: httpx.Response,
    attempt: int,
    base_delay_seconds: float,
) -> float:
    error = RetryableModelError(
        subtype=ModelFailureSubtype.TRANSIENT_HTTP,
        http_status=response.status_code,
        retry_after_seconds=_bounded_retry_after(response),
    )
    return _typed_retry_delay_seconds(error, attempt, base_delay_seconds)


def _parse_json_object(text: str) -> Mapping[str, Any]:
    value = _parse_json_value(text)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list):
        return {"queries": value}
    raise NonRetryableModelError(subtype=ModelFailureSubtype.INVALID_GENERATION_JSON)


def _parse_json_value(text: str) -> object:
    stripped = text.strip()
    loaded, value, recursive = _load_json(stripped)
    if recursive:
        raise NonRetryableModelError(
            subtype=ModelFailureSubtype.INVALID_GENERATION_JSON
        )
    if loaded:
        return value

    for match in re.finditer(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        loaded, value, recursive = _load_json(match.group(1).strip())
        if recursive:
            raise NonRetryableModelError(
                subtype=ModelFailureSubtype.INVALID_GENERATION_JSON
            )
        if loaded:
            return value

    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        loaded, value, recursive = _raw_decode_json(decoder, stripped[index:])
        if recursive:
            raise NonRetryableModelError(
                subtype=ModelFailureSubtype.INVALID_GENERATION_JSON
            )
        if loaded and isinstance(value, (Mapping, list)):
            return value
    raise NonRetryableModelError(subtype=ModelFailureSubtype.INVALID_GENERATION_JSON)


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
