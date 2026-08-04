"""Bounded client for an OpenAI-compatible public API."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from career_agent_workbench.errors import (
    LlmError,
    LlmTimeoutError,
    ModelFailureSubtype,
    NonRetryableModelError,
    RetryableModelError,
    TRANSIENT_MODEL_HTTP_STATUSES,
    TypedModelError,
)

TRANSIENT_HTTP_STATUSES = TRANSIENT_MODEL_HTTP_STATUSES
MAX_RESPONSE_BYTES = 2_000_000
MAX_RETRY_AFTER_SECONDS = 120.0
MAX_BACKOFF_SECONDS = 60.0

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


class _ResponseTooLarge(Exception):
    """Signal a response-size violation without retaining response data."""


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
                if response.status_code in TRANSIENT_HTTP_STATUSES:
                    status_error: TypedModelError = RetryableModelError(
                        subtype=ModelFailureSubtype.TRANSIENT_HTTP,
                        http_status=response.status_code,
                        retry_after_seconds=retry_after,
                    )
                else:
                    status_error = NonRetryableModelError(
                        subtype=ModelFailureSubtype.PERMANENT_HTTP,
                        http_status=response.status_code,
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

            data = _response_json(body)
            text, completion_error = _completion_text(data)
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


def _response_json(body: bytes) -> Mapping[str, Any]:
    loaded, value, _ = _load_json(body)
    if not loaded:
        raise NonRetryableModelError(subtype=ModelFailureSubtype.MALFORMED_ENVELOPE)
    if not isinstance(value, Mapping):
        raise NonRetryableModelError(subtype=ModelFailureSubtype.MALFORMED_ENVELOPE)
    return value


def _completion_text(
    data: Mapping[str, Any],
) -> tuple[str, TypedModelError | None]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return "", NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return "", NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE
        )
    text = message.get("content")
    if not isinstance(text, str):
        return "", NonRetryableModelError(
            subtype=ModelFailureSubtype.MALFORMED_ENVELOPE
        )
    stripped = _strip_thinking(text.strip())
    if not stripped:
        return "", RetryableModelError(subtype=ModelFailureSubtype.EMPTY_COMPLETION)
    return stripped, None


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
