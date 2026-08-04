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

from career_agent_workbench.errors import LlmError, LlmTimeoutError

TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
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
        parsed = json.loads(value)
    except RecursionError:
        recursion_error = True
    except (json.JSONDecodeError, UnicodeDecodeError):
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
    except json.JSONDecodeError:
        recursion_error = False
    else:
        return True, parsed, False
    return False, None, recursion_error


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

        completion_error = LlmError("API LLM returned an empty generation.")
        for attempt in range(1, self._retry_attempts + 1):
            request_error = None
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
                request_error = LlmError(
                    "API LLM response exceeds the public size limit."
                )
            except httpx.TimeoutException:
                request_error = LlmTimeoutError("API LLM request timed out.")
            except httpx.HTTPError:
                request_error = LlmError("API LLM request failed.")
            if request_error is not None:
                raise request_error

            if not 200 <= response.status_code < 300:
                if (
                    response.status_code in TRANSIENT_HTTP_STATUSES
                    and attempt < self._retry_attempts
                ):
                    await self._sleep(
                        _retry_delay_seconds(
                            response,
                            attempt,
                            self._retry_backoff_seconds,
                        )
                    )
                    continue
                raise LlmError(
                    f"API LLM request failed with HTTP status {response.status_code}."
                )

            data = _response_json(body)
            text, completion_error, retryable = _completion_text(data)
            if completion_error is None:
                return text
            if not retryable:
                raise completion_error
            if attempt < self._retry_attempts:
                await self._sleep(
                    _retry_delay_seconds(
                        response,
                        attempt,
                        self._retry_backoff_seconds,
                    )
                )

        raise completion_error


def _response_json(body: bytes) -> Mapping[str, Any]:
    loaded, value, _ = _load_json(body)
    if not loaded:
        raise LlmError("API LLM returned a non-JSON response.")
    if not isinstance(value, Mapping):
        raise LlmError("API LLM returned an invalid response object.")
    return value


def _completion_text(
    data: Mapping[str, Any],
) -> tuple[str, LlmError | None, bool]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", LlmError("API LLM returned no completion choices."), False
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return "", LlmError("API LLM returned malformed completion data."), False
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return "", LlmError("API LLM returned malformed completion data."), False
    text = message.get("content")
    if not isinstance(text, str):
        return "", LlmError("API LLM returned malformed completion data."), False
    stripped = _strip_thinking(text.strip())
    if not stripped:
        return "", LlmError("API LLM returned an empty generation."), True
    return stripped, None, False


def _retry_delay_seconds(
    response: httpx.Response,
    attempt: int,
    base_delay_seconds: float,
) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            parsed = float(retry_after)
        except (TypeError, ValueError, OverflowError):
            parsed = -1.0
        if math.isfinite(parsed) and parsed >= 0:
            return min(parsed, MAX_RETRY_AFTER_SECONDS)
    exponent = max(0, int(attempt) - 1)
    return min(base_delay_seconds * (2**exponent), MAX_BACKOFF_SECONDS)


def _parse_json_object(text: str) -> Mapping[str, Any]:
    value = _parse_json_value(text)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list):
        return {"queries": value}
    raise LlmError("API LLM did not return a JSON object or array.")


def _parse_json_value(text: str) -> object:
    stripped = text.strip()
    loaded, value, recursive = _load_json(stripped)
    if recursive:
        raise LlmError("API LLM did not return valid JSON.")
    if loaded:
        return value

    for match in re.finditer(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        loaded, value, recursive = _load_json(match.group(1).strip())
        if recursive:
            raise LlmError("API LLM did not return valid JSON.")
        if loaded:
            return value

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        loaded, value, recursive = _raw_decode_json(decoder, stripped[index:])
        if recursive:
            raise LlmError("API LLM did not return valid JSON.")
        if loaded and isinstance(value, (Mapping, list)):
            return value
    raise LlmError("API LLM did not return valid JSON.")


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
