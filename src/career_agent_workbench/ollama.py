"""Bounded Ollama generation client."""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from career_agent_workbench.api_client import (
    _bounded_http_request,
    _load_json,
    _raw_decode_json,
    _ResponseTooLarge,
)
from career_agent_workbench.errors import OllamaError, OllamaTimeoutError

MAX_RESPONSE_BYTES = 2_000_000


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    return bool(address is not None and address.is_loopback)


def _validated_ollama_base(value: str) -> str:
    parts = None
    hostname = None
    port = None
    try:
        parts = urlsplit(str(value).strip())
        hostname = parts.hostname
        port = parts.port
    except (TypeError, ValueError):
        pass
    scheme = parts.scheme.casefold() if parts is not None else ""
    if (
        parts is None
        or not hostname
        or scheme not in {"http", "https"}
        or scheme == "http"
        and not _is_loopback_host(hostname)
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port is not None
        and not 0 < port <= 65_535
    ):
        raise OllamaError("Ollama endpoint configuration is invalid.")
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, parts.netloc, path, "", ""))


class OllamaClient:
    """Own a clean client for one validated Ollama endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = _validated_ollama_base(base_url)
        self._model = str(model).strip()
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
            raise OllamaError("Ollama client configuration is invalid.")
        self._client = client

    def __repr__(self) -> str:
        return "OllamaClient(configured=True)"

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
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_response:
            payload["format"] = "json"
        request_error = None
        try:
            response, body = await _bounded_http_request(
                self._client,
                "POST",
                f"{self._base_url}/api/generate",
                max_response_bytes=MAX_RESPONSE_BYTES,
                json=payload,
            )
        except _ResponseTooLarge:
            request_error = OllamaError(
                "Ollama response exceeds the public size limit."
            )
        except httpx.TimeoutException:
            request_error = OllamaTimeoutError("Ollama request timed out.")
        except httpx.HTTPError:
            request_error = OllamaError("Ollama request failed.")
        if request_error is not None:
            raise request_error

        if not 200 <= response.status_code < 300:
            raise OllamaError(
                f"Ollama request failed with HTTP status {response.status_code}."
            )
        loaded, data, _ = _load_json(body)
        if not loaded:
            raise OllamaError("Ollama returned a non-JSON response.")
        if not isinstance(data, Mapping):
            raise OllamaError("Ollama returned an invalid response object.")
        text = data.get("response") or data.get("thinking")
        if not isinstance(text, str):
            raise OllamaError("Ollama returned an empty generation.")
        stripped = _strip_thinking(text.strip())
        if not stripped:
            raise OllamaError("Ollama returned an empty generation.")
        return stripped


def _parse_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    loaded, value, recursive = _load_json(stripped)
    if recursive:
        raise OllamaError("Ollama did not return a JSON object.")
    if not loaded:
        value = _extract_object(stripped)
    if not isinstance(value, Mapping):
        raise OllamaError("Ollama returned JSON that was not an object.")
    return value


def _extract_object(text: str) -> object:
    for match in re.finditer(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        loaded, value, recursive = _load_json(match.group(1).strip())
        if recursive:
            raise OllamaError("Ollama did not return a JSON object.")
        if not loaded:
            continue
        if isinstance(value, Mapping):
            return value
        raise OllamaError("Ollama returned JSON that was not an object.")

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        loaded, value, recursive = _raw_decode_json(decoder, text[index:])
        if recursive:
            raise OllamaError("Ollama did not return a JSON object.")
        if not loaded:
            continue
        if isinstance(value, Mapping):
            return value
        if isinstance(value, list):
            raise OllamaError("Ollama returned JSON that was not an object.")
    raise OllamaError("Ollama did not return a JSON object.")


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
