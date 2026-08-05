from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import replace

import httpx
import pytest

from career_agent_workbench.api_client import (
    ApiLlmClient,
    _retry_delay_seconds,
)
from career_agent_workbench.config import Settings
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
    OllamaError,
    OllamaTimeoutError,
    RetryableModelError,
    WorkflowError,
)
from career_agent_workbench.llm import build_llm_client, llm_settings_label
from career_agent_workbench.models import JobSearchQuery
from career_agent_workbench.ollama import OllamaClient
from career_agent_workbench.providers import LinkedInPublicJobsProvider
from career_agent_workbench.workflow_diagnostics import WorkflowStage
from career_agent_workbench.workflow_retry import run_model_operation

_HTTP_LOGGER_NAMES = (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)


def _api_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


class _TrackedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _BlockingAsyncStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await self.release.wait()
        yield b"{}"

    async def aclose(self) -> None:
        self.closed = True


class _DelayedTrackedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay_seconds: float) -> None:
        self._chunks = chunks
        self._delay_seconds = delay_seconds
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_seconds)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _assert_sanitized_exception(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _logger_state(logger: logging.Logger) -> tuple[object, ...]:
    return (
        logger.level,
        logger.propagate,
        logger.disabled,
        tuple(logger.handlers),
        tuple(logger.filters),
    )


def _http_logger_state() -> dict[str, tuple[object, ...]]:
    return {name: _logger_state(logging.getLogger(name)) for name in _HTTP_LOGGER_NAMES}


def test_api_request_shape_attribution_key_confinement_and_closure() -> None:
    marker_key = "SYNTHETIC-KEY-MARKER"
    marker_prompt = "SYNTHETIC-PROMPT-MARKER"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "https://api.example.invalid/api/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {marker_key}"
        assert (
            request.headers["http-referer"]
            == "https://github.com/mperkhou/career-agent-workbench"
        )
        assert request.headers["x-title"] == "career-agent-workbench"
        assert marker_key not in request.content.decode()
        body = json.loads(request.content)
        assert body["model"] == "synthetic-model"
        assert body["messages"] == [{"role": "user", "content": marker_prompt}]
        return _api_response("<think>private reasoning</think>Synthetic answer")

    client = ApiLlmClient(
        base_url="https://api.example.invalid/api/v1",
        model="synthetic-model",
        api_key=marker_key,
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_text(marker_prompt) == "Synthetic answer"
        await client.aclose()
        assert client._client.is_closed

    asyncio.run(scenario())
    assert len(requests) == 1
    assert marker_key not in repr(client)
    assert "api.example.invalid" not in repr(client)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"role": "synthetic"}', {"role": "synthetic"}),
        ('```json\n{"role": "synthetic"}\n```', {"role": "synthetic"}),
        ('prefix {"role": "synthetic"} suffix', {"role": "synthetic"}),
        ('[{"q": "synthetic"}]', {"queries": [{"q": "synthetic"}]}),
        (
            '```json\n[{"q": "synthetic"}]\n```',
            {"queries": [{"q": "synthetic"}]},
        ),
        (
            'prefix [{"q": "synthetic"}] suffix',
            {"queries": [{"q": "synthetic"}]},
        ),
    ],
)
def test_api_json_object_and_exact_array_wrapping(
    content: str,
    expected: dict[str, object],
) -> None:
    transport = httpx.MockTransport(lambda _: _api_response(content))
    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        transport=transport,
    )

    async def scenario() -> None:
        assert await client.generate_json("synthetic prompt") == expected
        await client.aclose()

    asyncio.run(scenario())


def test_api_transient_retry_empty_retry_delays_and_permanent_failure() -> None:
    calls = 0
    delays: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "999"})
        if calls == 2:
            return _api_response("<think>only reasoning</think>")
        return _api_response("Synthetic recovery")

    async def sleep(delay: float) -> None:
        delays.append(delay)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=99,
        retry_backoff_seconds=0.25,
        sleep=sleep,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_text("synthetic prompt") == "Synthetic recovery"
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 3
    assert delays == [120.0, 0.5]

    request = httpx.Request("POST", "https://api.example.invalid/")
    for retry_after in ("bad", "-1", "nan", "inf"):
        response = httpx.Response(
            429,
            headers={"retry-after": retry_after},
            request=request,
        )
        assert _retry_delay_seconds(response, 2, 0.5) == 1.0

    permanent_calls = 0

    def permanent(_: httpx.Request) -> httpx.Response:
        nonlocal permanent_calls
        permanent_calls += 1
        return httpx.Response(400, text="UPSTREAM-SENSITIVE-MARKER")

    permanent_client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=5,
        transport=httpx.MockTransport(permanent),
    )

    async def permanent_scenario() -> None:
        with pytest.raises(LlmError, match="HTTP status 400") as captured:
            await permanent_client.generate_text("synthetic prompt")
        assert "UPSTREAM-SENSITIVE-MARKER" not in str(captured.value)
        assert captured.value.__cause__ is None
        await permanent_client.aclose()

    asyncio.run(permanent_scenario())
    assert permanent_calls == 1


@pytest.mark.parametrize(
    ("client_type", "error_type"),
    [(ApiLlmClient, LlmTimeoutError), (OllamaClient, OllamaTimeoutError)],
)
def test_public_llm_timeout_boundaries_preserve_typed_timeouts(
    client_type,
    error_type: type[Exception],
) -> None:
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic private timeout detail")

    common = {
        "base_url": "https://api.example.invalid",
        "model": "synthetic-model",
        "timeout_seconds": 3,
        "transport": httpx.MockTransport(timeout),
    }
    client = (
        client_type(api_key="synthetic-key", retry_attempts=1, **common)
        if client_type is ApiLlmClient
        else client_type(**common)
    )

    async def scenario() -> None:
        with pytest.raises(error_type) as captured:
            await client.generate_text("synthetic prompt")
        assert "private timeout detail" not in str(captured.value)
        await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="UPSTREAM-SENSITIVE-MARKER"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, content=b"x" * 2_000_001),
    ],
)
def test_api_body_shape_bounds_and_sanitized_failures(
    response: httpx.Response,
) -> None:
    marker_prompt = "PROMPT-SENSITIVE-MARKER"
    marker_key = "KEY-SENSITIVE-MARKER"
    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key=marker_key,
        timeout_seconds=3,
        retry_attempts=1,
        transport=httpx.MockTransport(lambda _: response),
    )

    async def scenario() -> None:
        with pytest.raises(LlmError) as captured:
            await client.generate_text(marker_prompt)
        rendered = f"{captured.value!r} {captured.value}"
        for marker in (
            marker_prompt,
            marker_key,
            "api.example.invalid",
            "UPSTREAM-SENSITIVE-MARKER",
        ):
            assert marker not in rendered
        assert captured.value.__cause__ is None
        await client.aclose()

    asyncio.run(scenario())


def test_api_endpoint_retry_validation_and_transport_sanitization(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "CONFIG-SENSITIVE-MARKER"
    invalid = (
        f"http://{marker}.example.invalid",
        f"https://user:{marker}@api.example.invalid",
        f"https://api.example.invalid/path?token={marker}",
        f"https://api.example.invalid/path#{marker}",
        "https:///missing-host",
    )
    for endpoint in invalid:
        with pytest.raises(LlmError) as captured:
            ApiLlmClient(
                base_url=endpoint,
                model="synthetic-model",
                api_key="synthetic-key",
                timeout_seconds=3,
            )
        assert marker not in str(captured.value)
        assert endpoint not in str(captured.value)
        assert captured.value.__cause__ is None

    for backoff in (-1, float("nan"), float("inf")):
        with pytest.raises(LlmError, match="retry configuration"):
            ApiLlmClient(
                base_url="https://api.example.invalid",
                model="synthetic-model",
                api_key="synthetic-key",
                timeout_seconds=3,
                retry_backoff_seconds=backoff,
            )

    def transport_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(marker, request=request)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        transport=httpx.MockTransport(transport_failure),
    )

    async def scenario() -> None:
        with pytest.raises(LlmError) as captured:
            await client.generate_text(marker)
        assert marker not in str(captured.value)
        assert captured.value.__cause__ is None
        await client.aclose()

    asyncio.run(scenario())
    streams = capsys.readouterr()
    assert marker not in streams.out
    assert marker not in streams.err
    assert marker not in caplog.text


def test_concurrent_generation_logging_is_marker_safe_and_restores_logger(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    markers = (
        "API-URL-SYNTHETIC-MARKER",
        "OLLAMA-URL-SYNTHETIC-MARKER",
        "API-KEY-SYNTHETIC-MARKER",
        "PROMPT-SYNTHETIC-MARKER",
    )
    arrivals = 0
    both_arrived = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            both_arrived.set()
        await both_arrived.wait()
        if request.url.path.endswith("/chat/completions"):
            return _api_response("Synthetic answer")
        return httpx.Response(200, json={"response": "Synthetic answer"})

    caplog.set_level(logging.INFO, logger="httpx")
    logger = logging.getLogger("httpx")
    state_before = (
        logger.level,
        logger.propagate,
        logger.disabled,
        tuple(logger.handlers),
        tuple(logger.filters),
    )
    transport = httpx.MockTransport(handler)
    api = ApiLlmClient(
        base_url="https://api-url-synthetic-marker.example.invalid",
        model="synthetic-model",
        api_key="API-KEY-SYNTHETIC-MARKER",
        timeout_seconds=3,
        transport=transport,
    )
    ollama = OllamaClient(
        base_url="https://ollama-url-synthetic-marker.example.invalid",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=transport,
    )

    async def scenario() -> None:
        values = await asyncio.gather(
            api.generate_text("PROMPT-SYNTHETIC-MARKER"),
            ollama.generate_text("PROMPT-SYNTHETIC-MARKER"),
        )
        assert values == ["Synthetic answer", "Synthetic answer"]
        await api.aclose()
        await ollama.aclose()

    asyncio.run(scenario())
    state_after = (
        logger.level,
        logger.propagate,
        logger.disabled,
        tuple(logger.handlers),
        tuple(logger.filters),
    )
    assert state_after == state_before
    streams = capsys.readouterr()
    rendered = f"{caplog.text}\n{streams.out}\n{streams.err}"
    assert all(marker not in rendered for marker in markers)


def test_concurrent_public_clients_scope_all_http_logging_and_restore_state() -> None:
    package_marker = "PACKAGE-HTTP-LOG-SYNTHETIC-MARKER"
    unrelated_marker = "UNRELATED-HTTP-LOG-SYNTHETIC-MARKER"
    private_markers = (
        "API-HOST-SYNTHETIC-MARKER",
        "OLLAMA-HOST-SYNTHETIC-MARKER",
        "QUERY-SYNTHETIC-MARKER",
    )
    arrivals = 0
    all_arrived = asyncio.Event()
    release = asyncio.Event()
    capture = _LogCapture()
    loggers = [logging.getLogger(name) for name in _HTTP_LOGGER_NAMES]
    original_state = _http_logger_state()
    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(capture)
    configured_state = _http_logger_state()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal arrivals
        arrivals += 1
        if arrivals == 4:
            all_arrived.set()
        await all_arrived.wait()
        for name in _HTTP_LOGGER_NAMES:
            logging.getLogger(name).debug(package_marker)
        await release.wait()
        if request.url.path.endswith("/chat/completions"):
            return _api_response("Synthetic answer")
        if request.url.path.endswith("/api/generate"):
            return httpx.Response(200, json={"response": "Synthetic answer"})
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"<html></html>",
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html><h2>Synthetic role</h2></html>",
        )

    transport = httpx.MockTransport(handler)
    api = ApiLlmClient(
        base_url="https://api-host-synthetic-marker.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        transport=transport,
    )
    ollama = OllamaClient(
        base_url="https://ollama-host-synthetic-marker.example.invalid",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=transport,
    )
    provider = LinkedInPublicJobsProvider(
        user_agent="career-agent-workbench-tests",
        timeout_seconds=3,
        transport=transport,
    )
    query = JobSearchQuery(
        keywords="QUERY-SYNTHETIC-MARKER",
        location="Synthetic location",
    )

    async def unrelated_logging() -> None:
        await all_arrived.wait()
        logging.getLogger("httpcore.connection").debug(unrelated_marker)
        release.set()

    async def scenario() -> None:
        values = await asyncio.gather(
            api.generate_text("Synthetic prompt"),
            ollama.generate_text("Synthetic prompt"),
            provider.search_jobs(query),
            provider.get_job_details("123"),
            unrelated_logging(),
        )
        assert values[0] == "Synthetic answer"
        assert values[1] == "Synthetic answer"
        assert values[2] == []
        assert values[3].job_id == "123"
        await api.aclose()
        await ollama.aclose()
        await provider.aclose()

    try:
        asyncio.run(scenario())
        assert _http_logger_state() == configured_state
        assert unrelated_marker in capture.messages
        assert package_marker not in capture.messages
        assert all(
            marker.casefold() not in "\n".join(capture.messages).casefold()
            for marker in private_markers
        )
    finally:
        for logger in loggers:
            logger.removeHandler(capture)
            previous = original_state[logger.name]
            logger.setLevel(previous[0])
        assert _http_logger_state() == original_state


def test_http_logger_state_is_restored_on_every_request_exit_path() -> None:
    state_before = _http_logger_state()

    async def scenario() -> None:
        status_client = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            retry_attempts=1,
            transport=httpx.MockTransport(lambda _: httpx.Response(302)),
        )
        with pytest.raises(LlmError):
            await status_client.generate_text("Synthetic prompt")
        await status_client.aclose()
        assert _http_logger_state() == state_before

        def transport_failure(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("synthetic failure", request=request)

        failed_client = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            retry_attempts=1,
            transport=httpx.MockTransport(transport_failure),
        )
        with pytest.raises(LlmError):
            await failed_client.generate_text("Synthetic prompt")
        await failed_client.aclose()
        assert _http_logger_state() == state_before

        oversized_stream = _TrackedAsyncStream([b"unused"])
        oversized_client = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            retry_attempts=1,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={
                        "content-type": "application/json",
                        "content-length": "2000001",
                    },
                    stream=oversized_stream,
                )
            ),
        )
        with pytest.raises(LlmError):
            await oversized_client.generate_text("Synthetic prompt")
        assert oversized_stream.closed
        await oversized_client.aclose()
        assert _http_logger_state() == state_before

        blocked_stream = _BlockingAsyncStream()
        cancelled_client = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            retry_attempts=1,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    stream=blocked_stream,
                )
            ),
        )
        task = asyncio.create_task(cancelled_client.generate_text("Synthetic prompt"))
        await blocked_stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert blocked_stream.closed
        await cancelled_client.aclose()
        assert _http_logger_state() == state_before

    asyncio.run(scenario())


@pytest.mark.parametrize("client_kind", ["api", "ollama"])
def test_generation_stream_bounds_status_and_closure(client_kind: str) -> None:
    error_type = LlmError if client_kind == "api" else OllamaError

    def build_client(response: httpx.Response) -> ApiLlmClient | OllamaClient:
        transport = httpx.MockTransport(lambda _: response)
        if client_kind == "api":
            return ApiLlmClient(
                base_url="https://api.example.invalid",
                model="synthetic-model",
                api_key="synthetic-key",
                timeout_seconds=3,
                retry_attempts=1,
                transport=transport,
            )
        return OllamaClient(
            base_url="https://ollama.example.invalid",
            model="synthetic-local-model",
            timeout_seconds=3,
            transport=transport,
        )

    async def invoke(client: ApiLlmClient | OllamaClient) -> str:
        return await client.generate_text("synthetic prompt")

    async def scenario() -> None:
        declared = _TrackedAsyncStream([b'{"unused":true}'])
        client = build_client(
            httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "content-length": "2000001",
                },
                stream=declared,
            )
        )
        with pytest.raises(error_type) as captured:
            await invoke(client)
        _assert_sanitized_exception(captured.value)
        assert declared.consumed == 0
        assert declared.closed
        await client.aclose()

        crossing = _TrackedAsyncStream(
            [
                b"x" * 1_000_000,
                b"x" * 1_000_000,
                b"x",
                b"must-not-be-consumed",
            ]
        )
        client = build_client(
            httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "content-length": "1",
                },
                stream=crossing,
            )
        )
        with pytest.raises(error_type) as captured:
            await invoke(client)
        _assert_sanitized_exception(captured.value)
        assert crossing.consumed == 3
        assert crossing.closed
        await client.aclose()

        status_stream = _TrackedAsyncStream([b"redirect body"])
        client = build_client(
            httpx.Response(
                302,
                headers={"content-type": "application/json"},
                stream=status_stream,
            )
        )
        with pytest.raises(error_type) as captured:
            await invoke(client)
        _assert_sanitized_exception(captured.value)
        assert status_stream.consumed == 0
        assert status_stream.closed
        await client.aclose()

        invalid_stream = _TrackedAsyncStream([b"not-json"])
        client = build_client(
            httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=invalid_stream,
            )
        )
        with pytest.raises(error_type) as captured:
            await invoke(client)
        _assert_sanitized_exception(captured.value)
        assert invalid_stream.consumed == 1
        assert invalid_stream.closed
        await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("client_kind", ["api", "ollama"])
def test_generation_stream_closes_on_cancellation(client_kind: str) -> None:
    stream = _BlockingAsyncStream()
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=stream,
        )
    )
    if client_kind == "api":
        client: ApiLlmClient | OllamaClient = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            transport=transport,
        )
    else:
        client = OllamaClient(
            base_url="https://ollama.example.invalid",
            model="synthetic-local-model",
            timeout_seconds=3,
            transport=transport,
        )

    async def scenario() -> None:
        task = asyncio.create_task(client.generate_text("synthetic prompt"))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        await client.aclose()

    asyncio.run(scenario())


def test_generation_redirects_are_one_request_permanent_failures() -> None:
    api_calls = 0
    ollama_calls = 0
    sleeps: list[float] = []

    def api_handler(_: httpx.Request) -> httpx.Response:
        nonlocal api_calls
        api_calls += 1
        return httpx.Response(302, json={"choices": [{"message": {"content": "x"}}]})

    def ollama_handler(_: httpx.Request) -> httpx.Response:
        nonlocal ollama_calls
        ollama_calls += 1
        return httpx.Response(302, json={"response": "x"})

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    api = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=5,
        sleep=sleep,
        transport=httpx.MockTransport(api_handler),
    )
    ollama = OllamaClient(
        base_url="https://ollama.example.invalid",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=httpx.MockTransport(ollama_handler),
    )

    async def scenario() -> None:
        with pytest.raises(LlmError, match="HTTP status 302") as api_error:
            await api.generate_text("synthetic prompt")
        with pytest.raises(OllamaError, match="HTTP status 302") as ollama_error:
            await ollama.generate_text("synthetic prompt")
        _assert_sanitized_exception(api_error.value)
        _assert_sanitized_exception(ollama_error.value)
        await api.aclose()
        await ollama.aclose()

    asyncio.run(scenario())
    assert api_calls == 1
    assert ollama_calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "response_object",
    [
        {"choices": "invalid"},
        {"choices": [None]},
        {"choices": [{"message": {"content": 7}}]},
        {"choices": [{"finish_reason": 7, "message": {"content": "text"}}]},
    ],
)
def test_api_malformed_completion_is_not_retried(
    response_object: dict[str, object],
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=response_object)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=5,
        sleep=sleep,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(LlmError) as captured:
            await client.generate_text("synthetic prompt")
        _assert_sanitized_exception(captured.value)
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "empty_response",
    [
        {},
        {"choices": None},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_api_missing_null_and_empty_completion_remain_finitely_retryable(
    empty_response: dict[str, object],
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=empty_response)
        return _api_response("Synthetic recovery")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=2,
        retry_backoff_seconds=0,
        sleep=sleep,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_text("synthetic prompt") == "Synthetic recovery"
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 2
    assert sleeps == [0.0]


def test_api_valid_empty_completion_retains_finite_retry() -> None:
    responses = ["", "<think>synthetic reasoning</think>", "Synthetic recovery"]
    calls = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        value = responses[calls]
        calls += 1
        return _api_response(value)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=3,
        retry_backoff_seconds=0.25,
        sleep=sleep,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_text("synthetic prompt") == "Synthetic recovery"
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 3
    assert sleeps == [0.25, 0.5]


def test_llm_failures_remove_context_and_bound_json_recursion(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "SYNTHETIC-FAILURE-MARKER"

    def transport_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(marker, request=request)

    async def assert_failure(
        client: ApiLlmClient | OllamaClient,
        error_type: type[Exception],
        *,
        generated_json: bool = False,
    ) -> None:
        with pytest.raises(error_type) as captured:
            if generated_json:
                await client.generate_json(marker)
            else:
                await client.generate_text(marker)
        _assert_sanitized_exception(captured.value)
        assert marker not in f"{captured.value!r} {captured.value}"
        await client.aclose()

    async def scenario() -> None:
        api_failed = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            retry_attempts=1,
            transport=httpx.MockTransport(transport_failure),
        )
        ollama_failed = OllamaClient(
            base_url="https://ollama.example.invalid",
            model="synthetic-local-model",
            timeout_seconds=3,
            transport=httpx.MockTransport(transport_failure),
        )
        await assert_failure(api_failed, LlmError)
        await assert_failure(ollama_failed, OllamaError)

        deep_envelope = b"[" * 10_000 + b"0" + b"]" * 10_000
        api_envelope = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            retry_attempts=1,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=deep_envelope)
            ),
        )
        ollama_envelope = OllamaClient(
            base_url="https://ollama.example.invalid",
            model="synthetic-local-model",
            timeout_seconds=3,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=deep_envelope)
            ),
        )
        await assert_failure(api_envelope, LlmError)
        await assert_failure(ollama_envelope, OllamaError)

        deep_generated = "[" * 10_000 + "0" + "]" * 10_000
        api_generated = ApiLlmClient(
            base_url="https://api.example.invalid",
            model="synthetic-model",
            api_key="synthetic-key",
            timeout_seconds=3,
            transport=httpx.MockTransport(lambda _: _api_response(deep_generated)),
        )
        ollama_generated = OllamaClient(
            base_url="https://ollama.example.invalid",
            model="synthetic-local-model",
            timeout_seconds=3,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"response": deep_generated})
            ),
        )
        await assert_failure(api_generated, LlmError, generated_json=True)
        await assert_failure(
            ollama_generated,
            OllamaError,
            generated_json=True,
        )

    caplog.set_level(logging.INFO)
    asyncio.run(scenario())
    for endpoint, error_type, kwargs in (
        ("https://[", LlmError, {"api_key": "synthetic-key"}),
        ("https://[", OllamaError, {}),
    ):
        with pytest.raises(error_type) as captured:
            if error_type is LlmError:
                ApiLlmClient(
                    base_url=endpoint,
                    model="synthetic-model",
                    timeout_seconds=3,
                    **kwargs,
                )
            else:
                OllamaClient(
                    base_url=endpoint,
                    model="synthetic-model",
                    timeout_seconds=3,
                )
        _assert_sanitized_exception(captured.value)
    streams = capsys.readouterr()
    assert marker not in f"{caplog.text}\n{streams.out}\n{streams.err}"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"role": "synthetic"}', {"role": "synthetic"}),
        ('```json\n{"role": "synthetic"}\n```', {"role": "synthetic"}),
        ('prefix {"role": "synthetic"} suffix', {"role": "synthetic"}),
    ],
)
def test_ollama_text_json_thinking_and_closure(
    content: str,
    expected: dict[str, object],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://ollama.example.invalid/base/api/generate"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["format"] == "json"
        return httpx.Response(
            200,
            json={"response": f"<think>synthetic reasoning</think>{content}"},
        )

    client = OllamaClient(
        base_url="https://ollama.example.invalid/base",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_json("synthetic prompt") == expected
        await client.aclose()
        assert client._client.is_closed

    asyncio.run(scenario())
    assert len(requests) == 1
    assert "ollama.example.invalid" not in repr(client)


@pytest.mark.parametrize(
    "content",
    ["[]", '"scalar"', "7", "true", 'prefix [{"role": "synthetic"}] suffix'],
)
def test_ollama_rejects_nonmapping_json_without_wrapping(content: str) -> None:
    client = OllamaClient(
        base_url="https://ollama.example.invalid",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"response": content})
        ),
    )

    async def scenario() -> None:
        with pytest.raises(OllamaError, match="not an object") as captured:
            await client.generate_json("synthetic prompt")
        assert captured.value.__cause__ is None
        await client.aclose()

    asyncio.run(scenario())


def test_ollama_endpoint_response_bounds_empty_and_sanitization() -> None:
    invalid = (
        "http://ollama.example.invalid",
        "https://user:marker@ollama.example.invalid",
        "https://ollama.example.invalid/path?marker=value",
        "ftp://ollama.example.invalid",
    )
    for endpoint in invalid:
        with pytest.raises(OllamaError) as captured:
            OllamaClient(
                base_url=endpoint,
                model="synthetic-local-model",
                timeout_seconds=3,
            )
        assert "marker" not in str(captured.value)
        assert "ollama.example.invalid" not in str(captured.value)

    # Loopback HTTP is validation-only and remains backed by MockTransport.
    local = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"thinking": "Synthetic answer"})
        ),
    )
    assert asyncio.run(local.generate_text("synthetic prompt")) == "Synthetic answer"
    asyncio.run(local.aclose())

    responses = (
        httpx.Response(503, text="UPSTREAM-SENSITIVE-MARKER"),
        httpx.Response(200, text="UPSTREAM-SENSITIVE-MARKER"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"response": ""}),
        httpx.Response(200, content=b"x" * 2_000_001),
    )
    for response in responses:
        client = OllamaClient(
            base_url="https://ollama.example.invalid",
            model="synthetic-local-model",
            timeout_seconds=3,
            transport=httpx.MockTransport(lambda _, item=response: item),
        )

        async def scenario(selected: OllamaClient = client) -> None:
            with pytest.raises(OllamaError) as captured:
                await selected.generate_text("PROMPT-SENSITIVE-MARKER")
            rendered = f"{captured.value!r} {captured.value}"
            assert "UPSTREAM-SENSITIVE-MARKER" not in rendered
            assert "PROMPT-SENSITIVE-MARKER" not in rendered
            assert "ollama.example.invalid" not in rendered
            assert captured.value.__cause__ is None
            await selected.aclose()

        asyncio.run(scenario())

    marker = "OLLAMA-TRANSPORT-SENSITIVE-MARKER"

    def transport_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(marker, request=request)

    failed = OllamaClient(
        base_url="https://ollama.example.invalid",
        model="synthetic-local-model",
        timeout_seconds=3,
        transport=httpx.MockTransport(transport_failure),
    )

    async def failed_scenario() -> None:
        with pytest.raises(OllamaError) as captured:
            await failed.generate_text(marker)
        assert marker not in f"{captured.value!r} {captured.value}"
        assert captured.value.__cause__ is None
        await failed.aclose()

    asyncio.run(failed_scenario())


def test_settings_selection_safe_label_missing_key_and_zero_request() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _api_response("unexpected")

    transport = httpx.MockTransport(handler)
    settings = replace(
        Settings(),
        llm_api_base_url="https://api.example.invalid",
        llm_api_model="synthetic-api-model",
        llm_api_key="synthetic-key",
        llm_provider="api",
    )
    api = build_llm_client(settings, transport=transport)
    assert isinstance(api, ApiLlmClient)
    assert calls == 0
    assert llm_settings_label(settings) == "api:synthetic-api-model"
    asyncio.run(api.aclose())

    ollama_settings = replace(
        settings,
        llm_provider="ollama",
        ollama_base_url="https://ollama.example.invalid",
        ollama_model="synthetic-local-model",
    )
    ollama = build_llm_client(ollama_settings, transport=transport)
    assert isinstance(ollama, OllamaClient)
    assert calls == 0
    assert llm_settings_label(ollama_settings) == "ollama:synthetic-local-model"
    asyncio.run(ollama.aclose())

    with pytest.raises(
        WorkflowError,
        match="CAREER_AGENT_WORKBENCH_LLM_API_KEY",
    ):
        build_llm_client(replace(settings, llm_api_key=""), transport=transport)
    with pytest.raises(WorkflowError, match="Unsupported LLM provider"):
        build_llm_client(
            replace(settings, llm_provider="synthetic-unsupported"),
            transport=transport,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "response",
    (
        httpx.Response(429),
        _api_response("<think>reasoning without a completion</think>"),
    ),
)
def test_workflow_api_client_does_not_hide_non_timeout_retries(
    response: httpx.Response,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    settings = replace(
        Settings(),
        llm_api_base_url="https://api.example.invalid",
        llm_api_model="synthetic-api-model",
        llm_api_key="synthetic-key",
        llm_provider="api",
    )
    client = build_llm_client(settings, transport=httpx.MockTransport(handler))

    async def scenario() -> None:
        with pytest.raises(LlmError):
            await client.generate_text("synthetic prompt")
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1


@pytest.mark.parametrize(
    "operation",
    [
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.PERMANENT_HTTP,
            http_status=400,
        ),
        lambda: RetryableModelError(subtype=ModelFailureSubtype.MALFORMED_ENVELOPE),
        lambda: NonRetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=429,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=400,
        ),
        lambda: NonRetryableModelError(
            subtype=ModelFailureSubtype.PERMANENT_HTTP,
            http_status=503,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.TRANSPORT_CONNECT,
            http_status=503,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.TRANSPORT_READ,
            retry_after_seconds=1,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=429,
            retry_after_seconds=121,
        ),
        lambda: RetryableModelError(subtype=ModelFailureSubtype.TIMEOUT),
        lambda: NonRetryableModelError(
            subtype=ModelFailureSubtype.PERMANENT_HTTP,
            http_status=200,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_PERMANENT,
        ),
        lambda: NonRetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
        ),
        lambda: NonRetryableModelError(
            subtype=ModelFailureSubtype.INVALID_GENERATION_JSON,
        ),
        lambda: RetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
            retry_after_seconds=121,
        ),
    ],
)
def test_typed_model_failure_rejects_contradictory_metadata(operation) -> None:
    with pytest.raises(ValueError, match="Model failure metadata is invalid"):
        operation()


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_typed_transient_http_contract_accepts_only_allowlisted_statuses(
    status: int,
) -> None:
    error = RetryableModelError(
        subtype=ModelFailureSubtype.TRANSIENT_HTTP,
        http_status=status,
        retry_after_seconds=0,
    )
    assert error.http_status == status
    assert error.retry_after_seconds == 0


def test_response_summary_rejects_contradictory_closed_metadata() -> None:
    with pytest.raises(ValueError, match="response metadata"):
        ModelResponseSummary(
            http_status=200,
            error_presence=ModelResponseErrorPresence.TOP_LEVEL,
            error_code=400,
            error_type=ModelResponseErrorType.RATE_LIMIT,
            finish_reason=ModelResponseFinishReason.UNAVAILABLE,
            choices_count=None,
            content_state=ModelResponseContentState.UNAVAILABLE,
        )
    with pytest.raises(ValueError, match="response metadata"):
        ModelResponseSummary(
            http_status=200,
            error_presence=ModelResponseErrorPresence.TOP_LEVEL,
            error_code=429,
            error_type=ModelResponseErrorType.SERVER,
            finish_reason=ModelResponseFinishReason.UNAVAILABLE,
            choices_count=None,
            content_state=ModelResponseContentState.UNAVAILABLE,
        )
    permanent = ModelResponseSummary(
        http_status=200,
        error_presence=ModelResponseErrorPresence.TOP_LEVEL,
        error_code=400,
        error_type=ModelResponseErrorType.PERMANENT_REQUEST,
        finish_reason=ModelResponseFinishReason.UNAVAILABLE,
        choices_count=None,
        content_state=ModelResponseContentState.UNAVAILABLE,
    )
    with pytest.raises(ValueError, match="failure metadata"):
        RetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
            response_summary=permanent,
        )


def test_typed_embedded_error_rechecks_pair_and_preserves_type_only_shape() -> None:
    type_only = ModelResponseSummary(
        http_status=200,
        error_presence=ModelResponseErrorPresence.TOP_LEVEL,
        error_code=None,
        error_type=ModelResponseErrorType.SERVER,
        finish_reason=ModelResponseFinishReason.UNAVAILABLE,
        choices_count=None,
        content_state=ModelResponseContentState.UNAVAILABLE,
    )
    accepted = RetryableModelError(
        subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
        response_summary=type_only,
    )
    assert accepted.response_summary is type_only

    mismatched = ModelResponseSummary(
        http_status=200,
        error_presence=ModelResponseErrorPresence.TOP_LEVEL,
        error_code=500,
        error_type=ModelResponseErrorType.SERVER,
        finish_reason=ModelResponseFinishReason.UNAVAILABLE,
        choices_count=None,
        content_state=ModelResponseContentState.UNAVAILABLE,
    )
    object.__setattr__(mismatched, "error_code", 429)
    with pytest.raises(ValueError, match="failure metadata"):
        RetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
            response_summary=mismatched,
        )


@pytest.mark.parametrize(
    ("error_factory", "expected_subtype", "expected_calls"),
    [
        (
            lambda request: httpx.ReadTimeout(
                "synthetic timeout",
                request=request,
            ),
            ModelFailureSubtype.TIMEOUT,
            1,
        ),
        (
            lambda request: httpx.ConnectError(
                "synthetic connection interruption",
                request=request,
            ),
            ModelFailureSubtype.TRANSPORT_CONNECT,
            1,
        ),
        (
            lambda request: httpx.ReadError(
                "synthetic read interruption",
                request=request,
            ),
            ModelFailureSubtype.TRANSPORT_READ,
            1,
        ),
        (
            lambda request: httpx.RemoteProtocolError(
                "synthetic remote interruption",
                request=request,
            ),
            ModelFailureSubtype.TRANSPORT_PROTOCOL,
            1,
        ),
        (
            lambda request: httpx.LocalProtocolError(
                "synthetic local misuse",
                request=request,
            ),
            ModelFailureSubtype.UNEXPECTED_MODEL,
            1,
        ),
    ],
)
def test_standalone_api_transport_contract_does_not_retry_request_failure(
    error_factory,
    expected_subtype: ModelFailureSubtype,
    expected_calls: int,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_factory(request)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=2,
        retry_backoff_seconds=0,
        sleep=sleep,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(LlmError) as captured:
            await client.generate_text("synthetic prompt")
        assert captured.value.subtype is expected_subtype
        await client.aclose()

    asyncio.run(scenario())
    assert calls == expected_calls
    assert len(sleeps) == expected_calls - 1


@pytest.mark.parametrize(
    ("case", "expected_subtype", "generated_json"),
    [
        ("timeout", ModelFailureSubtype.TIMEOUT, False),
        ("connect", ModelFailureSubtype.TRANSPORT_CONNECT, False),
        ("read", ModelFailureSubtype.TRANSPORT_READ, False),
        ("remote_protocol", ModelFailureSubtype.TRANSPORT_PROTOCOL, False),
        ("local_protocol", ModelFailureSubtype.UNEXPECTED_MODEL, False),
        ("unexpected", ModelFailureSubtype.UNEXPECTED_MODEL, False),
        ("transient_http", ModelFailureSubtype.TRANSIENT_HTTP, False),
        ("permanent_http", ModelFailureSubtype.PERMANENT_HTTP, False),
        ("empty", ModelFailureSubtype.EMPTY_COMPLETION, False),
        ("malformed", ModelFailureSubtype.MALFORMED_ENVELOPE, False),
        ("too_large", ModelFailureSubtype.RESPONSE_TOO_LARGE, False),
        ("invalid_json", ModelFailureSubtype.INVALID_GENERATION_JSON, True),
    ],
)
def test_api_failure_translation_matrix_is_typed_and_content_free(
    case: str,
    expected_subtype: ModelFailureSubtype,
    generated_json: bool,
) -> None:
    marker = "SYNTHETIC-UPSTREAM-PRIVATE-MARKER"

    def handler(request: httpx.Request) -> httpx.Response:
        if case == "timeout":
            raise httpx.ReadTimeout(marker, request=request)
        if case == "connect":
            raise httpx.ConnectError(marker, request=request)
        if case == "read":
            raise httpx.ReadError(marker, request=request)
        if case == "remote_protocol":
            raise httpx.RemoteProtocolError(marker, request=request)
        if case == "local_protocol":
            raise httpx.LocalProtocolError(marker, request=request)
        if case == "unexpected":
            raise RuntimeError(marker)
        if case == "transient_http":
            return httpx.Response(503, text=marker)
        if case == "permanent_http":
            return httpx.Response(400, text=marker)
        if case == "empty":
            return _api_response("<think>bounded reasoning</think>")
        if case == "malformed":
            return httpx.Response(200, content=marker.encode())
        if case == "too_large":
            return httpx.Response(
                200,
                headers={"content-length": "2000001"},
                content=b"unused",
            )
        if case == "invalid_json":
            return _api_response(marker)
        raise AssertionError("unknown synthetic case")

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=1,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(LlmError) as captured:
            if generated_json:
                await client.generate_json("synthetic prompt")
            else:
                await client.generate_text("synthetic prompt")
        assert captured.value.subtype is expected_subtype
        assert marker not in f"{captured.value!r} {captured.value}"
        assert captured.value.__cause__ is None
        await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("generation", ['{"score": NaN}', '{"score": Infinity}'])
def test_api_generation_surfaces_retryable_syntax_failure_without_internal_retry(
    generation: str,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _api_response(generation)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=5,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(RetryableModelError) as captured:
            await client.generate_json("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.INVALID_GENERATION_JSON
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1


def test_api_generation_valid_json_wrong_shape_remains_nonretryable() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _api_response('"synthetic scalar"')

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=5,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(NonRetryableModelError) as captured:
            await client.generate_json("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.UNEXPECTED_MODEL
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1


@pytest.mark.parametrize(
    ("code", "error_type", "expected_type"),
    [
        (429, "rate_limit_exceeded", ModelResponseErrorType.RATE_LIMIT),
        (503, "provider_overloaded", ModelResponseErrorType.OVERLOADED),
        (502, "provider_unavailable", ModelResponseErrorType.PROVIDER_UNAVAILABLE),
        (500, "server", ModelResponseErrorType.SERVER),
        (504, "timeout", ModelResponseErrorType.UPSTREAM_TIMEOUT),
        (500, "unmapped", ModelResponseErrorType.UNMAPPED),
    ],
)
def test_api_canonical_embedded_transient_pair_retries_with_closed_summary(
    code: int,
    error_type: str,
    expected_type: ModelResponseErrorType,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def embedded_response() -> httpx.Response:
        return httpx.Response(
            200,
            headers={"retry-after": "999"},
            json={
                "error": {
                    "code": code,
                    "metadata": {"error_type": error_type},
                    "message": "RAW-ERROR-MESSAGE",
                    "provider": "PRIVATE-PROVIDER-MARKER",
                }
            },
        )

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return embedded_response()
        return _api_response("Synthetic recovery")

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=2,
        sleep=sleep,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_text("synthetic prompt") == "Synthetic recovery"
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 2
    assert sleeps == [120.0]

    inspection_client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=1,
        transport=httpx.MockTransport(lambda _: embedded_response()),
    )

    async def inspect_failure() -> RetryableModelError:
        with pytest.raises(RetryableModelError) as captured:
            await inspection_client.generate_text("synthetic prompt")
        await inspection_client.aclose()
        return captured.value

    error = asyncio.run(inspect_failure())
    assert error.subtype is ModelFailureSubtype.EMBEDDED_TRANSIENT
    assert error.response_summary == ModelResponseSummary(
        http_status=200,
        error_presence=ModelResponseErrorPresence.TOP_LEVEL,
        error_code=code,
        error_type=expected_type,
        finish_reason=ModelResponseFinishReason.UNAVAILABLE,
        choices_count=None,
        content_state=ModelResponseContentState.UNAVAILABLE,
    )
    rendered = f"{error!r} {error}"
    assert error.retry_after_seconds == 120
    assert "RAW-ERROR-MESSAGE" not in rendered
    assert "PRIVATE-PROVIDER-MARKER" not in rendered


@pytest.mark.parametrize(
    ("error_object", "expected_type"),
    [
        ({"code": 429}, ModelResponseErrorType.RATE_LIMIT),
        ({"code": 502}, ModelResponseErrorType.PROVIDER_UNAVAILABLE),
        ({"code": 503}, ModelResponseErrorType.OVERLOADED),
        ({"code": 500}, ModelResponseErrorType.SERVER),
        ({"code": 504}, ModelResponseErrorType.UPSTREAM_TIMEOUT),
        (
            {"metadata": {"error_type": "timeout"}},
            ModelResponseErrorType.UPSTREAM_TIMEOUT,
        ),
    ],
)
def test_api_supported_code_only_and_type_only_embedded_shapes(
    error_object: dict[str, object],
    expected_type: ModelResponseErrorType,
) -> None:
    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=1,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"error": error_object})
        ),
    )

    async def scenario() -> None:
        with pytest.raises(RetryableModelError) as captured:
            await client.generate_text("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.EMBEDDED_TRANSIENT
        assert captured.value.response_summary is not None
        assert captured.value.response_summary.error_type is expected_type
        await client.aclose()

    asyncio.run(scenario())


def test_api_choice_embedded_error_requires_error_finish_reason() -> None:
    responses = [
        {
            "choices": [
                {
                    "error": {
                        "code": 503,
                        "metadata": {"error_type": "provider_overloaded"},
                    },
                    "finish_reason": "error",
                    "message": {"content": "partial invalid {"},
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": "Synthetic recovery"},
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        response = httpx.Response(200, json=responses[calls])
        calls += 1
        return response

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=2,
        retry_backoff_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        assert await client.generate_text("synthetic prompt") == "Synthetic recovery"
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 2

    contradictory = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "error": {"metadata": {"error_type": "timeout"}},
                            "finish_reason": "stop",
                            "message": {"content": "partial content"},
                        }
                    ]
                },
            )
        ),
    )

    async def contradictory_scenario() -> None:
        with pytest.raises(NonRetryableModelError) as captured:
            await contradictory.generate_text("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.MALFORMED_ENVELOPE
        await contradictory.aclose()

    asyncio.run(contradictory_scenario())


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        (403, "permission_denied"),
        (400, "invalid_prompt"),
        (400, "content_policy_violation"),
        (402, "payment_required"),
        (400, "max_tokens_exceeded"),
        (400, "token_limit_exceeded"),
        (400, "string_too_long"),
        (404, "not_found"),
        (412, "precondition_failed"),
        (413, "payload_too_large"),
        (422, "unprocessable"),
        (400, "refusal"),
        (400, "invalid_image"),
        (400, "image_too_large"),
        (400, "image_too_small"),
        (400, "unsupported_image_format"),
        (404, "image_not_found"),
        (400, "image_download_failed"),
    ],
)
def test_api_canonical_permanent_embedded_error_never_retries(
    code: int,
    error_type: str,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "error": {
                    "code": code,
                    "metadata": {"error_type": error_type},
                    "message": "RAW-PROVIDER-ERROR",
                },
                "choices": [
                    {
                        "message": {"content": "{invalid generated JSON"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(NonRetryableModelError) as captured:
            await client.generate_json("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.EMBEDDED_PERMANENT
        assert captured.value.response_summary is not None
        assert captured.value.response_summary.error_code == code
        assert captured.value.response_summary.error_type is (
            ModelResponseErrorType.PERMANENT_REQUEST
        )
        assert captured.value.response_summary.content_state is (
            ModelResponseContentState.PRESENT
        )
        assert "RAW-PROVIDER-ERROR" not in repr(captured.value)
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1


@pytest.mark.parametrize(
    "error_object",
    [
        {"code": 400, "metadata": {"error_type": "invalid_request_error"}},
        {"type": "policy_error"},
        {"metadata": {"error_type": "context_length_exceeded"}},
        {"metadata": {"error_type": "image_url_fetch_failed"}},
    ],
)
def test_api_supported_permanent_alias_never_retries(
    error_object: dict[str, object],
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"error": error_object})

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(NonRetryableModelError) as captured:
            await client.generate_text("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.EMBEDDED_PERMANENT
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1


@pytest.mark.parametrize(
    "error_object",
    [
        {},
        {"type": "unknown-private-provider-type"},
        {"code": "429", "metadata": {"error_type": "timeout"}},
        {
            "type": "rate_limit",
            "metadata": {"error_type": "invalid_request_error"},
        },
        {"code": 400, "metadata": {"error_type": "timeout"}},
        {"code": 429, "metadata": {"error_type": "server"}},
        {"code": 500, "metadata": {"error_type": "rate_limit_exceeded"}},
        {"code": 502, "metadata": {"error_type": "provider_overloaded"}},
        {"code": 503, "metadata": {"error_type": "provider_unavailable"}},
        {"code": 504, "metadata": {"error_type": "unmapped"}},
        {"code": 500, "metadata": {"error_type": "permission_denied"}},
        {"code": 429, "metadata": {"error_type": "unknown-provider-type"}},
        {"metadata": "invalid"},
    ],
)
def test_api_malformed_or_contradictory_embedded_error_never_retries(
    error_object: dict[str, object],
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"error": error_object})

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=3,
        retry_attempts=5,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(NonRetryableModelError) as captured:
            await client.generate_text("synthetic prompt")
        assert captured.value.subtype is ModelFailureSubtype.MALFORMED_ENVELOPE
        rendered = f"{captured.value!r} {captured.value}"
        assert "unknown-private-provider-type" not in rendered
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 1


def test_outer_deadline_closes_delayed_stream_then_uses_next_boundary() -> None:
    calls = 0
    first_stream = _DelayedTrackedStream(
        [b'{"choices":[', b'{"message":', b'{"content":"late"}}]}'],
        delay_seconds=0.02,
    )

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, stream=first_stream)
        return _api_response("Synthetic recovery")

    client = ApiLlmClient(
        base_url="https://api.example.invalid",
        model="synthetic-model",
        api_key="synthetic-key",
        timeout_seconds=0.05,
        retry_attempts=1,
        transport=httpx.MockTransport(handler),
    )

    async def no_backoff(_delay: float) -> None:
        return None

    async def scenario() -> None:
        assert (
            await run_model_operation(
                lambda: client.generate_text("synthetic prompt"),
                retries=1,
                stage=WorkflowStage.V1_EXPERIENCE,
                sleep=no_backoff,
                timeout_seconds=0.03,
            )
            == "Synthetic recovery"
        )
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 2
    assert first_stream.closed is True


def test_workflow_client_embedded_transient_uses_exact_outer_http_budget() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"error": {"metadata": {"error_type": "timeout"}}},
        )

    settings = replace(
        Settings(),
        llm_api_base_url="https://api.example.invalid",
        llm_api_model="synthetic-api-model",
        llm_api_key="synthetic-key",
        llm_provider="api",
    )
    client = build_llm_client(settings, transport=httpx.MockTransport(handler))

    async def no_backoff(_delay: float) -> None:
        return None

    async def scenario() -> None:
        with pytest.raises(RetryableModelError) as captured:
            await run_model_operation(
                lambda: client.generate_text("synthetic prompt"),
                retries=2,
                stage=WorkflowStage.V1_JOD,
                sleep=no_backoff,
                timeout_seconds=1,
            )
        assert captured.value.subtype is ModelFailureSubtype.EMBEDDED_TRANSIENT
        await client.aclose()

    asyncio.run(scenario())
    assert calls == 3
