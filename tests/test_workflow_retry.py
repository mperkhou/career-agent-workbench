from __future__ import annotations

import json

import pytest

from career_agent_workbench import workflow_retry
from career_agent_workbench.errors import (
    LlmError,
    LlmTimeoutError,
    ModelFailureSubtype,
    NonRetryableModelError,
    OllamaError,
    OllamaTimeoutError,
    RetryableModelError,
)
from career_agent_workbench.workflow_diagnostics import WorkflowStage
from career_agent_workbench.workflow_retry import run_model_operation


def _diagnostics(captured: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def _retryable_error(subtype: ModelFailureSubtype) -> RetryableModelError:
    if subtype is ModelFailureSubtype.TRANSIENT_HTTP:
        return RetryableModelError(subtype=subtype, http_status=429)
    return RetryableModelError(subtype=subtype)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=status,
            retry_after_seconds=0,
        )
        for status in (429, 502, 503, 504)
    ]
    + [
        _retryable_error(ModelFailureSubtype.TRANSPORT_CONNECT),
        _retryable_error(ModelFailureSubtype.TRANSPORT_READ),
        _retryable_error(ModelFailureSubtype.TRANSPORT_PROTOCOL),
        _retryable_error(ModelFailureSubtype.EMPTY_COMPLETION),
        LlmTimeoutError(),
    ],
)
async def test_retryable_provider_failure_then_success_uses_two_boundaries(
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0
    sleeps: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "accepted"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    assert (
        await run_model_operation(
            operation,
            retries=2,
            stage=WorkflowStage.V1_EXPERIENCE,
            sleep=sleep,
        )
        == "accepted"
    )
    events = _diagnostics(capsys.readouterr().err)
    starts = [item for item in events if item["event"] == "attempt_start"]
    assert calls == len(starts) == 2
    assert [item["attempt"] for item in starts] == [1, 2]
    assert all(item["total_attempts"] == 3 for item in starts)
    decisions = [item for item in events if item["event"] == "retry_decision"]
    assert len(decisions) == 1
    assert decisions[0]["retry"] is True
    assert decisions[0]["failure_subtype"] == (
        ModelFailureSubtype.TIMEOUT.value
        if isinstance(error, LlmTimeoutError)
        else error.subtype.value
    )
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_retryable_failure_exhaustion_never_exceeds_configured_budget(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=503,
            retry_after_seconds=0,
        )

    async def sleep(_delay: float) -> None:
        return None

    with pytest.raises(RetryableModelError):
        await run_model_operation(
            operation,
            retries=2,
            stage=WorkflowStage.V1_JOD,
            sleep=sleep,
        )
    events = _diagnostics(capsys.readouterr().err)
    assert calls == 3
    assert len([item for item in events if item["event"] == "attempt_start"]) == 3
    assert [item["retry"] for item in events if item["event"] == "retry_decision"] == [
        True,
        True,
        False,
    ]
    failure = [item for item in events if item["event"] == "failure"]
    assert len(failure) == 1
    assert failure[0]["failure_subtype"] == "transient_http"
    assert failure[0]["category"] == "model"


@pytest.mark.asyncio
async def test_retry_delays_honor_bounded_retry_after_and_fallback_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    failures = [
        RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=429,
            retry_after_seconds=120,
        ),
        RetryableModelError(subtype=ModelFailureSubtype.TRANSPORT_READ),
        RetryableModelError(subtype=ModelFailureSubtype.TRANSPORT_READ),
    ]
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if failures:
            raise failures.pop(0)
        return "accepted"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(workflow_retry, "DEFAULT_WORKFLOW_BACKOFF_SECONDS", 40.0)
    assert (
        await run_model_operation(
            operation,
            retries=3,
            stage=WorkflowStage.V1_CORE,
            sleep=sleep,
        )
        == "accepted"
    )
    assert calls == 4
    assert sleeps == [120.0, 60.0, 60.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        NonRetryableModelError(
            subtype=ModelFailureSubtype.PERMANENT_HTTP,
            http_status=400,
        ),
        NonRetryableModelError(subtype=ModelFailureSubtype.MALFORMED_ENVELOPE),
        NonRetryableModelError(subtype=ModelFailureSubtype.RESPONSE_TOO_LARGE),
        NonRetryableModelError(subtype=ModelFailureSubtype.INVALID_GENERATION_JSON),
        NonRetryableModelError(subtype=ModelFailureSubtype.UNEXPECTED_MODEL),
        LlmError("synthetic legacy model failure"),
    ],
)
async def test_nonretryable_model_failure_stops_after_one_boundary(
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        await run_model_operation(
            operation,
            retries=3,
            stage=WorkflowStage.V2_CRITIQUE,
        )
    events = _diagnostics(capsys.readouterr().err)
    assert calls == 1
    decision = next(item for item in events if item["event"] == "retry_decision")
    assert decision["retry"] is False
    assert decision["category"] == "model"


@pytest.mark.asyncio
async def test_non_model_exception_escapes_without_model_relabel_or_retry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("synthetic application validation")

    with pytest.raises(ValueError):
        await run_model_operation(
            operation,
            retries=3,
            stage=WorkflowStage.V2_CRITIQUE,
        )
    events = _diagnostics(capsys.readouterr().err)
    assert calls == 1
    assert not any(item["event"] in {"failure", "retry_decision"} for item in events)
    assert all(item.get("category") != "model" for item in events)


@pytest.mark.asyncio
async def test_ollama_timeout_retries_but_other_ollama_failure_does_not(
    capsys: pytest.CaptureFixture[str],
) -> None:
    timeout_calls = 0

    async def timeout_then_success() -> str:
        nonlocal timeout_calls
        timeout_calls += 1
        if timeout_calls == 1:
            raise OllamaTimeoutError("unsafe timeout detail")
        return "accepted"

    async def sleep(_delay: float) -> None:
        return None

    assert (
        await run_model_operation(
            timeout_then_success,
            retries=2,
            stage=WorkflowStage.V1_CORE,
            sleep=sleep,
        )
        == "accepted"
    )
    timeout_events = _diagnostics(capsys.readouterr().err)
    assert timeout_calls == 2
    assert (
        next(item for item in timeout_events if item["event"] == "retry_decision")[
            "failure_subtype"
        ]
        == "timeout"
    )

    generic_calls = 0

    async def generic_failure() -> str:
        nonlocal generic_calls
        generic_calls += 1
        raise OllamaError("HTTP 503 /private/path raw response")

    with pytest.raises(OllamaError):
        await run_model_operation(
            generic_failure,
            retries=2,
            stage=WorkflowStage.V1_CORE,
            sleep=sleep,
        )
    rendered = capsys.readouterr().err
    events = _diagnostics(rendered)
    assert generic_calls == 1
    decision = next(item for item in events if item["event"] == "retry_decision")
    assert decision["retry"] is False
    assert decision["category"] == "model"
    assert decision["failure_subtype"] == "unexpected_model"
    assert "private" not in rendered
    assert "503" not in rendered


@pytest.mark.asyncio
async def test_retry_classification_does_not_parse_exception_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise LlmError(
            "HTTP 503 timeout empty completion connect /private/path secret-key"
        )

    with pytest.raises(LlmError):
        await run_model_operation(
            operation,
            retries=3,
            stage=WorkflowStage.V1_CORE,
        )
    rendered = capsys.readouterr().err
    events = _diagnostics(rendered)
    assert calls == 1
    assert (
        next(item for item in events if item["event"] == "failure")["failure_subtype"]
        == "unexpected_model"
    )
    for marker in ("503", "private", "secret", "empty completion", "connect"):
        assert marker not in rendered
