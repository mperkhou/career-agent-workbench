"""Single-owner retry policy for bounded provider/model operations."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import TypeVar

from career_agent_workbench.errors import (
    LlmError,
    ModelFailureSubtype,
    ModelTimeoutError,
    OllamaError,
    RetryableModelError,
    model_failure_subtype,
)
from career_agent_workbench.workflow_diagnostics import (
    DiagnosticEvent,
    FailureCategory,
    WorkflowStage,
    attempt_event,
    emit_diagnostic,
)

T = TypeVar("T")
MAX_WORKFLOW_BACKOFF_SECONDS = 60.0
DEFAULT_WORKFLOW_BACKOFF_SECONDS = 1.0


async def run_model_operation(
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    stage: WorkflowStage,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run one logical operation with exactly one visible retry owner."""

    retry_count = max(0, min(retries, 3))
    total_attempts = retry_count + 1
    for attempt in range(1, total_attempts + 1):
        emit_diagnostic(
            attempt_event(
                event=DiagnosticEvent.ATTEMPT_START,
                stage=stage,
                attempt=attempt,
                total_attempts=total_attempts,
            )
        )
        started = monotonic()
        try:
            result = await operation()
        except Exception as error:
            elapsed = monotonic() - started
            if not isinstance(error, (LlmError, OllamaError, ModelTimeoutError)):
                emit_diagnostic(
                    attempt_event(
                        event=DiagnosticEvent.ATTEMPT_ELAPSED,
                        stage=stage,
                        attempt=attempt,
                        total_attempts=total_attempts,
                        elapsed_seconds=elapsed,
                    )
                )
                raise
            subtype = model_failure_subtype(error)
            category = (
                FailureCategory.TIMEOUT
                if subtype is ModelFailureSubtype.TIMEOUT
                else FailureCategory.MODEL
            )
            emit_diagnostic(
                attempt_event(
                    event=DiagnosticEvent.ATTEMPT_ELAPSED,
                    stage=stage,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    elapsed_seconds=elapsed,
                )
            )
            if subtype is ModelFailureSubtype.TIMEOUT:
                emit_diagnostic(
                    attempt_event(
                        event=DiagnosticEvent.TIMEOUT,
                        stage=stage,
                        attempt=attempt,
                        total_attempts=total_attempts,
                        category=category,
                        failure_subtype=subtype,
                    )
                )
            retry = isinstance(error, (RetryableModelError, ModelTimeoutError)) and (
                attempt < total_attempts
            )
            emit_diagnostic(
                attempt_event(
                    event=DiagnosticEvent.RETRY_DECISION,
                    stage=stage,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    retry=retry,
                    category=category,
                    failure_subtype=subtype,
                )
            )
            if retry:
                await sleep(_workflow_retry_delay(error, attempt))
                continue
            emit_diagnostic(
                attempt_event(
                    event=DiagnosticEvent.FAILURE,
                    stage=stage,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    category=category,
                    failure_subtype=subtype,
                )
            )
            raise
        elapsed = monotonic() - started
        emit_diagnostic(
            attempt_event(
                event=DiagnosticEvent.ATTEMPT_ELAPSED,
                stage=stage,
                attempt=attempt,
                total_attempts=total_attempts,
                elapsed_seconds=elapsed,
            )
        )
        emit_diagnostic(
            attempt_event(
                event=DiagnosticEvent.ATTEMPT_COMPLETION,
                stage=stage,
                attempt=attempt,
                total_attempts=total_attempts,
            )
        )
        return result
    raise AssertionError("unreachable retry loop")


def _workflow_retry_delay(error: BaseException, attempt: int) -> float:
    if isinstance(error, RetryableModelError) and error.retry_after_seconds is not None:
        return error.retry_after_seconds
    exponent = max(0, int(attempt) - 1)
    delay = DEFAULT_WORKFLOW_BACKOFF_SECONDS * (2**exponent)
    if not math.isfinite(delay):
        return MAX_WORKFLOW_BACKOFF_SECONDS
    return min(delay, MAX_WORKFLOW_BACKOFF_SECONDS)


__all__ = [
    "DEFAULT_WORKFLOW_BACKOFF_SECONDS",
    "MAX_WORKFLOW_BACKOFF_SECONDS",
    "run_model_operation",
]
