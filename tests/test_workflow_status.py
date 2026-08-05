from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from career_agent_workbench.errors import (
    ModelFailureSubtype,
    ModelResponseContentState,
    ModelResponseErrorPresence,
    ModelResponseErrorType,
    ModelResponseFinishReason,
    ModelResponseSummary,
)
from career_agent_workbench.workflow_diagnostics import (
    ConfigurationSource,
    DiagnosticEvent,
    FailureCategory,
    WorkflowStage,
    attempt_event,
    configuration_event,
)
from career_agent_workbench.workflow_status import (
    MAX_STATUS_RUNS,
    WorkflowStatusError,
    WorkflowStatusStore,
    captured_diagnostic_event,
    local_status_event,
    normalized_status_event,
    status_event_message,
)


def _tmp_member(tmp_path: Path) -> Path:
    temporary = tmp_path / "workspace-tmp"
    temporary.mkdir(mode=0o700)
    return temporary.absolute()


def _run(run_id: str, *, started_at: str) -> dict[str, object]:
    event = local_status_event(
        "queued",
        timestamp="2042-04-10T10:00:00+00:00",
    )
    diagnostic = configuration_event(
        stage=WorkflowStage.V1_CORE,
        model="synthetic/model",
        effort="",
        timeout_seconds=300,
        retry_count=1,
        sources={
            "model": ConfigurationSource.DEFAULT,
            "effort": ConfigurationSource.DEFAULT,
            "timeout": ConfigurationSource.DEFAULT,
            "retry_count": ConfigurationSource.DEFAULT,
        },
        workspace_configured=True,
    )
    return {
        "id": run_id,
        "title": "Synthetic background action",
        "status": "completed",
        "started_at": started_at,
        "finished_at": "2042-04-10T10:01:00+00:00",
        "return_code": 0,
        "events": [event, diagnostic],
    }


def test_status_store_modes_round_trip_metadata_and_retention(tmp_path: Path) -> None:
    temporary = _tmp_member(tmp_path)
    store = WorkflowStatusStore(temporary)
    for index in range(MAX_STATUS_RUNS + 3):
        store.save(
            _run(
                f"{index:032x}",
                started_at=f"2042-04-{index + 1:02d}T10:00:00+00:00",
            )
        )

    directory = temporary / "workflow-status"
    files = tuple(directory.glob("*.json"))
    retained = store.load()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert len(files) == len(retained) == MAX_STATUS_RUNS
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)
    assert retained[0]["id"] == f"{MAX_STATUS_RUNS + 2:032x}"
    metadata = store.metadata(str(retained[0]["id"]))
    assert metadata == {"log_available": True, "log_event_count": 2}
    assert "workspace-tmp" not in json.dumps(retained)


def test_status_store_rejects_directory_and_target_replacement(tmp_path: Path) -> None:
    temporary = _tmp_member(tmp_path)
    store = WorkflowStatusStore(temporary)
    payload = _run("a" * 32, started_at="2042-04-10T10:00:00+00:00")
    store.save(payload)
    directory = temporary / "workflow-status"
    displaced = temporary / "workflow-status-displaced"
    directory.rename(displaced)
    replacement = temporary / "replacement"
    replacement.mkdir()
    directory.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(WorkflowStatusError):
        store.save(payload)
    assert tuple(replacement.iterdir()) == ()

    directory.unlink()
    displaced.rename(directory)
    fresh = WorkflowStatusStore(temporary)
    target = directory / f"{'b' * 32}.json"
    external = tmp_path / "external.json"
    external.write_text("preserve", encoding="utf-8")
    target.symlink_to(external)
    with pytest.raises(WorkflowStatusError):
        fresh.save(_run("b" * 32, started_at="2042-04-11T10:00:00+00:00"))
    assert external.read_text(encoding="utf-8") == "preserve"


def test_status_event_validation_and_messages_are_content_free() -> None:
    unsafe = {
        "event": "failure",
        "stage": "v1_core",
        "attempt": 1,
        "total_attempts": 2,
        "category": "model",
        "path": "/private/operator/path",
    }
    assert normalized_status_event(unsafe) is None
    event = local_status_event(
        "stage_failure",
        timestamp="2042-04-10T10:00:00+00:00",
        stage="v1",
        category="workflow",
    )
    message = status_event_message(event)
    assert "v1 stage failed" in message
    assert "private" not in message
    assert "/" not in message


def test_child_diagnostic_receives_and_preserves_capture_timestamp() -> None:
    diagnostic = configuration_event(
        stage=WorkflowStage.V1_CORE,
        model="synthetic/model",
        effort="",
        timeout_seconds=300,
        retry_count=1,
        sources={
            "model": ConfigurationSource.DEFAULT,
            "effort": ConfigurationSource.DEFAULT,
            "timeout": ConfigurationSource.DEFAULT,
            "retry_count": ConfigurationSource.DEFAULT,
        },
        workspace_configured=True,
    )
    captured = captured_diagnostic_event(
        diagnostic,
        timestamp="2042-04-10T10:00:01+00:00",
    )

    assert captured is not None
    assert captured["timestamp"] == "2042-04-10T10:00:01+00:00"
    assert normalized_status_event(captured) == captured
    assert status_event_message(captured).startswith("10:00:01 v1 core configured")
    assert (
        captured_diagnostic_event(
            {**diagnostic, "path": "/private/operator/path"},
            timestamp="2042-04-10T10:00:01+00:00",
        )
        is None
    )


def test_status_store_accepts_legacy_model_and_optional_subtype(
    tmp_path: Path,
) -> None:
    legacy = {
        "event": "failure",
        "stage": "v1_core",
        "attempt": 1,
        "total_attempts": 2,
        "category": "model",
    }
    typed = attempt_event(
        event=DiagnosticEvent.FAILURE,
        stage=WorkflowStage.V1_JOD,
        attempt=1,
        total_attempts=2,
        category=FailureCategory.MODEL,
        failure_subtype=ModelFailureSubtype.TRANSIENT_HTTP,
    )
    response_summary = ModelResponseSummary(
        http_status=200,
        error_presence=ModelResponseErrorPresence.CHOICE,
        error_code=503,
        error_type=ModelResponseErrorType.PROVIDER_UNAVAILABLE,
        finish_reason=ModelResponseFinishReason.ERROR,
        choices_count=1,
        content_state=ModelResponseContentState.PRESENT,
    )
    with pytest.raises(ValueError, match="diagnostic event"):
        attempt_event(
            event=DiagnosticEvent.ATTEMPT_START,
            stage=WorkflowStage.V1_JOD,
            attempt=1,
            total_attempts=2,
            category=FailureCategory.MODEL,
            failure_subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
            response_summary=response_summary,
        )
    detailed = attempt_event(
        event=DiagnosticEvent.RETRY_DECISION,
        stage=WorkflowStage.V1_JOD,
        attempt=1,
        total_attempts=2,
        retry=True,
        category=FailureCategory.MODEL,
        failure_subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
        response_summary=response_summary,
    )
    legacy_capture = captured_diagnostic_event(
        legacy,
        timestamp="2042-04-10T10:00:01+00:00",
    )
    typed_capture = captured_diagnostic_event(
        typed,
        timestamp="2042-04-10T10:00:02+00:00",
    )
    detailed_capture = captured_diagnostic_event(
        detailed,
        timestamp="2042-04-10T10:00:03+00:00",
    )
    assert legacy_capture is not None
    assert typed_capture is not None
    assert "failure_subtype" not in legacy_capture
    assert typed_capture["failure_subtype"] == "transient_http"
    assert detailed_capture is not None
    assert detailed_capture["response_summary"] == response_summary.as_dict()
    assert status_event_message(legacy_capture) == "10:00:01 v1 core failed (model)."
    assert (
        status_event_message(typed_capture)
        == "10:00:02 v1 jod failed (model; transient http)."
    )
    assert (
        captured_diagnostic_event(
            {**typed, "failure_subtype": "unknown"},
            timestamp="2042-04-10T10:00:03+00:00",
        )
        is None
    )
    assert (
        captured_diagnostic_event(
            {**detailed, "response_summary": {"raw_response": "unsafe"}},
            timestamp="2042-04-10T10:00:03+00:00",
        )
        is None
    )
    assert (
        captured_diagnostic_event(
            {key: value for key, value in detailed.items() if key != "failure_subtype"},
            timestamp="2042-04-10T10:00:03+00:00",
        )
        is None
    )
    assert (
        captured_diagnostic_event(
            {**typed, "raw_response": "unsafe"},
            timestamp="2042-04-10T10:00:03+00:00",
        )
        is None
    )

    payload = _run("f" * 32, started_at="2042-04-10T10:00:00+00:00")
    payload["events"] = [legacy_capture, typed_capture, detailed_capture]
    store = WorkflowStatusStore(_tmp_member(tmp_path))
    store.save(payload)
    recovered = store.load()[0]["events"]
    assert "failure_subtype" not in recovered[0]
    assert recovered[1]["failure_subtype"] == "transient_http"
    assert recovered[2]["response_summary"] == response_summary.as_dict()
    rendered = json.dumps(recovered, sort_keys=True)
    for marker in ("unsafe", "raw_response", "/private", "provider-message"):
        assert marker not in rendered


def test_status_attempt_messages_render_optional_validated_subtype() -> None:
    legacy_retry = attempt_event(
        event=DiagnosticEvent.RETRY_DECISION,
        stage=WorkflowStage.V1_CORE,
        attempt=1,
        total_attempts=2,
        retry=True,
        category=FailureCategory.MODEL,
    )
    typed_retry = {
        **legacy_retry,
        "failure_subtype": ModelFailureSubtype.TRANSIENT_HTTP.value,
    }
    typed_timeout = attempt_event(
        event=DiagnosticEvent.TIMEOUT,
        stage=WorkflowStage.V1_CORE,
        attempt=1,
        total_attempts=2,
        category=FailureCategory.TIMEOUT,
        failure_subtype=ModelFailureSubtype.TIMEOUT,
    )

    assert status_event_message(legacy_retry) == (
        "--:--:-- v1 core attempt 1/2: retrying."
    )
    assert status_event_message(typed_retry) == (
        "--:--:-- v1 core attempt 1/2: retrying (transient http)."
    )
    assert status_event_message(typed_timeout) == (
        "--:--:-- v1 core attempt 1/2 timed out (timeout)."
    )
    assert status_event_message({**legacy_retry, "failure_subtype": "unknown"}) == (
        "--:--:-- v1 core attempt 1/2: retrying."
    )


def test_status_store_does_not_follow_existing_hardlink(tmp_path: Path) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks unavailable")
    temporary = _tmp_member(tmp_path)
    store = WorkflowStatusStore(temporary)
    directory = temporary / "workflow-status"
    external = tmp_path / "external.json"
    external.write_text("preserve", encoding="utf-8")
    target = directory / f"{'c' * 32}.json"
    os.link(external, target)

    with pytest.raises(WorkflowStatusError):
        store.save(_run("c" * 32, started_at="2042-04-12T10:00:00+00:00"))
    assert external.read_text(encoding="utf-8") == "preserve"


def test_status_store_closes_filesystem_failure_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WorkflowStatusStore(_tmp_member(tmp_path))

    def unsafe_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("/private/operator/path secret-marker")

    monkeypatch.setattr(os, "replace", unsafe_replace)
    with pytest.raises(WorkflowStatusError) as captured:
        store.save(_run("d" * 32, started_at="2042-04-13T10:00:00+00:00"))

    rendered = str(captured.value)
    assert rendered == "Workflow status could not be persisted."
    assert "secret-marker" not in rendered
    assert "/private" not in rendered


def test_status_store_cleans_only_owned_orphan_temporary_files(
    tmp_path: Path,
) -> None:
    temporary = _tmp_member(tmp_path)
    WorkflowStatusStore(temporary)
    directory = temporary / "workflow-status"
    orphan = directory / f".{'e' * 32}.{'f' * 16}.tmp"
    unrelated = directory / ".unrelated.tmp"
    external = tmp_path / "external.tmp"
    linked = directory / f".{'a' * 32}.{'b' * 16}.tmp"
    orphan.write_text("orphan", encoding="utf-8")
    unrelated.write_text("preserve", encoding="utf-8")
    external.write_text("preserve", encoding="utf-8")
    linked.symlink_to(external)

    WorkflowStatusStore(temporary)

    assert not orphan.exists()
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert linked.is_symlink()
    assert external.read_text(encoding="utf-8") == "preserve"
