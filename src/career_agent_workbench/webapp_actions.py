"""App-scoped allowlisted workflow actions and ATS recalculation."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from career_agent_workbench.application_state import (
    ApplicationStateStore,
    AtsFields,
)
from career_agent_workbench.ats import AtsDiagnostics, calculate_ats_diagnostics
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.jod import usable_job_description
from career_agent_workbench.resume_manual_profiles import parse_manual_pass_profile

CommandExecutor = Callable[[Sequence[str]], int]

ACTION_TARGETS = {
    "regenerate-draft-resumes": "Regenerate draft resumes",
    "regenerate-resumes": "Regenerate and refine resumes",
    "refine-draft-resumes": "Refine draft resumes",
    "regenerate-aro-objects": "Regenerate application resume objects",
    "sync-draft-to-aro": "Sync drafts to application resume objects",
    "highlight-draft-resumes": "Highlight draft resumes",
    "manual-pass-resumes": "Run manual resume pass",
}
COMPOSITE_ACTIONS = {
    "v1-v2": "Run v1 then v2",
    "v1-v2-manual": "Run v1 then v2 then manual",
}
ACTION_OPTIONS = {**ACTION_TARGETS, **COMPOSITE_ACTIONS}
ATS_ACTION = "recalculate-ats"

_PATH_ASSIGNMENTS = (
    ("root", "WORKSPACE"),
    ("database", "DATABASE"),
    ("output_dir", "OUTPUT_DIR"),
    ("profile_dir", "PROFILE_DIR"),
    ("master_resume", "MASTER_RESUME"),
    ("master_resume_text", "MASTER_RESUME_TEXT"),
    ("blacklist", "BLACKLIST"),
    ("tmp_dir", "TMP_DIR"),
)
_COMPOSITE_TARGETS = {
    "v1-v2": ("regenerate-draft-resumes", "refine-draft-resumes"),
    "v1-v2-manual": (
        "regenerate-draft-resumes",
        "refine-draft-resumes",
        "manual-pass-resumes",
    ),
}
_HIGHLIGHT_VARIANTS = {
    "regenerate-draft-resumes": "v1",
    "regenerate-resumes": "v2",
    "manual-pass-resumes": "manual",
    "v1-v2": "v2",
    "v1-v2-manual": "manual",
}
_FIRST_DRAFT_TARGETS = {"regenerate-draft-resumes", "regenerate-resumes"}
_MAX_ACTIONS = 32
_MAX_MESSAGES = 24


@dataclass(frozen=True, slots=True)
class CommandStage:
    """One allowlisted Make stage with already-bound arguments."""

    label: str
    argv: tuple[str, ...] = field(repr=False)


@dataclass(slots=True)
class ActionRecord:
    """Bounded app-session action progress."""

    action_id: str
    target_label: str
    status: str
    queued_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    current_stage: str | None = None
    completed_stages: int = 0
    total_stages: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    message: str = "Action queued."
    messages: list[str] = field(default_factory=lambda: ["Action queued."])


@dataclass(slots=True)
class ActionRegistry:
    """One Flask application's isolated bounded action history."""

    actions: dict[str, ActionRecord] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class AtsRecalculationResult:
    """Count-only ATS recalculation result."""

    updated: int
    skipped: int


def action_snapshots(registry: ActionRegistry) -> list[dict[str, object]]:
    """Return newest-first content-free action snapshots."""

    with registry.lock:
        records = tuple(reversed(tuple(registry.actions.values())))
        return [
            {
                "id": record.action_id,
                "target": record.target_label,
                "status": record.status,
                "queued_at": record.queued_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "return_code": record.return_code,
                "current_stage": record.current_stage,
                "completed_stages": record.completed_stages,
                "total_stages": record.total_stages,
                "updated_count": record.updated_count,
                "skipped_count": record.skipped_count,
                "message": record.message,
                "messages": list(record.messages[-8:]),
            }
            for record in records
        ]


def create_action(
    registry: ActionRegistry,
    *,
    label: str,
    total_stages: int,
) -> ActionRecord:
    """Create one queued action and trim only this app's history."""

    if not label or not 1 <= total_stages <= 8:
        raise ValueError("Action request is invalid.")
    record = ActionRecord(
        action_id=uuid.uuid4().hex,
        target_label=label,
        status="queued",
        queued_at=_timestamp(),
        total_stages=total_stages,
    )
    with registry.lock:
        registry.actions[record.action_id] = record
        while len(registry.actions) > _MAX_ACTIONS:
            registry.actions.pop(next(iter(registry.actions)))
    return record


def build_action_stages(
    *,
    project_root: Path,
    workflow: str,
    job_ids: tuple[str, ...],
    paths: WorkspacePaths,
    manual_profile: str,
    highlight: bool,
) -> tuple[CommandStage, ...]:
    """Build only the seven existing Make targets and two bounded composites."""

    if workflow in ACTION_TARGETS:
        targets = (workflow,)
    else:
        targets = _COMPOSITE_TARGETS.get(workflow)
    if targets is None or not job_ids:
        raise ValueError("Action request is invalid.")
    if highlight and workflow not in _HIGHLIGHT_VARIANTS:
        raise ValueError("Action request is invalid.")
    parsed_profile = (
        parse_manual_pass_profile(manual_profile).key.value
        if "manual-pass-resumes" in targets
        else "regular"
    )
    stages = [
        CommandStage(
            label=ACTION_TARGETS[target],
            argv=_workflow_argv(
                project_root=project_root,
                target=target,
                job_ids=job_ids,
                paths=paths,
                manual_profile=parsed_profile,
            ),
        )
        for target in targets
    ]
    if highlight:
        stages.append(
            CommandStage(
                label=ACTION_TARGETS["highlight-draft-resumes"],
                argv=_workflow_argv(
                    project_root=project_root,
                    target="highlight-draft-resumes",
                    job_ids=job_ids,
                    paths=paths,
                    manual_profile="regular",
                    highlight_variant=_HIGHLIGHT_VARIANTS[workflow],
                ),
            )
        )
    return tuple(stages)


def build_seed_argv(
    *,
    project_root: Path,
    location: str,
    date_posted: str,
    limit_per_query: int,
    max_queries: int,
    max_jobs: int,
    paths: WorkspacePaths,
) -> tuple[str, ...]:
    """Build the one existing matching Make target from validated fields."""

    argv = [
        "make",
        "-C",
        str(project_root),
        "seed-jobs",
        f"LOCATION={location}",
        f"DATE_POSTED={date_posted}",
        f"LIMIT_PER_QUERY={limit_per_query}",
        f"MAX_QUERIES={max_queries}",
        f"MAX_JOBS={max_jobs}",
    ]
    return (*argv, *_workspace_assignments(paths))


def run_command_action(
    registry: ActionRegistry,
    action_id: str,
    executor: CommandExecutor,
    stages: tuple[CommandStage, ...],
) -> None:
    """Run stages sequentially and stop on the first content-free failure."""

    _start_action(registry, action_id)
    for index, stage in enumerate(stages, start=1):
        _stage_started(registry, action_id, stage.label, index=index)
        try:
            return_code = executor(stage.argv)
            if type(return_code) is not int:
                return_code = 1
        except Exception:  # noqa: BLE001 - failures stay content-free.
            return_code = 1
        if return_code != 0:
            _finish_action(
                registry,
                action_id,
                status="failed",
                return_code=return_code,
                message=f"Stage {index} of {len(stages)} failed.",
            )
            return
        _stage_completed(registry, action_id, stage.label, index=index)
    _finish_action(
        registry,
        action_id,
        status="completed",
        return_code=0,
        message="Action completed.",
    )


def run_ats_action(
    registry: ActionRegistry,
    action_id: str,
    store: ApplicationStateStore,
    job_ids: tuple[str, ...],
) -> None:
    """Recalculate selected/default ATS projections with count-only progress."""

    _start_action(registry, action_id)
    _stage_started(registry, action_id, "Recalculate ATS", index=1)
    try:
        result = recalculate_selected_ats(store=store, job_ids=job_ids)
    except Exception:  # noqa: BLE001 - state/calculator failures stay content-free.
        _finish_action(
            registry,
            action_id,
            status="failed",
            return_code=1,
            message="ATS recalculation failed.",
        )
        return
    with registry.lock:
        record = registry.actions[action_id]
        record.updated_count = result.updated
        record.skipped_count = result.skipped
    _stage_completed(registry, action_id, "Recalculate ATS", index=1)
    _finish_action(
        registry,
        action_id,
        status="completed",
        return_code=0,
        message=(
            f"ATS recalculation completed: {result.updated} updated, "
            f"{result.skipped} skipped."
        ),
    )


def recalculate_selected_ats(
    *,
    store: ApplicationStateStore,
    job_ids: tuple[str, ...],
) -> AtsRecalculationResult:
    """Update only ATS fields for usable selected/default projections."""

    records = store.fetch_job_records(job_ids)
    if tuple(record.job_id for record in records) != job_ids:
        raise ValueError("ATS request is invalid.")
    updated = 0
    skipped = 0
    for record in records:
        prompt = usable_job_description(record.prompt_job_description)
        source = usable_job_description(record.job_description)
        job_description = prompt or source
        if job_description is None or record.resume_pdf is None:
            skipped += 1
            continue
        diagnostics = calculate_ats_diagnostics(
            resume_pdf=record.resume_pdf,
            job_description=job_description,
        )
        store.store_ats(record.job_id, _ats_fields(diagnostics))
        updated += 1
    return AtsRecalculationResult(updated=updated, skipped=skipped)


def _workflow_argv(
    *,
    project_root: Path,
    target: str,
    job_ids: tuple[str, ...],
    paths: WorkspacePaths,
    manual_profile: str,
    highlight_variant: str | None = None,
) -> tuple[str, ...]:
    if target not in ACTION_TARGETS:
        raise ValueError("Action request is invalid.")
    argv = [
        "make",
        "-C",
        str(project_root),
        target,
        f"JOB_IDS={' '.join(job_ids)}",
    ]
    if target in _FIRST_DRAFT_TARGETS:
        argv.append("FIRST_DRAFT_FORCE=1")
    if target == "manual-pass-resumes":
        argv.append(f"MANUAL_PASS_PROFILE={manual_profile}")
    if target == "highlight-draft-resumes" and highlight_variant is not None:
        argv.append(f"HIGHLIGHT_RESUME_VARIANT={highlight_variant}")
    return (*argv, *_workspace_assignments(paths))


def _workspace_assignments(paths: WorkspacePaths) -> tuple[str, ...]:
    assignments: list[str] = []
    for member, assignment in _PATH_ASSIGNMENTS:
        value = getattr(paths, member)
        if value is not None:
            assignments.append(f"{assignment}={value}")
    return tuple(assignments)


def _ats_fields(diagnostics: AtsDiagnostics) -> AtsFields:
    score = diagnostics.score
    return AtsFields(
        score=score.overall_score,
        parsing_score=score.parsing_score,
        keyword_score=score.keyword_match_score,
        semantic_score=score.semantic_match_score,
        formatting_risk=score.formatting_risk,
        missing_terms=", ".join(score.missing_high_value_terms),
        diagnostics={
            "component_scores": {
                "overall_score": diagnostics.component_scores.overall_score,
                "parsing_score": diagnostics.component_scores.parsing_score,
                "keyword_match_score": diagnostics.component_scores.keyword_match_score,
                "semantic_match_score": diagnostics.component_scores.semantic_match_score,
                "formatting_score": diagnostics.component_scores.formatting_score,
                "formatting_risk": diagnostics.component_scores.formatting_risk,
            },
            "matched_terms": [
                {"term": item.term, "weight": item.weight}
                for item in diagnostics.matched_terms
            ],
            "unmatched_weighted_terms": [
                {"term": item.term, "weight": item.weight}
                for item in diagnostics.unmatched_weighted_terms
            ],
        },
        updated_at=_timestamp(),
    )


def _start_action(registry: ActionRegistry, action_id: str) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.status = "running"
        record.started_at = _timestamp()
        _message(record, "Action running.")


def _stage_started(
    registry: ActionRegistry,
    action_id: str,
    label: str,
    *,
    index: int,
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.current_stage = label
        _message(record, f"Stage {index} of {record.total_stages} running.")


def _stage_completed(
    registry: ActionRegistry,
    action_id: str,
    label: str,
    *,
    index: int,
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.current_stage = label
        record.completed_stages = index
        _message(record, f"Stage {index} of {record.total_stages} completed.")


def _finish_action(
    registry: ActionRegistry,
    action_id: str,
    *,
    status: str,
    return_code: int,
    message: str,
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.status = status
        record.return_code = return_code
        record.finished_at = _timestamp()
        _message(record, message)


def _message(record: ActionRecord, message: str) -> None:
    record.message = message
    record.messages.append(message)
    if len(record.messages) > _MAX_MESSAGES:
        del record.messages[: len(record.messages) - _MAX_MESSAGES]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


__all__ = [
    "ACTION_OPTIONS",
    "ACTION_TARGETS",
    "ATS_ACTION",
    "ActionRegistry",
    "AtsRecalculationResult",
    "CommandExecutor",
    "CommandStage",
    "action_snapshots",
    "build_action_stages",
    "build_seed_argv",
    "create_action",
    "recalculate_selected_ats",
    "run_ats_action",
    "run_command_action",
]
