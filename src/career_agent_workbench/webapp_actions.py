"""App-scoped allowlisted workflow actions and ATS recalculation."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
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

_MAX_CAPTURED_OUTPUT_CHARS = 1_000_000


@dataclass(frozen=True, slots=True)
class CommandExecution:
    """One bounded command outcome whose output is never exposed in status."""

    return_code: int
    output: str = field(default="", repr=False)


CommandExecutor = Callable[[Sequence[str]], int | CommandExecution]

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
_MAX_ACTION_JOBS = 50
_TERMINAL_STATUSES = {"completed", "partial", "failed"}


@dataclass(frozen=True, slots=True)
class CommandStage:
    """One allowlisted Make stage with already-bound arguments."""

    label: str
    argv: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class WorkflowComposition:
    """Validated ingestion-time downstream workflow selection."""

    run_v1: bool
    run_v2: bool
    run_manual: bool
    run_highlight: bool
    manual_profile: str

    @property
    def selected(self) -> bool:
        return self.run_v1 or self.run_v2 or self.run_manual or self.run_highlight


@dataclass(frozen=True, slots=True)
class ActionFailure:
    """Retryable content-free failure coordinate."""

    job_id: str
    stage_index: int
    stage_label: str


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
    current_job_id: str | None = None
    completed_jobs: int = 0
    total_jobs: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    retry_of: str | None = None
    message: str = "Action queued."
    messages: list[str] = field(default_factory=lambda: ["Action queued."])
    job_ids: tuple[str, ...] = field(default=(), repr=False)
    stages: tuple[CommandStage, ...] = field(default=(), repr=False)
    retry_starts: dict[str, int] = field(default_factory=dict, repr=False)
    failures: list[ActionFailure] = field(default_factory=list, repr=False)


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
        newest = tuple(reversed(tuple(registry.actions.values())))
        records = tuple(
            record for record in newest if record.status not in _TERMINAL_STATUSES
        ) + tuple(record for record in newest if record.status in _TERMINAL_STATUSES)
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
                "current_job_id": record.current_job_id,
                "completed_jobs": record.completed_jobs,
                "total_jobs": record.total_jobs,
                "successful_steps": record.successful_steps,
                "failed_steps": record.failed_steps,
                "skipped_steps": record.skipped_steps,
                "progress_current": (
                    record.successful_steps + record.failed_steps + record.skipped_steps
                ),
                "progress_total": _progress_total(record),
                "retry_of": record.retry_of,
                "retryable": bool(
                    record.failures
                    and record.stages
                    and record.status in _TERMINAL_STATUSES
                ),
                "failed_work": [
                    {
                        "job_id": failure.job_id,
                        "stage": failure.stage_label,
                        "stage_index": failure.stage_index,
                    }
                    for failure in record.failures[:_MAX_ACTION_JOBS]
                ],
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
    job_ids: tuple[str, ...] = (),
    stages: tuple[CommandStage, ...] = (),
    retry_of: str | None = None,
    retry_starts: Mapping[str, int] | None = None,
) -> ActionRecord:
    """Create one queued action and trim only this app's history."""

    if (
        not label
        or not 1 <= total_stages <= 8
        or len(job_ids) > _MAX_ACTION_JOBS
        or len(set(job_ids)) != len(job_ids)
        or (stages and len(stages) != total_stages)
    ):
        raise ValueError("Action request is invalid.")
    starts = dict(retry_starts or {})
    if any(job_id not in job_ids for job_id in starts) or any(
        type(index) is not int or not 0 <= index < total_stages
        for index in starts.values()
    ):
        raise ValueError("Action request is invalid.")
    record = ActionRecord(
        action_id=uuid.uuid4().hex,
        target_label=label,
        status="queued",
        queued_at=_timestamp(),
        total_stages=total_stages,
        total_jobs=len(job_ids),
        retry_of=retry_of,
        job_ids=job_ids,
        stages=stages,
        retry_starts=starts,
    )
    with registry.lock:
        registry.actions[record.action_id] = record
        while len(registry.actions) > _MAX_ACTIONS:
            registry.actions.pop(next(iter(registry.actions)))
    return record


def parse_workflow_composition(
    *,
    run_v1: bool,
    run_v2: bool,
    run_manual: bool,
    run_highlight: bool,
    manual_profile: str,
) -> WorkflowComposition:
    """Validate dependent ingestion options before any execution boundary."""

    if run_v2 and not run_v1:
        raise ValueError("Workflow composition is invalid.")
    if run_manual and not (run_v1 and run_v2):
        raise ValueError("Workflow composition is invalid.")
    if run_highlight and not run_v1:
        raise ValueError("Workflow composition is invalid.")
    selected_profile = (
        parse_manual_pass_profile(manual_profile).key.value if run_manual else "regular"
    )
    return WorkflowComposition(
        run_v1=run_v1,
        run_v2=run_v2,
        run_manual=run_manual,
        run_highlight=run_highlight,
        manual_profile=selected_profile,
    )


def build_ingestion_stages(
    *,
    project_root: Path,
    job_ids: tuple[str, ...],
    paths: WorkspacePaths,
    composition: WorkflowComposition,
) -> tuple[CommandStage, ...]:
    """Build only the explicitly selected dependent downstream stages."""

    if not composition.selected or not job_ids:
        return ()
    targets: list[tuple[str, str | None]] = []
    if composition.run_v1:
        targets.append(("regenerate-draft-resumes", None))
    if composition.run_v2:
        targets.append(("refine-draft-resumes", None))
    if composition.run_manual:
        targets.append(("manual-pass-resumes", None))
    if composition.run_highlight:
        variant = (
            "manual"
            if composition.run_manual
            else ("v2" if composition.run_v2 else "v1")
        )
        targets.append(("highlight-draft-resumes", variant))
    return tuple(
        CommandStage(
            label=ACTION_TARGETS[target],
            argv=_workflow_argv(
                project_root=project_root,
                target=target,
                job_ids=job_ids,
                paths=paths,
                manual_profile=composition.manual_profile,
                highlight_variant=highlight_variant,
            ),
        )
        for target, highlight_variant in targets
    )


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
    """Run every job independently so one failure cannot stop survivors."""

    _start_action(registry, action_id)
    _run_job_stages(registry, action_id, executor, stages)


def run_seed_action(
    registry: ActionRegistry,
    action_id: str,
    executor: CommandExecutor,
    seed_stage: CommandStage,
    store: ApplicationStateStore,
    existing_job_ids: tuple[str, ...],
    project_root: Path,
    paths: WorkspacePaths,
    composition: WorkflowComposition,
) -> None:
    """Run seed, identify only newly created rows, then run selected stages."""

    _start_action(registry, action_id)
    _stage_started(registry, action_id, seed_stage.label, index=1, job_id=None)
    result = _execute(executor, seed_stage.argv)
    if result.return_code != 0:
        _finish_action(
            registry,
            action_id,
            status="failed",
            return_code=result.return_code,
            message="Seed stage failed.",
        )
        return
    seeded_count = _seeded_count(result.output)
    try:
        after = store.list_applications("all")
    except Exception:  # noqa: BLE001 - store failures stay content-free.
        after = ()
    existing = set(existing_job_ids)
    job_ids = tuple(record.job_id for record in after if record.job_id not in existing)
    if seeded_count is None or seeded_count != len(job_ids):
        if not composition.selected and seeded_count is None:
            seeded_count = len(job_ids)
        else:
            _finish_action(
                registry,
                action_id,
                status="failed",
                return_code=1,
                message="Seed result could not be verified.",
            )
            return
    with registry.lock:
        record = registry.actions[action_id]
        record.successful_steps = 1
        record.completed_stages = 1
        record.completed_jobs = 0
        record.total_jobs = len(job_ids)
        record.job_ids = job_ids
        _message(record, "Seed stage completed.")
    if not composition.selected or not job_ids:
        _finish_action(
            registry,
            action_id,
            status="completed",
            return_code=0,
            message="Seed action completed.",
        )
        return
    try:
        stages = build_ingestion_stages(
            project_root=project_root,
            job_ids=job_ids,
            paths=paths,
            composition=composition,
        )
    except Exception:  # noqa: BLE001 - policy failures stay content-free.
        _finish_action(
            registry,
            action_id,
            status="failed",
            return_code=1,
            message="Seed workflow could not be prepared.",
        )
        return
    with registry.lock:
        record = registry.actions[action_id]
        record.stages = stages
        record.total_stages = len(stages)
        record.successful_steps = 0
        record.completed_stages = 0
    _run_job_stages(registry, action_id, executor, stages)


def create_retry_action(
    registry: ActionRegistry,
    action_id: str,
    *,
    repeat_completed: bool,
) -> ActionRecord:
    """Create a bounded retry that starts at each failed stage by default."""

    with registry.lock:
        original = registry.actions.get(action_id)
        if (
            original is None
            or original.status not in _TERMINAL_STATUSES
            or not original.stages
            or (not repeat_completed and not original.failures)
        ):
            raise ValueError("Retry request is invalid.")
        stages = original.stages
        if repeat_completed:
            job_ids = original.job_ids
            starts: dict[str, int] = {}
        else:
            failures = {failure.job_id: failure for failure in original.failures}
            job_ids = tuple(job_id for job_id in original.job_ids if job_id in failures)
            starts = {job_id: failures[job_id].stage_index - 1 for job_id in job_ids}
        label = (
            f"Repeat {original.target_label}"
            if repeat_completed
            else f"Retry failed {original.target_label}"
        )
    return create_action(
        registry,
        label=label,
        total_stages=len(stages),
        job_ids=job_ids,
        stages=stages,
        retry_of=action_id,
        retry_starts=starts,
    )


def dismiss_action(registry: ActionRegistry, action_id: str) -> bool:
    """Dismiss only a completed app-session action."""

    with registry.lock:
        record = registry.actions.get(action_id)
        if record is None or record.status not in _TERMINAL_STATUSES:
            return False
        del registry.actions[action_id]
        return True


def run_ats_action(
    registry: ActionRegistry,
    action_id: str,
    store: ApplicationStateStore,
    job_ids: tuple[str, ...],
) -> None:
    """Recalculate selected/default ATS projections with count-only progress."""

    _start_action(registry, action_id)
    _stage_started(registry, action_id, "Recalculate ATS", index=1, job_id=None)
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
        record.successful_steps = 1
        record.completed_jobs = result.updated
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


def _run_job_stages(
    registry: ActionRegistry,
    action_id: str,
    executor: CommandExecutor,
    stages: tuple[CommandStage, ...],
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        job_ids = record.job_ids
        starts = dict(record.retry_starts)
    if not stages or not job_ids:
        _finish_action(
            registry,
            action_id,
            status="completed",
            return_code=0,
            message="Action completed.",
        )
        return

    failed_job_ids: set[str] = set()
    last_return_code = 0
    for stage_index, stage in enumerate(stages):
        eligible = tuple(
            job_id
            for job_id in job_ids
            if job_id not in failed_job_ids and starts.get(job_id, 0) <= stage_index
        )
        if not eligible:
            continue
        stage_failures = 0
        for job_id in eligible:
            _stage_started(
                registry,
                action_id,
                stage.label,
                index=stage_index + 1,
                job_id=job_id,
            )
            try:
                argv = _stage_argv_for_job(stage.argv, job_id)
            except Exception:  # noqa: BLE001 - invalid stages stay content-free.
                result = CommandExecution(return_code=1)
            else:
                result = _execute(executor, argv)
            if result.return_code != 0:
                failed_job_ids.add(job_id)
                stage_failures += 1
                last_return_code = result.return_code
                _record_failure(
                    registry,
                    action_id,
                    job_id=job_id,
                    stage=stage,
                    stage_index=stage_index,
                    remaining_stages=len(stages) - stage_index - 1,
                )
                continue
            _record_success(registry, action_id)
        _stage_completed(
            registry,
            action_id,
            stage.label,
            index=stage_index + 1,
            attempted=len(eligible),
            failed=stage_failures,
        )

    completed_jobs = len(job_ids) - len(failed_job_ids)
    with registry.lock:
        record = registry.actions[action_id]
        record.completed_jobs = completed_jobs
        record.current_job_id = None
    if failed_job_ids and completed_jobs:
        _finish_action(
            registry,
            action_id,
            status="partial",
            return_code=last_return_code or 1,
            message=(
                f"Action completed with {len(failed_job_ids)} failed job(s); "
                f"{completed_jobs} survivor(s) completed."
            ),
        )
    elif failed_job_ids:
        _finish_action(
            registry,
            action_id,
            status="failed",
            return_code=last_return_code or 1,
            message="Action failed for every selected job.",
        )
    else:
        _finish_action(
            registry,
            action_id,
            status="completed",
            return_code=0,
            message="Action completed.",
        )


def _execute(executor: CommandExecutor, argv: Sequence[str]) -> CommandExecution:
    try:
        outcome = executor(argv)
    except Exception:  # noqa: BLE001 - failures stay content-free.
        return CommandExecution(return_code=1)
    if type(outcome) is int:
        return CommandExecution(return_code=outcome)
    if type(outcome) is not CommandExecution or type(outcome.return_code) is not int:
        return CommandExecution(return_code=1)
    if type(outcome.output) is not str:
        return CommandExecution(return_code=1)
    return CommandExecution(
        return_code=outcome.return_code,
        output=outcome.output[:_MAX_CAPTURED_OUTPUT_CHARS],
    )


def _stage_argv_for_job(argv: tuple[str, ...], job_id: str) -> tuple[str, ...]:
    replacements = 0
    result: list[str] = []
    for item in argv:
        if item.startswith("JOB_IDS="):
            result.append(f"JOB_IDS={job_id}")
            replacements += 1
        else:
            result.append(item)
    if replacements != 1:
        raise ValueError("Action stage is invalid.")
    return tuple(result)


def _seeded_count(output: str) -> int | None:
    if not output:
        return None
    decoder = json.JSONDecoder()
    selected: int | None = None
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if type(value) is not dict:
            continue
        count = value.get("jobs_seeded")
        if type(count) is int and 0 <= count <= _MAX_ACTION_JOBS:
            selected = count
    return selected


def _record_success(registry: ActionRegistry, action_id: str) -> None:
    with registry.lock:
        registry.actions[action_id].successful_steps += 1


def _record_failure(
    registry: ActionRegistry,
    action_id: str,
    *,
    job_id: str,
    stage: CommandStage,
    stage_index: int,
    remaining_stages: int,
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.failed_steps += 1
        record.skipped_steps += remaining_stages
        record.failures.append(
            ActionFailure(
                job_id=job_id,
                stage_index=stage_index + 1,
                stage_label=stage.label,
            )
        )


def _progress_total(record: ActionRecord) -> int:
    if record.job_ids and record.stages:
        return sum(
            len(record.stages) - record.retry_starts.get(job_id, 0)
            for job_id in record.job_ids
        )
    return max(record.total_stages, 1)


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
    job_id: str | None,
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.current_stage = label
        record.current_job_id = job_id
        _message(
            record,
            f"Stage {index} of {record.total_stages} running."
            if job_id is None
            else f"Stage {index} of {record.total_stages} processing one job.",
        )


def _stage_completed(
    registry: ActionRegistry,
    action_id: str,
    label: str,
    *,
    index: int,
    attempted: int = 1,
    failed: int = 0,
) -> None:
    with registry.lock:
        record = registry.actions[action_id]
        record.current_stage = label
        record.completed_stages = index
        _message(
            record,
            f"Stage {index} of {record.total_stages} completed for "
            f"{attempted - failed} job(s); {failed} failed.",
        )


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
    "CommandExecution",
    "CommandExecutor",
    "CommandStage",
    "WorkflowComposition",
    "action_snapshots",
    "build_action_stages",
    "build_ingestion_stages",
    "build_seed_argv",
    "create_action",
    "create_retry_action",
    "dismiss_action",
    "parse_workflow_composition",
    "recalculate_selected_ats",
    "run_ats_action",
    "run_command_action",
    "run_seed_action",
]
