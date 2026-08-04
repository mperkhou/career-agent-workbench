#!/usr/bin/env python3
"""Generate governed v1 resume variants from stored public job descriptions."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from career_agent_workbench.artifact_exports import export_rendered_resume
from career_agent_workbench.application_resume import (
    CORE_SKILLS_PROMPT_JOD_MAX_CHARS,
    apply_core_skill_jod_matches,
    attach_job_opening_description_object,
    build_core_skills_jod_match_prompt,
    build_experience_job_bullet_rewrite_prompt,
    build_jod_requirements_target_prompt,
    create_job_opening_description_object,
    experience_jobs_for_jod_bullet_rewrite,
    initialize_application_resume_object,
    replace_experience_job_bullets_from_text_response,
)
from career_agent_workbench.application_state import (
    MAX_QUERY_RESULTS,
    ApplicationStateError,
    ApplicationStateStore,
    ApplicationWorkflowSnapshot,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.ats import AtsError, calculate_ats_diagnostics
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
    resolve_private_workspace_path,
)
from career_agent_workbench.config import WorkspaceMember, WorkspacePaths
from career_agent_workbench.errors import LlmError, OllamaError, WorkflowError
from career_agent_workbench.jod import usable_job_description
from career_agent_workbench.llm import build_llm_client
from career_agent_workbench.resume_rendering import (
    ResumeRenderingError,
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)

T = TypeVar("T")


class _FailureStage(StrEnum):
    CANDIDATE_READ = "candidate_read"
    RESUME_INITIALIZE = "resume_initialize"
    CORE_REQUEST = "core_request"
    CORE_APPLY = "core_apply"
    JOD_REQUEST = "jod_request"
    JOD_APPLY = "jod_apply"
    EXPERIENCE_REWRITE_REQUEST = "experience_rewrite_request"
    EXPERIENCE_REWRITE_APPLY = "experience_rewrite_apply"
    HTML_RENDER = "html_render"
    PDF_RENDER = "pdf_render"
    ATS = "ats"
    STATE_WRITE = "state_write"
    ARTIFACT_EXPORT = "artifact_export"


class _FailureCategory(StrEnum):
    MODEL = "model"
    PARSE = "parse"
    POLICY = "policy"
    RENDER = "render"
    ATS = "ats"
    STATE = "state"
    ARTIFACT = "artifact"
    LOCAL_IO = "local_io"
    UNEXPECTED = "unexpected"


class _FirstDraftFailure(Exception):
    __slots__ = ("category", "stage")

    def __init__(
        self,
        *,
        stage: _FailureStage,
        category: _FailureCategory,
    ) -> None:
        super().__init__("First-draft record processing failed.")
        self.stage = stage
        self.category = category


def _failure_category(
    error: Exception,
    *,
    stage: _FailureStage,
) -> _FailureCategory:
    if isinstance(error, (LlmError, OllamaError)):
        return _FailureCategory.MODEL
    if isinstance(error, ApplicationStateError):
        return _FailureCategory.STATE
    if isinstance(error, ResumeRenderingError):
        return _FailureCategory.RENDER
    if isinstance(error, AtsError):
        return _FailureCategory.ATS
    if stage is _FailureStage.ARTIFACT_EXPORT and isinstance(
        error, (OSError, TypeError, ValueError)
    ):
        return _FailureCategory.ARTIFACT
    if isinstance(error, PermissionError):
        return _FailureCategory.POLICY
    if isinstance(
        error,
        (
            json.JSONDecodeError,
            yaml.YAMLError,
            UnicodeError,
            TypeError,
            ValueError,
            KeyError,
        ),
    ):
        return _FailureCategory.PARSE
    if isinstance(error, WorkflowError):
        return _FailureCategory.POLICY
    if isinstance(error, OSError):
        return _FailureCategory.LOCAL_IO
    return _FailureCategory.UNEXPECTED


def _stage_failure(stage: _FailureStage, error: Exception) -> _FirstDraftFailure:
    if isinstance(error, _FirstDraftFailure):
        return error
    return _FirstDraftFailure(
        stage=stage,
        category=_failure_category(error, stage=stage),
    )


def _run_stage(stage: _FailureStage, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except Exception as error:
        raise _stage_failure(stage, error) from None


async def _run_async_stage(
    stage: _FailureStage,
    operation: Callable[[], Awaitable[T]],
) -> T:
    try:
        return await operation()
    except Exception as error:
        raise _stage_failure(stage, error) from None


def _failure_diagnostic(error: _FirstDraftFailure) -> dict[str, str]:
    return {
        "stage": error.stage.value,
        "category": error.category.value,
    }


class _EligibleCandidate:
    __slots__ = ("description", "job_id", "snapshot")

    def __init__(
        self,
        *,
        job_id: str,
        snapshot: ApplicationWorkflowSnapshot,
        description: str,
    ) -> None:
        self.job_id = job_id
        self.snapshot = snapshot
        self.description = description

    def __repr__(self) -> str:
        return "_EligibleCandidate(content_hidden=True)"


def _eligible_candidate(
    store: ApplicationStateStore,
    *,
    job_id: str,
    force: bool,
) -> _EligibleCandidate | None:
    """Return the one shared dry-run and execution eligibility snapshot."""

    snapshot = store.get_workflow_snapshot(job_id)
    if not force and any(item.variant_key == "v1" for item in snapshot.variants):
        return None
    description: str | None = None
    for value in (
        snapshot.application.prompt_job_description,
        snapshot.application.job_description,
    ):
        try:
            description = usable_job_description(value)
        except (TypeError, ValueError):
            return None
        if description is not None:
            break
    if description is None:
        return None
    return _EligibleCandidate(
        job_id=job_id,
        snapshot=snapshot,
        description=description,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate governed first-draft resume variants."
    )
    add_runtime_path_arguments(
        parser,
        "workspace",
        "database",
        "output_dir",
        "master_resume",
    )
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--job-id", action="append", dest="job_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-model")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-jod-chars", type=int, default=CORE_SKILLS_PROMPT_JOD_MAX_CHARS
    )
    parser.add_argument("--jod-model")
    parser.add_argument("--llm-timeout-seconds", type=float)
    parser.add_argument("--llm-retries", type=int, default=1)
    return parser


async def _with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
) -> T:
    for attempt in range(max(0, min(retries, 3)) + 1):
        try:
            return await operation()
        except Exception:
            if attempt >= max(0, min(retries, 3)):
                raise
    raise AssertionError("unreachable retry loop")


async def _generate_one(
    *,
    store: ApplicationStateStore,
    paths: WorkspacePaths,
    candidate: _EligibleCandidate,
    core_client: Any,
    jod_client: Any,
    jod_model: str,
    template: Path | None,
    artifact_dir: Path | None,
    max_jod_chars: int,
    retries: int,
) -> bool:
    snapshot = candidate.snapshot
    job_id = candidate.job_id
    description = candidate.description
    resume = _run_stage(
        _FailureStage.RESUME_INITIALIZE,
        lambda: initialize_application_resume_object(
            paths.require(WorkspaceMember.MASTER_RESUME)
        ),
    )
    core_prompt = _run_stage(
        _FailureStage.CORE_REQUEST,
        lambda: build_core_skills_jod_match_prompt(
            application_resume=resume,
            trimmed_job_description=description,
            max_jod_chars=max_jod_chars,
        ),
    )
    core_response = await _run_async_stage(
        _FailureStage.CORE_REQUEST,
        lambda: _with_retries(
            lambda: core_client.generate_json(core_prompt), retries=retries
        ),
    )
    resume = _run_stage(
        _FailureStage.CORE_APPLY,
        lambda: apply_core_skill_jod_matches(
            application_resume=resume,
            core_skill_response=core_response,
        ),
    )
    target_prompt = _run_stage(
        _FailureStage.JOD_REQUEST,
        lambda: build_jod_requirements_target_prompt(
            trimmed_job_description=description,
            max_jod_chars=max_jod_chars,
        ),
    )
    target_response = await _run_async_stage(
        _FailureStage.JOD_REQUEST,
        lambda: _with_retries(
            lambda: jod_client.generate_json(target_prompt), retries=retries
        ),
    )
    jod = _run_stage(
        _FailureStage.JOD_APPLY,
        lambda: create_job_opening_description_object(
            trimmed_job_description=description,
            requirements_response=target_response,
            model=jod_model,
        ),
    )
    resume = _run_stage(
        _FailureStage.JOD_APPLY,
        lambda: attach_job_opening_description_object(
            application_resume=resume,
            job_opening_description=jod,
        ),
    )
    experience_jobs = _run_stage(
        _FailureStage.EXPERIENCE_REWRITE_APPLY,
        lambda: experience_jobs_for_jod_bullet_rewrite(resume),
    )
    for job in experience_jobs:
        prompt = _run_stage(
            _FailureStage.EXPERIENCE_REWRITE_REQUEST,
            lambda job=job: build_experience_job_bullet_rewrite_prompt(
                job_opening_description=jod,
                job=job,
            ),
        )
        response = await _run_async_stage(
            _FailureStage.EXPERIENCE_REWRITE_REQUEST,
            lambda prompt=prompt: _with_retries(
                lambda: jod_client.generate_text(prompt), retries=retries
            ),
        )
        resume = _run_stage(
            _FailureStage.EXPERIENCE_REWRITE_APPLY,
            lambda job=job, response=response: (
                replace_experience_job_bullets_from_text_response(
                    application_resume=resume,
                    job_order=job.get("order"),
                    bullet_response=response,
                )
            ),
        )
    html = _run_stage(
        _FailureStage.HTML_RENDER,
        lambda: render_resume_html_from_mapping(
            resume=resume,
            template_path=template,
        ),
    )
    pdf = await _run_async_stage(
        _FailureStage.PDF_RENDER,
        lambda: asyncio.to_thread(render_resume_pdf_from_html, html),
    )
    diagnostics = _run_stage(
        _FailureStage.ATS,
        lambda: calculate_ats_diagnostics(
            resume_pdf=pdf,
            job_description=description,
        ),
    )
    score = _run_stage(_FailureStage.ATS, lambda: diagnostics.score)

    def store_variant() -> None:
        store.upsert_resume_variant_if_revision(
            job_id,
            ResumeVariantWrite(
                variant_key="v1",
                variant_label="Governed first draft",
                source="governed_first_draft",
                application_resume_yaml=yaml.safe_dump(
                    resume, sort_keys=False, allow_unicode=False
                ),
                resume_html=html,
                resume_pdf=pdf,
                ats=AtsFields(
                    score=score.overall_score,
                    parsing_score=score.parsing_score,
                    keyword_score=score.keyword_match_score,
                    semantic_score=score.semantic_match_score,
                    formatting_risk=score.formatting_risk,
                    missing_terms=", ".join(score.missing_high_value_terms),
                ),
                ats_diagnostics=asdict(diagnostics),
                model_metadata={
                    "workflow": "first_draft",
                    "core_model": getattr(core_client, "model", "configured"),
                    "jod_model": jod_model,
                    "review_state": "awaiting_user_review",
                    "requires_human_review": True,
                },
            ),
            expected_revision=snapshot.revision,
        )

    _run_stage(_FailureStage.STATE_WRITE, store_variant)
    if artifact_dir is not None:
        _run_stage(
            _FailureStage.ARTIFACT_EXPORT,
            lambda: export_rendered_resume(
                paths=paths,
                output_dir=artifact_dir,
                job_id=job_id,
                resume=resume,
                template_path=template,
            ),
        )
    return True


async def main_async(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.llm_retries <= 3:
            raise CliConfigurationError("Model execution configuration is invalid.")
        config = load_command_config(
            args,
            required=(
                WorkspaceMember.DATABASE,
                WorkspaceMember.OUTPUT_DIR,
                WorkspaceMember.MASTER_RESUME,
            ),
            setting_overrides={
                "core_skill_model": args.api_model,
                "jod_model": args.jod_model,
                "llm_api_timeout_seconds": args.llm_timeout_seconds,
            },
        )
        template = (
            resolve_private_workspace_path(
                config.paths,
                args.template,
                must_exist=True,
            )
            if args.template is not None
            else None
        )
        artifact_dir = (
            resolve_private_workspace_path(
                config.paths,
                args.artifact_dir,
                directory=True,
            )
            if args.artifact_dir is not None
            else None
        )
        store = ApplicationStateStore(config.paths)
        selected = set(args.job_ids or ())
        records = [
            item
            for item in store.list_applications("active", limit=MAX_QUERY_RESULTS)
            if not selected or item.job_id in selected
        ]
        limit = None if args.limit is None else max(0, args.limit)
        if args.dry_run:
            eligible: list[_EligibleCandidate] = []
            if limit != 0:
                for record in records:
                    candidate = _eligible_candidate(
                        store,
                        job_id=record.job_id,
                        force=args.force,
                    )
                    if candidate is None:
                        continue
                    eligible.append(candidate)
                    if limit is not None and len(eligible) >= limit:
                        break
            print(json.dumps({"candidates": len(eligible), "dry_run": True}))
            return 0
        core_model = config.settings.core_skill_model
        jod_model = config.settings.jod_model
        core_client = build_llm_client(config.settings, api_model=core_model)
        jod_client = build_llm_client(config.settings, api_model=jod_model)
        processed = 0
        failures = 0
        failure_diagnostics: list[dict[str, str]] = []
        admitted = 0
        try:
            for record in records:
                if limit is not None and admitted >= limit:
                    break
                try:
                    candidate = _eligible_candidate(
                        store,
                        job_id=record.job_id,
                        force=args.force,
                    )
                    if candidate is None:
                        continue
                    admitted += 1
                    generated = await _generate_one(
                        store=store,
                        paths=config.paths,
                        candidate=candidate,
                        core_client=core_client,
                        jod_client=jod_client,
                        jod_model=jod_model,
                        template=template,
                        artifact_dir=artifact_dir,
                        max_jod_chars=args.max_jod_chars,
                        retries=args.llm_retries,
                    )
                    processed += int(generated)
                except Exception as error:
                    failures += 1
                    failure = _stage_failure(_FailureStage.CANDIDATE_READ, error)
                    failure_diagnostics.append(_failure_diagnostic(failure))
                    if args.fail_fast:
                        raise failure from None
        finally:
            await core_client.aclose()
            await jod_client.aclose()
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("First-draft generation could not be completed.")
    result: dict[str, object] = {"processed": processed, "failed": failures}
    if failure_diagnostics:
        result["failure_diagnostics"] = failure_diagnostics
    print(json.dumps(result, sort_keys=True))
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
