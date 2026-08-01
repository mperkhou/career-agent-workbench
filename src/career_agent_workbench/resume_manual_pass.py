"""Patch-based manual-review candidate generation.

The workflow always derives ``manual`` from the coherent stored ``v2`` and
stores it as awaiting human review.  It never selects or approves a variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.codex_cli import CodexModelConfig, ModelRunner
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.resume_refinement import (
    ResumeRefinementError,
    _assert_evidence_unchanged,
    _bounded_json_value,
    _calculate_ats,
    _clone_inert,
    _clone_resume,
    _freeze,
    _render_and_score,
    _run_patch_model,
    _snapshot_job_description,
    _validate_workflow_boundary,
    _variant_map,
    _variant_write,
    _workflow_model_metadata,
    build_resume_patch_prompt,
    collect_resume_patch_targets,
    read_resume_evidence_snapshot,
    validate_and_apply_resume_patches,
)

MANUAL_PASS_SCHEMA_VERSION = "governed_manual_pass.v1"


class ResumeManualPassError(ResumeRefinementError):
    """Raised when a manual-review candidate cannot be produced safely."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ManualPassResult:
    """Content-hidden proposal or committed manual-review candidate."""

    changed_count: int
    ats_score: int
    dry_run: bool
    stored_variant: str | None
    review_state: str = "awaiting_user_review"
    requires_human_review: bool = True
    candidate: Mapping[str, Any] = field(repr=False, default_factory=dict)
    job_id: str = field(repr=False, default="")

    def __repr__(self) -> str:
        return (
            "ManualPassResult("
            f"changed_count={self.changed_count}, ats_score={self.ats_score}, "
            f"dry_run={self.dry_run}, stored={self.stored_variant is not None}, "
            "review_state='awaiting_user_review', "
            "requires_human_review=True, content_hidden=True)"
        )


def run_manual_resume_pass(
    *,
    store: ApplicationStateStore,
    paths: WorkspacePaths,
    job_id: str,
    runner: ModelRunner,
    model_config: CodexModelConfig,
    dry_run: bool = False,
) -> ManualPassResult:
    """Derive one patch-validated manual candidate from exact stored v2."""

    validated_id = _validate_workflow_boundary(
        store=store,
        paths=paths,
        job_id=job_id,
        model_config=model_config,
        dry_run=dry_run,
    )
    snapshot = store.get_workflow_snapshot(validated_id)
    evidence = read_resume_evidence_snapshot(paths)
    variants = _variant_map(snapshot)
    v1 = variants.get("v1")
    v2 = variants.get("v2")
    if v1 is None or v2 is None:
        raise ResumeManualPassError("Manual resume workflow input is invalid.")

    base_resume = _clone_resume(v2.application_resume)
    targets = collect_resume_patch_targets(base_resume)
    job_description = _snapshot_job_description(snapshot)
    initial_diagnostics = _calculate_ats(v2.resume_pdf, job_description)
    context = _manual_context(v1=v1, v2=v2)
    prompt = build_resume_patch_prompt(
        workflow="manual",
        snapshot=snapshot,
        base_variant=v2,
        base_resume=base_resume,
        targets=targets,
        evidence=evidence,
        ats_diagnostics=initial_diagnostics,
        external_critique=None,
        additional_context=context,
    )
    parsed, model_metadata = _run_patch_model(
        runner=runner,
        config=model_config,
        prompt=prompt,
    )
    candidate, audit = validate_and_apply_resume_patches(
        application_resume=base_resume,
        response=parsed,
        evidence=evidence,
    )
    html, pdf, diagnostics = _render_and_score(candidate, job_description)
    result = ManualPassResult(
        changed_count=len(parsed.changes),
        ats_score=diagnostics.score.overall_score,
        dry_run=dry_run,
        stored_variant=None if dry_run else "manual",
        candidate=_freeze(candidate),
        job_id=validated_id,
    )
    if dry_run:
        return result

    variant = _variant_write(
        variant_key="manual",
        variant_label="Manual review candidate",
        source="governed_manual_pass",
        parent_variant_key="v2",
        candidate=candidate,
        html=html,
        pdf=pdf,
        diagnostics=diagnostics,
        evidence=evidence,
        audit=audit,
        model_metadata=_workflow_model_metadata(
            model_metadata,
            workflow="manual",
            review_state="awaiting_user_review",
        ),
        external_critique=None,
    )
    _assert_evidence_unchanged(paths, evidence)
    store.upsert_resume_variant_if_revision(
        validated_id,
        variant,
        expected_revision=snapshot.revision,
    )
    return result


def run_manual_resume_pass_for_job(**kwargs: Any) -> ManualPassResult:
    """Keyword-only compatibility spelling for the callable domain workflow."""

    return run_manual_resume_pass(**kwargs)


def _manual_context(*, v1: Any, v2: Any) -> Mapping[str, Any]:
    """Return bounded v1/v2 context without exposing raw model traffic."""

    context = {
        "schema_version": MANUAL_PASS_SCHEMA_VERSION,
        "v1_application_resume": _clone_resume(v1.application_resume),
        "v2_application_resume": _clone_resume(v2.application_resume),
        "v2_structured_audit": {
            "evidence_packet": _clone_inert(v2.evidence_packet),
            "external_critique": _clone_inert(v2.external_critique),
            "critique": _clone_inert(v2.critique),
            "validation": _clone_inert(v2.validation),
            "model_metadata": _clone_inert(v2.model_metadata),
        },
        "review_state": "awaiting_user_review",
        "requires_human_review": True,
    }
    return _bounded_json_value(context)
