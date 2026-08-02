"""Evidence-grounded, caller-injected resume refinement.

This module owns no path discovery, process construction, or durable artifact
output.  A caller supplies one initialized state store, one exact workspace
snapshot, and one bounded model runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml

from career_agent_workbench.application_state import (
    ApplicationRecord,
    ApplicationStateStore,
    ApplicationWorkflowSnapshot,
    AtsFields,
    ResumeVariantRecord,
    ResumeVariantWrite,
)
from career_agent_workbench.ats import (
    AtsComponentScores,
    AtsDiagnostics,
    AtsProxyScore,
    AtsWeightedTerm,
    calculate_ats_diagnostics,
)
from career_agent_workbench.codex_cli import (
    CodexModelConfig,
    ModelRequest,
    ModelResult,
    ModelRunner,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.errors import WorkflowError
from career_agent_workbench.resume_rendering import (
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)

RESUME_PATCH_SCHEMA_VERSION = "governed_resume_patch.v1"
RESUME_EVIDENCE_SCHEMA_VERSION = "governed_resume_evidence.v1"
RESUME_VALIDATION_SCHEMA_VERSION = "governed_resume_validation.v1"

MAX_EVIDENCE_YAML_BYTES = 2_000_000
MAX_EVIDENCE_SOURCE_BYTES = 100_000
MAX_EXTERNAL_CRITIQUE_CHARS = 20_000
MAX_MODEL_RESPONSE_CHARS = 200_000
MAX_MODEL_PROMPT_CHARS = 240_000
MAX_PATCH_CHANGES = 64
MAX_EVIDENCE_ITEMS = 256
MAX_EVIDENCE_ITEM_CHARS = 3_000
MAX_EVIDENCE_PACKET_CHARS = 130_000
MAX_PATCH_TEXT_CHARS = 12_000
MAX_PATCH_ID_CHARS = 128
MAX_PATCH_RATIONALE_CHARS = 4_000
MAX_EVIDENCE_REFS_PER_CHANGE = 24
MAX_JSON_DEPTH = 18
MAX_JSON_NODES = 4_000
MAX_JSON_COLLECTION_ITEMS = 512
MAX_JSON_STRING_CHARS = 200_000
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_METRIC_RE = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?:%|\+)?(?![A-Za-z0-9])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]*")
_CLAIM_TOKEN_RE = re.compile(r"[^\W_]+(?:[+#./-][^\W_]+)*", re.UNICODE)
_MOVEMENT_ATOM_RE = re.compile(
    r"(?:[^\W_]+(?:[+#/-][^\W_]+)*(?:[+#%])?|"
    r"[^\W_]*\d(?:\.\d+)+(?:[+#%])?|[A-Za-z]\+\+)\Z",
    re.UNICODE,
)
_PATH_TYPE = type(Path())
_NEGATION_TOKENS = frozenset({"no", "nor", "not", "never", "without"})
_SUMMARY_EXTRACTABLE_PREFIXES = frozenset(
    {
        "built",
        "builds",
        "created",
        "creates",
        "delivered",
        "delivers",
        "designed",
        "designs",
        "developed",
        "develops",
        "engineered",
        "maintained",
        "maintains",
    }
)
_SUMMARY_ACTION_TOKENS = frozenset(
    {
        "administered",
        "architected",
        "audited",
        "build",
        "builds",
        "built",
        "contributed",
        "created",
        "creates",
        "delivered",
        "delivers",
        "deployed",
        "designed",
        "designs",
        "developed",
        "develops",
        "directed",
        "engineered",
        "helped",
        "implemented",
        "increased",
        "led",
        "managed",
        "maintained",
        "maintains",
        "migrated",
        "owned",
        "reduced",
        "secured",
        "supervised",
        "supported",
    }
)
_MOVABLE_PREPOSITION_TOKENS = frozenset(
    {
        "for",
    }
)
_CLAUSE_BOUNDARY_TOKENS = frozenset(
    {
        "after",
        "although",
        "and",
        "because",
        "before",
        "but",
        "or",
        "that",
        "when",
        "while",
        "which",
        "who",
    }
)

_TECHNICAL_OR_REGULATED_PHRASES = frozenset(
    {
        "ai",
        "ansible",
        "api",
        "aws",
        "azure",
        "certificate",
        "certified",
        "certification",
        "ci/cd",
        "compliance",
        "credential",
        "docker",
        "fedramp",
        "gcp",
        "github actions",
        "gpu",
        "hipaa",
        "java",
        "javascript",
        "kafka",
        "kubernetes",
        "linux",
        "oracle",
        "pci",
        "postgresql",
        "python",
        "rbac",
        "regulated",
        "rust",
        "soc 2",
        "sql",
        "terraform",
        "typescript",
    }
)
_RESPONSIBILITY_OR_OUTCOME_PHRASES = frozenset(
    {
        "administered",
        "architected",
        "audited",
        "certified",
        "complied",
        "deployed",
        "directed",
        "governed",
        "implemented",
        "increased",
        "led",
        "leadership",
        "managed",
        "migrated",
        "owned",
        "reduced",
        "secured",
        "supervised",
    }
)

_WORKFLOW_ERROR = "Resume workflow could not be completed."
_WORKFLOW_INPUT_ERROR = "Resume workflow input is invalid."
_WORKFLOW_EVIDENCE_ERROR = "Resume evidence could not be loaded."
_WORKFLOW_MODEL_ERROR = "Resume model result is invalid."
_WORKFLOW_PATCH_ERROR = "Resume patch validation failed."
_WORKFLOW_CONFLICT_ERROR = "Resume workflow input changed."
_WORKFLOW_RENDER_ERROR = "Resume candidate could not be rendered."


class ResumeRefinementError(WorkflowError):
    """Base class for stable, content-free refinement failures."""

    __slots__ = ()


class ResumeEvidenceError(ResumeRefinementError):
    """Raised when exact bounded evidence cannot be snapshotted."""

    __slots__ = ()


class ResumePatchError(ResumeRefinementError):
    """Raised when a model response is not one exact safe patch set."""

    __slots__ = ()


class ResumeWorkflowConflictError(ResumeRefinementError):
    """Raised when evidence or state changes before the conditional write."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ResumeEvidenceItem:
    """One exact canonical MRO claim with role/employer ownership."""

    evidence_id: str
    role_id: str
    employer: str = field(repr=False)
    role: str = field(repr=False)
    text: str = field(repr=False)

    def __repr__(self) -> str:
        return "ResumeEvidenceItem(configured=True)"


@dataclass(frozen=True, slots=True)
class ResumeEvidenceSnapshot:
    """Immutable in-memory snapshot of both exact evidence files."""

    master_resume: Mapping[str, Any] = field(repr=False)
    source_text: str = field(repr=False)
    mro_sha256: str = field(repr=False)
    source_text_sha256: str = field(repr=False)
    items: tuple[ResumeEvidenceItem, ...] = field(repr=False)

    def __repr__(self) -> str:
        return "ResumeEvidenceSnapshot(configured=True, content_hidden=True)"


@dataclass(frozen=True, slots=True)
class ResumePatchTarget:
    """One allowlisted exact ARO text target."""

    section: Literal[
        "professional_summary",
        "professional_experience",
        "core_technical_skills",
    ]
    field: Literal["paragraph", "text", "items"]
    job_order: str | None
    bullet_order: str | None

    @property
    def target_id(self) -> str:
        if self.section == "professional_summary":
            return f"summary:{self.field}"
        if self.section == "core_technical_skills":
            return f"skills:{self.job_order}:items"
        return f"experience:{self.job_order}:bullet:{self.bullet_order}"


@dataclass(frozen=True, slots=True)
class ResumePatch:
    """One exact-target, evidence-referenced proposed rewrite."""

    change_id: str
    operation: Literal[
        "rewrite_summary",
        "rewrite_bullet",
        "replace_skill_items",
    ]
    target: ResumePatchTarget
    current_text: str = field(repr=False)
    proposed_text: str = field(repr=False)
    rationale: str = field(repr=False)
    evidence_refs: tuple[str, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            "ResumePatch("
            f"operation={self.operation!r}, target={self.target!r}, "
            "content_hidden=True)"
        )


@dataclass(frozen=True, slots=True)
class ResumePatchResponse:
    """One fully parsed strict patch response."""

    schema_version: str
    changes: tuple[ResumePatch, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            "ResumePatchResponse("
            f"change_count={len(self.changes)}, content_hidden=True)"
        )


@dataclass(frozen=True, slots=True)
class ResumeRefinementResult:
    """Content-hidden proposal or committed v2 result."""

    changed_count: int
    ats_score: int
    dry_run: bool
    stored_variant: str | None
    requires_human_review: bool = True
    candidate: Mapping[str, Any] = field(repr=False, default_factory=dict)
    job_id: str = field(repr=False, default="")

    def __repr__(self) -> str:
        return (
            "ResumeRefinementResult("
            f"changed_count={self.changed_count}, ats_score={self.ats_score}, "
            f"dry_run={self.dry_run}, stored={self.stored_variant is not None}, "
            "requires_human_review=True, content_hidden=True)"
        )


@dataclass(frozen=True, slots=True)
class _ResumeTarget:
    target: ResumePatchTarget
    text: str = field(repr=False)
    role_id: str | None = field(repr=False)
    employer: str = field(repr=False)
    role: str = field(repr=False)
    location: tuple[object, ...] = field(repr=False)


def refine_resume_for_job(
    *,
    store: ApplicationStateStore,
    paths: WorkspacePaths,
    job_id: str,
    runner: ModelRunner,
    model_config: CodexModelConfig,
    external_critique: str | None = None,
    template_path: Path | None = None,
    dry_run: bool = False,
) -> ResumeRefinementResult:
    """Derive an evidence-grounded v2 candidate from the exact stored v1."""

    validated_id = _validate_workflow_boundary(
        store=store,
        paths=paths,
        job_id=job_id,
        model_config=model_config,
        dry_run=dry_run,
    )
    critique = _optional_external_critique(external_critique)
    snapshot = store.get_workflow_snapshot(validated_id)
    evidence = read_resume_evidence_snapshot(paths)
    variants = _variant_map(snapshot)
    v1 = variants.get("v1")
    if v1 is None:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)

    base_resume = _clone_resume(v1.application_resume)
    targets = collect_resume_patch_targets(base_resume)
    job_description = _snapshot_job_description(snapshot)
    initial_diagnostics = _calculate_ats(v1.resume_pdf, job_description)
    prompt = build_resume_patch_prompt(
        workflow="refinement",
        snapshot=snapshot,
        base_variant=v1,
        base_resume=base_resume,
        targets=targets,
        evidence=evidence,
        ats_diagnostics=initial_diagnostics,
        external_critique=critique,
        additional_context=None,
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
    html, pdf, diagnostics = _render_and_score(
        candidate,
        job_description,
        template_path=template_path,
    )

    result = ResumeRefinementResult(
        changed_count=len(parsed.changes),
        ats_score=diagnostics.score.overall_score,
        dry_run=dry_run,
        stored_variant=None if dry_run else "v2",
        candidate=_freeze(candidate),
        job_id=validated_id,
    )
    if dry_run:
        return result

    variant = _variant_write(
        variant_key="v2",
        variant_label="Evidence-grounded refinement",
        source="governed_refinement",
        parent_variant_key="v1",
        candidate=candidate,
        html=html,
        pdf=pdf,
        diagnostics=diagnostics,
        evidence=evidence,
        audit=audit,
        model_metadata=_workflow_model_metadata(
            model_metadata,
            workflow="refinement",
            review_state="awaiting_user_review",
        ),
        external_critique=critique,
    )
    _assert_evidence_unchanged(paths, evidence)
    store.upsert_resume_variant_if_revision(
        validated_id,
        variant,
        expected_revision=snapshot.revision,
    )
    return result


def run_resume_refinement(**kwargs: Any) -> ResumeRefinementResult:
    """Keyword-only compatibility spelling for the callable domain workflow."""

    return refine_resume_for_job(**kwargs)


def read_resume_evidence_snapshot(paths: WorkspacePaths) -> ResumeEvidenceSnapshot:
    """Read the exact MRO and source-text files once each into bounded memory."""

    if type(paths) is not WorkspacePaths:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    master_path = _required_absolute_path(paths.master_resume)
    source_path = _required_absolute_path(paths.master_resume_text)
    master_bytes = _read_regular_file(master_path, MAX_EVIDENCE_YAML_BYTES)
    source_bytes = _read_regular_file(source_path, MAX_EVIDENCE_SOURCE_BYTES)
    master_text = _decode_text(master_bytes)
    source_text = _decode_text(source_bytes)
    _validate_text(source_text, max_chars=MAX_EVIDENCE_SOURCE_BYTES)

    try:
        loaded = yaml.safe_load(master_text)
    except (yaml.YAMLError, RecursionError, TypeError, ValueError):
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR) from None
    try:
        master_resume = _materialize_inert(loaded)
    except (TypeError, ValueError, RecursionError):
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR) from None
    if type(master_resume) is not dict:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)

    items = _collect_mro_evidence_items(master_resume)
    serialized_items = json.dumps(
        [_evidence_prompt_item(item) for item in items],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized_items) > MAX_EVIDENCE_PACKET_CHARS:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    return ResumeEvidenceSnapshot(
        master_resume=_freeze(master_resume),
        source_text=source_text,
        mro_sha256=hashlib.sha256(master_bytes).hexdigest(),
        source_text_sha256=hashlib.sha256(source_bytes).hexdigest(),
        items=items,
    )


def collect_resume_patch_targets(
    application_resume: Mapping[str, Any],
    *,
    include_skill_targets: bool = False,
) -> tuple[_ResumeTarget, ...]:
    """Collect summary and rendered bullet targets with stable exact identities."""

    if type(include_skill_targets) is not bool:
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    resume = _clone_resume(application_resume)
    targets: list[_ResumeTarget] = []
    seen: set[str] = set()

    summary = resume.get("professional_summary")
    if type(summary) is dict:
        summary_fields = [
            name
            for name in ("paragraph", "text")
            if type(summary.get(name)) is str and summary.get(name)
        ]
        if len(summary_fields) > 1:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        if summary_fields:
            field_name = summary_fields[0]
            target = ResumePatchTarget(
                section="professional_summary",
                field=field_name,
                job_order=None,
                bullet_order=None,
            )
            targets.append(
                _ResumeTarget(
                    target=target,
                    text=summary[field_name],
                    role_id=None,
                    employer="",
                    role="",
                    location=("professional_summary", field_name),
                )
            )
            seen.add(target.target_id)

    if include_skill_targets:
        _collect_skill_patch_targets(resume, targets, seen)

    experience = resume.get("professional_experience")
    jobs = experience.get("jobs") if type(experience) is dict else None
    if jobs is None:
        return tuple(targets)
    if type(jobs) is not list:
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    seen_jobs: set[str] = set()
    for job_index, job in enumerate(jobs):
        if type(job) is not dict:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        if not _is_rendered(job):
            continue
        job_order = _exact_order(job.get("order"))
        if job_order in seen_jobs:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        seen_jobs.add(job_order)
        employer, role = _job_ownership(job)
        bullets = job.get("bullet_points")
        if type(bullets) is not list:
            continue
        seen_bullets: set[str] = set()
        for bullet_index, bullet in enumerate(bullets):
            if type(bullet) is not dict:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            if not _is_rendered(bullet):
                continue
            bullet_order = _exact_order(bullet.get("order"))
            if bullet_order in seen_bullets:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            seen_bullets.add(bullet_order)
            text = bullet.get("text")
            if type(text) is not str or not text:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            _validate_text(text, max_chars=MAX_PATCH_TEXT_CHARS)
            target = ResumePatchTarget(
                section="professional_experience",
                field="text",
                job_order=job_order,
                bullet_order=bullet_order,
            )
            if target.target_id in seen:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            seen.add(target.target_id)
            targets.append(
                _ResumeTarget(
                    target=target,
                    text=text,
                    role_id=f"mro:job:{job_order}",
                    employer=employer,
                    role=role,
                    location=(
                        "professional_experience",
                        "jobs",
                        job_index,
                        "bullet_points",
                        bullet_index,
                        "text",
                    ),
                )
            )
    return tuple(targets)


def _collect_skill_patch_targets(
    resume: dict[str, Any],
    targets: list[_ResumeTarget],
    seen: set[str],
) -> None:
    skills = resume.get("core_technical_skills")
    categories = skills.get("bullet_points") if type(skills) is dict else None
    if categories is None:
        return
    if type(categories) is not list:
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    for category_index, category in enumerate(categories):
        if type(category) is not dict or not _is_rendered(category):
            continue
        category_name = category.get("category")
        items = category.get("items")
        if (
            type(category_name) is not str
            or not category_name
            or type(items) is not dict
        ):
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        _validate_text(category_name, max_chars=1_024)
        visible = _visible_skill_items(items)
        if not visible:
            continue
        target = ResumePatchTarget(
            section="core_technical_skills",
            field="items",
            job_order=str(category_index + 1),
            bullet_order=None,
        )
        if target.target_id in seen:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        seen.add(target.target_id)
        targets.append(
            _ResumeTarget(
                target=target,
                text=_serialize_skill_items(visible),
                role_id="mro:skills",
                employer="Canonical MRO",
                role=category_name,
                location=(
                    "core_technical_skills",
                    "bullet_points",
                    category_index,
                    "items",
                ),
            )
        )


def _visible_skill_items(items: dict[str, Any]) -> dict[str, list[str]]:
    visible: dict[str, list[str]] = {}
    for key in ("primary", "additional"):
        values = items.get(key)
        if values is None:
            continue
        if type(values) is not list or len(values) > MAX_JSON_COLLECTION_ITEMS:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        parsed: list[str] = []
        for value in values:
            if type(value) is not str:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            parsed.append(
                _bounded_plain_string(
                    value,
                    max_chars=1_024,
                    allow_empty=False,
                )
            )
        visible[key] = parsed
    return visible


def _serialize_skill_items(items: Mapping[str, list[str]]) -> str:
    return json.dumps(
        dict(items),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_resume_patch_prompt(
    *,
    workflow: Literal["refinement", "manual"],
    snapshot: ApplicationWorkflowSnapshot,
    base_variant: ResumeVariantRecord,
    base_resume: Mapping[str, Any],
    targets: tuple[_ResumeTarget, ...],
    evidence: ResumeEvidenceSnapshot,
    ats_diagnostics: AtsDiagnostics,
    external_critique: str | None,
    additional_context: Mapping[str, Any] | None,
) -> str:
    """Build a bounded patch-only request with exact ownership evidence."""

    validated_evidence = _validate_patch_prompt_inputs(
        workflow=workflow,
        snapshot=snapshot,
        base_variant=base_variant,
        base_resume=base_resume,
        targets=targets,
        evidence=evidence,
        ats_diagnostics=ats_diagnostics,
        external_critique=external_critique,
        additional_context=additional_context,
    )
    application = snapshot.application
    payload = {
        "workflow": workflow,
        "job": {
            "company": application.company,
            "title": application.job_title,
            "job_description": _snapshot_job_description(snapshot),
        },
        "base_variant": base_variant.variant_key,
        "base_resume": _bounded_json_value(base_resume),
        "targets": [
            {
                "target": _target_payload(item.target),
                "target_id": item.target.target_id,
                "current_text": item.text,
                "role_id": item.role_id,
                "employer": item.employer,
                "role": item.role,
            }
            for item in targets
        ],
        "canonical_mro_evidence": [
            _evidence_prompt_item(item) for item in validated_evidence.items
        ],
        "source_text_corroboration": validated_evidence.source_text,
        "ats_context": _ats_diagnostics_payload(ats_diagnostics),
        "external_critique": external_critique,
        "additional_context": _bounded_json_value(additional_context)
        if additional_context is not None
        else None,
    }
    manual_skill_policy = (
        " For manual skill targets, replace only the rendered primary/additional "
        "item lists represented by the exact current_text JSON object; preserve "
        "category identity and non-rendered metadata. Keep pruning and "
        "de-duplicating the skills section to avoid bloat; do not copy v2's skills "
        "section wholesale. While pruning, preserve or include DevOps, Scalability, "
        "CI/CD pipelines, cloud environments, and GitHub Actions when the term is "
        "already present in v2 or requested by the JOD and supported by MRO/ARO "
        "evidence. Rewrite inflated surrounding wording truthfully while retaining "
        "only supported terms; never retain an unsupported claim merely to retain a "
        "term. Use operation replace_skill_items, section core_technical_skills, "
        "field items, the supplied job_order, null bullet_order, and JSON-encoded "
        "current_text/proposed_text objects with the same list keys."
        if workflow == "manual"
        else ""
    )
    prompt = (
        "Return only one strict JSON object. Propose patch objects, never a full "
        "resume. Only rewrite an existing professional summary, an existing "
        "professional-experience bullet, or an explicitly supplied manual skill "
        "target. Preserve target identity, ordering, and "
        "topology. Every change must cite canonical_mro_evidence IDs. A bullet may "
        "cite only evidence with the exact same canonical role_id, employer, and "
        "role. Source text, the job description, ATS context, current resumes, "
        "external critique, and model commentary are context only and cannot "
        "authorize a claim. Do not invent a metric, tool, employer, certification, "
        "credential, responsibility, leadership, compliance claim, outcome, or "
        "identity. Return exact keys and this schema: "
        '{"schema_version":"governed_resume_patch.v1","changes":['
        '{"change_id":"change-1","operation":"rewrite_bullet","target":'
        '{"section":"professional_experience","field":"text","job_order":"1",'
        '"bullet_order":"1"},"current_text":"exact current text",'
        '"proposed_text":"supported replacement","rationale":"bounded reason",'
        '"evidence_refs":["mro:job:1:bullet:1"]}]}. ' + manual_skill_policy + " "
        "For a summary, use operation rewrite_summary, section "
        "professional_summary, field paragraph or text, and null job_order and "
        "bullet_order. Zero changes is valid. Payload: "
        + json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    if len(prompt) > MAX_MODEL_PROMPT_CHARS:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return prompt


def _validate_patch_prompt_inputs(
    *,
    workflow: object,
    snapshot: object,
    base_variant: object,
    base_resume: object,
    targets: object,
    evidence: object,
    ats_diagnostics: object,
    external_critique: object,
    additional_context: object,
) -> ResumeEvidenceSnapshot:
    if type(workflow) is not str or workflow not in {"refinement", "manual"}:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if (
        type(snapshot) is not ApplicationWorkflowSnapshot
        or type(snapshot.application) is not ApplicationRecord
        or type(snapshot.application.company) is not str
        or type(snapshot.application.job_title) is not str
        or (
            snapshot.application.prompt_job_description is not None
            and type(snapshot.application.prompt_job_description) is not str
        )
        or (
            snapshot.application.job_description is not None
            and type(snapshot.application.job_description) is not str
        )
        or type(base_variant) is not ResumeVariantRecord
        or type(base_variant.variant_key) is not str
        or type(base_resume) is not dict
        or type(targets) is not tuple
        or len(targets) > MAX_EVIDENCE_ITEMS
        or external_critique is not None
        and type(external_critique) is not str
        or additional_context is not None
        and type(additional_context) is not dict
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    _bounded_plain_string(
        snapshot.application.company,
        max_chars=1_024,
        allow_empty=True,
    )
    _bounded_plain_string(
        snapshot.application.job_title,
        max_chars=1_024,
        allow_empty=True,
    )
    _bounded_plain_string(
        base_variant.variant_key,
        max_chars=128,
        allow_empty=False,
    )
    if external_critique is not None:
        _optional_external_critique(external_critique)
    validated_evidence = _validate_evidence_snapshot_object(evidence)
    _ats_diagnostics_payload(ats_diagnostics)
    seen_targets: set[str] = set()
    for item in targets:
        if (
            type(item) is not _ResumeTarget
            or type(item.target) is not ResumePatchTarget
            or type(item.text) is not str
            or (item.role_id is not None and type(item.role_id) is not str)
            or type(item.employer) is not str
            or type(item.role) is not str
            or type(item.location) is not tuple
            or any(type(part) not in {str, int} for part in item.location)
        ):
            raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
        target_id = _validated_prompt_target_id(item.target)
        if target_id in seen_targets:
            raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
        seen_targets.add(target_id)
        if item.target.section == "professional_summary":
            role_identity_valid = (
                item.role_id is None and not item.employer and not item.role
            )
        elif item.target.section == "core_technical_skills":
            role_identity_valid = (
                item.role_id == "mro:skills"
                and item.employer == "Canonical MRO"
                and bool(item.role)
            )
        else:
            role_identity_valid = (
                type(item.role_id) is str
                and item.role_id == f"mro:job:{item.target.job_order}"
                and bool(item.employer)
                and bool(item.role)
            )
        if not role_identity_valid:
            raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
        _validate_text(item.text, max_chars=MAX_PATCH_TEXT_CHARS)
        _bounded_plain_string(
            item.employer,
            max_chars=1_024,
            allow_empty=True,
        )
        _bounded_plain_string(
            item.role,
            max_chars=1_024,
            allow_empty=True,
        )
    _bounded_json_value(base_resume)
    target_derivation_failed = False
    canonical_targets: tuple[_ResumeTarget, ...] = ()
    try:
        canonical_targets = collect_resume_patch_targets(
            base_resume,
            include_skill_targets=any(
                item.target.section == "core_technical_skills" for item in targets
            ),
        )
    except (ResumeEvidenceError, ResumePatchError, ResumeRefinementError):
        target_derivation_failed = True
    if target_derivation_failed or canonical_targets != targets:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if additional_context is not None:
        _bounded_json_value(additional_context)
    return validated_evidence


def _ats_diagnostics_payload(value: object) -> dict[str, Any]:
    if (
        type(value) is not AtsDiagnostics
        or type(value.score) is not AtsProxyScore
        or type(value.component_scores) is not AtsComponentScores
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)

    score = value.score
    components = value.component_scores
    integer_values = (
        score.overall_score,
        score.parsing_score,
        score.keyword_match_score,
        score.semantic_match_score,
        components.overall_score,
        components.parsing_score,
        components.keyword_match_score,
        components.semantic_match_score,
        components.formatting_score,
    )
    if (
        any(type(item) is not int or not 0 <= item <= 100 for item in integer_values)
        or type(score.formatting_risk) is not str
        or type(components.formatting_risk) is not str
        or type(score.missing_high_value_terms) is not tuple
        or any(type(item) is not str for item in score.missing_high_value_terms)
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    _bounded_plain_string(
        score.formatting_risk,
        max_chars=128,
        allow_empty=False,
    )
    _bounded_plain_string(
        components.formatting_risk,
        max_chars=128,
        allow_empty=False,
    )
    missing = [
        _bounded_plain_string(item, max_chars=1_024, allow_empty=False)
        for item in score.missing_high_value_terms
    ]

    def terms(items: object) -> list[dict[str, str | float]]:
        if type(items) is not tuple or len(items) > 2_000:
            raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
        result: list[dict[str, str | float]] = []
        for item in items:
            if (
                type(item) is not AtsWeightedTerm
                or type(item.term) is not str
                or type(item.weight) not in {int, float}
            ):
                raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
            if type(item.weight) is float:
                valid_weight = math.isfinite(item.weight)
            else:
                valid_weight = abs(item.weight) <= 1_000_000
            if not valid_weight or abs(item.weight) > 1_000_000:
                raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
            result.append(
                {
                    "term": _bounded_plain_string(
                        item.term,
                        max_chars=1_024,
                        allow_empty=False,
                    ),
                    "weight": float(item.weight),
                }
            )
        return result

    return {
        "score": {
            "overall_score": score.overall_score,
            "parsing_score": score.parsing_score,
            "keyword_match_score": score.keyword_match_score,
            "semantic_match_score": score.semantic_match_score,
            "formatting_risk": score.formatting_risk,
            "missing_high_value_terms": missing,
        },
        "component_scores": {
            "overall_score": components.overall_score,
            "parsing_score": components.parsing_score,
            "keyword_match_score": components.keyword_match_score,
            "semantic_match_score": components.semantic_match_score,
            "formatting_score": components.formatting_score,
            "formatting_risk": components.formatting_risk,
        },
        "matched_terms": terms(value.matched_terms),
        "unmatched_weighted_terms": terms(value.unmatched_weighted_terms),
        "repeated_phrase_terms": terms(value.repeated_phrase_terms),
        "likely_noisy_phrase_matches": terms(value.likely_noisy_phrase_matches),
    }


def parse_resume_patch_response(response_text: str) -> ResumePatchResponse:
    """Parse one exact, duplicate-free, bounded versioned JSON response."""

    payload = _strict_json_object(response_text)
    if set(payload) != {"schema_version", "changes"}:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    if payload.get("schema_version") != RESUME_PATCH_SCHEMA_VERSION:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    raw_changes = payload.get("changes")
    if type(raw_changes) is not list or len(raw_changes) > MAX_PATCH_CHANGES:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)

    changes: list[ResumePatch] = []
    change_ids: set[str] = set()
    target_ids: set[str] = set()
    for raw in raw_changes:
        if type(raw) is not dict or set(raw) != {
            "change_id",
            "operation",
            "target",
            "current_text",
            "proposed_text",
            "rationale",
            "evidence_refs",
        }:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        change_id = _bounded_plain_string(
            raw["change_id"],
            max_chars=MAX_PATCH_ID_CHARS,
            allow_empty=False,
        )
        if not _IDENTIFIER_RE.fullmatch(change_id) or change_id in change_ids:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        change_ids.add(change_id)
        operation = raw["operation"]
        if type(operation) is not str or operation not in {
            "rewrite_summary",
            "rewrite_bullet",
            "replace_skill_items",
        }:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        target = _parse_patch_target(raw["target"])
        if target.target_id in target_ids:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        target_ids.add(target.target_id)
        if operation == "rewrite_summary" and target.section != "professional_summary":
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        if (
            operation == "rewrite_bullet"
            and target.section != "professional_experience"
        ):
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        if (
            operation == "replace_skill_items"
            and target.section != "core_technical_skills"
        ):
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)

        current_text = _bounded_plain_string(
            raw["current_text"],
            max_chars=MAX_PATCH_TEXT_CHARS,
            allow_empty=False,
        )
        proposed_text = _bounded_plain_string(
            raw["proposed_text"],
            max_chars=MAX_PATCH_TEXT_CHARS,
            allow_empty=False,
        )
        rationale = _bounded_plain_string(
            raw["rationale"],
            max_chars=MAX_PATCH_RATIONALE_CHARS,
            allow_empty=False,
        )
        refs = raw["evidence_refs"]
        if (
            type(refs) is not list
            or not refs
            or len(refs) > MAX_EVIDENCE_REFS_PER_CHANGE
        ):
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        parsed_refs: list[str] = []
        seen_refs: set[str] = set()
        for ref in refs:
            parsed_ref = _bounded_plain_string(
                ref,
                max_chars=MAX_PATCH_ID_CHARS,
                allow_empty=False,
            )
            if parsed_ref in seen_refs:
                raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
            seen_refs.add(parsed_ref)
            parsed_refs.append(parsed_ref)
        changes.append(
            ResumePatch(
                change_id=change_id,
                operation=operation,
                target=target,
                current_text=current_text,
                proposed_text=proposed_text,
                rationale=rationale,
                evidence_refs=tuple(parsed_refs),
            )
        )
    return ResumePatchResponse(
        schema_version=RESUME_PATCH_SCHEMA_VERSION,
        changes=tuple(changes),
    )


def validate_and_apply_resume_patches(
    *,
    application_resume: Mapping[str, Any],
    response: ResumePatchResponse,
    evidence: ResumeEvidenceSnapshot,
    allow_skill_updates: bool = False,
    job_description: str | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Validate every patch first, then apply all changes to a defensive copy."""

    if (
        type(application_resume) is not dict
        or type(allow_skill_updates) is not bool
        or (job_description is not None and type(job_description) is not str)
    ):
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    _validate_patch_response_object(response)
    validated_evidence = _validate_evidence_snapshot_object(evidence)
    candidate = _clone_resume(application_resume)
    targets = collect_resume_patch_targets(
        candidate,
        include_skill_targets=allow_skill_updates,
    )
    target_by_id = {target.target.target_id: target for target in targets}
    evidence_by_id = {item.evidence_id: item for item in validated_evidence.items}
    if len(evidence_by_id) != len(validated_evidence.items):
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)

    for change in response.changes:
        target = target_by_id.get(change.target.target_id)
        if target is None or target.target != change.target:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        if change.current_text != target.text:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        if change.proposed_text == change.current_text:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        cited: list[ResumeEvidenceItem] = []
        for ref in change.evidence_refs:
            item = evidence_by_id.get(ref)
            if item is None:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            cited.append(item)
        if target.target.section == "core_technical_skills":
            if not allow_skill_updates or job_description is None:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            current_items = _parse_skill_items(
                change.current_text,
                require_unique=False,
            )
            proposed_items = _parse_skill_items(
                change.proposed_text,
                expected_keys=frozenset(current_items),
            )
            if not _skill_items_are_supported(
                current_items=current_items,
                proposed_items=proposed_items,
                job_description=job_description,
                master_resume=validated_evidence.master_resume,
            ):
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        else:
            role_ids = {item.role_id for item in cited}
            ownerships = {(item.employer, item.role) for item in cited}
            if len(role_ids) != 1 or len(ownerships) != 1:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            if target.target.section == "professional_experience" and (
                role_ids != {target.role_id}
                or ownerships != {(target.employer, target.role)}
                or not any(not item.evidence_id.endswith(":header") for item in cited)
            ):
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            if not _new_claims_are_supported(
                current_text=change.current_text,
                proposed_text=change.proposed_text,
                evidence_items=tuple(cited),
                summary_target=target.target.section == "professional_summary",
            ):
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)

    for change in response.changes:
        target = target_by_id[change.target.target_id]
        if target.target.section == "core_technical_skills":
            existing = _value_at_location(candidate, target.location)
            if type(existing) is not dict:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            updated = _clone_inert(existing)
            if type(updated) is not dict:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            for key, values in _parse_skill_items(change.proposed_text).items():
                updated[key] = list(values)
            _set_exact_location(candidate, target.location, updated)
        else:
            _set_exact_location(candidate, target.location, change.proposed_text)

    audit = {
        "schema_version": RESUME_VALIDATION_SCHEMA_VERSION,
        "is_valid": True,
        "requires_human_review": True,
        "change_count": len(response.changes),
        "changes": [
            {
                "change_id": change.change_id,
                "operation": change.operation,
                "target_id": change.target.target_id,
                "evidence_refs": list(change.evidence_refs),
            }
            for change in response.changes
        ],
    }
    return candidate, MappingProxyType(audit)


def _validate_workflow_boundary(
    *,
    store: ApplicationStateStore,
    paths: WorkspacePaths,
    job_id: str,
    model_config: CodexModelConfig,
    dry_run: bool,
) -> str:
    if type(store) is not ApplicationStateStore:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if type(paths) is not WorkspacePaths:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if type(model_config) is not CodexModelConfig or type(dry_run) is not bool:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    validated_id = _bounded_plain_string(
        job_id,
        max_chars=128,
        allow_empty=False,
    )
    if not _IDENTIFIER_RE.fullmatch(validated_id):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    # This must be the first capability invocation.  Exact path lstat/open and state
    # reads happen only after the hidden store binding succeeds.
    store.assert_workspace_binding(paths)
    for configured in (
        paths.database,
        paths.master_resume,
        paths.master_resume_text,
        paths.output_dir,
        paths.tmp_dir,
    ):
        _required_absolute_path(configured)
    _require_exact_directory(paths.output_dir)
    _require_exact_directory(paths.tmp_dir)
    return validated_id


def _run_patch_model(
    *,
    runner: ModelRunner,
    config: CodexModelConfig,
    prompt: str,
) -> tuple[ResumePatchResponse, Mapping[str, Any]]:
    try:
        result = runner.run(
            ModelRequest(
                prompt=prompt,
                config=config,
                timeout_seconds=120,
                max_attempts=1,
            )
        )
    except Exception:  # noqa: BLE001 - normalize an injected capability boundary.
        raise ResumeRefinementError(_WORKFLOW_ERROR) from None
    if type(result) is not ModelResult:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    parsed = parse_resume_patch_response(result.response)
    return parsed, _safe_model_metadata(result.model_metadata)


def _render_and_score(
    candidate: Mapping[str, Any],
    job_description: str,
    *,
    template_path: Path | None = None,
) -> tuple[str, bytes, AtsDiagnostics]:
    try:
        render_options = (
            {} if template_path is None else {"template_path": template_path}
        )
        html = render_resume_html_from_mapping(resume=candidate, **render_options)
        pdf = render_resume_pdf_from_html(html)
        diagnostics = calculate_ats_diagnostics(
            resume_pdf=pdf,
            job_description=job_description,
        )
    except Exception:  # noqa: BLE001 - P07 backend details stay content-free.
        raise ResumeRefinementError(_WORKFLOW_RENDER_ERROR) from None
    if type(html) is not str or type(pdf) is not bytes:
        raise ResumeRefinementError(_WORKFLOW_RENDER_ERROR)
    return html, pdf, diagnostics


def _calculate_ats(
    resume_pdf: bytes | None,
    job_description: str,
) -> AtsDiagnostics:
    if type(resume_pdf) is not bytes or not resume_pdf:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    try:
        return calculate_ats_diagnostics(
            resume_pdf=resume_pdf,
            job_description=job_description,
        )
    except Exception:  # noqa: BLE001 - P07 parser details stay content-free.
        raise ResumeRefinementError(_WORKFLOW_RENDER_ERROR) from None


def _variant_write(
    *,
    variant_key: str,
    variant_label: str,
    source: str,
    parent_variant_key: str,
    candidate: Mapping[str, Any],
    html: str,
    pdf: bytes,
    diagnostics: AtsDiagnostics,
    evidence: ResumeEvidenceSnapshot,
    audit: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    external_critique: str | None,
) -> ResumeVariantWrite:
    validated_evidence = _validate_evidence_snapshot_object(evidence)
    try:
        aro_yaml = yaml.safe_dump(
            _clone_resume(candidate),
            sort_keys=False,
            allow_unicode=True,
        )
    except Exception:  # noqa: BLE001 - YAML implementation details stay private.
        raise ResumeRefinementError(_WORKFLOW_RENDER_ERROR) from None
    score = diagnostics.score
    ats = AtsFields(
        score=score.overall_score,
        parsing_score=score.parsing_score,
        keyword_score=score.keyword_match_score,
        semantic_score=score.semantic_match_score,
        formatting_risk=score.formatting_risk,
        missing_terms=", ".join(score.missing_high_value_terms),
        diagnostics=_bounded_json_value(asdict(diagnostics)),
    )
    evidence_packet = {
        "schema_version": RESUME_EVIDENCE_SCHEMA_VERSION,
        "mro_sha256": validated_evidence.mro_sha256,
        "source_text_sha256": validated_evidence.source_text_sha256,
        "canonical_evidence_ids": [
            item.evidence_id for item in validated_evidence.items
        ],
    }
    critique = {
        "schema_version": RESUME_PATCH_SCHEMA_VERSION,
        "accepted_change_ids": [item["change_id"] for item in audit["changes"]],
    }
    external = (
        {
            "present": True,
            "sha256": hashlib.sha256(external_critique.encode("utf-8")).hexdigest(),
        }
        if external_critique is not None
        else None
    )
    return ResumeVariantWrite(
        variant_key=variant_key,
        variant_label=variant_label,
        source=source,
        parent_variant_key=parent_variant_key,
        application_resume_yaml=aro_yaml,
        resume_html=html,
        resume_pdf=pdf,
        ats=ats,
        ats_diagnostics=_bounded_json_value(asdict(diagnostics)),
        evidence_packet=evidence_packet,
        external_critique=external,
        critique_prompt=None,
        critique_response=None,
        critique=critique,
        validation=_clone_inert(audit),
        model_metadata=_clone_inert(model_metadata),
    )


def _workflow_model_metadata(
    metadata: Mapping[str, Any],
    *,
    workflow: str,
    review_state: str,
) -> Mapping[str, Any]:
    result = _clone_inert(metadata)
    if type(result) is not dict:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    result["workflow"] = workflow
    result["review_state"] = review_state
    result["requires_human_review"] = True
    return MappingProxyType(result)


def _safe_model_metadata(value: object) -> Mapping[str, Any]:
    try:
        copied = _materialize_inert(value)
    except (TypeError, ValueError, RecursionError):
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR) from None
    if type(copied) is not dict:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    allowed = {
        "workflow",
        "profile",
        "model",
        "reasoning_effort",
        "attempt",
        "timestamp",
        "version",
    }
    if not set(copied).issubset(allowed):
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    encoded = json.dumps(copied, sort_keys=True, separators=(",", ":"))
    if len(encoded) > 4_000:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    return MappingProxyType(copied)


def _collect_mro_evidence_items(
    master_resume: Mapping[str, Any],
) -> tuple[ResumeEvidenceItem, ...]:
    experience = master_resume.get("professional_experience")
    jobs = experience.get("jobs") if type(experience) is dict else None
    if type(jobs) is not list:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    items: list[ResumeEvidenceItem] = []
    seen_ids: set[str] = set()
    for job_index, job in enumerate(jobs, start=1):
        if type(job) is not dict:
            raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
        job_order = _evidence_order(job.get("order"), fallback=job_index)
        employer, role = _job_ownership(job)
        role_id = f"mro:job:{job_order}"
        header_text = " | ".join(part for part in (employer, role) if part)
        if header_text:
            item = ResumeEvidenceItem(
                evidence_id=f"{role_id}:header",
                role_id=role_id,
                employer=employer,
                role=role,
                text=header_text,
            )
            _append_evidence_item(items, seen_ids, item)
        bullets = job.get("bullet_points")
        if type(bullets) is not list:
            continue
        for bullet_index, bullet in enumerate(bullets, start=1):
            if type(bullet) is str:
                text = bullet
                bullet_order = str(bullet_index)
            elif type(bullet) is dict:
                text = bullet.get("text")
                bullet_order = _evidence_order(
                    bullet.get("order"),
                    fallback=bullet_index,
                )
            else:
                raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
            if type(text) is not str or not text:
                continue
            _validate_text(text, max_chars=MAX_EVIDENCE_ITEM_CHARS)
            item = ResumeEvidenceItem(
                evidence_id=f"{role_id}:bullet:{bullet_order}",
                role_id=role_id,
                employer=employer,
                role=role,
                text=text,
            )
            _append_evidence_item(items, seen_ids, item)
    if len(items) > MAX_EVIDENCE_ITEMS:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    return tuple(items)


def _append_evidence_item(
    items: list[ResumeEvidenceItem],
    seen_ids: set[str],
    item: ResumeEvidenceItem,
) -> None:
    if item.evidence_id in seen_ids:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    seen_ids.add(item.evidence_id)
    items.append(item)


def _new_claims_are_supported(
    *,
    current_text: str,
    proposed_text: str,
    evidence_items: tuple[ResumeEvidenceItem, ...],
    summary_target: bool,
) -> bool:
    if (
        type(current_text) is not str
        or type(proposed_text) is not str
        or type(evidence_items) is not tuple
        or not evidence_items
        or type(summary_target) is not bool
    ):
        return False
    evidence_text = "\n".join(item.text for item in evidence_items)
    proposed = _claim_text(proposed_text)
    evidence = _claim_text(evidence_text)
    proposed_tokens = _claim_tokens(proposed_text)
    if not proposed_tokens:
        return False
    evidence_token_sequences = tuple(
        _claim_tokens(item.text)
        for item in evidence_items
        if not item.evidence_id.endswith(":header")
    )
    exact_canonical = any(
        proposed_text == item.text
        for item in evidence_items
        if not item.evidence_id.endswith(":header")
    )
    safe_summary_extraction = summary_target and any(
        _is_safe_summary_nominal_extraction(proposed_text, item.text)
        for item in evidence_items
        if not item.evidence_id.endswith(":header")
    )
    safe_prepositional_movement = any(
        _is_safe_prepositional_phrase_movement(
            proposed_text,
            item.text,
            proposed_tokens,
            _claim_tokens(item.text),
        )
        for item in evidence_items
        if not item.evidence_id.endswith(":header")
    )
    if (
        not evidence_token_sequences
        or not exact_canonical
        and not safe_summary_extraction
        and not safe_prepositional_movement
    ):
        return False

    evidence_metrics = {
        _normalize_metric(value) for value in _METRIC_RE.findall(evidence_text)
    }
    for metric in _METRIC_RE.findall(proposed_text):
        normalized = _normalize_metric(metric)
        if normalized not in evidence_metrics:
            return False

    for phrase in _TECHNICAL_OR_REGULATED_PHRASES | _RESPONSIBILITY_OR_OUTCOME_PHRASES:
        if not _contains_phrase(proposed, phrase):
            continue
        if not _contains_phrase(evidence, phrase):
            return False

    evidence_acronyms = set(re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", evidence_text))
    for acronym in re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", proposed_text):
        if acronym not in evidence_acronyms:
            return False

    evidence_words = {_claim_text(word) for word in _WORD_RE.findall(evidence_text)}
    for match in _WORD_RE.finditer(proposed_text):
        word = match.group(0)
        new_word = _claim_text(word) not in evidence_words
        if (
            len(word) >= 2
            and (word.isupper() or any(char.isdigit() for char in word))
            and new_word
        ):
            return False
        if (
            match.start() > 0
            and word[0].isupper()
            and any(character.islower() for character in word[1:])
            and new_word
        ):
            return False
    return bool(evidence_items)


def _claim_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize(
            "NFC",
            match.group(0),
        ).casefold()
        for match in _CLAIM_TOKEN_RE.finditer(value)
    )


def _is_safe_summary_nominal_extraction(
    proposed_text: str,
    canonical_text: str,
) -> bool:
    """Remove one exact action prefix and preserve the complete remaining text."""

    if type(proposed_text) is not str or type(canonical_text) is not str:
        return False
    delimiter = canonical_text.find(" ")
    if delimiter <= 0 or delimiter == len(canonical_text) - 1:
        return False
    raw_prefix = canonical_text[:delimiter]
    remainder = canonical_text[delimiter + 1 :]
    if remainder.startswith(" "):
        return False

    if raw_prefix in _SUMMARY_EXTRACTABLE_PREFIXES:
        normalized_prefix = raw_prefix
    elif "A" <= raw_prefix[0] <= "Z" and raw_prefix[1:] == raw_prefix[1:].lower():
        normalized_prefix = raw_prefix[0].lower() + raw_prefix[1:]
    else:
        return False
    if normalized_prefix not in _SUMMARY_EXTRACTABLE_PREFIXES:
        return False

    expected = remainder
    if "a" <= remainder[0] <= "z":
        expected = chr(ord(remainder[0]) - 32) + remainder[1:]
    if proposed_text != expected:
        return False

    proposed = _claim_tokens(proposed_text)
    canonical = _claim_tokens(canonical_text)
    return not (
        len(proposed) < 3
        or len(canonical) != len(proposed) + 1
        or canonical[0] != normalized_prefix
        or proposed != canonical[1:]
        or any(token in _NEGATION_TOKENS for token in canonical)
        or any(token in _SUMMARY_ACTION_TOKENS for token in proposed)
    )


def _is_safe_prepositional_phrase_movement(
    proposed_text: str,
    canonical_text: str,
    proposed: tuple[str, ...],
    canonical: tuple[str, ...],
) -> bool:
    """Allow one exact comma-delimited trailing ``for`` adjunct movement."""

    if (
        type(proposed_text) is not str
        or type(canonical_text) is not str
        or len(canonical) < 6
        or len(proposed) != len(canonical)
        or canonical[0] not in _SUMMARY_ACTION_TOKENS
        or canonical.count("for") != 1
        or any(token in _NEGATION_TOKENS for token in canonical)
        or any(token in _CLAUSE_BOUNDARY_TOKENS for token in canonical)
    ):
        return False
    for index in range(3, len(canonical) - 1):
        if canonical[index] not in _MOVABLE_PREPOSITION_TOKENS:
            continue
        leading = canonical[:index]
        adjunct = canonical[index:]
        if (
            len(adjunct) < 2
            or any(token in _SUMMARY_ACTION_TOKENS for token in adjunct)
            or proposed != adjunct + leading
        ):
            continue
        terminal = "." if canonical_text.endswith(".") else ""
        canonical_body = canonical_text[:-1] if terminal else canonical_text
        if not _is_exact_movement_body(canonical_body):
            return False
        raw_atoms = tuple(canonical_body.split(" "))
        canonical_from_atoms: list[str] = []
        for atom in raw_atoms:
            tokens = _claim_tokens(atom)
            if len(tokens) != 1:
                return False
            canonical_from_atoms.append(tokens[0])
        if (
            tuple(canonical_from_atoms) != canonical
            or len(raw_atoms) != len(canonical)
            or raw_atoms[index] != "for"
        ):
            return False
        action_text = " ".join(raw_atoms[:index])
        adjunct_text = " ".join(raw_atoms[index:])
        if (
            not action_text
            or not adjunct_text.startswith("for ")
            or action_text[0] < "A"
            or action_text[0] > "Z"
            or canonical_body != f"{action_text} {adjunct_text}"
        ):
            return False
        expected = (
            f"F{adjunct_text[1:]}, {action_text[0].lower()}{action_text[1:]}{terminal}"
        )
        return proposed_text == expected
    return False


def _is_exact_movement_body(value: str) -> bool:
    if (
        not value
        or value.startswith(" ")
        or value.endswith(" ")
        or "  " in value
        or any(character.isspace() and character != " " for character in value)
    ):
        return False
    return all(_MOVEMENT_ATOM_RE.fullmatch(atom) for atom in value.split(" "))


def _validate_patch_response_object(value: object) -> None:
    if (
        type(value) is not ResumePatchResponse
        or type(value.schema_version) is not str
        or value.schema_version != RESUME_PATCH_SCHEMA_VERSION
        or type(value.changes) is not tuple
        or len(value.changes) > MAX_PATCH_CHANGES
    ):
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    change_ids: set[str] = set()
    target_ids: set[str] = set()
    for change in value.changes:
        if (
            type(change) is not ResumePatch
            or type(change.change_id) is not str
            or type(change.operation) is not str
            or type(change.target) is not ResumePatchTarget
            or type(change.current_text) is not str
            or type(change.proposed_text) is not str
            or type(change.rationale) is not str
            or type(change.evidence_refs) is not tuple
            or not change.evidence_refs
            or len(change.evidence_refs) > MAX_EVIDENCE_REFS_PER_CHANGE
            or any(type(ref) is not str for ref in change.evidence_refs)
        ):
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        target = change.target
        if (
            type(target.section) is not str
            or type(target.field) is not str
            or (target.job_order is not None and type(target.job_order) is not str)
            or (
                target.bullet_order is not None and type(target.bullet_order) is not str
            )
        ):
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        if (
            not _IDENTIFIER_RE.fullmatch(change.change_id)
            or change.change_id in change_ids
            or change.operation
            not in {"rewrite_summary", "rewrite_bullet", "replace_skill_items"}
            or not change.current_text
            or len(change.current_text) > MAX_PATCH_TEXT_CHARS
            or not change.proposed_text
            or len(change.proposed_text) > MAX_PATCH_TEXT_CHARS
            or not change.rationale
            or len(change.rationale) > MAX_PATCH_RATIONALE_CHARS
            or any(
                _has_forbidden_control(item)
                for item in (
                    change.change_id,
                    change.current_text,
                    change.proposed_text,
                    change.rationale,
                )
            )
            or any(
                not ref or len(ref) > MAX_PATCH_ID_CHARS or _has_forbidden_control(ref)
                for ref in change.evidence_refs
            )
            or len(set(change.evidence_refs)) != len(change.evidence_refs)
        ):
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        if target.section == "professional_summary":
            target_valid = (
                change.operation == "rewrite_summary"
                and target.field in {"paragraph", "text"}
                and target.job_order is None
                and target.bullet_order is None
            )
        elif target.section == "professional_experience":
            target_valid = (
                change.operation == "rewrite_bullet"
                and target.field == "text"
                and type(target.job_order) is str
                and bool(target.job_order)
                and len(target.job_order) <= 128
                and type(target.bullet_order) is str
                and bool(target.bullet_order)
                and len(target.bullet_order) <= 128
                and not _has_forbidden_control(target.job_order)
                and not _has_forbidden_control(target.bullet_order)
            )
        elif target.section == "core_technical_skills":
            target_valid = (
                change.operation == "replace_skill_items"
                and target.field == "items"
                and type(target.job_order) is str
                and 0 < len(target.job_order) <= 128
                and target.bullet_order is None
                and not _has_forbidden_control(target.job_order)
            )
        else:
            target_valid = False
        if not target_valid or target.target_id in target_ids:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        change_ids.add(change.change_id)
        target_ids.add(target.target_id)


def _validate_evidence_snapshot_object(
    value: object,
) -> ResumeEvidenceSnapshot:
    validation_failed = False
    try:
        if type(value) is not ResumeEvidenceSnapshot:
            raise ValueError
        (
            master_resume,
            source_text,
            mro_sha256,
            source_text_sha256,
            items,
        ) = _snapshot_public_evidence_fields(value)
        copied_items = _snapshot_evidence_items(items)
        if (
            type(master_resume) is not MappingProxyType
            or type(source_text) is not str
            or type(mro_sha256) is not str
            or type(source_text_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", mro_sha256) is None
            or re.fullmatch(r"[0-9a-f]{64}", source_text_sha256) is None
            or len(source_text.encode("utf-8", errors="strict"))
            > MAX_EVIDENCE_SOURCE_BYTES
        ):
            raise ValueError
        _validate_text(source_text, max_chars=MAX_EVIDENCE_SOURCE_BYTES)
        materialized_master = _materialize_inert(master_resume)
        if type(materialized_master) is not dict:
            raise ValueError
        serialized_master = yaml.safe_dump(
            materialized_master,
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8", errors="strict")
        if len(serialized_master) > MAX_EVIDENCE_YAML_BYTES:
            raise ValueError
        derived_items = _collect_mro_evidence_items(materialized_master)
        if not _exact_evidence_items_equal(copied_items, derived_items):
            raise ValueError
        serialized_items = json.dumps(
            [_evidence_prompt_item(item) for item in copied_items],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized_items) > MAX_EVIDENCE_PACKET_CHARS:
            raise ValueError
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        yaml.YAMLError,
        ResumeEvidenceError,
        ResumePatchError,
        RecursionError,
        RuntimeError,
    ):
        validation_failed = True
    if validation_failed:
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)

    return value


def _snapshot_public_evidence_fields(
    value: ResumeEvidenceSnapshot,
) -> tuple[object, object, object, object, object]:
    try:
        return (
            object.__getattribute__(value, "master_resume"),
            object.__getattribute__(value, "source_text"),
            object.__getattribute__(value, "mro_sha256"),
            object.__getattribute__(value, "source_text_sha256"),
            object.__getattribute__(value, "items"),
        )
    except (AttributeError, TypeError):
        raise ValueError from None


def _snapshot_evidence_items(
    value: object,
) -> tuple[ResumeEvidenceItem, ...]:
    if type(value) is not tuple or len(value) > MAX_EVIDENCE_ITEMS:
        raise ValueError
    copied: list[ResumeEvidenceItem] = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not ResumeEvidenceItem:
            raise ValueError
        try:
            fields = (
                object.__getattribute__(item, "evidence_id"),
                object.__getattribute__(item, "role_id"),
                object.__getattribute__(item, "employer"),
                object.__getattribute__(item, "role"),
                object.__getattribute__(item, "text"),
            )
        except (AttributeError, TypeError):
            raise ValueError from None
        evidence_id, role_id, employer, role, text = fields
        if (
            any(type(field_value) is not str for field_value in fields)
            or not evidence_id
            or len(evidence_id) > MAX_PATCH_ID_CHARS
            or evidence_id in seen
            or not role_id
            or len(role_id) > MAX_PATCH_ID_CHARS
            or not employer
            or len(employer) > 1_024
            or not role
            or len(role) > 1_024
            or not text
            or len(text) > MAX_EVIDENCE_ITEM_CHARS
            or any(_has_forbidden_control(field_value) for field_value in fields)
        ):
            raise ValueError
        seen.add(evidence_id)
        copied.append(
            ResumeEvidenceItem(
                evidence_id=evidence_id,
                role_id=role_id,
                employer=employer,
                role=role,
                text=text,
            )
        )
    return tuple(copied)


def _exact_evidence_items_equal(
    left: tuple[ResumeEvidenceItem, ...],
    right: tuple[ResumeEvidenceItem, ...],
) -> bool:
    return len(left) == len(right) and all(
        (
            left_item.evidence_id,
            left_item.role_id,
            left_item.employer,
            left_item.role,
            left_item.text,
        )
        == (
            right_item.evidence_id,
            right_item.role_id,
            right_item.employer,
            right_item.role,
            right_item.text,
        )
        for left_item, right_item in zip(left, right, strict=True)
    )


def _strict_json_object(response_text: str) -> dict[str, Any]:
    if type(response_text) is not str:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    if not response_text or len(response_text) > MAX_MODEL_RESPONSE_CHARS:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    if response_text != response_text.strip():
        # JSON whitespace around the object is harmless but accepting it makes prose
        # and fence extraction mistakes harder to distinguish.  Keep the boundary exact.
        response_text = response_text.strip()
    if not response_text.startswith("{") or not response_text.endswith("}"):
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    try:
        payload = json.loads(
            response_text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, RecursionError):
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR) from None
    if type(payload) is not dict:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    _validate_json_tree(payload)
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _validate_json_tree(root: object) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        value_type = type(value)
        if value is None or value_type is bool or value_type is int:
            continue
        if value_type is float:
            if not math.isfinite(value):
                raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
            continue
        if value_type is str:
            if len(value) > MAX_JSON_STRING_CHARS or _has_forbidden_control(value):
                raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
            continue
        if value_type is list:
            if len(value) > MAX_JSON_COLLECTION_ITEMS:
                raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
            stack.extend((item, depth + 1) for item in value)
            continue
        if value_type is dict:
            if len(value) > MAX_JSON_COLLECTION_ITEMS:
                raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
            for key, item in value.items():
                if (
                    type(key) is not str
                    or len(key) > MAX_PATCH_ID_CHARS
                    or _has_forbidden_control(key)
                ):
                    raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
                stack.append((item, depth + 1))
            continue
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)


def _parse_patch_target(value: object) -> ResumePatchTarget:
    if type(value) is not dict or set(value) != {
        "section",
        "field",
        "job_order",
        "bullet_order",
    }:
        raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
    section = value["section"]
    field_name = value["field"]
    if section == "professional_summary":
        if field_name not in {"paragraph", "text"}:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        if value["job_order"] is not None or value["bullet_order"] is not None:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        return ResumePatchTarget(
            section="professional_summary",
            field=field_name,
            job_order=None,
            bullet_order=None,
        )
    if section == "professional_experience":
        if field_name != "text":
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        job_order = _bounded_plain_string(
            value["job_order"],
            max_chars=128,
            allow_empty=False,
        )
        bullet_order = _bounded_plain_string(
            value["bullet_order"],
            max_chars=128,
            allow_empty=False,
        )
        return ResumePatchTarget(
            section="professional_experience",
            field="text",
            job_order=job_order,
            bullet_order=bullet_order,
        )
    if section == "core_technical_skills":
        if field_name != "items" or value["bullet_order"] is not None:
            raise ResumePatchError(_WORKFLOW_MODEL_ERROR)
        job_order = _bounded_plain_string(
            value["job_order"],
            max_chars=128,
            allow_empty=False,
        )
        return ResumePatchTarget(
            section="core_technical_skills",
            field="items",
            job_order=job_order,
            bullet_order=None,
        )
    raise ResumePatchError(_WORKFLOW_MODEL_ERROR)


def _target_payload(target: ResumePatchTarget) -> dict[str, str | None]:
    return {
        "section": target.section,
        "field": target.field,
        "job_order": target.job_order,
        "bullet_order": target.bullet_order,
    }


def _validated_prompt_target_id(target: ResumePatchTarget) -> str:
    if (
        type(target.section) is not str
        or type(target.field) is not str
        or (target.job_order is not None and type(target.job_order) is not str)
        or (target.bullet_order is not None and type(target.bullet_order) is not str)
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if target.section == "professional_summary":
        valid = (
            target.field in {"paragraph", "text"}
            and target.job_order is None
            and target.bullet_order is None
        )
    elif target.section == "professional_experience":
        valid = (
            target.field == "text"
            and type(target.job_order) is str
            and 0 < len(target.job_order) <= 128
            and type(target.bullet_order) is str
            and 0 < len(target.bullet_order) <= 128
            and not _has_forbidden_control(target.job_order)
            and not _has_forbidden_control(target.bullet_order)
        )
    elif target.section == "core_technical_skills":
        valid = (
            target.field == "items"
            and type(target.job_order) is str
            and 0 < len(target.job_order) <= 128
            and target.bullet_order is None
            and not _has_forbidden_control(target.job_order)
        )
    else:
        valid = False
    if not valid:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return target.target_id


def _variant_map(
    snapshot: ApplicationWorkflowSnapshot,
) -> dict[str, ResumeVariantRecord]:
    if (
        type(snapshot) is not ApplicationWorkflowSnapshot
        or type(snapshot.variants) is not tuple
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    result: dict[str, ResumeVariantRecord] = {}
    for variant in snapshot.variants:
        if (
            type(variant) is not ResumeVariantRecord
            or type(variant.variant_key) is not str
            or variant.variant_key in result
        ):
            raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
        result[variant.variant_key] = variant
    return result


def _snapshot_job_description(snapshot: ApplicationWorkflowSnapshot) -> str:
    if (
        type(snapshot) is not ApplicationWorkflowSnapshot
        or type(snapshot.application) is not ApplicationRecord
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    prompt_value = snapshot.application.prompt_job_description
    source_value = snapshot.application.job_description
    if (
        prompt_value is not None
        and type(prompt_value) is not str
        or source_value is not None
        and type(source_value) is not str
    ):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    value = prompt_value if prompt_value else source_value
    if type(value) is not str or not value or len(value) > 100_000:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    _validate_text(value, max_chars=100_000)
    return value


def _assert_evidence_unchanged(
    paths: WorkspacePaths,
    expected: ResumeEvidenceSnapshot,
) -> None:
    expected_invalid = False
    try:
        validated_expected = _validate_evidence_snapshot_object(expected)
    except ResumePatchError:
        expected_invalid = True
    if expected_invalid:
        raise ResumeWorkflowConflictError(_WORKFLOW_CONFLICT_ERROR)

    current_invalid = False
    try:
        validated_current = _validate_evidence_snapshot_object(
            read_resume_evidence_snapshot(paths)
        )
    except (ResumeEvidenceError, ResumePatchError):
        current_invalid = True
    if current_invalid:
        raise ResumeWorkflowConflictError(_WORKFLOW_CONFLICT_ERROR)
    if (
        validated_current.mro_sha256 != validated_expected.mro_sha256
        or validated_current.source_text_sha256 != validated_expected.source_text_sha256
    ):
        raise ResumeWorkflowConflictError(_WORKFLOW_CONFLICT_ERROR)


def _required_absolute_path(value: object) -> Path:
    if type(value) is not _PATH_TYPE or not value.is_absolute():
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return value


def _require_exact_directory(value: object) -> Path:
    path = _required_absolute_path(value)
    try:
        metadata = path.lstat()
    except (OSError, ValueError):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR) from None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return path


def _read_regular_file(path: Path, max_bytes: int) -> bytes:
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        visible = os.stat(path, follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (visible.st_dev, visible.st_ino) != identity
            or not stat.S_ISREG(visible.st_mode)
            or metadata.st_size > max_bytes
        ):
            raise OSError
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        visible = os.stat(path, follow_symlinks=False)
        if (visible.st_dev, visible.st_ino) != identity or not stat.S_ISREG(
            visible.st_mode
        ):
            raise OSError
    except (OSError, ValueError):
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if type(payload) is not bytes or len(payload) > max_bytes:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    return payload


def _decode_text(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR) from None


def _optional_external_critique(value: object) -> str | None:
    if value is None:
        return None
    return _bounded_plain_string(
        value,
        max_chars=MAX_EXTERNAL_CRITIQUE_CHARS,
        allow_empty=True,
    )


def _bounded_plain_string(
    value: object,
    *,
    max_chars: int,
    allow_empty: bool,
) -> str:
    if type(value) is not str:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if len(value) > max_chars or (not allow_empty and not value):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    if _has_forbidden_control(value):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return value


def _validate_text(value: str, *, max_chars: int) -> None:
    if type(value) is not str or len(value) > max_chars:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    if _has_forbidden_control(value, allow_layout=True):
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)


def _has_forbidden_control(value: str, *, allow_layout: bool = False) -> bool:
    allowed = {"\t", "\n", "\r"} if allow_layout else set()
    return any(
        character not in allowed
        and (
            ord(character) <= 0x1F
            or 0x7F <= ord(character) <= 0x9F
            or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        )
        for character in value
    )


def _exact_order(value: object) -> str:
    if type(value) is int:
        if value < 0:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        return str(value)
    if type(value) is str and value and len(value) <= 128:
        if _has_forbidden_control(value):
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        return value
    raise ResumePatchError(_WORKFLOW_PATCH_ERROR)


def _evidence_order(value: object, *, fallback: int) -> str:
    if value is None:
        return str(fallback)
    try:
        return _exact_order(value)
    except ResumePatchError:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR) from None


def _job_ownership(job: Mapping[str, Any]) -> tuple[str, str]:
    line_1 = job.get("line_1")
    if type(line_1) is not dict:
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    employer = line_1.get("company_name_text")
    role = line_1.get("position_name_text")
    if (
        type(employer) is not str
        or type(role) is not str
        or not employer
        or not role
        or len(employer) > 1_024
        or len(role) > 1_024
        or _has_forbidden_control(employer)
        or _has_forbidden_control(role)
    ):
        raise ResumeEvidenceError(_WORKFLOW_EVIDENCE_ERROR)
    return employer, role


def _is_rendered(value: Mapping[str, Any]) -> bool:
    raw = value.get("render")
    if raw is None:
        return True
    if type(raw) is not bool:
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    return raw


def _evidence_prompt_item(item: ResumeEvidenceItem) -> dict[str, str]:
    return {
        "evidence_id": item.evidence_id,
        "role_id": item.role_id,
        "employer": item.employer,
        "role": item.role,
        "text": item.text,
    }


def _set_exact_location(
    resume: dict[str, Any],
    location: tuple[object, ...],
    value: object,
) -> None:
    current: object = resume
    for part in location[:-1]:
        if type(part) is int:
            if type(current) is not list:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            current = current[part]
        else:
            if type(current) is not dict or type(part) is not str:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            current = current[part]
    final = location[-1]
    if type(current) is not dict or type(final) is not str:
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    current[final] = value


def _value_at_location(
    resume: dict[str, Any],
    location: tuple[object, ...],
) -> object:
    current: object = resume
    for part in location:
        if type(part) is int:
            if type(current) is not list or not 0 <= part < len(current):
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            current = current[part]
        else:
            if (
                type(current) is not dict
                or type(part) is not str
                or part not in current
            ):
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            current = current[part]
    return current


def _parse_skill_items(
    value: str,
    *,
    expected_keys: frozenset[str] | None = None,
    require_unique: bool = True,
) -> dict[str, tuple[str, ...]]:
    loaded = _strict_json_object(value)
    if (
        type(loaded) is not dict
        or not loaded
        or not set(loaded).issubset({"primary", "additional"})
        or expected_keys is not None
        and set(loaded) != expected_keys
    ):
        raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    parsed: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for key in ("primary", "additional"):
        if key not in loaded:
            continue
        values = loaded[key]
        if type(values) is not list or len(values) > MAX_JSON_COLLECTION_ITEMS:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        items: list[str] = []
        for item in values:
            parsed_item = _bounded_plain_string(
                item,
                max_chars=1_024,
                allow_empty=False,
            )
            normalized = _claim_text(parsed_item)
            if require_unique and normalized in seen:
                raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
            seen.add(normalized)
            items.append(parsed_item)
        parsed[key] = tuple(items)
    return parsed


def _skill_items_are_supported(
    *,
    current_items: Mapping[str, tuple[str, ...]],
    proposed_items: Mapping[str, tuple[str, ...]],
    job_description: str,
    master_resume: Mapping[str, Any],
) -> bool:
    current_evidence = tuple(
        item for values in current_items.values() for item in values
    )
    canonical_evidence = tuple(_iter_resume_strings(master_resume))
    normalized_job = _claim_text(job_description)
    for proposed in (item for values in proposed_items.values() for item in values):
        normalized = _claim_text(proposed)
        eligible = any(
            _contains_phrase(_claim_text(existing), normalized)
            for existing in current_evidence
        ) or _contains_phrase(normalized_job, normalized)
        supported = any(
            _contains_phrase(_claim_text(evidence), normalized)
            for evidence in (*current_evidence, *canonical_evidence)
        )
        if not eligible or not supported:
            return False
    return True


def _iter_resume_strings(value: object) -> tuple[str, ...]:
    result: list[str] = []
    stack: list[object] = [value]
    remaining = 100_000
    while stack:
        remaining -= 1
        if remaining < 0:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
        current = stack.pop()
        if type(current) is str:
            result.append(current)
        elif type(current) in {list, tuple}:
            stack.extend(current)
        elif type(current) in {dict, MappingProxyType}:
            stack.extend(current.values())
        elif current is not None and type(current) not in {bool, int, float, bytes}:
            raise ResumePatchError(_WORKFLOW_PATCH_ERROR)
    return tuple(result)


def _materialize_inert(
    value: object,
    *,
    depth: int = 0,
    remaining: list[int] | None = None,
) -> Any:
    if remaining is None:
        remaining = [100_000]
    remaining[0] -= 1
    if depth > 64 or remaining[0] < 0:
        raise ValueError
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str, bytes}:
        return value
    if value_type is list or value_type is tuple:
        if len(value) > 100_000:
            raise ValueError
        return [
            _materialize_inert(
                item,
                depth=depth + 1,
                remaining=remaining,
            )
            for item in value
        ]
    if value_type is dict or value_type is MappingProxyType:
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError
            result[key] = _materialize_inert(
                item,
                depth=depth + 1,
                remaining=remaining,
            )
        return result
    raise ValueError


def _clone_inert(value: object) -> Any:
    try:
        return _materialize_inert(value)
    except (TypeError, ValueError, RecursionError):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR) from None


def _clone_resume(value: object) -> dict[str, Any]:
    copied = _clone_inert(value)
    if type(copied) is not dict:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return copied


def _freeze(value: object) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str, bytes}:
        return value
    if value_type is list or value_type is tuple:
        return tuple(_freeze(item) for item in value)
    if value_type is dict or value_type is MappingProxyType:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)


def _bounded_json_value(value: object) -> Any:
    copied = _clone_inert(value)
    try:
        encoded = json.dumps(copied, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR) from None
    if len(encoded) > 100_000:
        raise ResumeRefinementError(_WORKFLOW_INPUT_ERROR)
    return copied


def _claim_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+#/%]+", " ", value.casefold()).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = _claim_text(phrase)
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
            text,
        )
        is not None
    )


def _normalize_metric(value: str) -> str:
    return value.casefold().replace(",", "")
