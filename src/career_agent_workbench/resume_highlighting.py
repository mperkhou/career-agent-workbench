"""Exact-target ``<strong>`` highlighting without claim mutation."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from career_agent_workbench.application_state import (
    ApplicationRecord,
    ApplicationStateStore,
    ApplicationWorkflowSnapshot,
    AtsFields,
    ResumeVariantRecord,
    ResumeVariantWrite,
)
from career_agent_workbench.codex_cli import (
    CodexModelConfig,
    ModelRequest,
    ModelResult,
    ModelRunner,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.resume_refinement import (
    ResumeRefinementError,
    _assert_evidence_unchanged,
    _bounded_json_value,
    _clone_inert,
    _clone_resume,
    _exact_order,
    _freeze,
    _has_forbidden_control,
    _is_rendered,
    _render_and_score,
    _safe_model_metadata,
    _set_exact_location,
    _snapshot_job_description,
    _strict_json_object,
    _validate_workflow_boundary,
    _variant_map,
    read_resume_evidence_snapshot,
)

HIGHLIGHT_RESPONSE_SCHEMA_VERSION = "governed_resume_highlighting.v1"
HIGHLIGHT_AUDIT_SCHEMA_VERSION = "governed_resume_highlighting_audit.v1"
DEFAULT_MAX_STRONG_SPANS_PER_BULLET = 3
MAX_TOTAL_STRONG_SPANS = 128
MAX_HIGHLIGHT_BULLETS = 128
MAX_HIGHLIGHT_TEXT_CHARS = 12_000
MAX_HIGHLIGHT_PROMPT_CHARS = 240_000

_HIGHLIGHT_ERROR = "Resume highlighting could not be completed."
_HIGHLIGHT_INPUT_ERROR = "Resume highlighting input is invalid."
_HIGHLIGHT_RESPONSE_ERROR = "Resume highlighting response is invalid."


class ResumeHighlightError(ResumeRefinementError):
    """Raised when exact highlighting cannot be applied atomically."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class HighlightBullet:
    """One exact rendered professional-experience target."""

    target_id: str
    job_order: str
    bullet_order: str
    text: str = field(repr=False)
    location: tuple[object, ...] = field(repr=False)

    def __repr__(self) -> str:
        return f"HighlightBullet(target_id={self.target_id!r}, content_hidden=True)"


@dataclass(frozen=True, slots=True)
class HighlightUpdate:
    """One strict model-proposed wrapper insertion."""

    target_id: str
    current_text: str = field(repr=False)
    highlighted_text: str = field(repr=False)

    def __repr__(self) -> str:
        return f"HighlightUpdate(target_id={self.target_id!r}, content_hidden=True)"


@dataclass(frozen=True, slots=True)
class HighlightResponse:
    """One complete exact update set."""

    updates: tuple[HighlightUpdate, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"HighlightResponse(update_count={len(self.updates)}, content_hidden=True)"
        )


@dataclass(frozen=True, slots=True)
class HighlightStats:
    """Bounded non-content highlighting counts."""

    bullet_count: int
    strong_span_count: int


@dataclass(frozen=True, slots=True)
class HighlightResult:
    """Content-hidden proposal or committed target-variant update."""

    target_variant: str
    bullet_count: int
    strong_span_count: int
    ats_score: int
    dry_run: bool
    requires_human_review: bool = True
    candidate: Mapping[str, Any] = field(repr=False, default_factory=dict)
    job_id: str = field(repr=False, default="")

    def __repr__(self) -> str:
        return (
            "HighlightResult("
            f"target_variant={self.target_variant!r}, "
            f"bullet_count={self.bullet_count}, "
            f"strong_span_count={self.strong_span_count}, "
            f"ats_score={self.ats_score}, dry_run={self.dry_run}, "
            "requires_human_review=True, content_hidden=True)"
        )


def highlight_resume_for_job(
    *,
    store: ApplicationStateStore,
    paths: WorkspacePaths,
    job_id: str,
    runner: ModelRunner,
    model_config: CodexModelConfig,
    variant_override: str | None = None,
    following_variant: str | None = None,
    combined_variants: tuple[str, ...] = (),
    dry_run: bool = False,
) -> HighlightResult:
    """Apply exact wrapper-only emphasis to one snapshotted existing variant."""

    validated_id = _validate_workflow_boundary(
        store=store,
        paths=paths,
        job_id=job_id,
        model_config=model_config,
        dry_run=dry_run,
    )
    snapshot = store.get_workflow_snapshot(validated_id)
    evidence = read_resume_evidence_snapshot(paths)
    target_key = resolve_highlight_target(
        snapshot,
        variant_override=variant_override,
        following_variant=following_variant,
        combined_variants=combined_variants,
    )
    variants = _variant_map(snapshot)
    target_variant = variants[target_key]
    base_resume = _clone_resume(target_variant.application_resume)
    bullets = collect_highlight_bullets(base_resume)
    prompt = build_resume_highlight_prompt(
        snapshot=snapshot,
        target_variant=target_variant,
        bullets=bullets,
    )
    response, logical_metadata = _run_highlight_model(
        runner=runner,
        config=model_config,
        prompt=prompt,
    )
    candidate, stats = apply_highlight_response(
        base_resume,
        response,
        max_strong_spans_per_bullet=DEFAULT_MAX_STRONG_SPANS_PER_BULLET,
    )
    job_description = _snapshot_job_description(snapshot)
    html, pdf, diagnostics = _render_and_score(candidate, job_description)
    result = HighlightResult(
        target_variant=target_key,
        bullet_count=stats.bullet_count,
        strong_span_count=stats.strong_span_count,
        ats_score=diagnostics.score.overall_score,
        dry_run=dry_run,
        candidate=_freeze(candidate),
        job_id=validated_id,
    )
    if dry_run:
        return result

    write = _highlight_variant_write(
        original=target_variant,
        candidate=candidate,
        html=html,
        pdf=pdf,
        diagnostics=diagnostics,
        logical_metadata=logical_metadata,
        stats=stats,
        mro_sha256=evidence.mro_sha256,
        source_text_sha256=evidence.source_text_sha256,
    )
    _assert_evidence_unchanged(paths, evidence)
    store.upsert_resume_variant_if_revision(
        validated_id,
        write,
        expected_revision=snapshot.revision,
    )
    return result


def run_resume_highlighting(**kwargs: Any) -> HighlightResult:
    """Keyword-only compatibility spelling for the callable domain workflow."""

    return highlight_resume_for_job(**kwargs)


def resolve_highlight_target(
    snapshot: ApplicationWorkflowSnapshot,
    *,
    variant_override: str | None = None,
    following_variant: str | None = None,
    combined_variants: tuple[str, ...] = (),
) -> str:
    """Resolve one deterministic target without reading state a second time."""

    variants = _variant_map(snapshot)
    allowed = {"v1", "v2", "manual"}
    if type(combined_variants) is not tuple:
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    for item in combined_variants:
        if type(item) is not str or item not in {"v2", "manual"}:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    if len(set(combined_variants)) != len(combined_variants):
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    if combined_variants:
        if variant_override is not None or following_variant is not None:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        resolved = "manual" if "manual" in combined_variants else "v2"
    elif following_variant is not None:
        if variant_override is not None:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        if type(following_variant) is not str or following_variant not in {
            "v2",
            "manual",
        }:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        resolved = following_variant
    elif variant_override is not None:
        if type(variant_override) is not str or variant_override not in allowed:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        resolved = variant_override
    else:
        resolved = snapshot.application.selected_resume_variant
        if type(resolved) is not str or resolved not in allowed:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    if resolved not in variants:
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    return resolved


def resolve_highlight_variant(
    snapshot: ApplicationWorkflowSnapshot,
    **kwargs: Any,
) -> str:
    """Compatibility spelling for deterministic target resolution."""

    return resolve_highlight_target(snapshot, **kwargs)


def collect_highlight_bullets(
    application_resume: Mapping[str, Any],
) -> tuple[HighlightBullet, ...]:
    """Collect exact-ID rendered bullets and reject ambiguous identities."""

    resume = _clone_resume(application_resume)
    experience = resume.get("professional_experience")
    jobs = experience.get("jobs") if type(experience) is dict else None
    if type(jobs) is not list:
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    results: list[HighlightBullet] = []
    seen_jobs: set[str] = set()
    seen_targets: set[str] = set()
    for job_index, job in enumerate(jobs):
        if type(job) is not dict:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        if not _is_rendered(job):
            continue
        try:
            job_order = _exact_order(job.get("order"))
        except ResumeRefinementError:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR) from None
        if job_order in seen_jobs:
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        seen_jobs.add(job_order)
        bullets = job.get("bullet_points")
        if type(bullets) is not list:
            continue
        seen_bullets: set[str] = set()
        for bullet_index, bullet in enumerate(bullets):
            if type(bullet) is not dict:
                raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
            if not _is_rendered(bullet):
                continue
            try:
                bullet_order = _exact_order(bullet.get("order"))
            except ResumeRefinementError:
                raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR) from None
            if bullet_order in seen_bullets:
                raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
            seen_bullets.add(bullet_order)
            text = bullet.get("text")
            if (
                type(text) is not str
                or not text
                or len(text) > MAX_HIGHLIGHT_TEXT_CHARS
                or _has_forbidden_control(text)
            ):
                raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
            target_id = f"experience:{job_order}:bullet:{bullet_order}"
            if target_id in seen_targets:
                raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
            seen_targets.add(target_id)
            results.append(
                HighlightBullet(
                    target_id=target_id,
                    job_order=job_order,
                    bullet_order=bullet_order,
                    text=text,
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
    if not results or len(results) > MAX_HIGHLIGHT_BULLETS:
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    return tuple(results)


def build_resume_highlight_prompt(
    *,
    snapshot: ApplicationWorkflowSnapshot,
    target_variant: ResumeVariantRecord,
    bullets: tuple[HighlightBullet, ...],
) -> str:
    """Build one bounded exact-wrapper request."""

    if (
        type(snapshot) is not ApplicationWorkflowSnapshot
        or type(snapshot.application) is not ApplicationRecord
        or type(snapshot.application.company) is not str
        or type(snapshot.application.job_title) is not str
        or type(target_variant) is not ResumeVariantRecord
        or type(target_variant.variant_key) is not str
        or type(bullets) is not tuple
        or not bullets
        or len(bullets) > MAX_HIGHLIGHT_BULLETS
    ):
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    if (
        len(snapshot.application.company) > 1_024
        or len(snapshot.application.job_title) > 1_024
        or _has_forbidden_control(snapshot.application.company)
        or _has_forbidden_control(snapshot.application.job_title)
        or len(target_variant.variant_key) > 128
        or _has_forbidden_control(target_variant.variant_key)
    ):
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    seen: set[str] = set()
    for bullet in bullets:
        if (
            type(bullet) is not HighlightBullet
            or type(bullet.target_id) is not str
            or type(bullet.job_order) is not str
            or type(bullet.bullet_order) is not str
            or type(bullet.text) is not str
            or type(bullet.location) is not tuple
            or any(type(part) not in {str, int} for part in bullet.location)
            or not bullet.target_id
            or len(bullet.target_id) > 384
            or bullet.target_id in seen
            or not bullet.text
            or len(bullet.text) > MAX_HIGHLIGHT_TEXT_CHARS
            or any(
                _has_forbidden_control(item)
                for item in (
                    bullet.target_id,
                    bullet.job_order,
                    bullet.bullet_order,
                    bullet.text,
                )
            )
        ):
            raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
        seen.add(bullet.target_id)
    payload = {
        "job": {
            "company": snapshot.application.company,
            "title": snapshot.application.job_title,
            "job_description": _snapshot_job_description(snapshot),
        },
        "target_variant": target_variant.variant_key,
        "bullets": [
            {
                "target_id": bullet.target_id,
                "current_text": bullet.text,
            }
            for bullet in bullets
        ],
    }
    prompt = (
        "Return only one strict JSON object. Return exactly one update for every "
        "input target and no extras. Preserve every input code point. The only "
        "permitted change is insertion of literal lowercase <strong> and </strong> "
        "pairs. Use one to three balanced, nonnested spans per bullet. Do not use "
        "attributes, other tags, comments, entities to alter bytes, controls, or "
        "Unicode tag lookalikes. A span must be nonempty, contain an alphanumeric "
        "character, and must not cover the whole bullet. Schema: "
        '{"schema_version":"governed_resume_highlighting.v1","updates":['
        '{"target_id":"experience:1:bullet:1","current_text":"exact input",'
        '"highlighted_text":"exact <strong>input phrase</strong>"}]}. Payload: '
        + json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    if len(prompt) > MAX_HIGHLIGHT_PROMPT_CHARS:
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    return prompt


def parse_highlight_response(response_text: str) -> HighlightResponse:
    """Parse one strict duplicate-free highlight response."""

    payload = _strict_json_object(response_text)
    if set(payload) != {"schema_version", "updates"}:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    if payload["schema_version"] != HIGHLIGHT_RESPONSE_SCHEMA_VERSION:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    updates = payload["updates"]
    if type(updates) is not list or not updates or len(updates) > MAX_HIGHLIGHT_BULLETS:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    result: list[HighlightUpdate] = []
    seen: set[str] = set()
    for update in updates:
        if type(update) is not dict or set(update) != {
            "target_id",
            "current_text",
            "highlighted_text",
        }:
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        target_id = update["target_id"]
        current = update["current_text"]
        highlighted = update["highlighted_text"]
        if (
            type(target_id) is not str
            or not target_id
            or len(target_id) > 384
            or target_id in seen
            or type(current) is not str
            or not current
            or len(current) > MAX_HIGHLIGHT_TEXT_CHARS
            or type(highlighted) is not str
            or not highlighted
            or len(highlighted) > MAX_HIGHLIGHT_TEXT_CHARS + 256
        ):
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        seen.add(target_id)
        result.append(
            HighlightUpdate(
                target_id=target_id,
                current_text=current,
                highlighted_text=highlighted,
            )
        )
    return HighlightResponse(updates=tuple(result))


def apply_highlight_response(
    application_resume: Mapping[str, Any],
    response: HighlightResponse | str,
    *,
    max_strong_spans_per_bullet: int = DEFAULT_MAX_STRONG_SPANS_PER_BULLET,
) -> tuple[dict[str, Any], HighlightStats]:
    """Validate the complete response, then apply all exact wrapper insertions."""

    if (
        type(application_resume) is not dict
        or type(max_strong_spans_per_bullet) is not int
        or not (1 <= max_strong_spans_per_bullet <= DEFAULT_MAX_STRONG_SPANS_PER_BULLET)
    ):
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    parsed = parse_highlight_response(response) if type(response) is str else response
    _validate_highlight_response_object(parsed)
    bullets = collect_highlight_bullets(application_resume)
    update_by_id = {update.target_id: update for update in parsed.updates}
    if len(update_by_id) != len(parsed.updates):
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    expected = {bullet.target_id for bullet in bullets}
    if set(update_by_id) != expected:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)

    span_count = 0
    validated_text: dict[str, str] = {}
    for bullet in bullets:
        update = update_by_id[bullet.target_id]
        if update.current_text != bullet.text:
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        count = validate_highlighted_text(
            original_text=bullet.text,
            highlighted_text=update.highlighted_text,
            max_strong_spans=max_strong_spans_per_bullet,
        )
        span_count += count
        if span_count > MAX_TOTAL_STRONG_SPANS:
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        validated_text[bullet.target_id] = update.highlighted_text

    candidate = _clone_resume(application_resume)
    for bullet in bullets:
        _set_exact_location(
            candidate,
            bullet.location,
            validated_text[bullet.target_id],
        )
    return candidate, HighlightStats(
        bullet_count=len(bullets),
        strong_span_count=span_count,
    )


def _validate_highlight_response_object(value: object) -> None:
    if (
        type(value) is not HighlightResponse
        or type(value.updates) is not tuple
        or not value.updates
        or len(value.updates) > MAX_HIGHLIGHT_BULLETS
    ):
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    seen: set[str] = set()
    for update in value.updates:
        if (
            type(update) is not HighlightUpdate
            or type(update.target_id) is not str
            or not update.target_id
            or len(update.target_id) > 384
            or type(update.current_text) is not str
            or not update.current_text
            or len(update.current_text) > MAX_HIGHLIGHT_TEXT_CHARS
            or type(update.highlighted_text) is not str
            or not update.highlighted_text
            or len(update.highlighted_text) > MAX_HIGHLIGHT_TEXT_CHARS + 256
            or _has_forbidden_control(update.target_id)
            or _has_forbidden_control(update.current_text)
            or _has_forbidden_control(update.highlighted_text)
            or update.target_id in seen
        ):
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        seen.add(update.target_id)


def validate_highlighted_text(
    *,
    original_text: str,
    highlighted_text: str,
    min_strong_spans: int = 1,
    max_strong_spans: int = DEFAULT_MAX_STRONG_SPANS_PER_BULLET,
) -> int:
    """Validate exact wrapper insertion without tolerant HTML normalization."""

    if (
        type(original_text) is not str
        or type(highlighted_text) is not str
        or not original_text
        or not highlighted_text
        or len(original_text) > MAX_HIGHLIGHT_TEXT_CHARS
        or len(highlighted_text) > MAX_HIGHLIGHT_TEXT_CHARS + 256
        or type(min_strong_spans) is not int
        or type(max_strong_spans) is not int
        or not 1 <= min_strong_spans <= max_strong_spans
        or max_strong_spans > DEFAULT_MAX_STRONG_SPANS_PER_BULLET
        or _has_forbidden_control(original_text)
        or _has_forbidden_control(highlighted_text)
        or _has_tag_angle_lookalike(original_text)
        or _has_tag_angle_lookalike(highlighted_text)
        or "<strong>" in original_text
        or "</strong>" in original_text
    ):
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)

    plain: list[str] = []
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    index = 0
    while index < len(highlighted_text):
        if highlighted_text.startswith("<strong>", index):
            if open_at is not None:
                raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
            open_at = len(plain)
            index += len("<strong>")
            continue
        if highlighted_text.startswith("</strong>", index):
            if open_at is None:
                raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
            spans.append((open_at, len(plain)))
            open_at = None
            index += len("</strong>")
            continue
        character = highlighted_text[index]
        if character in {"<", ">"} or unicodedata.category(character) in {
            "Cc",
            "Cf",
            "Zl",
            "Zp",
        }:
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        plain.append(character)
        index += 1
    if open_at is not None or "".join(plain) != original_text:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    if not min_strong_spans <= len(spans) <= max_strong_spans:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)

    covered: set[int] = set()
    for start, end in spans:
        if start >= end:
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        span_text = original_text[start:end]
        if not any(character.isalnum() for character in span_text):
            raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
        covered.update(range(start, end))
    if not any(
        character.isalnum() and index not in covered
        for index, character in enumerate(original_text)
    ):
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    return len(spans)


def _has_tag_angle_lookalike(value: str) -> bool:
    return any(
        character not in {"<", ">"}
        and unicodedata.normalize("NFKC", character) in {"<", ">"}
        for character in value
    )


def _run_highlight_model(
    *,
    runner: ModelRunner,
    config: CodexModelConfig,
    prompt: str,
) -> tuple[HighlightResponse, Mapping[str, Any]]:
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
        raise ResumeHighlightError(_HIGHLIGHT_ERROR) from None
    if type(result) is not ModelResult:
        raise ResumeHighlightError(_HIGHLIGHT_RESPONSE_ERROR)
    return (
        parse_highlight_response(result.response),
        _safe_model_metadata(result.model_metadata),
    )


def _highlight_variant_write(
    *,
    original: ResumeVariantRecord,
    candidate: Mapping[str, Any],
    html: str,
    pdf: bytes,
    diagnostics: Any,
    logical_metadata: Mapping[str, Any],
    stats: HighlightStats,
    mro_sha256: str,
    source_text_sha256: str,
) -> ResumeVariantWrite:
    try:
        aro_yaml = yaml.safe_dump(
            _clone_resume(candidate),
            sort_keys=False,
            allow_unicode=True,
        )
    except Exception:  # noqa: BLE001 - YAML implementation details stay private.
        raise ResumeHighlightError(_HIGHLIGHT_ERROR) from None
    metadata = _clone_inert(original.model_metadata)
    if metadata is None:
        metadata = {}
    if type(metadata) is not dict:
        raise ResumeHighlightError(_HIGHLIGHT_INPUT_ERROR)
    metadata["highlighting"] = {
        "schema_version": HIGHLIGHT_AUDIT_SCHEMA_VERSION,
        "requires_human_review": True,
        "bullet_count": stats.bullet_count,
        "strong_span_count": stats.strong_span_count,
        "mro_sha256": mro_sha256,
        "source_text_sha256": source_text_sha256,
        "model": _clone_inert(logical_metadata),
    }
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
    return ResumeVariantWrite(
        variant_key=original.variant_key,
        variant_label=original.variant_label,
        source=original.source,
        parent_variant_key=original.parent_variant_key,
        application_resume_yaml=aro_yaml,
        resume_html=html,
        resume_pdf=pdf,
        source_resume_html_path=original.source_resume_html_path,
        source_resume_path=original.source_resume_path,
        ats=ats,
        ats_diagnostics=_bounded_json_value(asdict(diagnostics)),
        evidence_packet=_clone_inert(original.evidence_packet),
        external_critique=_clone_inert(original.external_critique),
        critique_prompt=original.critique_prompt,
        critique_response=original.critique_response,
        critique=_clone_inert(original.critique),
        validation=_clone_inert(original.validation),
        model_metadata=metadata,
    )
