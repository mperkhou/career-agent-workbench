"""JOD and active-resume editor orchestration for the local web adapter."""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml

from career_agent_workbench.application_state import (
    MAX_ARO_YAML_BYTES,
    ApplicationWorkflowSnapshot,
    AtsFields,
)
from career_agent_workbench.ats import AtsDiagnostics, calculate_ats_diagnostics
from career_agent_workbench.jod import usable_job_description
from career_agent_workbench.resume_rendering import (
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)

_MAX_DIFF_LINES = 600
_MAX_TREE_DEPTH = 64
_MAX_TREE_NODES = 100_000

ResumeHtmlRenderer = Callable[..., str]
ResumePdfRenderer = Callable[[str], bytes]
AtsCalculator = Callable[..., AtsDiagnostics]


class WebEditorError(ValueError):
    """Content-free editor input or render failure."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class JodComparison:
    """Bounded line diff and removed-line summary."""

    unified_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderedResume:
    """Caller-produced active resume artifacts and optional ATS fields."""

    yaml_text: str
    mapping: Mapping[str, Any]
    html: str
    pdf: bytes
    ats: AtsFields | None


def compare_jod(source_text: str | None, prompt_text: str | None) -> JodComparison:
    """Return one bounded, presentation-only JOD comparison."""

    source_lines = str(source_text or "").splitlines()
    prompt_lines = str(prompt_text or "").splitlines()
    unified = tuple(
        list(
            difflib.unified_diff(
                source_lines,
                prompt_lines,
                fromfile="source",
                tofile="prompt",
                lineterm="",
            )
        )[:_MAX_DIFF_LINES]
    )
    removed = tuple(
        line[1:]
        for line in unified
        if line.startswith("-") and not line.startswith("---")
    )
    return JodComparison(unified_lines=unified, removed_lines=removed)


def active_resume_target(snapshot: ApplicationWorkflowSnapshot) -> str:
    """Return the exact target identity represented by one snapshot."""

    return snapshot.application.selected_resume_variant or "fallback"


def resume_yaml_text(value: Mapping[str, Any]) -> str:
    """Serialize one immutable state mapping without lossy field projection."""

    try:
        return yaml.safe_dump(_materialize(value), sort_keys=False, allow_unicode=False)
    except Exception:  # noqa: BLE001 - mapping details remain private.
        raise WebEditorError("Resume data is invalid.") from None


def render_resume_edit(
    yaml_text: str,
    *,
    prompt_jod: str | None,
    source_jod: str | None,
    html_renderer: ResumeHtmlRenderer = render_resume_html_from_mapping,
    pdf_renderer: ResumePdfRenderer = render_resume_pdf_from_html,
    ats_calculator: AtsCalculator = calculate_ats_diagnostics,
) -> RenderedResume:
    """Parse, render, and optionally score one exact YAML mapping."""

    mapping = _parse_yaml_mapping(yaml_text)
    try:
        html = html_renderer(resume=mapping)
        pdf = pdf_renderer(html)
        if type(html) is not str or not html or type(pdf) is not bytes or not pdf:
            raise ValueError
        ats = calculate_optional_ats(
            pdf,
            prompt_jod=prompt_jod,
            source_jod=source_jod,
            ats_calculator=ats_calculator,
        )
    except Exception:  # noqa: BLE001 - renderer and content failures stay generic.
        raise WebEditorError("Resume could not be rendered.") from None
    return RenderedResume(
        yaml_text=yaml_text,
        mapping=mapping,
        html=html,
        pdf=pdf,
        ats=ats,
    )


def calculate_optional_ats(
    pdf: bytes,
    *,
    prompt_jod: str | None,
    source_jod: str | None,
    ats_calculator: AtsCalculator = calculate_ats_diagnostics,
) -> AtsFields | None:
    """Calculate ATS from prompt JOD with source fallback, or preserve state."""

    description = usable_job_description(prompt_jod) or usable_job_description(
        source_jod
    )
    if description is None:
        return None
    diagnostics = ats_calculator(resume_pdf=pdf, job_description=description)
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
                "keyword_match_score": (
                    diagnostics.component_scores.keyword_match_score
                ),
                "semantic_match_score": (
                    diagnostics.component_scores.semantic_match_score
                ),
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
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _parse_yaml_mapping(value: object) -> dict[str, Any]:
    if type(value) is not str:
        raise WebEditorError("Resume YAML is invalid.")
    try:
        if len(value.encode("utf-8")) > MAX_ARO_YAML_BYTES:
            raise ValueError
        parsed = yaml.safe_load(value)
        if type(parsed) is not dict:
            raise ValueError
        normalized, nodes = _validate_tree(parsed, depth=0)
        if nodes > _MAX_TREE_NODES or type(normalized) is not dict:
            raise ValueError
        return normalized
    except Exception:  # noqa: BLE001 - parser details remain private.
        raise WebEditorError("Resume YAML is invalid.") from None


def _validate_tree(value: Any, *, depth: int) -> tuple[Any, int]:
    if depth > _MAX_TREE_DEPTH:
        raise ValueError
    if value is None or type(value) in {str, bool, int, float}:
        return value, 1
    if type(value) is list:
        values: list[Any] = []
        nodes = 1
        for item in value:
            normalized, count = _validate_tree(item, depth=depth + 1)
            values.append(normalized)
            nodes += count
            if nodes > _MAX_TREE_NODES:
                raise ValueError
        return values, nodes
    if type(value) is dict:
        values: dict[str, Any] = {}
        nodes = 1
        for key, item in value.items():
            if type(key) is not str or key in values:
                raise ValueError
            normalized, count = _validate_tree(item, depth=depth + 1)
            values[key] = normalized
            nodes += count
            if nodes > _MAX_TREE_NODES:
                raise ValueError
        return values, nodes
    raise ValueError


def _materialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_materialize(item) for item in value]
    return value


__all__ = [
    "AtsCalculator",
    "JodComparison",
    "RenderedResume",
    "ResumeHtmlRenderer",
    "ResumePdfRenderer",
    "WebEditorError",
    "active_resume_target",
    "calculate_optional_ats",
    "compare_jod",
    "render_resume_edit",
    "resume_yaml_text",
]
