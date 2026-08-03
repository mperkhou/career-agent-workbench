"""Bounded artifact and resume-variant helpers for the local web adapter."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from career_agent_workbench.application_state import (
    ApplicationStateStore,
    ApplicationWorkflowSnapshot,
    ResumeVariantRecord,
)
from career_agent_workbench.cli_paths import resolve_private_workspace_path
from career_agent_workbench.config import WorkspacePaths

_MAX_CHANGED_FIELDS = 64


class WebArtifactError(ValueError):
    """Content-free artifact or comparison failure."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """One exact database-backed artifact prepared for an HTTP response."""

    content: bytes
    mime_type: str
    filename: str


def selected_resume_artifact(
    store: ApplicationStateStore,
    job_id: str,
    kind: Literal["html", "pdf"],
) -> StoredArtifact:
    """Return the exact selected/default resume artifact."""

    application = store.get_application(job_id)
    if kind == "html" and application.resume_html is not None:
        return StoredArtifact(
            application.resume_html.encode("utf-8"),
            application.resume_html_mime_type,
            application.resume_html_filename,
        )
    if kind == "pdf" and application.resume_pdf is not None:
        return StoredArtifact(
            application.resume_pdf,
            application.resume_pdf_mime_type,
            application.resume_pdf_filename,
        )
    raise WebArtifactError("Artifact is unavailable.")


def variant_resume_artifact(
    store: ApplicationStateStore,
    job_id: str,
    variant_key: str,
    kind: Literal["html", "pdf"],
) -> StoredArtifact:
    """Return one exact requested variant artifact without substitution."""

    variant = store.get_resume_variant(job_id, variant_key)
    if kind == "html" and variant.resume_html is not None:
        return StoredArtifact(
            variant.resume_html.encode("utf-8"),
            variant.resume_html_mime_type,
            variant.resume_html_filename,
        )
    if kind == "pdf" and variant.resume_pdf is not None:
        return StoredArtifact(
            variant.resume_pdf,
            variant.resume_pdf_mime_type,
            variant.resume_pdf_filename,
        )
    raise WebArtifactError("Artifact is unavailable.")


def cover_letter_artifact(
    store: ApplicationStateStore,
    job_id: str,
) -> StoredArtifact:
    """Return the exact stored cover-letter PDF artifact."""

    application = store.get_application(job_id)
    if application.cover_letter_pdf is None:
        raise WebArtifactError("Artifact is unavailable.")
    return StoredArtifact(
        application.cover_letter_pdf,
        application.cover_letter_mime_type,
        application.cover_letter_filename,
    )


def variant_review(snapshot: ApplicationWorkflowSnapshot) -> tuple[dict[str, Any], ...]:
    """Build bounded declared-parent comparisons without exposing raw values."""

    variants_by_key = {variant.variant_key: variant for variant in snapshot.variants}
    if len(variants_by_key) != len(snapshot.variants):
        raise WebArtifactError("Variant comparison is unavailable.")
    result: list[dict[str, Any]] = []
    for variant in snapshot.variants:
        aro_comparison: dict[str, Any] | None = None
        ats_comparison: dict[str, Any] | None = None
        parent: ResumeVariantRecord | None = None
        if variant.parent_variant_key is not None:
            parent = variants_by_key.get(variant.parent_variant_key)
            if parent is None:
                raise WebArtifactError("Variant comparison is unavailable.")
        if parent is not None:
            parent_mapping = _materialize(parent.application_resume)
            current_mapping = _materialize(variant.application_resume)
            aro_comparison = _aro_comparison(parent_mapping, current_mapping)
            ats_comparison = _ats_comparison(parent, variant)
        result.append(
            {
                "variant": variant,
                "parent": variant.parent_variant_key,
                "aro_comparison": aro_comparison,
                "ats_comparison": ats_comparison,
                "evidence": _evidence_summary(variant),
                "review_metadata": _review_metadata(variant),
            }
        )
    return tuple(result)


def copy_artifact_to_downloads(
    paths: WorkspacePaths,
    artifact: StoredArtifact,
) -> None:
    """Atomically copy one deterministic artifact inside the private workspace."""

    if not artifact.filename or Path(artifact.filename).name != artifact.filename:
        raise WebArtifactError("Artifact copy is unavailable.")
    download_dir = paths.download_dir
    if download_dir is None:
        raise WebArtifactError("Artifact copy is unavailable.")
    temporary_path: Path | None = None
    try:
        selected = resolve_private_workspace_path(paths, download_dir, directory=True)
        selected.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(selected, 0o700)
        selected = resolve_private_workspace_path(paths, selected, directory=True)
        destination = resolve_private_workspace_path(
            paths, selected / artifact.filename
        )
        if destination.exists() and (
            destination.is_symlink() or not destination.is_file()
        ):
            raise ValueError
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".workbench-copy-",
            dir=selected,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(artifact.content)
            temporary.flush()
            os.fsync(temporary.fileno())
        if os.name == "posix":
            os.chmod(temporary_path, 0o600)
        resolve_private_workspace_path(paths, temporary_path, must_exist=True)
        os.replace(temporary_path, destination)
        temporary_path = None
    except Exception:  # noqa: BLE001 - keep private paths and content hidden.
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise WebArtifactError("Artifact copy could not be completed.") from None


def _aro_comparison(
    parent: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    parent_keys = set(parent)
    current_keys = set(current)
    added = tuple(sorted(current_keys - parent_keys))
    removed = tuple(sorted(parent_keys - current_keys))
    changed = tuple(
        sorted(key for key in parent_keys & current_keys if parent[key] != current[key])
    )
    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added_fields": added[:_MAX_CHANGED_FIELDS],
        "removed_fields": removed[:_MAX_CHANGED_FIELDS],
        "changed_fields": changed[:_MAX_CHANGED_FIELDS],
        "truncated": any(
            len(values) > _MAX_CHANGED_FIELDS for values in (added, removed, changed)
        ),
    }


def _ats_comparison(
    parent: ResumeVariantRecord,
    current: ResumeVariantRecord,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, int | None]] = {}
    for label, attribute in (
        ("overall", "score"),
        ("parsing", "parsing_score"),
        ("keywords", "keyword_score"),
        ("semantic", "semantic_score"),
    ):
        before = getattr(parent.ats, attribute)
        after = getattr(current.ats, attribute)
        metrics[label] = {
            "parent": before,
            "current": after,
            "delta": after - before
            if before is not None and after is not None
            else None,
        }
    return {
        "metrics": metrics,
        "formatting_risk_changed": (
            parent.ats.formatting_risk != current.ats.formatting_risk
        ),
        "parent_missing_term_count": _term_count(parent.ats.missing_terms),
        "current_missing_term_count": _term_count(current.ats.missing_terms),
    }


def _term_count(value: str | None) -> int | None:
    if value is None:
        return None
    return len(tuple(item for item in value.split(",") if item.strip()))


def _evidence_summary(variant: ResumeVariantRecord) -> dict[str, Any]:
    validation = variant.validation if isinstance(variant.validation, Mapping) else None
    critique = variant.critique if isinstance(variant.critique, Mapping) else None
    accepted = _first_sequence(
        critique,
        validation,
        key="accepted_change_ids",
    )
    rejected = _first_sequence(
        critique,
        validation,
        key="rejected_changes",
    )
    return {
        "evidence_packet": "recorded"
        if variant.evidence_packet is not None
        else "not recorded",
        "external_critique": (
            "recorded" if variant.external_critique is not None else "not recorded"
        ),
        "critique": "recorded" if variant.critique is not None else "not recorded",
        "validation": "recorded" if validation is not None else "not recorded",
        "accepted_count": None if accepted is None else len(accepted),
        "rejected_count": None if rejected is None else len(rejected),
    }


def _first_sequence(
    *values: Mapping[str, Any] | None,
    key: str,
) -> list[Any] | tuple[Any, ...] | None:
    for value in values:
        if value is None:
            continue
        candidate = value.get(key)
        if type(candidate) in {list, tuple}:
            return candidate
    return None


def _review_metadata(variant: ResumeVariantRecord) -> dict[str, Any]:
    validation = variant.validation if isinstance(variant.validation, Mapping) else None
    outcome = None
    if validation is not None:
        candidate = validation.get("is_valid", validation.get("valid"))
        if type(candidate) is bool:
            outcome = candidate
    state = None
    metadata = variant.model_metadata
    if isinstance(metadata, Mapping) and "review_state" in metadata:
        candidate = metadata.get("review_state")
        if type(candidate) is str and candidate in {
            "awaiting_user_review",
            "accepted",
            "rejected",
            "reviewed",
            "draft",
        }:
            state = candidate
        else:
            state = "recorded"
    return {
        "created_at": variant.created_at,
        "updated_at": variant.updated_at,
        "html_updated_at": variant.resume_html_updated_at,
        "pdf_updated_at": variant.resume_pdf_updated_at,
        "ats_updated_at": variant.ats.updated_at,
        "validation_outcome": outcome,
        "review_state": state or "not recorded",
    }


def _materialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_materialize(item) for item in value]
    return value


__all__ = [
    "StoredArtifact",
    "WebArtifactError",
    "copy_artifact_to_downloads",
    "cover_letter_artifact",
    "selected_resume_artifact",
    "variant_resume_artifact",
    "variant_review",
]
