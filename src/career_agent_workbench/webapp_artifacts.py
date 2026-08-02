"""Bounded artifact and resume-variant helpers for the local web adapter."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from career_agent_workbench.application_state import (
    ApplicationStateStore,
    ApplicationWorkflowSnapshot,
    ResumeVariantRecord,
)
from career_agent_workbench.cli_paths import resolve_private_workspace_path
from career_agent_workbench.config import WorkspacePaths

_MAX_DIFF_LINES = 400
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
    """Build bounded structured and YAML comparisons in canonical order."""

    variants_by_key = {variant.variant_key: variant for variant in snapshot.variants}
    if len(variants_by_key) != len(snapshot.variants):
        raise WebArtifactError("Variant comparison is unavailable.")
    result: list[dict[str, Any]] = []
    for variant in snapshot.variants:
        changed_fields: tuple[str, ...] = ()
        unified_diff: tuple[str, ...] = ()
        parent: ResumeVariantRecord | None = None
        if variant.parent_variant_key is not None:
            parent = variants_by_key.get(variant.parent_variant_key)
            if parent is None:
                raise WebArtifactError("Variant comparison is unavailable.")
        if parent is not None:
            parent_mapping = _materialize(parent.application_resume)
            current_mapping = _materialize(variant.application_resume)
            changed_fields = tuple(
                sorted(
                    key
                    for key in {*parent_mapping, *current_mapping}
                    if parent_mapping.get(key) != current_mapping.get(key)
                )[:_MAX_CHANGED_FIELDS]
            )
            unified_diff = _yaml_diff(
                parent.variant_key,
                parent_mapping,
                variant.variant_key,
                current_mapping,
            )
        result.append(
            {
                "variant": variant,
                "parent": variant.parent_variant_key,
                "changed_fields": changed_fields,
                "unified_diff": unified_diff,
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


def _yaml_diff(
    old_name: str,
    old_value: Mapping[str, Any],
    new_name: str,
    new_value: Mapping[str, Any],
) -> tuple[str, ...]:
    import difflib

    old_lines = yaml.safe_dump(
        old_value, sort_keys=False, allow_unicode=False
    ).splitlines()
    new_lines = yaml.safe_dump(
        new_value, sort_keys=False, allow_unicode=False
    ).splitlines()
    return tuple(
        list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=old_name,
                tofile=new_name,
                lineterm="",
            )
        )[:_MAX_DIFF_LINES]
    )


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
