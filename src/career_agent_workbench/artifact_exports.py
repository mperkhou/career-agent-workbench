"""Explicit rendered-only resume exports beneath a private workspace."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from career_agent_workbench.cli_paths import resolve_private_workspace_path
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.resume_rendering import (
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)

_SAFE_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class ResumeExportResult:
    """Content-free confirmation for one YAML/HTML/PDF export set."""

    file_count: int = 3
    yaml_written: bool = True
    html_written: bool = True
    pdf_written: bool = True


def _materialize_resume(value: object) -> object:
    """Convert governed immutable containers into renderer/YAML-safe containers."""

    if isinstance(value, Mapping):
        return {key: _materialize_resume(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialize_resume(item) for item in value]
    return value


def export_rendered_resume(
    *,
    paths: WorkspacePaths,
    output_dir: Path,
    job_id: str,
    resume: Mapping[str, Any],
    template_path: Path | None = None,
) -> ResumeExportResult:
    """Write rendered artifacts only; raw model traffic is never accepted."""

    if (
        type(job_id) is not str
        or not _SAFE_JOB_ID_RE.fullmatch(job_id)
        or job_id in {".", ".."}
    ):
        raise ValueError("Rendered resume export could not be completed.")
    selected = resolve_private_workspace_path(paths, output_dir, directory=True)
    try:
        selected.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(selected, 0o700)
        selected = resolve_private_workspace_path(paths, selected, directory=True)
        materialized = _materialize_resume(resume)
        if type(materialized) is not dict:
            raise ValueError
        resume_yaml = yaml.safe_dump(
            materialized,
            sort_keys=False,
            allow_unicode=False,
        )
        render_options = (
            {} if template_path is None else {"template_path": template_path}
        )
        html = render_resume_html_from_mapping(resume=materialized, **render_options)
        pdf = render_resume_pdf_from_html(html)
        yaml_output = resolve_private_workspace_path(paths, selected / f"{job_id}.yml")
        html_output = resolve_private_workspace_path(paths, selected / f"{job_id}.html")
        pdf_output = resolve_private_workspace_path(paths, selected / f"{job_id}.pdf")
        yaml_output.write_text(resume_yaml, encoding="utf-8")
        html_output.write_text(html, encoding="utf-8")
        pdf_output.write_bytes(pdf)
    except Exception:  # noqa: BLE001 - keep private content and paths hidden
        raise ValueError("Rendered resume export could not be completed.") from None
    return ResumeExportResult()


__all__ = ["ResumeExportResult", "export_rendered_resume"]
