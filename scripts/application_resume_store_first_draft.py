#!/usr/bin/env python3
"""Store one caller-reviewed v1 YAML through the governed state boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import yaml

from career_agent_workbench.application_state import (
    ApplicationStateStore,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.ats import calculate_ats_diagnostics
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
    resolve_private_workspace_path,
)
from career_agent_workbench.config import WorkspaceMember, WorkspacePaths
from career_agent_workbench.jod import usable_job_description
from career_agent_workbench.resume_rendering import (
    load_resume,
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Store one governed first-draft ARO.")
    add_runtime_path_arguments(parser, "workspace", "database", "output_dir")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument(
        "--output-yaml",
        type=Path,
        help="Write the reviewed YAML beneath the private workspace.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Write rendered HTML beneath the private workspace.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        help="Write rendered PDF beneath the private workspace.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_command_config(
            args,
            required=(WorkspaceMember.DATABASE, WorkspaceMember.OUTPUT_DIR),
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
        output_yaml = _private_output(config.paths, args.output_yaml)
        output_html = _private_output(config.paths, args.output_html)
        output_pdf = _private_output(config.paths, args.output_pdf)
        store = ApplicationStateStore(config.paths)
        snapshot = store.get_workflow_snapshot(args.job_id)
        resume = load_resume(args.input)
        resume_yaml = yaml.safe_dump(
            resume,
            sort_keys=False,
            allow_unicode=False,
        )
        html = render_resume_html_from_mapping(
            resume=resume,
            template_path=template,
        )
        pdf = render_resume_pdf_from_html(html)
        description = usable_job_description(
            snapshot.application.prompt_job_description
            or snapshot.application.job_description
        )
        if description is None:
            raise ValueError
        diagnostics = calculate_ats_diagnostics(
            resume_pdf=pdf,
            job_description=description,
        )
        score = diagnostics.score
        store.upsert_resume_variant_if_revision(
            args.job_id,
            ResumeVariantWrite(
                variant_key="v1",
                variant_label="Governed first draft",
                source="reviewed_first_draft",
                application_resume_yaml=resume_yaml,
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
                    "workflow": "first_draft_import",
                    "review_state": "awaiting_user_review",
                    "requires_human_review": True,
                },
            ),
            expected_revision=snapshot.revision,
        )
        if output_yaml is not None:
            output_yaml.write_text(resume_yaml, encoding="utf-8")
        if output_html is not None:
            output_html.write_text(html, encoding="utf-8")
        if output_pdf is not None:
            output_pdf.write_bytes(pdf)
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("First-draft storage could not be completed.")
    return 0


def _private_output(paths: WorkspacePaths, value: Path | None) -> Path | None:
    if value is None:
        return None
    selected = resolve_private_workspace_path(paths, value)
    try:
        selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        raise CliConfigurationError("Private workspace path is invalid.") from None
    return resolve_private_workspace_path(paths, selected)


if __name__ == "__main__":
    raise SystemExit(main())
