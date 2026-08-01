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
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import WorkspaceMember
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
        "--output-html",
        type=Path,
        help="Retained compatibility option; governed artifact output is disabled.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        help="Retained compatibility option; governed artifact output is disabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.output_html is not None or args.output_pdf is not None:
        parser.error("Compatibility artifact output is disabled.")
    try:
        config = load_command_config(
            args,
            required=(WorkspaceMember.DATABASE, WorkspaceMember.OUTPUT_DIR),
        )
        store = ApplicationStateStore(config.paths)
        snapshot = store.get_workflow_snapshot(args.job_id)
        resume = load_resume(args.input)
        resume_yaml = yaml.safe_dump(
            resume,
            sort_keys=False,
            allow_unicode=False,
        )
        html = render_resume_html_from_mapping(
            resume=resume, template_path=args.template
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
    except Exception:
        parser.error("First-draft storage could not be completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
