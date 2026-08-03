#!/usr/bin/env python3
"""Re-render stored selected ARO variants through the packaged renderer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import yaml

from career_agent_workbench.application_state import (
    MAX_QUERY_RESULTS,
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
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync selected ARO variants to governed rendered drafts."
    )
    add_runtime_path_arguments(parser, "workspace", "database", "output_dir")
    parser.add_argument("--job-id", action="append", dest="job_ids")
    parser.add_argument("--limit", type=int)
    return parser


def _materialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_materialize(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_command_config(
            args,
            required=(WorkspaceMember.DATABASE, WorkspaceMember.OUTPUT_DIR),
        )
        store = ApplicationStateStore(config.paths)
        selected = set(args.job_ids or ())
        records = [
            record
            for record in store.list_applications("active", limit=MAX_QUERY_RESULTS)
            if not selected or record.job_id in selected
        ]
        if args.limit is not None:
            records = records[: max(0, args.limit)]
        processed = 0
        for record in records:
            snapshot = store.get_workflow_snapshot(record.job_id)
            key = record.selected_resume_variant or "v1"
            variant = next(
                (item for item in snapshot.variants if item.variant_key == key), None
            )
            if variant is None:
                continue
            description = usable_job_description(
                record.prompt_job_description or record.job_description
            )
            if description is None:
                continue
            resume = _materialize(variant.application_resume)
            html = render_resume_html_from_mapping(resume=resume)
            pdf = render_resume_pdf_from_html(html)
            diagnostics = calculate_ats_diagnostics(
                resume_pdf=pdf, job_description=description
            )
            score = diagnostics.score
            store.upsert_resume_variant_if_revision(
                record.job_id,
                ResumeVariantWrite(
                    variant_key=variant.variant_key,
                    variant_label=variant.variant_label,
                    source="governed_draft_sync",
                    parent_variant_key=variant.parent_variant_key,
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
                    ats_diagnostics=_materialize(asdict(diagnostics)),
                    evidence_packet=_optional_materialized(variant.evidence_packet),
                    external_critique=_optional_materialized(variant.external_critique),
                    critique=_optional_materialized(variant.critique),
                    validation=_optional_materialized(variant.validation),
                    model_metadata={
                        **(_optional_materialized(variant.model_metadata) or {}),
                        "render_sync": "packaged_template",
                        "review_state": "awaiting_user_review",
                    },
                ),
                expected_revision=snapshot.revision,
            )
            processed += 1
    except Exception:
        parser.error("Draft synchronization could not be completed.")
    print(json.dumps({"processed": processed}, sort_keys=True))
    return 0


def _optional_materialized(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    materialized = _materialize(value)
    if type(materialized) is not dict:
        raise ValueError("Stored review metadata is invalid.")
    return materialized


if __name__ == "__main__":
    raise SystemExit(main())
