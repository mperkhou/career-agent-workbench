#!/usr/bin/env python3
"""Refresh governed v1 variants from the configured master resume without a model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import yaml

from career_agent_workbench.application_resume import (
    apply_core_skill_jod_matches,
    initialize_application_resume_object,
)
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
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate governed v1 AROs without model calls."
    )
    add_runtime_path_arguments(
        parser, "workspace", "database", "output_dir", "master_resume"
    )
    parser.add_argument("--job-id", action="append", dest="job_ids")
    parser.add_argument("--limit", type=int)
    return parser


def _core_skill_response(resume: Mapping[str, Any]) -> dict[str, object]:
    core = resume.get("core_technical_skills")
    buckets = core.get("bullet_points") if isinstance(core, Mapping) else ()
    values: list[dict[str, object]] = []
    if isinstance(buckets, Sequence) and not isinstance(buckets, (str, bytes)):
        for bucket in buckets:
            if not isinstance(bucket, Mapping):
                continue
            category = str(bucket.get("category") or "").strip()
            matched = bucket.get("jod_matched_items")
            if category:
                values.append(
                    {
                        "category": category,
                        "jod_matched_items": (
                            [str(item) for item in matched]
                            if isinstance(matched, Sequence)
                            and not isinstance(matched, (str, bytes))
                            else []
                        ),
                    }
                )
    return {"core_technical_skills": values}


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
            required=(
                WorkspaceMember.DATABASE,
                WorkspaceMember.OUTPUT_DIR,
                WorkspaceMember.MASTER_RESUME,
            ),
        )
        store = ApplicationStateStore(config.paths)
        selected = set(args.job_ids or ())
        records = [
            record
            for record in store.list_applications("active", limit=10_000)
            if not selected or record.job_id in selected
        ]
        if args.limit is not None:
            records = records[: max(0, args.limit)]
        processed = 0
        for record in records:
            snapshot = store.get_workflow_snapshot(record.job_id)
            old = next(
                (item for item in snapshot.variants if item.variant_key == "v1"), None
            )
            if old is None:
                continue
            fresh = initialize_application_resume_object(
                config.paths.require(WorkspaceMember.MASTER_RESUME)
            )
            refreshed = apply_core_skill_jod_matches(
                application_resume=fresh,
                core_skill_response=_core_skill_response(old.application_resume),
            )
            for key in ("job_opening_description", "professional_experience"):
                value = old.application_resume.get(key)
                if isinstance(value, Mapping):
                    refreshed[key] = _materialize(value)
            description = usable_job_description(
                record.prompt_job_description or record.job_description
            )
            if description is None:
                continue
            html = render_resume_html_from_mapping(resume=refreshed)
            pdf = render_resume_pdf_from_html(html)
            diagnostics = calculate_ats_diagnostics(
                resume_pdf=pdf, job_description=description
            )
            score = diagnostics.score
            store.upsert_resume_variant_if_revision(
                record.job_id,
                ResumeVariantWrite(
                    variant_key="v1",
                    variant_label=old.variant_label,
                    source="master_resume_regeneration",
                    application_resume_yaml=yaml.safe_dump(
                        refreshed, sort_keys=False, allow_unicode=False
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
                    ats_diagnostics=asdict(diagnostics),
                    model_metadata={
                        "workflow": "master_resume_regeneration",
                        "review_state": "awaiting_user_review",
                        "requires_human_review": True,
                    },
                ),
                expected_revision=snapshot.revision,
            )
            processed += 1
    except Exception:
        parser.error("ARO regeneration could not be completed.")
    print(json.dumps({"processed": processed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
