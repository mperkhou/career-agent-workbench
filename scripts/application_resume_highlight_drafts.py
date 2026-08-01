#!/usr/bin/env python3
"""Run exact-target highlighting through the governed model boundary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    build_codex_runner,
    load_command_config,
    reject_compatibility_path,
    with_model_request_policy,
)
from career_agent_workbench.codex_cli import resolve_codex_model_config
from career_agent_workbench.config import WorkspaceMember
from career_agent_workbench.resume_highlighting import (
    DEFAULT_MAX_STRONG_SPANS_PER_BULLET,
    highlight_resume_for_job,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add governed exact-target highlighting to resume variants."
    )
    add_runtime_path_arguments(
        parser,
        "workspace",
        "database",
        "output_dir",
        "master_resume",
        "master_resume_text",
        "tmp_dir",
    )
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--job-id", action="append", dest="job_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variant-key")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument(
        "--max-strong-spans-per-bullet",
        type=int,
        default=DEFAULT_MAX_STRONG_SPANS_PER_BULLET,
    )
    parser.add_argument("--experience-company")
    parser.add_argument("--experience-job-order")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        reject_compatibility_path(args.artifact_dir)
        if args.template is not None:
            raise CliConfigurationError("Custom workflow templates are unsupported.")
        if args.experience_company is not None or args.experience_job_order is not None:
            raise CliConfigurationError("Partial highlight targets are unsupported.")
        if args.max_strong_spans_per_bullet != DEFAULT_MAX_STRONG_SPANS_PER_BULLET:
            raise CliConfigurationError("Custom highlight span limits are unsupported.")
        config = load_command_config(
            args,
            required=(
                WorkspaceMember.ROOT,
                WorkspaceMember.DATABASE,
                WorkspaceMember.OUTPUT_DIR,
                WorkspaceMember.MASTER_RESUME,
                WorkspaceMember.MASTER_RESUME_TEXT,
                WorkspaceMember.TMP_DIR,
            ),
        )
        store = ApplicationStateStore(config.paths)
        runner = with_model_request_policy(
            build_codex_runner(
                command=args.codex_command,
                working_directory=config.paths.require(WorkspaceMember.ROOT),
                tmp_dir=config.paths.require(WorkspaceMember.TMP_DIR),
            ),
            timeout_seconds=args.timeout_seconds,
            retry_count=args.retry_count,
        )
        model = resolve_codex_model_config(
            default_model="gpt-5.6-sol",
            default_reasoning_effort="high",
            workflow_model_override=args.codex_model,
            workflow_reasoning_effort_override=args.codex_reasoning_effort,
            workflow="highlighting",
        )
        selected = set(args.job_ids or ())
        records = [
            item
            for item in store.list_applications("active", limit=10_000)
            if not selected or item.job_id in selected
        ]
        if args.limit is not None:
            records = records[: max(0, args.limit)]
        processed = 0
        failures = 0
        for record in records:
            try:
                highlight_resume_for_job(
                    store=store,
                    paths=config.paths,
                    job_id=record.job_id,
                    runner=runner,
                    model_config=model,
                    variant_override=args.variant_key,
                    dry_run=args.dry_run,
                )
                processed += 1
            except Exception:
                failures += 1
                if args.fail_fast:
                    raise
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("Resume highlighting could not be completed.")
    print(
        json.dumps(
            {
                "processed": processed,
                "failed": failures,
                "dry_run": args.dry_run,
                "selection_changed": False,
                "requires_human_review": True,
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
