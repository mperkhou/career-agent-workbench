#!/usr/bin/env python3
"""Run governed manual resume candidates through an explicit Codex runner."""

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
from career_agent_workbench.config import WorkspaceMember
from career_agent_workbench.resume_manual_pass import run_manual_resume_pass
from career_agent_workbench.resume_manual_profiles import (
    DEFAULT_MANUAL_PASS_PROFILE,
    parse_manual_pass_profile_key,
    resolve_manual_pass_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run governed manual resume review candidates."
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
    parser.add_argument("--job-id", action="append", dest="job_ids", required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument(
        "--manual-pass-profile",
        type=parse_manual_pass_profile_key,
        default=None,
    )
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--retry-count", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        reject_compatibility_path(args.artifact_dir)
        if args.template is not None:
            raise CliConfigurationError("Custom workflow templates are unsupported.")
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
            setting_overrides={
                "manual_pass_codex_model": args.codex_model,
                "manual_pass_codex_reasoning_effort": (args.codex_reasoning_effort),
            },
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
        model = resolve_manual_pass_config(
            profile=args.manual_pass_profile or DEFAULT_MANUAL_PASS_PROFILE,
            workflow_model_override=(config.settings.manual_pass_codex_model or None),
            workflow_reasoning_effort_override=(
                config.settings.manual_pass_codex_reasoning_effort or None
            ),
        ).to_model_config()
        job_ids = list(
            dict.fromkeys(value.strip() for value in args.job_ids if value.strip())
        )
        processed = 0
        failures = 0
        for job_id in job_ids:
            try:
                run_manual_resume_pass(
                    store=store,
                    paths=config.paths,
                    job_id=job_id,
                    runner=runner,
                    model_config=model,
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
        parser.error("Manual resume workflow could not be completed.")
    print(
        json.dumps(
            {
                "processed": processed,
                "failed": failures,
                "dry_run": args.dry_run,
                "stored_variant": None if args.dry_run else "manual",
                "selection_changed": False,
                "requires_human_review": True,
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
