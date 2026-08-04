#!/usr/bin/env python3
"""Run exact-target highlighting through the governed model boundary."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from career_agent_workbench.artifact_exports import export_rendered_resume
from career_agent_workbench.application_state import (
    MAX_QUERY_RESULTS,
    ApplicationStateStore,
)
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    build_codex_runner,
    load_command_config,
    resolve_private_workspace_path,
    with_model_request_policy,
)
from career_agent_workbench.codex_cli import resolve_codex_model_config
from career_agent_workbench.config import WorkspaceMember
from career_agent_workbench.resume_highlighting import (
    DEFAULT_MAX_STRONG_SPANS_PER_BULLET,
    highlight_resume_for_job,
)
from career_agent_workbench.workflow_diagnostics import (
    ConfigurationSource,
    WorkflowStage,
    configuration_event,
    emit_diagnostic,
    invocation_argument_source,
)

DEFAULT_CODEX_TIMEOUT_SECONDS = 900.0
DEFAULT_WORKFLOW_RETRY_COUNT = 1


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
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Write rendered YAML/HTML/PDF only beneath the private workspace.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_CODEX_TIMEOUT_SECONDS,
    )
    parser.add_argument("--retry-count", type=int, default=DEFAULT_WORKFLOW_RETRY_COUNT)
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        timeout_seconds = args.timeout_seconds
        retry_count = args.retry_count
        timeout_explicit = _argument_present(raw_argv, "--timeout-seconds")
        retry_explicit = _argument_present(raw_argv, "--retry-count")
        config = load_command_config(
            args,
            required=(
                ()
                if args.config_only
                else (
                    WorkspaceMember.ROOT,
                    WorkspaceMember.DATABASE,
                    WorkspaceMember.OUTPUT_DIR,
                    WorkspaceMember.MASTER_RESUME,
                    WorkspaceMember.MASTER_RESUME_TEXT,
                    WorkspaceMember.TMP_DIR,
                )
            ),
            setting_overrides={
                "highlight_codex_model": args.codex_model,
                "highlight_codex_reasoning_effort": (args.codex_reasoning_effort),
            },
        )
        model = resolve_codex_model_config(
            default_model=config.settings.highlight_codex_model,
            default_reasoning_effort=(config.settings.highlight_codex_reasoning_effort),
            workflow="highlighting",
        )
        emit_diagnostic(
            configuration_event(
                stage=WorkflowStage.HIGHLIGHT,
                model=model.model,
                effort=model.reasoning_effort,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                sources={
                    "model": config.setting_source("highlight_codex_model"),
                    "effort": config.setting_source("highlight_codex_reasoning_effort"),
                    "timeout": (
                        ConfigurationSource.DEFAULT
                        if not timeout_explicit
                        else invocation_argument_source()
                    ),
                    "retry_count": (
                        ConfigurationSource.DEFAULT
                        if not retry_explicit
                        else invocation_argument_source()
                    ),
                },
                workspace_configured=config.paths.root is not None,
            )
        )
        if args.config_only:
            print(json.dumps({"config_only": True}, sort_keys=True))
            return 0
        template = (
            resolve_private_workspace_path(
                config.paths,
                args.template,
                must_exist=True,
            )
            if args.template is not None
            else None
        )
        artifact_dir = (
            resolve_private_workspace_path(
                config.paths,
                args.artifact_dir,
                directory=True,
            )
            if args.artifact_dir is not None
            else None
        )
        store = ApplicationStateStore(config.paths)
        runner = with_model_request_policy(
            build_codex_runner(
                command=args.codex_command,
                working_directory=config.paths.require(WorkspaceMember.ROOT),
                tmp_dir=config.paths.require(WorkspaceMember.TMP_DIR),
            ),
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
        )
        selected = set(args.job_ids or ())
        records = [
            item
            for item in store.list_applications("active", limit=MAX_QUERY_RESULTS)
            if not selected or item.job_id in selected
        ]
        if args.limit is not None:
            records = records[: max(0, args.limit)]
        processed = 0
        failures = 0
        for record in records:
            try:
                result = highlight_resume_for_job(
                    store=store,
                    paths=config.paths,
                    job_id=record.job_id,
                    runner=runner,
                    model_config=model,
                    variant_override=args.variant_key,
                    template_path=template,
                    max_strong_spans_per_bullet=(args.max_strong_spans_per_bullet),
                    experience_company=args.experience_company,
                    experience_job_order=args.experience_job_order,
                    dry_run=args.dry_run,
                )
                if artifact_dir is not None:
                    export_rendered_resume(
                        paths=config.paths,
                        output_dir=artifact_dir,
                        job_id=record.job_id,
                        resume=result.candidate,
                        template_path=template,
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


def _argument_present(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


if __name__ == "__main__":
    raise SystemExit(main())
