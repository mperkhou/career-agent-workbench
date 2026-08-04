#!/usr/bin/env python3
"""Run governed manual resume candidates through an explicit Codex runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from career_agent_workbench.artifact_exports import export_rendered_resume
from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    build_codex_runner,
    load_command_config,
    resolve_private_workspace_path,
    with_model_request_policy,
)
from career_agent_workbench.config import WorkspaceMember
from career_agent_workbench.resume_manual_pass import run_manual_resume_pass
from career_agent_workbench.resume_manual_profiles import (
    DEFAULT_MANUAL_PASS_PROFILE,
    parse_manual_pass_profile_key,
    resolve_manual_pass_config,
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
    parser.add_argument("--job-id", action="append", dest="job_ids")
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
    parser.add_argument(
        "--manual-pass-profile",
        type=parse_manual_pass_profile_key,
        default=None,
    )
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_CODEX_TIMEOUT_SECONDS,
    )
    parser.add_argument("--retry-count", type=int, default=DEFAULT_WORKFLOW_RETRY_COUNT)
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
        profile_explicit = _argument_present(raw_argv, "--manual-pass-profile")
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
                "manual_pass_codex_model": args.codex_model,
                "manual_pass_codex_reasoning_effort": (args.codex_reasoning_effort),
            },
        )
        profile = args.manual_pass_profile or DEFAULT_MANUAL_PASS_PROFILE
        model_source = config.setting_source("manual_pass_codex_model")
        effort_source = config.setting_source("manual_pass_codex_reasoning_effort")
        model_override = (
            config.settings.manual_pass_codex_model
            if model_source != ConfigurationSource.DEFAULT.value
            and config.settings.manual_pass_codex_model
            else None
        )
        effort_override = (
            config.settings.manual_pass_codex_reasoning_effort
            if effort_source != ConfigurationSource.DEFAULT.value
            else None
        )
        resolved_model = resolve_manual_pass_config(
            profile=profile,
            workflow_model_override=model_override,
            workflow_reasoning_effort_override=effort_override,
        )
        profile_source = (
            invocation_argument_source()
            if profile_explicit
            else ConfigurationSource.DEFAULT
        )
        emit_diagnostic(
            configuration_event(
                stage=WorkflowStage.MANUAL,
                model=resolved_model.model,
                effort=resolved_model.reasoning_effort,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                sources={
                    "model": (
                        profile_source
                        if model_source == ConfigurationSource.DEFAULT.value
                        else model_source
                    ),
                    "effort": (
                        profile_source
                        if effort_source == ConfigurationSource.DEFAULT.value
                        else effort_source
                    ),
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
                profile=resolved_model.profile.key.value,
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
        model = resolved_model.to_model_config()
        job_ids = list(
            dict.fromkeys(
                value.strip() for value in args.job_ids or [] if value.strip()
            )
        )
        if not job_ids:
            raise CliConfigurationError("Pass at least one --job-id.")
        processed = 0
        failures = 0
        for job_id in job_ids:
            try:
                result = run_manual_resume_pass(
                    store=store,
                    paths=config.paths,
                    job_id=job_id,
                    runner=runner,
                    model_config=model,
                    template_path=template,
                    dry_run=args.dry_run,
                )
                if artifact_dir is not None:
                    export_rendered_resume(
                        paths=config.paths,
                        output_dir=artifact_dir,
                        job_id=job_id,
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


def _argument_present(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


if __name__ == "__main__":
    raise SystemExit(main())
