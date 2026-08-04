"""Command composition for the governed second-pass resume workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from career_agent_workbench.artifact_exports import export_rendered_resume
from career_agent_workbench.application_state import (
    MAX_QUERY_RESULTS,
    ApplicationStateStore,
)
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
    resolve_private_workspace_path,
)
from career_agent_workbench.codex_cli import CodexModelConfig, ModelRequest, ModelResult
from career_agent_workbench.config import Settings, WorkspaceMember
from career_agent_workbench.errors import ModelFailureSubtype, NonRetryableModelError
from career_agent_workbench.llm import build_llm_client
from career_agent_workbench.resume_refinement import refine_resume_for_job
from career_agent_workbench.workflow_diagnostics import (
    ConfigurationSource,
    WorkflowStage,
    configuration_event,
    emit_diagnostic,
    invocation_argument_source,
)
from career_agent_workbench.workflow_retry import run_model_operation

DEFAULT_SECOND_PASS_TIMEOUT_SECONDS = 600.0
DEFAULT_WORKFLOW_RETRY_COUNT = 1


class _ConfiguredLlmRunner:
    """Synchronous domain adapter around one-call async configured clients."""

    __slots__ = ("_api_model", "_retries", "_settings", "_timeout_seconds")

    def __init__(
        self,
        settings: Settings,
        *,
        api_model: str | None,
        retries: int,
        timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._api_model = api_model
        self._retries = max(0, min(retries, 3))
        self._timeout_seconds = timeout_seconds

    def run(self, request: ModelRequest, /) -> ModelResult:
        async def generate() -> tuple[str, str, int]:
            client = build_llm_client(
                self._settings,
                api_model=self._api_model,
                timeout_seconds=self._timeout_seconds,
            )
            try:
                attempt_count = 0

                async def invoke() -> str:
                    nonlocal attempt_count
                    attempt_count += 1
                    response = await client.generate_text(request.prompt)
                    _validate_generation_json_syntax(response)
                    return response

                response = await run_model_operation(
                    invoke,
                    retries=self._retries,
                    stage=WorkflowStage.V2_CRITIQUE,
                )
                return response, client.model, attempt_count
            finally:
                await client.aclose()

        response, model, attempt = asyncio.run(generate())
        return ModelResult(
            response=response,
            model_metadata={
                "workflow": request.config.workflow,
                "model": model,
                "reasoning_effort": request.config.reasoning_effort,
                "attempt": attempt,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "version": 1,
            },
        )


def _validate_generation_json_syntax(response: str) -> None:
    try:
        json.loads(response, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        raise NonRetryableModelError(
            subtype=ModelFailureSubtype.INVALID_GENERATION_JSON
        ) from None


def _reject_json_constant(_value: str) -> None:
    raise ValueError("Non-standard JSON constants are not accepted.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run governed evidence-grounded second-pass resume refinement."
    )
    add_runtime_path_arguments(
        parser,
        "workspace",
        "database",
        "output_dir",
        "master_resume",
        "master_resume_text",
    )
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--job-id", action="append")
    parser.add_argument("--all-active", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Write rendered YAML/HTML/PDF only beneath the private workspace.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Compatibility flag; v2 is stored but selection remains unchanged.",
    )
    parser.add_argument("--api-model")
    parser.add_argument("--api-timeout-seconds", type=float)
    parser.add_argument("--api-retries", type=int)
    parser.add_argument("--external-critique-file", type=Path)
    parser.add_argument("--external-critique-text")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    return parser


def _external_critique(args: argparse.Namespace) -> str | None:
    values: list[str] = []
    if args.external_critique_file is not None:
        try:
            data = args.external_critique_file.read_bytes()
            if len(data) > 1_000_000:
                raise ValueError
            values.append(data.decode("utf-8", errors="strict"))
        except (OSError, UnicodeError, ValueError):
            raise CliConfigurationError("External critique input is invalid.") from None
    if args.external_critique_text:
        values.append(args.external_critique_text)
    value = "\n".join(item.strip() for item in values if item.strip()).strip()
    return value or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        retries = (
            DEFAULT_WORKFLOW_RETRY_COUNT
            if args.api_retries is None
            else args.api_retries
        )
        if not 0 <= retries <= 3:
            raise CliConfigurationError("Model execution configuration is invalid.")
        config = load_command_config(
            args,
            required=(
                ()
                if args.config_only
                else (
                    WorkspaceMember.DATABASE,
                    WorkspaceMember.OUTPUT_DIR,
                    WorkspaceMember.MASTER_RESUME,
                    WorkspaceMember.MASTER_RESUME_TEXT,
                )
            ),
            setting_overrides={
                "second_pass_model": args.api_model,
                "llm_api_timeout_seconds": args.api_timeout_seconds,
                "ollama_timeout_seconds": args.api_timeout_seconds,
            },
        )
        provider = config.settings.llm_provider
        timeout_field = (
            "ollama_timeout_seconds"
            if provider == "ollama"
            else "llm_api_timeout_seconds"
        )
        timeout_source = config.setting_source(timeout_field)
        timeout_seconds = getattr(config.settings, timeout_field)
        if args.api_timeout_seconds is None and timeout_source == "default":
            timeout_seconds = DEFAULT_SECOND_PASS_TIMEOUT_SECONDS
        if provider == "ollama":
            model = config.settings.ollama_model
            model_source = config.setting_source("ollama_model")
        else:
            model = config.settings.second_pass_model
            model_source = config.setting_source("second_pass_model")
        emit_diagnostic(
            configuration_event(
                stage=WorkflowStage.V2_CRITIQUE,
                model=model,
                effort="",
                timeout_seconds=timeout_seconds,
                retry_count=retries,
                sources={
                    "model": model_source,
                    "effort": ConfigurationSource.DEFAULT,
                    "timeout": timeout_source,
                    "retry_count": (
                        ConfigurationSource.DEFAULT
                        if args.api_retries is None
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
        selected = [item.strip() for item in args.job_id or [] if item.strip()]
        if args.all_active:
            selected.extend(
                item.job_id
                for item in store.list_applications("active", limit=MAX_QUERY_RESULTS)
            )
        job_ids = list(dict.fromkeys(selected))
        if not job_ids:
            parser.error("Pass --job-id or --all-active.")
        runner = _ConfiguredLlmRunner(
            config.settings,
            api_model=model,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        results = []
        failures = 0
        external_critique = _external_critique(args)
        for job_id in job_ids:
            try:
                result = refine_resume_for_job(
                    store=store,
                    paths=config.paths,
                    job_id=job_id,
                    runner=runner,
                    model_config=CodexModelConfig(
                        model=model,
                        reasoning_effort="",
                        workflow="refinement",
                    ),
                    external_critique=external_critique,
                    template_path=template,
                    dry_run=args.dry_run,
                )
                if artifact_dir is not None:
                    export_rendered_resume(
                        paths=config.paths,
                        output_dir=artifact_dir,
                        job_id=result.job_id,
                        resume=result.candidate,
                        template_path=template,
                    )
                results.append(result)
            except Exception:
                failures += 1
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("Resume refinement could not be completed.")
    print(
        json.dumps(
            {
                "processed": len(results),
                "failed": failures,
                "dry_run": args.dry_run,
                "stored_variant": None if args.dry_run else "v2",
                "selection_changed": False,
                "requires_human_review": True,
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
