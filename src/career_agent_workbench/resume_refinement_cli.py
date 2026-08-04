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
from career_agent_workbench.llm import build_llm_client
from career_agent_workbench.resume_refinement import refine_resume_for_job


class _ConfiguredLlmRunner:
    """Synchronous domain adapter around one-call async configured clients."""

    __slots__ = ("_api_model", "_retries", "_settings")

    def __init__(
        self, settings: Settings, *, api_model: str | None, retries: int
    ) -> None:
        self._settings = settings
        self._api_model = api_model
        self._retries = max(0, min(retries, 3))

    def run(self, request: ModelRequest, /) -> ModelResult:
        async def generate() -> tuple[str, str]:
            client = build_llm_client(self._settings, api_model=self._api_model)
            try:
                for attempt in range(self._retries + 1):
                    try:
                        return await client.generate_text(request.prompt), client.model
                    except Exception:
                        if attempt >= self._retries:
                            raise
                raise AssertionError("unreachable retry loop")
            finally:
                await client.aclose()

        response, model = asyncio.run(generate())
        return ModelResult(
            response=response,
            model_metadata={
                "workflow": request.config.workflow,
                "model": model,
                "reasoning_effort": request.config.reasoning_effort,
                "attempt": 1,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "version": 1,
            },
        )


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
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--external-critique-file", type=Path)
    parser.add_argument("--external-critique-text")
    parser.add_argument("--dry-run", action="store_true")
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
        if not 0 <= args.api_retries <= 3:
            raise CliConfigurationError("Model execution configuration is invalid.")
        config = load_command_config(
            args,
            required=(
                WorkspaceMember.DATABASE,
                WorkspaceMember.OUTPUT_DIR,
                WorkspaceMember.MASTER_RESUME,
                WorkspaceMember.MASTER_RESUME_TEXT,
            ),
            setting_overrides={
                "second_pass_model": args.api_model,
                "llm_api_timeout_seconds": args.api_timeout_seconds,
            },
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
        model = config.settings.second_pass_model
        runner = _ConfiguredLlmRunner(
            config.settings,
            api_model=model,
            retries=args.api_retries,
        )
        results = [
            refine_resume_for_job(
                store=store,
                paths=config.paths,
                job_id=job_id,
                runner=runner,
                model_config=CodexModelConfig(
                    model=model,
                    reasoning_effort="",
                    workflow="refinement",
                ),
                external_critique=_external_critique(args),
                template_path=template,
                dry_run=args.dry_run,
            )
            for job_id in job_ids
        ]
        if artifact_dir is not None:
            for result in results:
                export_rendered_resume(
                    paths=config.paths,
                    output_dir=artifact_dir,
                    job_id=result.job_id,
                    resume=result.candidate,
                    template_path=template,
                )
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("Resume refinement could not be completed.")
    print(
        json.dumps(
            {
                "processed": len(results),
                "dry_run": args.dry_run,
                "stored_variant": None if args.dry_run else "v2",
                "selection_changed": False,
                "requires_human_review": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
