#!/usr/bin/env python3
"""Apply one caller-supplied core-skill response without model execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import yaml

from career_agent_workbench.application_resume import (
    CORE_SKILLS_PROMPT_JOD_MAX_CHARS,
    apply_core_skill_jod_matches,
    build_core_skills_jod_match_prompt,
    initialize_application_resume_object,
)
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
    resolve_private_workspace_path,
)
from career_agent_workbench.config import WorkspaceMember, WorkspacePaths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one local first-pass ARO YAML.")
    add_runtime_path_arguments(parser, "workspace", "master_resume")
    parser.add_argument("--trimmed-jod", type=Path, required=True)
    parser.add_argument("--core-skill-response", type=Path)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-jod-chars",
        type=int,
        default=CORE_SKILLS_PROMPT_JOD_MAX_CHARS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if not args.prompt_only and (
            args.core_skill_response is None or args.output is None
        ):
            raise CliConfigurationError(
                "First-pass response and output are required unless --prompt-only is used."
            )
        config = load_command_config(
            args,
            required=(WorkspaceMember.MASTER_RESUME,),
        )
        base = initialize_application_resume_object(
            config.paths.require(WorkspaceMember.MASTER_RESUME)
        )
        job_description = args.trimmed_jod.read_text(encoding="utf-8")
        prompt = build_core_skills_jod_match_prompt(
            application_resume=base,
            trimmed_job_description=job_description,
            max_jod_chars=args.max_jod_chars,
        )
        if args.prompt_output is not None:
            prompt_output = _private_output(config.paths, args.prompt_output)
            prompt_output.write_text(prompt, encoding="utf-8")
        elif args.prompt_only:
            print(prompt)
        if args.prompt_only:
            return 0
        response = json.loads(args.core_skill_response.read_text(encoding="utf-8"))
        updated = apply_core_skill_jod_matches(
            application_resume=base,
            core_skill_response=response,
        )
        output = _private_output(config.paths, args.output)
        output.write_text(
            yaml.safe_dump(updated, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("First-pass resume generation could not be completed.")
    return 0


def _private_output(paths: WorkspacePaths, value: Path | None) -> Path:
    if value is None:
        raise CliConfigurationError("Private workspace path is invalid.")
    selected = resolve_private_workspace_path(paths, value)
    try:
        selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        raise CliConfigurationError("Private workspace path is invalid.") from None
    return resolve_private_workspace_path(paths, selected)


if __name__ == "__main__":
    raise SystemExit(main())
