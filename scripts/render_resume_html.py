#!/usr/bin/env python3
"""Render configured resume YAML to HTML with the packaged template by default."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import WorkspaceMember
from career_agent_workbench.resume_rendering import render_resume_html


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render resume YAML to HTML.")
    add_runtime_path_arguments(parser, "workspace", "master_resume", "tmp_dir")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_command_config(
            args,
            required=(
                (WorkspaceMember.MASTER_RESUME, WorkspaceMember.TMP_DIR)
                if args.output is None
                else (WorkspaceMember.MASTER_RESUME,)
            ),
            master_resume=args.input,
        )
        input_path = config.paths.require(WorkspaceMember.MASTER_RESUME)
        output_path = (
            args.output.resolve(strict=False)
            if args.output is not None
            else config.paths.require(WorkspaceMember.TMP_DIR) / "resume.html"
        )
        template = (
            args.template.resolve(strict=False) if args.template is not None else None
        )
        rendered = render_resume_html(input_path, template_path=template)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    except CliConfigurationError as exc:
        parser.error(str(exc))
    except Exception:
        parser.error("Resume rendering could not be completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
