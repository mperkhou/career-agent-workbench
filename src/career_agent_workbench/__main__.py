"""Minimal command-line entry point for Career Agent Workbench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from career_agent_workbench import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free command-line parser."""
    parser = argparse.ArgumentParser(
        prog="career-agent-workbench",
        description="Career Agent Workbench",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and exit successfully."""
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
