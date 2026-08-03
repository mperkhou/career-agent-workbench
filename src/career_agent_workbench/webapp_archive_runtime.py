"""Public runtime boundary for the archive-first Flask application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from flask import Flask

from career_agent_workbench import _archived_flask_source as archived
from career_agent_workbench.application_state import (
    ApplicationStateError,
    ApplicationStateNotInitializedError,
    ApplicationStateStore,
)
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import RuntimeConfig, WorkspaceMember
from career_agent_workbench import resume_rendering

_PATH_ENV_KEYS = {
    "root": "CAREER_AGENT_WORKBENCH_WORKSPACE",
    "profile_dir": "CAREER_AGENT_WORKBENCH_PROFILE_DIR",
    "master_resume": "CAREER_AGENT_WORKBENCH_MASTER_RESUME",
    "master_resume_text": "CAREER_AGENT_WORKBENCH_MASTER_RESUME_TEXT",
    "output_dir": "CAREER_AGENT_WORKBENCH_OUTPUT_DIR",
    "database": "CAREER_AGENT_WORKBENCH_DATABASE",
    "blacklist": "CAREER_AGENT_WORKBENCH_BLACKLIST",
    "tmp_dir": "CAREER_AGENT_WORKBENCH_TMP_DIR",
    "download_dir": "CAREER_AGENT_WORKBENCH_DOWNLOAD_DIR",
}


def _port(value: str) -> int:
    try:
        selected = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.") from None
    if not 1 <= selected <= 65_535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the archive-first web command parser without loading config."""

    parser = argparse.ArgumentParser(
        description="Launch the local Career Agent Workbench tracker."
    )
    add_runtime_path_arguments(
        parser,
        "workspace",
        "database",
        "output_dir",
        "profile_dir",
        "master_resume",
        "master_resume_text",
        "blacklist",
        "tmp_dir",
        "download_dir",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--project-root", type=Path, default=None)
    return parser


def create_app(runtime: RuntimeConfig, *, project_root: Path | None = None) -> Flask:
    """Activate the archived app around already-resolved public paths."""

    paths = runtime.paths
    database_path = paths.require(WorkspaceMember.DATABASE)
    output_dir = paths.require(WorkspaceMember.OUTPUT_DIR)
    download_dir = paths.require(WorkspaceMember.DOWNLOAD_DIR)
    bound_root = _project_root(project_root)
    template_path = _packaged_resume_template()
    state_store = ApplicationStateStore(paths)
    try:
        state_store.list_applications(limit=1)
    except ApplicationStateNotInitializedError:
        try:
            state_store.initialize()
        except ApplicationStateError:
            raise ValueError("Web application database is unavailable.") from None
    except ApplicationStateError:
        raise ValueError("Web application database is unavailable.") from None
    archived.configure_runtime_boundaries(
        project_root=bound_root,
        process_env=_runtime_process_env(runtime),
    )
    app = archived.create_app(
        database_path=database_path,
        output_dir=output_dir,
        download_dir=download_dir,
        resume_template_path=template_path,
    )
    app.extensions["career_agent_workbench.runtime"] = runtime
    app.extensions["career_agent_workbench.application_state"] = state_store
    return app


def _project_root(value: Path | None) -> Path:
    candidate = value if value is not None else Path(__file__).resolve().parents[2]
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("Web application project root is invalid.") from None
    if not (resolved / "Makefile").is_file():
        raise ValueError("Web application project root is invalid.")
    return resolved


def _packaged_resume_template() -> Path:
    candidate = (
        Path(resume_rendering.__file__).resolve(strict=True).parent
        / "templates"
        / "resume"
        / "master_resume.html.j2"
    )
    if not candidate.is_file():
        raise ValueError("Packaged resume template is unavailable.")
    return candidate


def _runtime_process_env(runtime: RuntimeConfig) -> dict[str, str]:
    configured: dict[str, str] = {}
    for field_name, key in _PATH_ENV_KEYS.items():
        value = getattr(runtime.paths, field_name)
        if value is not None:
            configured[key] = str(value)
    return configured


def main(argv: Sequence[str] | None = None) -> int:
    """Load central configuration once and run the archived Flask UI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        runtime = load_command_config(
            args,
            required=(
                WorkspaceMember.DATABASE,
                WorkspaceMember.OUTPUT_DIR,
                WorkspaceMember.DOWNLOAD_DIR,
            ),
        )
        app = create_app(runtime, project_root=args.project_root)
    except (CliConfigurationError, ValueError):
        parser.error("Web application configuration is invalid.")
    if args.open_browser:
        archived._schedule_browser_open(host=args.host, port=args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
