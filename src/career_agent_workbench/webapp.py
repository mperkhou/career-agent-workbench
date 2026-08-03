"""Compatibility imports for the active archive-first Flask workbench."""

from career_agent_workbench.webapp_archive_runtime import (
    build_arg_parser,
    create_app,
    main,
)

__all__ = ["build_arg_parser", "create_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
