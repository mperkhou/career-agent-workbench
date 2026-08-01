"""Audit governed application rows against the canonical JOD cleaner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
)
from career_agent_workbench.config import WorkspaceMember
from career_agent_workbench.jod import job_description_context, usable_job_description
from career_agent_workbench.models import JobDetails


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit canonical prompt JODs in governed application state."
    )
    add_runtime_path_arguments(parser, "workspace", "database", "output_dir")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Store canonical JOD text through the governed state API.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Required with --apply; direct database backups are not performed.",
    )
    parser.add_argument(
        "--skip-ats",
        action="store_true",
        help="Retained compatibility flag; this audit never changes ATS fields.",
    )
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser


def audit_tracker_state(
    store: ApplicationStateStore,
    *,
    apply: bool,
    sample_limit: int,
) -> dict[str, object]:
    """Compare bounded stored source/prompt pairs without direct SQLite access."""

    records = store.list_applications("all", limit=10_000)
    changed: list[str] = []
    usable = 0
    for record in records:
        source = usable_job_description(record.job_description)
        if source is None:
            continue
        usable += 1
        try:
            cleaned = job_description_context(
                JobDetails(
                    job_id=record.job_id,
                    title=record.job_title,
                    company=record.company,
                    job_url=record.job_url,
                    source=record.source or "public",
                    description=source,
                )
            )
        except Exception:
            raise CliConfigurationError("JOD audit input is invalid.") from None
        if cleaned == (record.prompt_job_description or ""):
            continue
        changed.append(record.job_id)
        if apply:
            store.store_jod(record.job_id, source_text=source, prompt_text=cleaned)
    return {
        "total_rows": len(records),
        "usable_source_rows": usable,
        "changed_rows": len(changed),
        "applied_rows": len(changed) if apply else 0,
        "sample_changed_count": min(len(changed), max(0, sample_limit)),
        "ats_fields_changed": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.apply and not args.no_backup:
        parser.error("Governed apply requires explicit --no-backup.")
    try:
        config = load_command_config(
            args,
            required=(WorkspaceMember.DATABASE,),
        )
        store = ApplicationStateStore(config.paths)
        result = audit_tracker_state(
            store,
            apply=args.apply,
            sample_limit=args.sample_limit,
        )
    except CliConfigurationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
