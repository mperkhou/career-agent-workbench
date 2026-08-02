"""Audit governed application rows against the canonical JOD cleaner."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from career_agent_workbench.application_state import ApplicationStateStore, AtsFields
from career_agent_workbench.ats import calculate_ats_diagnostics
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
        help="Apply without the default SQLite backup.",
    )
    parser.add_argument(
        "--skip-ats",
        action="store_true",
        help="Skip the default ATS refresh for changed rows with rendered resumes.",
    )
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser


def audit_tracker_state(
    store: ApplicationStateStore,
    *,
    apply: bool,
    sample_limit: int,
    refresh_ats: bool = True,
    backup: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Compare bounded stored source/prompt pairs without direct SQLite access."""

    records = store.list_applications("all", limit=10_000)
    changed: list[str] = []
    usable = 0
    ats_fields_changed = 0
    backup_created = False
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
            if backup is not None and not backup_created:
                backup()
                backup_created = True
            ats = None
            selected_pdf = (
                record.selected_variant.resume_pdf
                if record.selected_variant is not None
                else record.resume_pdf
            )
            if refresh_ats and type(selected_pdf) is bytes and selected_pdf:
                try:
                    diagnostics = calculate_ats_diagnostics(
                        resume_pdf=selected_pdf,
                        job_description=cleaned,
                    )
                except Exception:
                    raise CliConfigurationError(
                        "JOD audit ATS refresh failed."
                    ) from None
                score = diagnostics.score
                ats = AtsFields(
                    score=score.overall_score,
                    parsing_score=score.parsing_score,
                    keyword_score=score.keyword_match_score,
                    semantic_score=score.semantic_match_score,
                    formatting_risk=score.formatting_risk,
                    missing_terms=", ".join(score.missing_high_value_terms),
                    diagnostics=asdict(diagnostics),
                    updated_at=datetime.now(UTC).isoformat(timespec="microseconds"),
                )
                ats_fields_changed += 1
            store.store_jod(
                record.job_id,
                source_text=source,
                prompt_text=cleaned,
                ats=ats,
            )
    return {
        "total_rows": len(records),
        "usable_source_rows": usable,
        "changed_rows": len(changed),
        "applied_rows": len(changed) if apply else 0,
        "sample_changed_count": min(len(changed), max(0, sample_limit)),
        "ats_fields_changed": ats_fields_changed,
        "backup_created": backup_created,
    }


def _backup_sqlite_database(database: Path) -> None:
    destination: Path | None = None
    source: sqlite3.Connection | None = None
    target: sqlite3.Connection | None = None
    complete = False
    try:
        if (
            type(database) is not type(Path())
            or not database.is_absolute()
            or database.is_symlink()
            or not database.is_file()
        ):
            raise ValueError
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = database.with_name(f"{database.name}.backup-{timestamp}")
        if destination.exists() or destination.is_symlink():
            raise ValueError
        source = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        target = sqlite3.connect(destination)
        try:
            source.execute("PRAGMA query_only = ON")
            source.backup(target)
            if target.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise sqlite3.DatabaseError
        finally:
            target.close()
            source.close()
        os.chmod(destination, 0o600)
        complete = True
    except Exception:
        raise CliConfigurationError("JOD audit backup failed.") from None
    finally:
        if not complete and destination is not None:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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
            refresh_ats=not args.skip_ats,
            backup=(
                None
                if not args.apply or args.no_backup
                else lambda: _backup_sqlite_database(config.paths.database)
            ),
        )
    except CliConfigurationError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
