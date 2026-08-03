from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape as html_escape
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import yaml
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from linkedin_career_mcp.ats import AtsProxyScore, calculate_ats_proxy_score
from linkedin_career_mcp.config import load_settings
from linkedin_career_mcp.errors import LinkedInCareerMcpError
from linkedin_career_mcp.generic_job_scraper import fetch_generic_job_details
from linkedin_career_mcp.jod import clean_job_description_for_prompt
from linkedin_career_mcp.models import JobDetails
from linkedin_career_mcp.providers.linkedin_public import (
    LinkedInPublicJobsProvider,
    extract_job_id,
)
from linkedin_career_mcp.resume_manual_profiles import (
    DEFAULT_MANUAL_PASS_PROFILE,
    ManualPassProfileKey,
    parse_manual_pass_profile,
)
from linkedin_career_mcp.resume_rendering import (
    render_resume_html_from_mapping,
    render_resume_pdf_from_html,
    rich_text,
    sanitize_resume_rich_text,
)

DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_DATABASE = Path("tracking/applications.sqlite3")
DEFAULT_RESUME_TEMPLATE = Path("templates/resume/master_resume.html.j2")
APPLICATION_STATUSES = {"No", "Yes", "N/A", "Rejected", "Accepted for interview"}
APPLICATION_STATUS_FILTERS = {"all", *APPLICATION_STATUSES}
APPLICATION_ARCHIVE_FILTERS = {"active", "archived", "all"}
VIEW_STATE_SORTS = {"company", "matched", "ats", "resume", "cover_letter"}
VIEW_STATE_DIRECTIONS = {"asc", "desc"}
VIEW_STATE_QUERY_KEYS = {"q", "status", "archive", "sort", "direction"}
DEFAULT_RESUME_VARIANT = "v1"
SECOND_PASS_RESUME_VARIANT = "v2"
MANUAL_PASS_RESUME_VARIANT = "manual"
AUTO_RESUME_VARIANT_SELECTION_MODE = "auto"
MANUAL_RESUME_VARIANT_SELECTION_MODE = "manual"
RESUME_VARIANT_DEFAULT_PRECEDENCE = (
    MANUAL_PASS_RESUME_VARIANT,
    SECOND_PASS_RESUME_VARIANT,
    DEFAULT_RESUME_VARIANT,
)
RESUME_VARIANT_LABELS = {
    DEFAULT_RESUME_VARIANT: "Draft v1",
    SECOND_PASS_RESUME_VARIANT: "Refined v2",
    MANUAL_PASS_RESUME_VARIANT: "Manual pass",
}
APPLICATION_EXTRA_COLUMNS = {
    "archived_at": "TEXT",
    "selected_resume_variant": f"TEXT NOT NULL DEFAULT '{DEFAULT_RESUME_VARIANT}'",
    "resume_variant_selection_mode": (
        f"TEXT NOT NULL DEFAULT '{AUTO_RESUME_VARIANT_SELECTION_MODE}'"
    ),
    "job_description": "TEXT",
    "prompt_job_description": "TEXT",
    "application_resume_object": "TEXT",
    "application_resume_updated_at": "TEXT",
    "application_resume_backup_object": "TEXT",
    "application_resume_backup_created_at": "TEXT",
    "resume_html_filename": "TEXT NOT NULL DEFAULT ''",
    "resume_html_content": "TEXT",
    "resume_html_mime_type": "TEXT NOT NULL DEFAULT 'text/html; charset=utf-8'",
    "source_resume_html_path": "TEXT NOT NULL DEFAULT ''",
    "resume_html_updated_at": "TEXT",
    "resume_updated_at": "TEXT",
    "cover_letter_object": "TEXT",
    "cover_letter_object_updated_at": "TEXT",
    "cover_letter_filename": "TEXT NOT NULL DEFAULT ''",
    "cover_letter_content": "BLOB",
    "cover_letter_mime_type": "TEXT NOT NULL DEFAULT 'application/pdf'",
    "source_cover_letter_path": "TEXT NOT NULL DEFAULT ''",
    "cover_letter_updated_at": "TEXT",
    "date_matched": "TEXT",
    "date_posted": "TEXT",
    "experience_level": "TEXT",
    "ats_score": "INTEGER",
    "ats_parsing_score": "INTEGER",
    "ats_keyword_score": "INTEGER",
    "ats_semantic_score": "INTEGER",
    "ats_formatting_risk": "TEXT",
    "ats_missing_terms": "TEXT",
    "ats_updated_at": "TEXT",
}
REGENERATE_ACTION_TARGETS = {
    "draft_resumes": "regenerate-draft-resumes",
    "resume_variants": "regenerate-resumes",
    "refine_drafts": "refine-draft-resumes",
    "aro_objects": "regenerate-aro-objects",
    "sync_draft_to_aro": "sync-draft-to-aro",
    "highlight_drafts": "highlight-draft-resumes",
    "manual_pass": "manual-pass-resumes",
}
_DRAFT_REGENERATE_MODES = {"draft_resumes", "resume_variants"}
COMPOSITE_ACTION_MODES = {"resume_variants_manual_pass"}
CHAINABLE_HIGHLIGHT_MODES = {
    "draft_resumes",
    "resume_variants",
    "resume_variants_manual_pass",
    "manual_pass",
}
COVER_LETTER_OBJECT_SCHEMA_VERSION = "cover_letter_object.v0.1"
EMERALD_ACCENT = HexColor("#57ba86")
RESUME_BODY_COLOR = HexColor("#111827")
COVER_LETTER_BODY_FONT_SIZE = 9
COVER_LETTER_BODY_LEADING = 11.5
COVER_LETTER_PARAGRAPH_SPACE_AFTER = 7
MAX_ACTION_RUNS = 8
MAX_ACTION_MESSAGES = 160
DEFAULT_SEED_MAX_JOBS = 5
MAX_SEED_JOBS_FROM_UI = 50
DEFAULT_SEED_DATE_POSTED = "past_week"
SEED_DATE_POSTED_OPTIONS = (
    ("past_24_hours", "Last 24 hours"),
    ("past_week", "Past week"),
    ("past_month", "Past month"),
)
MANUAL_PASS_PROFILE_OPTIONS = (
    (ManualPassProfileKey.ECONOMY.value, "Economy — Terra / High"),
    (
        ManualPassProfileKey.REGULAR.value,
        "Regular — Sol / High (Recommended)",
    ),
    (ManualPassProfileKey.PREMIUM.value, "Premium — Sol / X-High"),
)


@dataclass(frozen=True)
class ApplicationJobRecord:
    job_id: str
    company: str
    job_title: str
    linkedin_url: str
    job_description: str | None
    prompt_job_description: str | None
    date_matched: str | None
    date_posted: str | None
    experience_level: str | None


@dataclass(frozen=True)
class DescriptionDiffRow:
    status: str
    left_line_no: int | None
    right_line_no: int | None
    left_text: str
    right_text: str


@dataclass
class BackgroundActionRun:
    run_id: str
    title: str
    status: str
    started_at: str
    finished_at: str | None = None
    return_code: int | None = None
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BackgroundWorkflowStage:
    label: str
    regenerate_mode: str | None = None
    highlight_variant_key: str | None = None
    manual_pass_profile: ManualPassProfileKey | None = None


BackgroundActionRunner = Callable[..., None]


_ACTION_RUNS: dict[str, BackgroundActionRun] = {}
_ACTION_RUN_LOCK = threading.Lock()


def connect_database(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    init_database(connection)
    return connection


def init_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            job_id TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            job_title TEXT NOT NULL,
            linkedin_url TEXT NOT NULL,
            job_description TEXT,
            prompt_job_description TEXT,
            application_resume_object TEXT,
            application_resume_updated_at TEXT,
            application_resume_backup_object TEXT,
            application_resume_backup_created_at TEXT,
            resume_html_filename TEXT NOT NULL DEFAULT '',
            resume_html_content TEXT,
            resume_html_mime_type TEXT NOT NULL DEFAULT 'text/html; charset=utf-8',
            source_resume_html_path TEXT NOT NULL DEFAULT '',
            resume_html_updated_at TEXT,
            resume_filename TEXT NOT NULL,
            resume_content BLOB,
            resume_mime_type TEXT NOT NULL DEFAULT 'application/pdf',
            source_resume_path TEXT NOT NULL,
            resume_updated_at TEXT,
            cover_letter_object TEXT,
            cover_letter_object_updated_at TEXT,
            cover_letter_filename TEXT NOT NULL DEFAULT '',
            cover_letter_content BLOB,
            cover_letter_mime_type TEXT NOT NULL DEFAULT 'application/pdf',
            source_cover_letter_path TEXT NOT NULL DEFAULT '',
            cover_letter_updated_at TEXT,
            date_matched TEXT,
            date_posted TEXT,
            experience_level TEXT,
            ats_score INTEGER,
            ats_parsing_score INTEGER,
            ats_keyword_score INTEGER,
            ats_semantic_score INTEGER,
            ats_formatting_risk TEXT,
            ats_missing_terms TEXT,
            ats_updated_at TEXT,
            applied_to TEXT NOT NULL DEFAULT 'No',
            date_applied TEXT,
            notes TEXT NOT NULL DEFAULT '',
            archived_at TEXT,
            imported_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _ensure_application_columns(connection)
    _ensure_resume_variant_table(connection)
    _backfill_v1_resume_variants(connection)
    _backfill_legacy_manual_resume_variants(connection)
    _apply_resume_variant_default_precedence(connection)
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS applications_unique_linkedin_job_id
        ON applications(job_id)
        """
    )
    connection.commit()


def _ensure_resume_variant_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS application_resume_variants (
            job_id TEXT NOT NULL,
            variant_key TEXT NOT NULL,
            variant_label TEXT NOT NULL,
            source TEXT NOT NULL,
            parent_variant_key TEXT,
            application_resume_object TEXT NOT NULL,
            resume_html_filename TEXT NOT NULL DEFAULT '',
            resume_html_content TEXT,
            resume_html_mime_type TEXT NOT NULL DEFAULT 'text/html; charset=utf-8',
            source_resume_html_path TEXT NOT NULL DEFAULT '',
            resume_html_updated_at TEXT,
            resume_filename TEXT NOT NULL DEFAULT '',
            resume_content BLOB,
            resume_mime_type TEXT NOT NULL DEFAULT 'application/pdf',
            source_resume_path TEXT NOT NULL DEFAULT '',
            resume_updated_at TEXT,
            ats_score INTEGER,
            ats_parsing_score INTEGER,
            ats_keyword_score INTEGER,
            ats_semantic_score INTEGER,
            ats_formatting_risk TEXT,
            ats_missing_terms TEXT,
            ats_updated_at TEXT,
            ats_diagnostics_json TEXT,
            evidence_packet_json TEXT,
            external_critique_json TEXT,
            critique_prompt TEXT,
            critique_response TEXT,
            critique_json TEXT,
            validation_json TEXT,
            model_metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (job_id, variant_key)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS application_resume_variants_variant_key
        ON application_resume_variants(variant_key)
        """
    )


def _backfill_v1_resume_variants(connection: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    connection.execute(
        """
        UPDATE applications
        SET selected_resume_variant = ?
        WHERE selected_resume_variant IS NULL
           OR TRIM(selected_resume_variant) = ''
        """,
        (DEFAULT_RESUME_VARIANT,),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO application_resume_variants (
            job_id, variant_key, variant_label, source, parent_variant_key,
            application_resume_object, resume_html_filename, resume_html_content,
            resume_html_mime_type, source_resume_html_path, resume_html_updated_at,
            resume_filename, resume_content, resume_mime_type, source_resume_path,
            resume_updated_at, ats_score, ats_parsing_score, ats_keyword_score,
            ats_semantic_score, ats_formatting_risk, ats_missing_terms, ats_updated_at,
            created_at, updated_at
        )
        SELECT
            job_id, ?, 'Draft v1', 'first_draft', NULL,
            application_resume_object, resume_html_filename, resume_html_content,
            resume_html_mime_type, source_resume_html_path, resume_html_updated_at,
            resume_filename, resume_content, resume_mime_type, source_resume_path,
            resume_updated_at, ats_score, ats_parsing_score, ats_keyword_score,
            ats_semantic_score, ats_formatting_risk, ats_missing_terms, ats_updated_at,
            COALESCE(application_resume_updated_at, resume_updated_at, imported_at, updated_at, ?),
            COALESCE(application_resume_updated_at, resume_updated_at, updated_at, ?)
        FROM applications
        WHERE COALESCE(NULLIF(application_resume_object, ''), '') != ''
        """,
        (DEFAULT_RESUME_VARIANT, now, now),
    )


def _backfill_legacy_manual_resume_variants(connection: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    connection.execute(
        """
        INSERT OR IGNORE INTO application_resume_variants (
            job_id, variant_key, variant_label, source, parent_variant_key,
            application_resume_object, resume_html_filename, resume_html_content,
            resume_html_mime_type, source_resume_html_path, resume_html_updated_at,
            resume_filename, resume_content, resume_mime_type, source_resume_path,
            resume_updated_at, ats_score, ats_parsing_score, ats_keyword_score,
            ats_semantic_score, ats_formatting_risk, ats_missing_terms, ats_updated_at,
            created_at, updated_at
        )
        SELECT
            applications.job_id, ?, 'Manual pass', 'legacy_manual_pass',
            CASE
                WHEN v2.variant_key IS NOT NULL THEN ?
                WHEN v1.variant_key IS NOT NULL THEN ?
                ELSE NULL
            END,
            applications.application_resume_object,
            applications.resume_html_filename,
            applications.resume_html_content,
            applications.resume_html_mime_type,
            applications.source_resume_html_path,
            applications.resume_html_updated_at,
            applications.resume_filename,
            applications.resume_content,
            applications.resume_mime_type,
            applications.source_resume_path,
            applications.resume_updated_at,
            applications.ats_score,
            applications.ats_parsing_score,
            applications.ats_keyword_score,
            applications.ats_semantic_score,
            applications.ats_formatting_risk,
            applications.ats_missing_terms,
            applications.ats_updated_at,
            COALESCE(
                applications.application_resume_updated_at,
                applications.resume_updated_at,
                applications.updated_at,
                ?
            ),
            COALESCE(
                applications.application_resume_updated_at,
                applications.resume_updated_at,
                applications.updated_at,
                ?
            )
        FROM applications
        LEFT JOIN application_resume_variants AS v1
          ON v1.job_id = applications.job_id
         AND v1.variant_key = ?
        LEFT JOIN application_resume_variants AS v2
          ON v2.job_id = applications.job_id
         AND v2.variant_key = ?
        WHERE COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
          AND applications.resume_content IS NOT NULL
          AND (
              applications.selected_resume_variant = ?
              OR lower(COALESCE(applications.notes, '')) LIKE '%manual second pass%'
              OR lower(COALESCE(applications.notes, '')) LIKE '%manual passthrough%'
          )
        """,
        (
            MANUAL_PASS_RESUME_VARIANT,
            SECOND_PASS_RESUME_VARIANT,
            DEFAULT_RESUME_VARIANT,
            now,
            now,
            DEFAULT_RESUME_VARIANT,
            SECOND_PASS_RESUME_VARIANT,
            MANUAL_PASS_RESUME_VARIANT,
        ),
    )


def _apply_resume_variant_default_precedence(
    connection: sqlite3.Connection,
    *,
    job_ids: list[str] | None = None,
    force: bool = False,
) -> int:
    clauses = [
        "COALESCE(NULLIF(applications.resume_variant_selection_mode, ''), ?) = ?"
    ]
    params: list[Any] = [
        MANUAL_PASS_RESUME_VARIANT,
        SECOND_PASS_RESUME_VARIANT,
        DEFAULT_RESUME_VARIANT,
        AUTO_RESUME_VARIANT_SELECTION_MODE,
        AUTO_RESUME_VARIANT_SELECTION_MODE,
    ]
    if job_ids:
        placeholders = ", ".join("?" for _ in job_ids)
        clauses.append(f"applications.job_id IN ({placeholders})")
        params.extend(job_ids)
    rows = connection.execute(
        f"""
        SELECT
            applications.job_id,
            applications.selected_resume_variant,
            manual_variant.variant_key AS manual_variant_key,
            v2_variant.variant_key AS v2_variant_key,
            v1_variant.variant_key AS v1_variant_key
        FROM applications
        LEFT JOIN application_resume_variants AS manual_variant
          ON manual_variant.job_id = applications.job_id
         AND manual_variant.variant_key = ?
        LEFT JOIN application_resume_variants AS v2_variant
          ON v2_variant.job_id = applications.job_id
         AND v2_variant.variant_key = ?
        LEFT JOIN application_resume_variants AS v1_variant
          ON v1_variant.job_id = applications.job_id
         AND v1_variant.variant_key = ?
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchall()

    changed_count = 0
    for row in rows:
        selected_variant = str(row["selected_resume_variant"] or DEFAULT_RESUME_VARIANT)
        next_variant = _preferred_resume_variant_from_row(row)
        if next_variant is None:
            continue
        if force or selected_variant != next_variant:
            _select_application_resume_variant_on_connection(
                connection=connection,
                job_id=str(row["job_id"]),
                variant_key=next_variant,
                selection_mode=AUTO_RESUME_VARIANT_SELECTION_MODE,
            )
            changed_count += 1
    return changed_count


def _preferred_resume_variant_from_row(row: sqlite3.Row) -> str | None:
    available_variants = {
        MANUAL_PASS_RESUME_VARIANT: row["manual_variant_key"],
        SECOND_PASS_RESUME_VARIANT: row["v2_variant_key"],
        DEFAULT_RESUME_VARIANT: row["v1_variant_key"],
    }
    for variant_key in RESUME_VARIANT_DEFAULT_PRECEDENCE:
        if available_variants.get(variant_key):
            return variant_key
    return None


def _ensure_application_columns(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA table_info(applications)").fetchall()
    existing_columns = {str(row["name"]) for row in rows}
    for column, column_type in APPLICATION_EXTRA_COLUMNS.items():
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE applications ADD COLUMN {column} {column_type}")
    connection.execute(
        """
        UPDATE applications
        SET date_matched = imported_at
        WHERE (date_matched IS NULL OR TRIM(date_matched) = '')
          AND imported_at IS NOT NULL
        """
    )


def fetch_existing_resume_job_ids(database_path: Path) -> set[str]:
    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT job_id
            FROM applications
            WHERE resume_content IS NOT NULL
            """
        ).fetchall()
    return {str(row["job_id"]) for row in rows}


def fetch_existing_cover_letter_job_ids(database_path: Path) -> set[str]:
    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT job_id
            FROM applications
            WHERE cover_letter_content IS NOT NULL
            """
        ).fetchall()
    return {str(row["job_id"]) for row in rows}


def fetch_application_job_ids(database_path: Path) -> set[str]:
    with connect_database(database_path) as connection:
        rows = connection.execute("SELECT job_id FROM applications").fetchall()
    return {str(row["job_id"]) for row in rows}


def fetch_application_job_records(
    database_path: Path,
    *,
    job_ids: list[str] | None = None,
) -> list[ApplicationJobRecord]:
    with connect_database(database_path) as connection:
        if job_ids is None:
            rows = connection.execute(
                """
                SELECT job_id, company, job_title, linkedin_url, job_description,
                       prompt_job_description, date_matched, date_posted, experience_level
                FROM applications
                ORDER BY company COLLATE NOCASE ASC, job_title COLLATE NOCASE ASC
                """
            ).fetchall()
        elif not job_ids:
            rows = []
        else:
            placeholders = ", ".join("?" for _ in job_ids)
            rows = connection.execute(
                f"""
                SELECT job_id, company, job_title, linkedin_url, job_description,
                       prompt_job_description, date_matched, date_posted, experience_level
                FROM applications
                WHERE job_id IN ({placeholders})
                """,
                job_ids,
            ).fetchall()
            row_by_job_id = {str(row["job_id"]): row for row in rows}
            rows = [row_by_job_id[job_id] for job_id in job_ids if job_id in row_by_job_id]
    return [
        ApplicationJobRecord(
            job_id=str(row["job_id"]),
            company=str(row["company"] or ""),
            job_title=str(row["job_title"] or ""),
            linkedin_url=str(row["linkedin_url"] or ""),
            job_description=row["job_description"],
            prompt_job_description=row["prompt_job_description"],
            date_matched=row["date_matched"],
            date_posted=row["date_posted"],
            experience_level=row["experience_level"],
        )
        for row in rows
    ]


def upsert_application_artifact(
    *,
    database_path: Path,
    job_id: str,
    company: str,
    job_title: str,
    linkedin_url: str,
    resume_path: Path | None,
    cover_letter_path: Path | None = None,
    job_description: str | None = None,
    prompt_job_description: str | None = None,
    applied_to: str = "No",
    date_applied: str | None = None,
    date_matched: str | None = None,
    date_posted: str | None = None,
    experience_level: str | None = None,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    applied_to = _normalize_applied_to(applied_to)
    date_matched = _date_value(date_matched) or now
    date_posted = _date_value(date_posted)
    experience_level = str(experience_level or "").strip()
    resume_content = (
        resume_path.read_bytes()
        if resume_path is not None and resume_path.is_file()
        else None
    )
    resume_updated_at = _artifact_timestamp(resume_path) if resume_content is not None else None
    cover_letter_content = (
        cover_letter_path.read_bytes()
        if cover_letter_path is not None and cover_letter_path.is_file()
        else None
    )
    cover_letter_updated_at = (
        _artifact_timestamp(cover_letter_path)
        if cover_letter_content is not None
        else None
    )
    ats_score = _calculate_ats_score(
        resume_content=resume_content,
        job_description=prompt_job_description or job_description,
    )
    with connect_database(database_path) as connection:
        connection.execute(
            """
            INSERT INTO applications (
                job_id, company, job_title, linkedin_url, job_description,
                prompt_job_description, resume_filename, resume_content, resume_mime_type,
                source_resume_path, resume_updated_at, cover_letter_filename,
                cover_letter_content, cover_letter_mime_type, source_cover_letter_path,
                cover_letter_updated_at, date_matched, date_posted, experience_level,
                ats_score, ats_parsing_score, ats_keyword_score, ats_semantic_score,
                ats_formatting_risk, ats_missing_terms, ats_updated_at, applied_to,
                date_applied, imported_at, updated_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(job_id) DO UPDATE SET
                company = excluded.company,
                job_title = excluded.job_title,
                linkedin_url = excluded.linkedin_url,
                job_description = COALESCE(
                    excluded.job_description,
                    applications.job_description
                ),
                prompt_job_description = COALESCE(
                    excluded.prompt_job_description,
                    applications.prompt_job_description
                ),
                resume_filename = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.resume_filename
                    WHEN excluded.resume_filename != '' THEN excluded.resume_filename
                    ELSE applications.resume_filename
                END,
                resume_content = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.resume_content
                    ELSE COALESCE(excluded.resume_content, applications.resume_content)
                END,
                resume_mime_type = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.resume_mime_type
                    WHEN excluded.resume_content IS NOT NULL THEN excluded.resume_mime_type
                    ELSE applications.resume_mime_type
                END,
                source_resume_path = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.source_resume_path
                    WHEN excluded.source_resume_path != '' THEN excluded.source_resume_path
                    ELSE applications.source_resume_path
                END,
                resume_updated_at = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.resume_updated_at
                    ELSE COALESCE(
                        excluded.resume_updated_at,
                        applications.resume_updated_at
                    )
                END,
                cover_letter_filename = CASE
                    WHEN COALESCE(NULLIF(applications.cover_letter_object, ''), '') != ''
                    THEN applications.cover_letter_filename
                    WHEN excluded.cover_letter_filename != '' THEN excluded.cover_letter_filename
                    ELSE applications.cover_letter_filename
                END,
                cover_letter_content = CASE
                    WHEN COALESCE(NULLIF(applications.cover_letter_object, ''), '') != ''
                    THEN applications.cover_letter_content
                    ELSE COALESCE(
                        excluded.cover_letter_content,
                        applications.cover_letter_content
                    )
                END,
                cover_letter_mime_type = CASE
                    WHEN COALESCE(NULLIF(applications.cover_letter_object, ''), '') != ''
                    THEN applications.cover_letter_mime_type
                    WHEN excluded.cover_letter_content IS NOT NULL
                    THEN excluded.cover_letter_mime_type
                    ELSE applications.cover_letter_mime_type
                END,
                source_cover_letter_path = CASE
                    WHEN COALESCE(NULLIF(applications.cover_letter_object, ''), '') != ''
                    THEN applications.source_cover_letter_path
                    WHEN excluded.source_cover_letter_path != ''
                    THEN excluded.source_cover_letter_path
                    ELSE applications.source_cover_letter_path
                END,
                cover_letter_updated_at = CASE
                    WHEN COALESCE(NULLIF(applications.cover_letter_object, ''), '') != ''
                    THEN applications.cover_letter_updated_at
                    ELSE COALESCE(
                        excluded.cover_letter_updated_at,
                        applications.cover_letter_updated_at
                    )
                END,
                date_matched = COALESCE(
                    NULLIF(applications.date_matched, ''),
                    excluded.date_matched
                ),
                date_posted = COALESCE(
                    NULLIF(excluded.date_posted, ''),
                    applications.date_posted
                ),
                experience_level = COALESCE(
                    NULLIF(excluded.experience_level, ''),
                    applications.experience_level
                ),
                ats_score = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_score
                    ELSE COALESCE(excluded.ats_score, applications.ats_score)
                END,
                ats_parsing_score = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_parsing_score
                    ELSE COALESCE(
                        excluded.ats_parsing_score,
                        applications.ats_parsing_score
                    )
                END,
                ats_keyword_score = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_keyword_score
                    ELSE COALESCE(
                        excluded.ats_keyword_score,
                        applications.ats_keyword_score
                    )
                END,
                ats_semantic_score = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_semantic_score
                    ELSE COALESCE(
                        excluded.ats_semantic_score,
                        applications.ats_semantic_score
                    )
                END,
                ats_formatting_risk = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_formatting_risk
                    ELSE COALESCE(
                        excluded.ats_formatting_risk,
                        applications.ats_formatting_risk
                    )
                END,
                ats_missing_terms = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_missing_terms
                    ELSE COALESCE(
                        excluded.ats_missing_terms,
                        applications.ats_missing_terms
                    )
                END,
                ats_updated_at = CASE
                    WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') != ''
                    THEN applications.ats_updated_at
                    ELSE COALESCE(excluded.ats_updated_at, applications.ats_updated_at)
                END,
                applied_to = CASE
                    WHEN applications.applied_to != 'No' THEN applications.applied_to
                    ELSE excluded.applied_to
                END,
                date_applied = COALESCE(applications.date_applied, excluded.date_applied),
                updated_at = excluded.updated_at
            """,
            (
                str(job_id),
                company,
                job_title,
                linkedin_url,
                job_description,
                prompt_job_description,
                resume_path.name if resume_path is not None else "",
                resume_content,
                "application/pdf",
                str(resume_path) if resume_path is not None else "",
                resume_updated_at,
                cover_letter_path.name if cover_letter_path is not None else "",
                cover_letter_content,
                "application/pdf",
                str(cover_letter_path) if cover_letter_path is not None else "",
                cover_letter_updated_at,
                date_matched,
                date_posted,
                experience_level,
                ats_score.overall_score if ats_score is not None else None,
                ats_score.parsing_score if ats_score is not None else None,
                ats_score.keyword_match_score if ats_score is not None else None,
                ats_score.semantic_match_score if ats_score is not None else None,
                ats_score.formatting_risk if ats_score is not None else None,
                _format_missing_terms(ats_score) if ats_score is not None else None,
                now if ats_score is not None else None,
                applied_to,
                date_applied,
                now,
                now,
            ),
        )
        connection.commit()


def add_linkedin_application_from_url(
    *,
    database_path: Path,
    linkedin_url: str,
) -> str:
    url = str(linkedin_url or "").strip()
    job_id = extract_job_id(url)
    if not job_id:
        raise ValueError("Paste a valid LinkedIn job URL.")

    details = asyncio.run(_fetch_linkedin_job_details(url))
    raw_description = str(details.description or "").strip() or None
    prompt_description = (
        _clean_prompt_job_description(raw_description) if raw_description else None
    )
    stored_job_id = details.job_id or job_id
    upsert_application_artifact(
        database_path=database_path,
        job_id=stored_job_id,
        company=details.company or "",
        job_title=details.title or "Unknown title",
        linkedin_url=str(details.job_url or url),
        resume_path=None,
        cover_letter_path=None,
        job_description=raw_description,
        prompt_job_description=prompt_description,
        date_posted=details.listed_at,
        experience_level=details.seniority_level,
    )
    return stored_job_id


def add_generic_application_from_url(
    *,
    database_path: Path,
    job_url: str,
) -> str:
    url = str(job_url or "").strip()
    details = asyncio.run(_fetch_generic_job_details(url))
    raw_description = str(details.description or "").strip() or None
    if raw_description is None:
        raise ValueError("No usable job description was found at that URL.")
    prompt_description = _clean_prompt_job_description(raw_description)
    upsert_application_artifact(
        database_path=database_path,
        job_id=details.job_id,
        company=details.company or "",
        job_title=details.title or "Job Posting",
        linkedin_url=str(details.job_url or url),
        resume_path=None,
        cover_letter_path=None,
        job_description=raw_description,
        prompt_job_description=prompt_description,
        date_posted=details.listed_at,
        experience_level=details.seniority_level,
    )
    return details.job_id


async def _fetch_linkedin_job_details(linkedin_url: str) -> JobDetails:
    settings = load_settings()
    provider = LinkedInPublicJobsProvider(
        user_agent=settings.user_agent,
        timeout_seconds=settings.timeout_seconds,
    )
    try:
        return await provider.get_job_details(linkedin_url)
    finally:
        await provider.aclose()


async def _fetch_generic_job_details(job_url: str) -> JobDetails:
    settings = load_settings()
    return await fetch_generic_job_details(
        url=job_url,
        user_agent=settings.user_agent,
        timeout_seconds=settings.timeout_seconds,
    )


def _clean_prompt_job_description(description: str) -> str:
    return clean_job_description_for_prompt(description)


def store_application_resume_first_draft(
    *,
    database_path: Path,
    job_id: str,
    application_resume_object: str,
    resume_html: str,
    resume_pdf: bytes,
    resume_html_path: Path | None = None,
    resume_pdf_path: Path | None = None,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT job_id, job_title, prompt_job_description, job_description
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Application row was not found for job_id={job_id}.")

        ats_score = _calculate_ats_score(
            resume_content=resume_pdf,
            job_description=row["prompt_job_description"] or row["job_description"],
        )
        connection.execute(
            """
            UPDATE applications
            SET application_resume_object = ?,
                application_resume_updated_at = ?,
                resume_html_filename = ?,
                resume_html_content = ?,
                resume_html_mime_type = 'text/html; charset=utf-8',
                source_resume_html_path = ?,
                resume_html_updated_at = ?,
                resume_filename = ?,
                resume_content = ?,
                resume_mime_type = 'application/pdf',
                source_resume_path = ?,
                resume_updated_at = ?,
                ats_score = ?,
                ats_parsing_score = ?,
                ats_keyword_score = ?,
                ats_semantic_score = ?,
                ats_formatting_risk = ?,
                ats_missing_terms = ?,
                ats_updated_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                application_resume_object,
                now,
                (
                    resume_html_path.name
                    if resume_html_path is not None
                    else _resume_html_filename(row)
                ),
                resume_html,
                str(resume_html_path) if resume_html_path is not None else "",
                now,
                resume_pdf_path.name if resume_pdf_path is not None else _resume_pdf_filename(row),
                resume_pdf,
                str(resume_pdf_path) if resume_pdf_path is not None else "",
                now,
                ats_score.overall_score if ats_score is not None else None,
                ats_score.parsing_score if ats_score is not None else None,
                ats_score.keyword_match_score if ats_score is not None else None,
                ats_score.semantic_match_score if ats_score is not None else None,
                ats_score.formatting_risk if ats_score is not None else None,
                _format_missing_terms(ats_score) if ats_score is not None else None,
                now if ats_score is not None else None,
                now,
                job_id,
            ),
        )
        _upsert_application_resume_variant(
            connection=connection,
            row=row,
            variant_key=DEFAULT_RESUME_VARIANT,
            variant_label="Draft v1",
            source="first_draft",
            parent_variant_key=None,
            application_resume_object=application_resume_object,
            resume_html=resume_html,
            resume_pdf=resume_pdf,
            ats_score=ats_score,
            resume_html_filename=(
                resume_html_path.name
                if resume_html_path is not None
                else _resume_html_filename(row)
            ),
            resume_filename=(
                resume_pdf_path.name if resume_pdf_path is not None else _resume_pdf_filename(row)
            ),
            source_resume_html_path=str(resume_html_path) if resume_html_path is not None else "",
            source_resume_path=str(resume_pdf_path) if resume_pdf_path is not None else "",
        )
        _apply_resume_variant_default_precedence(
            connection,
            job_ids=[job_id],
            force=True,
        )
        connection.commit()


def backfill_application_resume_v1_variants(database_path: Path) -> int:
    with connect_database(database_path) as connection:
        before = _resume_variant_count(connection, variant_key=DEFAULT_RESUME_VARIANT)
        _backfill_v1_resume_variants(connection)
        after = _resume_variant_count(connection, variant_key=DEFAULT_RESUME_VARIANT)
        connection.commit()
    return max(0, after - before)


def store_application_resume_variant(
    *,
    database_path: Path,
    job_id: str,
    variant_key: str,
    variant_label: str,
    source: str,
    application_resume_object: str,
    resume_html: str,
    resume_pdf: bytes,
    parent_variant_key: str | None = None,
    ats_score: AtsProxyScore | None = None,
    ats_diagnostics: Any | None = None,
    evidence_packet: Any | None = None,
    external_critique: Any | None = None,
    critique_prompt: str | None = None,
    critique_response: str | None = None,
    critique: Any | None = None,
    validation: Any | None = None,
    model_metadata: Any | None = None,
    resume_html_filename: str | None = None,
    resume_filename: str | None = None,
    source_resume_html_path: str = "",
    source_resume_path: str = "",
    select_after_store: bool = False,
    selection_mode: str | None = None,
) -> AtsProxyScore | None:
    _parse_application_resume_yaml(application_resume_object)
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Application row was not found for job_id={job_id}.")
        if ats_score is None:
            ats_score = _calculate_ats_score(
                resume_content=resume_pdf,
                job_description=row["prompt_job_description"] or row["job_description"],
            )
        _upsert_application_resume_variant(
            connection=connection,
            row=row,
            variant_key=variant_key,
            variant_label=variant_label,
            source=source,
            parent_variant_key=parent_variant_key,
            application_resume_object=application_resume_object,
            resume_html=resume_html,
            resume_pdf=resume_pdf,
            ats_score=ats_score,
            resume_html_filename=resume_html_filename,
            resume_filename=resume_filename,
            source_resume_html_path=source_resume_html_path,
            source_resume_path=source_resume_path,
            ats_diagnostics=ats_diagnostics,
            evidence_packet=evidence_packet,
            external_critique=external_critique,
            critique_prompt=critique_prompt,
            critique_response=critique_response,
            critique=critique,
            validation=validation,
            model_metadata=model_metadata,
        )
        if select_after_store:
            _select_application_resume_variant_on_connection(
                connection=connection,
                job_id=job_id,
                variant_key=variant_key,
                selection_mode=selection_mode
                or str(row["resume_variant_selection_mode"] or AUTO_RESUME_VARIANT_SELECTION_MODE),
            )
        else:
            _apply_resume_variant_default_precedence(
                connection,
                job_ids=[job_id],
                force=True,
            )
        connection.commit()
    return ats_score


def fetch_active_resume_refinement_job_ids(database_path: Path) -> list[str]:
    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT job_id
            FROM applications
            WHERE archived_at IS NULL
              AND COALESCE(NULLIF(application_resume_object, ''), '') != ''
            ORDER BY
                CASE applied_to
                    WHEN 'No' THEN 0
                    WHEN 'Accepted for interview' THEN 1
                    WHEN 'N/A' THEN 2
                    WHEN 'Rejected' THEN 3
                    WHEN 'Yes' THEN 4
                    ELSE 5
                END,
                company COLLATE NOCASE ASC,
                job_title COLLATE NOCASE ASC
            """
        ).fetchall()
    return [str(row["job_id"]) for row in rows]


def fetch_resume_variant_comparisons(
    *,
    database_path: Path,
    job_ids: list[str] | None = None,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[str] = []
    if active_only:
        clauses.append("applications.archived_at IS NULL")
    if job_ids:
        placeholders = ", ".join("?" for _ in job_ids)
        clauses.append(f"applications.job_id IN ({placeholders})")
        params.extend(job_ids)
    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect_database(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                applications.job_id,
                applications.company,
                applications.job_title,
                applications.selected_resume_variant,
                v1.ats_score AS v1_ats_score,
                v1.ats_missing_terms AS v1_ats_missing_terms,
                v1.updated_at AS v1_updated_at,
                v2.ats_score AS v2_ats_score,
                v2.ats_missing_terms AS v2_ats_missing_terms,
                v2.validation_json AS v2_validation_json,
                v2.model_metadata_json AS v2_model_metadata_json,
                v2.updated_at AS v2_updated_at
            FROM applications
            LEFT JOIN application_resume_variants AS v1
              ON v1.job_id = applications.job_id
             AND v1.variant_key = ?
            LEFT JOIN application_resume_variants AS v2
              ON v2.job_id = applications.job_id
             AND v2.variant_key = ?
            {where_clause}
            ORDER BY
                applications.company COLLATE NOCASE ASC,
                applications.job_title COLLATE NOCASE ASC
            """,
            [DEFAULT_RESUME_VARIANT, SECOND_PASS_RESUME_VARIANT, *params],
        ).fetchall()

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        validation = _loads_json_mapping(row["v2_validation_json"])
        model_metadata = _loads_json_mapping(row["v2_model_metadata_json"])
        accepted_change_ids = validation.get("accepted_change_ids")
        rejected_changes = validation.get("rejected_changes")
        v1_score = row["v1_ats_score"]
        v2_score = row["v2_ats_score"]
        comparisons.append(
            {
                "job_id": row["job_id"],
                "company": row["company"],
                "job_title": row["job_title"],
                "selected_resume_variant": row["selected_resume_variant"]
                or DEFAULT_RESUME_VARIANT,
                "v1": {
                    "ats_score": v1_score,
                    "missing_terms": row["v1_ats_missing_terms"],
                    "updated_at": row["v1_updated_at"],
                },
                "v2": {
                    "ats_score": v2_score,
                    "missing_terms": row["v2_ats_missing_terms"],
                    "updated_at": row["v2_updated_at"],
                    "accepted_changes": (
                        len(accepted_change_ids) if isinstance(accepted_change_ids, list) else 0
                    ),
                    "rejected_changes": (
                        len(rejected_changes) if isinstance(rejected_changes, list) else 0
                    ),
                    "is_valid": validation.get("is_valid"),
                    "model": model_metadata.get("model"),
                },
                "ats_delta": (
                    int(v2_score) - int(v1_score)
                    if v1_score is not None and v2_score is not None
                    else None
                ),
            }
        )
    return comparisons


def fetch_application_resume_variants(
    *,
    database_path: Path,
    job_id: str,
) -> list[dict[str, Any]]:
    application = _fetch_application(database_path, job_id)
    if application is None:
        return []
    selected_variant = str(
        application["selected_resume_variant"] or DEFAULT_RESUME_VARIANT
    )
    with connect_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM application_resume_variants
            WHERE job_id = ?
            ORDER BY
                CASE variant_key
                    WHEN ? THEN 0
                    WHEN ? THEN 1
                    WHEN ? THEN 2
                    ELSE 3
                END,
                updated_at DESC,
                variant_key COLLATE NOCASE ASC
            """,
            (
                job_id,
                DEFAULT_RESUME_VARIANT,
                SECOND_PASS_RESUME_VARIANT,
                MANUAL_PASS_RESUME_VARIANT,
            ),
        ).fetchall()
    baseline_score = next(
        (
            row["ats_score"]
            for row in rows
            if row["variant_key"] == DEFAULT_RESUME_VARIANT
        ),
        None,
    )
    return [
        _resume_variant_summary(
            row=row,
            selected_variant=selected_variant,
            baseline_score=baseline_score,
        )
        for row in rows
    ]


def fetch_application_resume_variant(
    *,
    database_path: Path,
    job_id: str,
    variant_key: str,
) -> sqlite3.Row | None:
    with connect_database(database_path) as connection:
        return connection.execute(
            """
            SELECT *
            FROM application_resume_variants
            WHERE job_id = ?
              AND variant_key = ?
            """,
            (job_id, variant_key),
        ).fetchone()


def select_application_resume_variant(
    *,
    database_path: Path,
    job_id: str,
    variant_key: str,
) -> dict[str, Any]:
    with connect_database(database_path) as connection:
        selected = _select_application_resume_variant_on_connection(
            connection=connection,
            job_id=job_id,
            variant_key=variant_key,
            selection_mode=MANUAL_RESUME_VARIANT_SELECTION_MODE,
        )
        connection.commit()
    return selected


def _select_application_resume_variant_on_connection(
    *,
    connection: sqlite3.Connection,
    job_id: str,
    variant_key: str,
    selection_mode: str,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    application = connection.execute(
        """
        SELECT job_id
        FROM applications
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()
    if application is None:
        raise ValueError(f"Application row was not found for job_id={job_id}.")
    variant = connection.execute(
        """
        SELECT *
        FROM application_resume_variants
        WHERE job_id = ?
          AND variant_key = ?
        """,
        (job_id, variant_key),
    ).fetchone()
    if variant is None:
        raise ValueError(
            f"Resume variant {variant_key!r} was not found for job_id={job_id}."
        )
    connection.execute(
        """
        UPDATE applications
        SET selected_resume_variant = ?,
            resume_variant_selection_mode = ?,
            application_resume_object = ?,
            application_resume_updated_at = ?,
            resume_html_filename = ?,
            resume_html_content = ?,
            resume_html_mime_type = ?,
            source_resume_html_path = ?,
            resume_html_updated_at = ?,
            resume_filename = ?,
            resume_content = ?,
            resume_mime_type = ?,
            source_resume_path = ?,
            resume_updated_at = ?,
            ats_score = ?,
            ats_parsing_score = ?,
            ats_keyword_score = ?,
            ats_semantic_score = ?,
            ats_formatting_risk = ?,
            ats_missing_terms = ?,
            ats_updated_at = ?,
            updated_at = ?
        WHERE job_id = ?
        """,
        (
            variant_key,
            selection_mode,
            variant["application_resume_object"],
            variant["updated_at"] or now,
            variant["resume_html_filename"] or "",
            variant["resume_html_content"],
            variant["resume_html_mime_type"] or "text/html; charset=utf-8",
            variant["source_resume_html_path"] or "",
            variant["resume_html_updated_at"] or variant["updated_at"] or now,
            variant["resume_filename"] or "",
            variant["resume_content"],
            variant["resume_mime_type"] or "application/pdf",
            variant["source_resume_path"] or "",
            variant["resume_updated_at"] or variant["updated_at"] or now,
            variant["ats_score"],
            variant["ats_parsing_score"],
            variant["ats_keyword_score"],
            variant["ats_semantic_score"],
            variant["ats_formatting_risk"],
            variant["ats_missing_terms"],
            variant["ats_updated_at"] or variant["updated_at"],
            now,
            job_id,
        ),
    )
    return {
        "variant_key": variant_key,
        "variant_label": _resume_variant_label(variant_key, variant["variant_label"]),
    }


def _resume_variant_summary(
    *,
    row: sqlite3.Row,
    selected_variant: str,
    baseline_score: Any,
) -> dict[str, Any]:
    validation = _loads_json_mapping(row["validation_json"])
    critique = _loads_json_mapping(row["critique_json"])
    model_metadata = _loads_json_mapping(row["model_metadata_json"])
    accepted_change_ids = _string_list(validation.get("accepted_change_ids"))
    rejected_changes = _dict_list(validation.get("rejected_changes"))
    proposed_changes = _dict_list(critique.get("proposed_changes"))
    proposed_change_by_id = {
        str(change.get("change_id") or ""): change for change in proposed_changes
    }
    accepted_changes = [
        {
            **proposed_change_by_id.get(change_id, {}),
            "change_id": change_id,
        }
        for change_id in accepted_change_ids
    ]
    unsupported_claims = sorted(
        {
            claim
            for change in proposed_changes
            for claim in _string_list(change.get("unsupported_claims"))
        }
    )
    ats_score = row["ats_score"]
    ats_delta = (
        int(ats_score) - int(baseline_score)
        if ats_score is not None and baseline_score is not None
        else None
    )
    variant_key = str(row["variant_key"])
    return {
        "variant_key": variant_key,
        "variant_label": _resume_variant_label(variant_key, row["variant_label"]),
        "action_label": _resume_variant_action_label(variant_key, row["variant_label"]),
        "badge_class": _resume_variant_badge_class(variant_key),
        "source": row["source"],
        "parent_variant_key": row["parent_variant_key"],
        "is_selected": variant_key == selected_variant,
        "resume_filename": row["resume_filename"],
        "resume_html_filename": row["resume_html_filename"],
        "has_pdf": row["resume_content"] is not None,
        "has_html": bool(row["resume_html_content"]),
        "ats_score": ats_score,
        "ats_delta": ats_delta,
        "ats_parsing_score": row["ats_parsing_score"],
        "ats_keyword_score": row["ats_keyword_score"],
        "ats_semantic_score": row["ats_semantic_score"],
        "ats_formatting_risk": row["ats_formatting_risk"],
        "ats_missing_terms": row["ats_missing_terms"],
        "updated_at": row["updated_at"],
        "model": model_metadata.get("model"),
        "accepted_changes": accepted_changes,
        "rejected_changes": rejected_changes,
        "unsupported_claims": unsupported_claims,
        "validation_is_valid": validation.get("is_valid"),
    }


def _resume_variant_review_details(variants: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_changes: list[dict[str, Any]] = []
    rejected_changes: list[dict[str, Any]] = []
    unsupported_claims: list[str] = []
    seen_accepted: set[str] = set()
    seen_rejected: set[str] = set()
    seen_claims: set[str] = set()

    for variant in variants:
        for change in variant["accepted_changes"]:
            key = _stable_review_key(change)
            if key in seen_accepted:
                continue
            seen_accepted.add(key)
            accepted_changes.append(change)
        for rejected in variant["rejected_changes"]:
            key = _stable_review_key(rejected)
            if key in seen_rejected:
                continue
            seen_rejected.add(key)
            rejected_changes.append(rejected)
        for claim in variant["unsupported_claims"]:
            if claim in seen_claims:
                continue
            seen_claims.add(claim)
            unsupported_claims.append(claim)

    return {
        "accepted_changes": accepted_changes,
        "rejected_changes": rejected_changes,
        "unsupported_claims": unsupported_claims,
    }


def _stable_review_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _resume_variant_label(variant_key: str, fallback: Any = None) -> str:
    return RESUME_VARIANT_LABELS.get(variant_key, str(fallback or variant_key))


def _resume_variant_action_label(variant_key: str, fallback: Any = None) -> str:
    return {
        DEFAULT_RESUME_VARIANT: "Use v1 draft",
        SECOND_PASS_RESUME_VARIANT: "Use v2 draft",
        MANUAL_PASS_RESUME_VARIANT: "Use manual pass",
    }.get(variant_key, f"Use {_resume_variant_label(variant_key, fallback)}")


def _resume_variant_badge_class(variant_key: str) -> str:
    return {
        DEFAULT_RESUME_VARIANT: "is-v1",
        SECOND_PASS_RESUME_VARIANT: "is-v2",
        MANUAL_PASS_RESUME_VARIANT: "is-manual",
    }.get(variant_key, "is-other")


def _resume_variant_unified_diff(
    first_variant: sqlite3.Row | None,
    second_variant: sqlite3.Row | None,
) -> str:
    if first_variant is None or second_variant is None:
        return ""
    first_text = str(first_variant["application_resume_object"] or "")
    second_text = str(second_variant["application_resume_object"] or "")
    if first_text == second_text:
        return ""
    return "\n".join(
        difflib.unified_diff(
            first_text.splitlines(),
            second_text.splitlines(),
            fromfile=_resume_variant_label(str(first_variant["variant_key"])),
            tofile=_resume_variant_label(str(second_variant["variant_key"])),
            lineterm="",
        )
    )


def _resume_variant_count(connection: sqlite3.Connection, *, variant_key: str) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM application_resume_variants
        WHERE variant_key = ?
        """,
        (variant_key,),
    ).fetchone()
    return int(row["count"] or 0)


def _upsert_application_resume_variant(
    *,
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    variant_key: str,
    variant_label: str,
    source: str,
    parent_variant_key: str | None,
    application_resume_object: str,
    resume_html: str,
    resume_pdf: bytes,
    ats_score: AtsProxyScore | None,
    resume_html_filename: str | None = None,
    resume_filename: str | None = None,
    source_resume_html_path: str = "",
    source_resume_path: str = "",
    ats_diagnostics: Any | None = None,
    evidence_packet: Any | None = None,
    external_critique: Any | None = None,
    critique_prompt: str | None = None,
    critique_response: str | None = None,
    critique: Any | None = None,
    validation: Any | None = None,
    model_metadata: Any | None = None,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    connection.execute(
        """
        INSERT INTO application_resume_variants (
            job_id, variant_key, variant_label, source, parent_variant_key,
            application_resume_object, resume_html_filename, resume_html_content,
            resume_html_mime_type, source_resume_html_path, resume_html_updated_at,
            resume_filename, resume_content, resume_mime_type, source_resume_path,
            resume_updated_at, ats_score, ats_parsing_score, ats_keyword_score,
            ats_semantic_score, ats_formatting_risk, ats_missing_terms, ats_updated_at,
            ats_diagnostics_json, evidence_packet_json, external_critique_json,
            critique_prompt, critique_response, critique_json, validation_json,
            model_metadata_json, created_at, updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, 'text/html; charset=utf-8', ?, ?, ?, ?,
            'application/pdf', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(job_id, variant_key) DO UPDATE SET
            variant_label = excluded.variant_label,
            source = excluded.source,
            parent_variant_key = excluded.parent_variant_key,
            application_resume_object = excluded.application_resume_object,
            resume_html_filename = excluded.resume_html_filename,
            resume_html_content = excluded.resume_html_content,
            resume_html_mime_type = excluded.resume_html_mime_type,
            source_resume_html_path = excluded.source_resume_html_path,
            resume_html_updated_at = excluded.resume_html_updated_at,
            resume_filename = excluded.resume_filename,
            resume_content = excluded.resume_content,
            resume_mime_type = excluded.resume_mime_type,
            source_resume_path = excluded.source_resume_path,
            resume_updated_at = excluded.resume_updated_at,
            ats_score = excluded.ats_score,
            ats_parsing_score = excluded.ats_parsing_score,
            ats_keyword_score = excluded.ats_keyword_score,
            ats_semantic_score = excluded.ats_semantic_score,
            ats_formatting_risk = excluded.ats_formatting_risk,
            ats_missing_terms = excluded.ats_missing_terms,
            ats_updated_at = excluded.ats_updated_at,
            ats_diagnostics_json = excluded.ats_diagnostics_json,
            evidence_packet_json = excluded.evidence_packet_json,
            external_critique_json = excluded.external_critique_json,
            critique_prompt = excluded.critique_prompt,
            critique_response = excluded.critique_response,
            critique_json = excluded.critique_json,
            validation_json = excluded.validation_json,
            model_metadata_json = excluded.model_metadata_json,
            updated_at = excluded.updated_at
        """,
        (
            row["job_id"],
            variant_key,
            variant_label,
            source,
            parent_variant_key,
            application_resume_object,
            resume_html_filename or _variant_resume_html_filename(row, variant_key),
            resume_html,
            source_resume_html_path,
            now,
            resume_filename or _variant_resume_pdf_filename(row, variant_key),
            resume_pdf,
            source_resume_path,
            now,
            ats_score.overall_score if ats_score is not None else None,
            ats_score.parsing_score if ats_score is not None else None,
            ats_score.keyword_match_score if ats_score is not None else None,
            ats_score.semantic_match_score if ats_score is not None else None,
            ats_score.formatting_risk if ats_score is not None else None,
            _format_missing_terms(ats_score) if ats_score is not None else None,
            now if ats_score is not None else None,
            _json_dumps_or_none(ats_diagnostics),
            _json_dumps_or_none(evidence_packet),
            _json_dumps_or_none(external_critique),
            critique_prompt,
            critique_response,
            _json_dumps_or_none(critique),
            _json_dumps_or_none(validation),
            _json_dumps_or_none(model_metadata),
            now,
            now,
        ),
    )


def _json_dumps_or_none(value: Any | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _loads_json_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def store_application_resume_object(
    *,
    database_path: Path,
    job_id: str,
    application_resume_object: str,
) -> None:
    _parse_application_resume_yaml(application_resume_object)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE applications
            SET application_resume_object = ?,
                application_resume_updated_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (application_resume_object, now, now, job_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Application row was not found for job_id={job_id}.")
        connection.commit()


def save_application_resume_edit(
    *,
    database_path: Path,
    job_id: str,
    application_resume_object: str,
    template_path: Path | None = None,
    backup_current: bool = True,
) -> AtsProxyScore | None:
    resume = _parse_application_resume_yaml(application_resume_object)
    resume_html = render_resume_html_from_mapping(
        resume=resume,
        template_path=_resume_template_path(template_path),
    )
    resume_pdf = render_resume_pdf_from_html(resume_html)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Application row was not found for job_id={job_id}.")
        ats_score = _calculate_ats_score(
            resume_content=resume_pdf,
            job_description=row["prompt_job_description"] or row["job_description"],
        )
        connection.execute(
            """
            UPDATE applications
            SET application_resume_backup_object = ?,
                application_resume_backup_created_at = ?,
                application_resume_object = ?,
                application_resume_updated_at = ?,
                resume_html_filename = ?,
                resume_html_content = ?,
                resume_html_mime_type = 'text/html; charset=utf-8',
                source_resume_html_path = '',
                resume_html_updated_at = ?,
                resume_filename = ?,
                resume_content = ?,
                resume_mime_type = 'application/pdf',
                source_resume_path = '',
                resume_updated_at = ?,
                ats_score = ?,
                ats_parsing_score = ?,
                ats_keyword_score = ?,
                ats_semantic_score = ?,
                ats_formatting_risk = ?,
                ats_missing_terms = ?,
                ats_updated_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                row["application_resume_object"] if backup_current else row[
                    "application_resume_backup_object"
                ],
                now if backup_current else row["application_resume_backup_created_at"],
                application_resume_object,
                now,
                _resume_html_filename(row),
                resume_html,
                now,
                _resume_pdf_filename(row),
                resume_pdf,
                now,
                ats_score.overall_score if ats_score is not None else None,
                ats_score.parsing_score if ats_score is not None else None,
                ats_score.keyword_match_score if ats_score is not None else None,
                ats_score.semantic_match_score if ats_score is not None else None,
                ats_score.formatting_risk if ats_score is not None else None,
                _format_missing_terms(ats_score) if ats_score is not None else None,
                now if ats_score is not None else None,
                now,
                job_id,
            ),
        )
        connection.commit()
    return ats_score


def sync_application_resume_to_draft(
    *,
    database_path: Path,
    job_id: str,
    template_path: Path | None = None,
) -> AtsProxyScore | None:
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Application row was not found for job_id={job_id}.")
        application_resume_object = str(row["application_resume_object"] or "").strip()
        if not application_resume_object:
            raise ValueError("Application resume object was not found.")

        resume = _parse_application_resume_yaml(application_resume_object)
        resume_html = render_resume_html_from_mapping(
            resume=resume,
            template_path=_resume_template_path(template_path),
        )
        resume_pdf = render_resume_pdf_from_html(resume_html)
        ats_score = _calculate_ats_score(
            resume_content=resume_pdf,
            job_description=row["prompt_job_description"] or row["job_description"],
        )
        now = datetime.now(UTC).isoformat(timespec="seconds")
        connection.execute(
            """
            UPDATE applications
            SET resume_html_filename = ?,
                resume_html_content = ?,
                resume_html_mime_type = 'text/html; charset=utf-8',
                source_resume_html_path = '',
                resume_html_updated_at = ?,
                resume_filename = ?,
                resume_content = ?,
                resume_mime_type = 'application/pdf',
                source_resume_path = '',
                resume_updated_at = ?,
                ats_score = ?,
                ats_parsing_score = ?,
                ats_keyword_score = ?,
                ats_semantic_score = ?,
                ats_formatting_risk = ?,
                ats_missing_terms = ?,
                ats_updated_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                _resume_html_filename(row),
                resume_html,
                now,
                _resume_pdf_filename(row),
                resume_pdf,
                now,
                ats_score.overall_score if ats_score is not None else None,
                ats_score.parsing_score if ats_score is not None else None,
                ats_score.keyword_match_score if ats_score is not None else None,
                ats_score.semantic_match_score if ats_score is not None else None,
                ats_score.formatting_risk if ats_score is not None else None,
                _format_missing_terms(ats_score) if ats_score is not None else None,
                now if ats_score is not None else None,
                now,
                job_id,
            ),
        )
        connection.commit()
    return ats_score


def revert_application_resume_edit(
    *,
    database_path: Path,
    job_id: str,
    template_path: Path | None = None,
) -> AtsProxyScore | None:
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT application_resume_backup_object
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Application row was not found for job_id={job_id}.")
    backup_object = str(row["application_resume_backup_object"] or "").strip()
    if not backup_object:
        raise ValueError("No manual resume backup is available for this application.")
    return save_application_resume_edit(
        database_path=database_path,
        job_id=job_id,
        application_resume_object=backup_object,
        template_path=template_path,
        backup_current=True,
    )


def save_cover_letter_edit(
    *,
    database_path: Path,
    job_id: str,
    body_html: str,
) -> None:
    sanitized_html = _sanitize_cover_letter_body_html(body_html)
    body_text = _cover_letter_plain_text(sanitized_html)
    if not body_text:
        raise ValueError("Cover letter text is empty.")
    cover_letter_object = _dump_cover_letter_object(
        {
            "schema_version": COVER_LETTER_OBJECT_SCHEMA_VERSION,
            "source": "manual",
            "body_html": sanitized_html,
            "body_text": body_text,
        }
    )
    cover_letter_pdf = render_cover_letter_pdf_from_clo_html(sanitized_html)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Application row was not found for job_id={job_id}.")
        connection.execute(
            """
            UPDATE applications
            SET cover_letter_object = ?,
                cover_letter_object_updated_at = ?,
                cover_letter_filename = ?,
                cover_letter_content = ?,
                cover_letter_mime_type = 'application/pdf',
                source_cover_letter_path = '',
                cover_letter_updated_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                cover_letter_object,
                now,
                _cover_letter_pdf_filename(row),
                cover_letter_pdf,
                now,
                now,
                job_id,
            ),
        )
        connection.commit()


def save_description_edit(
    *,
    database_path: Path,
    job_id: str,
    job_description: str,
    prompt_job_description: str,
) -> AtsProxyScore | None:
    parsed_description = job_description.strip() or None
    prompt_description = prompt_job_description.strip() or None
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT resume_content
            FROM applications
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Application row was not found for job_id={job_id}.")

        score = _calculate_ats_score(
            resume_content=row["resume_content"],
            job_description=prompt_description or parsed_description,
        )
        connection.execute(
            """
            UPDATE applications
            SET job_description = ?,
                prompt_job_description = ?,
                ats_score = ?,
                ats_parsing_score = ?,
                ats_keyword_score = ?,
                ats_semantic_score = ?,
                ats_formatting_risk = ?,
                ats_missing_terms = ?,
                ats_updated_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                parsed_description,
                prompt_description,
                score.overall_score if score is not None else None,
                score.parsing_score if score is not None else None,
                score.keyword_match_score if score is not None else None,
                score.semantic_match_score if score is not None else None,
                score.formatting_risk if score is not None else None,
                _format_missing_terms(score) if score is not None else None,
                now if score is not None else None,
                now,
                job_id,
            ),
        )
        connection.commit()
    return score


def render_cover_letter_pdf_from_clo_html(body_html: str) -> bytes:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ManualCoverLetterBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=COVER_LETTER_BODY_FONT_SIZE,
        leading=COVER_LETTER_BODY_LEADING,
        spaceAfter=COVER_LETTER_PARAGRAPH_SPACE_AFTER,
        textColor=RESUME_BODY_COLOR,
    )
    story: list[Any] = [
        HRFlowable(
            width="100%",
            thickness=7.2,
            color=EMERALD_ACCENT,
            spaceBefore=0,
            spaceAfter=0,
        ),
        Spacer(1, 34),
    ]
    paragraphs = _cover_letter_paragraph_markup(body_html)
    if not paragraphs:
        paragraphs = ["Cover letter content was empty."]
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, body))

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=72,
        leftMargin=72,
        topMargin=54,
        bottomMargin=72,
    )
    document.build(story)
    return buffer.getvalue()


def delete_applications(
    *,
    database_path: Path,
    job_ids: list[str],
) -> int:
    normalized_job_ids = sorted({job_id.strip() for job_id in job_ids if job_id.strip()})
    if not normalized_job_ids:
        return 0

    placeholders = ", ".join("?" for _ in normalized_job_ids)
    with connect_database(database_path) as connection:
        cursor = connection.execute(
            f"DELETE FROM applications WHERE job_id IN ({placeholders})",
            normalized_job_ids,
        )
        deleted_count = cursor.rowcount
        connection.commit()

    return deleted_count


def archive_applications(
    *,
    database_path: Path,
    job_ids: list[str],
) -> int:
    normalized_job_ids = _selected_job_ids(job_ids)
    if not normalized_job_ids:
        return 0

    placeholders = ", ".join("?" for _ in normalized_job_ids)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        cursor = connection.execute(
            f"""
            UPDATE applications
            SET archived_at = ?,
                updated_at = ?
            WHERE job_id IN ({placeholders})
              AND archived_at IS NULL
            """,
            [now, now, *normalized_job_ids],
        )
        archived_count = cursor.rowcount
        connection.commit()

    return archived_count


def unarchive_applications(
    *,
    database_path: Path,
    job_ids: list[str],
) -> int:
    normalized_job_ids = _selected_job_ids(job_ids)
    if not normalized_job_ids:
        return 0

    placeholders = ", ".join("?" for _ in normalized_job_ids)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with connect_database(database_path) as connection:
        cursor = connection.execute(
            f"""
            UPDATE applications
            SET archived_at = NULL,
                updated_at = ?
            WHERE job_id IN ({placeholders})
              AND archived_at IS NOT NULL
            """,
            [now, *normalized_job_ids],
        )
        restored_count = cursor.rowcount
        connection.commit()

    return restored_count


def refresh_missing_ats_scores(
    database_path: Path,
    *,
    archive_filter: str = "all",
) -> int:
    archive_clause = {
        "active": "AND archived_at IS NULL",
        "archived": "AND archived_at IS NOT NULL",
        "all": "",
    }.get(archive_filter, "")
    with connect_database(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT job_id, resume_content, prompt_job_description, job_description
            FROM applications
            WHERE (ats_score IS NULL OR ats_missing_terms IS NULL)
              AND resume_content IS NOT NULL
              AND (
                prompt_job_description IS NOT NULL
                OR job_description IS NOT NULL
              )
              {archive_clause}
            """
        ).fetchall()
        updated_count = 0
        for row in rows:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            score = _calculate_ats_score(
                resume_content=row["resume_content"],
                job_description=row["prompt_job_description"] or row["job_description"],
            )
            if score is None:
                continue
            connection.execute(
                """
                UPDATE applications
                SET ats_score = ?,
                    ats_parsing_score = ?,
                    ats_keyword_score = ?,
                    ats_semantic_score = ?,
                    ats_formatting_risk = ?,
                    ats_missing_terms = ?,
                    ats_updated_at = ?
                WHERE job_id = ?
                """,
                (
                    score.overall_score,
                    score.parsing_score,
                    score.keyword_match_score,
                    score.semantic_match_score,
                    score.formatting_risk,
                    _format_missing_terms(score),
                    now,
                    row["job_id"],
                ),
            )
            updated_count += 1
        connection.commit()
    return updated_count


def start_background_action(
    *,
    regenerate_mode: str,
    job_ids: list[str],
    highlight_with_codex: bool = False,
    run_manual: bool = False,
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
    runner: BackgroundActionRunner | None = None,
) -> BackgroundActionRun:
    selected_manual_pass_profile = _manual_pass_profile_key(manual_pass_profile)
    run = _create_background_action_run(
        title=_background_action_title(
            regenerate_mode=regenerate_mode,
            job_ids=job_ids,
            highlight_with_codex=highlight_with_codex,
            run_manual=run_manual,
            manual_pass_profile=selected_manual_pass_profile,
        )
    )
    target = runner or _run_background_action
    thread = threading.Thread(
        target=target,
        kwargs={
            "run_id": run.run_id,
            "regenerate_mode": regenerate_mode,
            "job_ids": job_ids,
            "highlight_with_codex": highlight_with_codex,
            "run_manual": run_manual,
            "manual_pass_profile": selected_manual_pass_profile,
        },
        daemon=True,
    )
    thread.start()
    return run


def start_seed_background_action(
    *,
    max_jobs: int,
    date_posted: str,
    run_v1: bool,
    run_v2: bool,
    run_manual: bool,
    run_highlight: bool,
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
    runner: BackgroundActionRunner | None = None,
) -> BackgroundActionRun:
    selected_manual_pass_profile = _manual_pass_profile_key(manual_pass_profile)
    run = _create_background_action_run(
        title=_seed_background_action_title(
            max_jobs=max_jobs,
            date_posted=date_posted,
            run_v1=run_v1,
            run_v2=run_v2,
            run_manual=run_manual,
            run_highlight=run_highlight,
            manual_pass_profile=selected_manual_pass_profile,
        )
    )
    target = runner or _run_seed_workflow_action
    thread = threading.Thread(
        target=target,
        kwargs={
            "run_id": run.run_id,
            "max_jobs": max_jobs,
            "date_posted": date_posted,
            "run_v1": run_v1,
            "run_v2": run_v2,
            "run_manual": run_manual,
            "run_highlight": run_highlight,
            "manual_pass_profile": selected_manual_pass_profile,
        },
        daemon=True,
    )
    thread.start()
    return run


def background_action_snapshots() -> list[dict[str, object]]:
    with _ACTION_RUN_LOCK:
        runs = sorted(
            _ACTION_RUNS.values(),
            key=lambda run: run.started_at,
            reverse=True,
        )
        return [
            {
                "id": run.run_id,
                "title": run.title,
                "status": run.status,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "return_code": run.return_code,
                "messages": run.messages[-20:],
            }
            for run in runs
        ]


def _create_background_action_run(*, title: str) -> BackgroundActionRun:
    run = BackgroundActionRun(
        run_id=uuid.uuid4().hex,
        title=title,
        status="running",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    with _ACTION_RUN_LOCK:
        _ACTION_RUNS[run.run_id] = run
        _trim_background_action_runs_locked()
    _append_background_action_message(run.run_id, "Queued background action.")
    return run


def _trim_background_action_runs_locked() -> None:
    if len(_ACTION_RUNS) <= MAX_ACTION_RUNS:
        return
    old_runs = sorted(_ACTION_RUNS.values(), key=lambda run: run.started_at)
    for run in old_runs[: len(_ACTION_RUNS) - MAX_ACTION_RUNS]:
        del _ACTION_RUNS[run.run_id]


def _append_background_action_message(run_id: str, message: str) -> None:
    value = " ".join(str(message).split())
    if not value:
        return
    timestamp = datetime.now(UTC).strftime("%H:%M:%S")
    with _ACTION_RUN_LOCK:
        run = _ACTION_RUNS.get(run_id)
        if run is None:
            return
        run.messages.append(f"{timestamp} {value}")
        if len(run.messages) > MAX_ACTION_MESSAGES:
            run.messages = run.messages[-MAX_ACTION_MESSAGES:]


def _finish_background_action_run(
    run_id: str,
    *,
    status: str,
    return_code: int | None = None,
) -> None:
    with _ACTION_RUN_LOCK:
        run = _ACTION_RUNS.get(run_id)
        if run is None:
            return
        run.status = status
        run.return_code = return_code
        run.finished_at = datetime.now(UTC).isoformat(timespec="seconds")


def _run_background_action(
    *,
    run_id: str,
    regenerate_mode: str,
    job_ids: list[str],
    highlight_with_codex: bool = False,
    run_manual: bool = False,
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> None:
    try:
        _run_job_workflow_stages(
            run_id=run_id,
            job_ids=job_ids,
            stages=_background_workflow_stages(
                regenerate_mode=regenerate_mode,
                run_manual=run_manual,
                highlight_with_codex=highlight_with_codex,
                manual_pass_profile=manual_pass_profile,
            ),
        )
        _append_background_action_message(run_id, "Background action completed.")
        _finish_background_action_run(run_id, status="completed", return_code=0)
    except Exception as exc:
        _append_background_action_message(run_id, f"Background action failed: {exc}")
        _finish_background_action_run(run_id, status="failed", return_code=1)


def _run_regenerate_action(
    *,
    run_id: str,
    regenerate_mode: str,
    job_ids: list[str],
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> None:
    command = _regenerate_make_command(
        regenerate_mode=regenerate_mode,
        job_ids=job_ids,
        manual_pass_profile=manual_pass_profile,
    )
    _run_make_command(
        run_id=run_id,
        command=command,
        output_stream_error="regeneration command did not provide an output stream",
        failure_label="regeneration command",
        completion_message="Regeneration command completed.",
    )


def _run_highlight_action(
    *,
    run_id: str,
    job_ids: list[str],
    variant_key: str | None = None,
) -> None:
    command = _highlight_make_command(job_ids=job_ids, variant_key=variant_key)
    _run_make_command(
        run_id=run_id,
        command=command,
        output_stream_error="Codex highlighting command did not provide an output stream",
        failure_label="Codex highlighting command",
        completion_message="Codex highlighting command completed.",
    )


def _run_seed_workflow_action(
    *,
    run_id: str,
    max_jobs: int,
    date_posted: str,
    run_v1: bool,
    run_v2: bool,
    run_manual: bool,
    run_highlight: bool,
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> None:
    try:
        seed_output = _run_make_command(
            run_id=run_id,
            command=_seed_make_command(max_jobs=max_jobs, date_posted=date_posted),
            output_stream_error="seed command did not provide an output stream",
            failure_label="seed command",
            completion_message="Seed command completed.",
            collect_output=True,
        )
        job_ids = _extract_seeded_job_ids(seed_output)
        if not job_ids:
            _append_background_action_message(
                run_id,
                "Seed command returned no new job IDs; skipping selected workflow steps.",
            )
        elif run_v1 or run_v2 or run_manual or run_highlight:
            _append_background_action_message(
                run_id,
                f"Seeded job IDs: {' '.join(job_ids)}",
            )
            _run_job_workflow_stages(
                run_id=run_id,
                job_ids=job_ids,
                stages=_seed_workflow_stages(
                    run_v1=run_v1,
                    run_v2=run_v2,
                    run_manual=run_manual,
                    run_highlight=run_highlight,
                    manual_pass_profile=manual_pass_profile,
                ),
            )
        else:
            _append_background_action_message(
                run_id,
                "Seed-only workflow selected; no resume actions requested.",
            )
        _append_background_action_message(run_id, "Seed workflow completed.")
        _finish_background_action_run(run_id, status="completed", return_code=0)
    except Exception as exc:
        _append_background_action_message(run_id, f"Seed workflow failed: {exc}")
        _finish_background_action_run(run_id, status="failed", return_code=1)


def _run_job_workflow_stages(
    *,
    run_id: str,
    job_ids: list[str],
    stages: list[BackgroundWorkflowStage],
) -> None:
    if not stages:
        return
    if not job_ids:
        raise ValueError("At least one job id is required for workflow actions.")

    active_job_ids = list(job_ids)
    failures: list[dict[str, str]] = []
    for stage in stages:
        if not active_job_ids:
            break
        _append_background_action_message(
            run_id,
            f"Starting {stage.label} for {len(active_job_ids)} active job(s).",
        )
        succeeded: list[str] = []
        for job_id in active_job_ids:
            try:
                _run_background_workflow_stage(
                    run_id=run_id,
                    stage=stage,
                    job_id=job_id,
                )
            except Exception as exc:
                failures.append(
                    {
                        "job_id": job_id,
                        "stage": stage.label,
                        "error": str(exc),
                    }
                )
                _append_background_action_message(
                    run_id,
                    f"[{job_id}] {stage.label} failed: {exc}",
                )
                continue
            succeeded.append(job_id)
        active_job_ids = succeeded

    if failures and not active_job_ids:
        first_failure = failures[0]
        raise RuntimeError(
            "All workflow jobs failed before completing every requested stage. "
            f"First failure: {first_failure['job_id']} "
            f"{first_failure['stage']}: {first_failure['error']}"
        )
    if failures:
        _append_background_action_message(
            run_id,
            (
                "Background workflow completed with partial failures: "
                f"{len(active_job_ids)} succeeded, {len(failures)} failed stage(s)."
            ),
        )
        _append_background_action_message(
            run_id,
            f"Completed job IDs: {' '.join(active_job_ids)}",
        )


def _run_background_workflow_stage(
    *,
    run_id: str,
    stage: BackgroundWorkflowStage,
    job_id: str,
) -> None:
    if stage.regenerate_mode is not None:
        if stage.regenerate_mode == "manual_pass":
            _run_regenerate_action(
                run_id=run_id,
                regenerate_mode=stage.regenerate_mode,
                job_ids=[job_id],
                manual_pass_profile=(
                    stage.manual_pass_profile or DEFAULT_MANUAL_PASS_PROFILE
                ),
            )
        else:
            _run_regenerate_action(
                run_id=run_id,
                regenerate_mode=stage.regenerate_mode,
                job_ids=[job_id],
            )
        return
    _run_highlight_action(
        run_id=run_id,
        job_ids=[job_id],
        variant_key=stage.highlight_variant_key,
    )


def _run_make_command(
    *,
    run_id: str,
    command: list[str],
    output_stream_error: str,
    failure_label: str,
    completion_message: str,
    collect_output: bool = False,
) -> str:
    _append_background_action_message(run_id, f"Running {' '.join(command)}")
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=_project_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError(output_stream_error)
    collected_output: list[str] = []
    for line in process.stdout:
        if collect_output:
            collected_output.append(line)
        _append_background_action_message(run_id, line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{failure_label} exited with status {return_code}")
    _append_background_action_message(run_id, completion_message)
    return "".join(collected_output)


def _regenerate_make_command(
    *,
    regenerate_mode: str,
    job_ids: list[str],
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> list[str]:
    target = REGENERATE_ACTION_TARGETS.get(regenerate_mode)
    if target is None:
        raise ValueError(f"Unsupported regeneration mode: {regenerate_mode}")
    if not job_ids:
        raise ValueError("At least one job id is required for regeneration.")
    command = ["make", target, f"JOB_IDS={' '.join(job_ids)}"]
    if regenerate_mode in _DRAFT_REGENERATE_MODES:
        command.append("FIRST_DRAFT_FORCE=1")
    if regenerate_mode == "manual_pass":
        selected_manual_pass_profile = _manual_pass_profile_key(manual_pass_profile)
        command.append(f"MANUAL_PASS_PROFILE={selected_manual_pass_profile.value}")
    return command


def _highlight_make_command(
    *,
    job_ids: list[str],
    variant_key: str | None = None,
) -> list[str]:
    if not job_ids:
        raise ValueError("At least one job id is required for Codex highlighting.")
    command = ["make", "highlight-draft-resumes", f"JOB_IDS={' '.join(job_ids)}"]
    if variant_key:
        command.append(f"HIGHLIGHT_RESUME_VARIANT={variant_key}")
    return command


def _background_workflow_stages(
    *,
    regenerate_mode: str,
    run_manual: bool,
    highlight_with_codex: bool,
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> list[BackgroundWorkflowStage]:
    selected_manual_pass_profile = (
        _manual_pass_profile_key(manual_pass_profile)
        if regenerate_mode == "manual_pass" or run_manual
        else DEFAULT_MANUAL_PASS_PROFILE
    )
    stages: list[BackgroundWorkflowStage] = []
    if regenerate_mode:
        stages.append(
            BackgroundWorkflowStage(
                label=(
                    _manual_pass_stage_label(selected_manual_pass_profile)
                    if regenerate_mode == "manual_pass"
                    else _regenerate_stage_label(regenerate_mode)
                ),
                regenerate_mode=regenerate_mode,
                manual_pass_profile=(
                    selected_manual_pass_profile
                    if regenerate_mode == "manual_pass"
                    else None
                ),
            )
        )
    if run_manual:
        stages.append(
            BackgroundWorkflowStage(
                label=_manual_pass_stage_label(selected_manual_pass_profile),
                regenerate_mode="manual_pass",
                manual_pass_profile=selected_manual_pass_profile,
            )
        )
    if highlight_with_codex and regenerate_mode != "highlight_drafts":
        stages.append(
            BackgroundWorkflowStage(
                label="Codex highlighting",
                highlight_variant_key=_highlight_variant_for_chained_workflow(
                    regenerate_mode=regenerate_mode,
                    run_manual=run_manual,
                ),
            )
        )
    return stages


def _seed_workflow_stages(
    *,
    run_v1: bool,
    run_v2: bool,
    run_manual: bool,
    run_highlight: bool,
    manual_pass_profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> list[BackgroundWorkflowStage]:
    selected_manual_pass_profile = (
        _manual_pass_profile_key(manual_pass_profile)
        if run_manual
        else DEFAULT_MANUAL_PASS_PROFILE
    )
    stages: list[BackgroundWorkflowStage] = []
    if run_v1:
        stages.append(
            BackgroundWorkflowStage(
                label=_regenerate_stage_label("draft_resumes"),
                regenerate_mode="draft_resumes",
            )
        )
    if run_v2:
        stages.append(
            BackgroundWorkflowStage(
                label=_regenerate_stage_label("refine_drafts"),
                regenerate_mode="refine_drafts",
            )
        )
    if run_manual:
        stages.append(
            BackgroundWorkflowStage(
                label=_manual_pass_stage_label(selected_manual_pass_profile),
                regenerate_mode="manual_pass",
                manual_pass_profile=selected_manual_pass_profile,
            )
        )
    if run_highlight:
        stages.append(
            BackgroundWorkflowStage(
                label="Codex highlighting",
                highlight_variant_key=_highlight_variant_for_seed_workflow(
                    run_v2=run_v2,
                    run_manual=run_manual,
                ),
            )
        )
    return stages


def _regenerate_stage_label(regenerate_mode: str) -> str:
    return {
        "draft_resumes": "v1 draft generation",
        "resume_variants": "v1 and v2 resume workflow",
        "refine_drafts": "v2 resume refinement",
        "aro_objects": "ARO regeneration",
        "sync_draft_to_aro": "draft-to-ARO sync",
        "highlight_drafts": "Codex highlighting",
        "manual_pass": "Codex manual pass",
    }.get(regenerate_mode, regenerate_mode)


def _manual_pass_stage_label(manual_pass_profile: ManualPassProfileKey) -> str:
    return f"Codex manual pass ({manual_pass_profile.value})"


def _highlight_variant_for_chained_workflow(
    *,
    regenerate_mode: str,
    run_manual: bool,
) -> str | None:
    if run_manual:
        return MANUAL_PASS_RESUME_VARIANT
    if regenerate_mode == "manual_pass":
        return MANUAL_PASS_RESUME_VARIANT
    if regenerate_mode in {"resume_variants", "refine_drafts"}:
        return SECOND_PASS_RESUME_VARIANT
    return None


def _highlight_variant_for_seed_workflow(
    *,
    run_v2: bool,
    run_manual: bool,
) -> str | None:
    if run_manual:
        return MANUAL_PASS_RESUME_VARIANT
    if run_v2:
        return SECOND_PASS_RESUME_VARIANT
    return None


def _seed_make_command(*, max_jobs: int, date_posted: str) -> list[str]:
    if max_jobs < 1:
        raise ValueError("Seed job count must be at least 1.")
    if date_posted not in _seed_date_posted_values():
        raise ValueError(f"Unsupported seed date posted filter: {date_posted}")
    return ["make", "seed-jobs", f"MAX_JOBS={max_jobs}", f"DATE_POSTED={date_posted}"]


def _extract_seeded_job_ids(output: str) -> list[str]:
    decoder = json.JSONDecoder()
    payload: dict[str, Any] | None = None
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("seeded_applications"), list):
            payload = value

    if payload is None:
        return []

    job_ids: list[str] = []
    seen: set[str] = set()
    for item in payload["seeded_applications"]:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id") or "").strip()
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        job_ids.append(job_id)
    return job_ids


def _project_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "Makefile").is_file():
        return candidate
    return Path.cwd()


def _selected_job_ids(values: list[str]) -> list[str]:
    job_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        job_id = str(value or "").strip()
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        job_ids.append(job_id)
    return job_ids


def _parse_job_url_list(value: Any) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\s,]+", str(value or "")):
        url = part.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    if not urls:
        raise ValueError("Paste at least one job URL.")
    return urls


def _add_url_regenerate_mode(*, run_v2: bool) -> str:
    return "resume_variants" if run_v2 else "draft_resumes"


def _add_url_workflow_validation_error(*, run_v2: bool, run_manual: bool) -> str | None:
    if run_manual and not run_v2:
        return "Run v2 refinement must be selected before Codex manual pass."
    return None


def _format_url_add_failures(failures: list[tuple[str, str]]) -> str:
    if not failures:
        return ""
    shown = [f"{url}: {message}" for url, message in failures[:3]]
    if len(failures) > len(shown):
        shown.append(f"{len(failures) - len(shown)} more")
    return "; ".join(shown)


def _normalize_action_mode(regenerate_mode: str) -> tuple[str, bool]:
    if regenerate_mode == "resume_variants_manual_pass":
        return "resume_variants", True
    return regenerate_mode, False


def _manual_pass_profile_key(
    value: str | ManualPassProfileKey | None,
) -> ManualPassProfileKey:
    if value is None:
        value = DEFAULT_MANUAL_PASS_PROFILE
    return parse_manual_pass_profile(value).key


def _background_action_title(
    *,
    regenerate_mode: str,
    job_ids: list[str],
    highlight_with_codex: bool = False,
    run_manual: bool = False,
    manual_pass_profile: ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> str:
    parts: list[str] = []
    if regenerate_mode:
        label = {
            "draft_resumes": "regenerate draft v1 only",
            "resume_variants": "run v1 and v2 resume workflow",
            "refine_drafts": "run v2 resume refinement",
            "aro_objects": "regenerate ARO object(s)",
            "sync_draft_to_aro": "sync draft to ARO",
            "highlight_drafts": "Codex highlight selected resume",
            "manual_pass": (
                f"Codex manual pass resume ({manual_pass_profile.value})"
            ),
        }.get(regenerate_mode, "regenerate docs")
        parts.append(f"{label} for {len(job_ids)} job(s)")
    if run_manual:
        parts.append(
            "Codex manual pass resume "
            f"({manual_pass_profile.value}) for {len(job_ids)} job(s)"
        )
    if highlight_with_codex and regenerate_mode != "highlight_drafts":
        parts.append(f"Codex highlight selected resume for {len(job_ids)} job(s)")
    return " + ".join(parts) or "background action"


def _seed_background_action_title(
    *,
    max_jobs: int,
    date_posted: str,
    run_v1: bool,
    run_v2: bool,
    run_manual: bool,
    run_highlight: bool,
    manual_pass_profile: ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
) -> str:
    parts = [
        f"seed up to {max_jobs} job(s)",
        _seed_date_posted_label(date_posted).lower(),
    ]
    if run_v1:
        parts.append("v1 draft")
    if run_v2:
        parts.append("v2 refinement")
    if run_manual:
        parts.append(f"Codex manual pass ({manual_pass_profile.value})")
    if run_highlight:
        parts.append("Codex highlight")
    return " + ".join(parts)


def _parse_seed_max_jobs(value: Any) -> int:
    try:
        max_jobs = int(str(value or DEFAULT_SEED_MAX_JOBS).strip())
    except ValueError as exc:
        raise ValueError("Seed job count must be a whole number.") from exc
    if max_jobs < 1 or max_jobs > MAX_SEED_JOBS_FROM_UI:
        raise ValueError(f"Seed job count must be between 1 and {MAX_SEED_JOBS_FROM_UI}.")
    return max_jobs


def _parse_seed_date_posted(value: Any) -> str:
    date_posted = str(value or DEFAULT_SEED_DATE_POSTED).strip()
    if date_posted not in _seed_date_posted_values():
        raise ValueError("Choose a valid posting date window.")
    return date_posted


def _seed_date_posted_values() -> set[str]:
    return {value for value, _ in SEED_DATE_POSTED_OPTIONS}


def _seed_date_posted_label(value: str) -> str:
    return dict(SEED_DATE_POSTED_OPTIONS).get(value, value)


def _seed_workflow_validation_error(
    *,
    run_v1: bool,
    run_v2: bool,
    run_manual: bool,
    run_highlight: bool,
) -> str | None:
    if run_v2 and not run_v1:
        return "Run v1 draft must be selected before v2 refinement."
    if run_manual and (not run_v1 or not run_v2):
        return "Run v1 draft and v2 refinement must be selected before manual pass."
    if run_highlight and not run_v1:
        return "Run v1 draft must be selected before Codex highlighting."
    return None


def create_app(
    *,
    database_path: Path,
    output_dir: Path,
    background_action_runner: BackgroundActionRunner | None = None,
):
    from flask import (
        Flask,
        Response,
        abort,
        flash,
        jsonify,
        redirect,
        render_template_string,
        request,
        send_file,
    )

    app = Flask(__name__)
    app.config["SECRET_KEY"] = "linkedin-career-local-only"
    app.config["DATABASE_PATH"] = database_path
    app.config["OUTPUT_DIR"] = output_dir
    app.jinja_env.filters["display_date"] = _display_date
    app.jinja_env.filters["display_timestamp"] = _display_timestamp
    app.jinja_env.filters["rich_text"] = rich_text
    app.jinja_env.globals["bullet_text_for_editing"] = _bullet_text_for_editing
    app.jinja_env.globals["job_bulk_bullet_text"] = _job_bulk_bullet_text
    app.jinja_env.globals["resume_variant_label"] = _resume_variant_label
    app.jinja_env.globals["resume_variant_badge_class"] = _resume_variant_badge_class
    app.jinja_env.globals["application_status_row_class"] = _application_status_row_class

    def redirect_to_index_state():
        return redirect(_safe_index_return_path(request.values.get("return_to")))

    @app.get("/")
    def index():
        view_state = _view_state_from_args(request.args)
        refresh_missing_ats_scores(database_path, archive_filter=view_state["archive"])
        rows = _fetch_applications(database_path, archive_filter=view_state["archive"])
        archive_counts = _application_archive_counts(database_path)
        current_query = request.query_string.decode("utf-8")
        current_path = f"/?{current_query}" if current_query else "/"
        stats = {
            "total": len(rows),
            "active": archive_counts["active"],
            "archived": archive_counts["archived"],
            "applied": sum(1 for row in rows if row["applied_to"] == "Yes"),
            "pending": sum(1 for row in rows if row["applied_to"] == "No"),
            "not_applicable": sum(1 for row in rows if row["applied_to"] == "N/A"),
            "rejected": sum(1 for row in rows if row["applied_to"] == "Rejected"),
            "interview": sum(
                1 for row in rows if row["applied_to"] == "Accepted for interview"
            ),
        }
        return render_template_string(
            INDEX_TEMPLATE,
            rows=rows,
            stats=stats,
            view_state=view_state,
            current_path=current_path,
            manual_pass_profile_options=MANUAL_PASS_PROFILE_OPTIONS,
            default_manual_pass_profile=DEFAULT_MANUAL_PASS_PROFILE.value,
        )

    @app.post("/applications/<job_id>")
    def update_application(job_id: str):
        applied_to = _normalize_applied_to(request.form.get("applied_to"))
        date_applied = request.form.get("date_applied") or None
        notes = request.form.get("notes", "")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with connect_database(database_path) as connection:
            connection.execute(
                """
                UPDATE applications
                SET applied_to = ?, date_applied = ?, notes = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (applied_to, date_applied, notes, now, job_id),
            )
            connection.commit()
        if applied_to == "Yes":
            deleted_count = cleanup_downloaded_application_pdfs()
            flash(f"Application updated. Cleared {deleted_count} downloaded PDF file(s).")
        else:
            flash("Application updated.")
        return redirect_to_index_state()

    @app.get("/applications/add")
    def add_application():
        return render_template_string(
            ADD_APPLICATION_TEMPLATE,
            return_to=_safe_index_return_path(request.args.get("return_to")),
            default_seed_max_jobs=DEFAULT_SEED_MAX_JOBS,
            max_seed_jobs=MAX_SEED_JOBS_FROM_UI,
            seed_date_posted_options=SEED_DATE_POSTED_OPTIONS,
            default_seed_date_posted=DEFAULT_SEED_DATE_POSTED,
            manual_pass_profile_options=MANUAL_PASS_PROFILE_OPTIONS,
            default_manual_pass_profile=DEFAULT_MANUAL_PASS_PROFILE.value,
        )

    @app.post("/applications/add/seed")
    def add_seeded_applications():
        try:
            max_jobs = _parse_seed_max_jobs(request.form.get("max_jobs"))
            date_posted = _parse_seed_date_posted(request.form.get("date_posted"))
        except ValueError as exc:
            flash(f"Seed workflow failed: {exc}")
            return redirect(_add_application_return_path(request.form.get("return_to")))

        run_v1 = bool(request.form.get("run_v1"))
        run_v2 = bool(request.form.get("run_v2"))
        run_manual = bool(request.form.get("run_manual"))
        run_highlight = bool(request.form.get("run_highlight"))
        validation_error = _seed_workflow_validation_error(
            run_v1=run_v1,
            run_v2=run_v2,
            run_manual=run_manual,
            run_highlight=run_highlight,
        )
        if validation_error:
            flash(f"Seed workflow failed: {validation_error}")
            return redirect(_add_application_return_path(request.form.get("return_to")))
        try:
            manual_pass_profile = (
                _manual_pass_profile_key(request.form.get("manual_pass_profile"))
                if run_manual
                else DEFAULT_MANUAL_PASS_PROFILE
            )
        except ValueError as exc:
            flash(f"Seed workflow failed: {exc}")
            return redirect(_add_application_return_path(request.form.get("return_to")))

        run = start_seed_background_action(
            max_jobs=max_jobs,
            date_posted=date_posted,
            run_v1=run_v1,
            run_v2=run_v2,
            run_manual=run_manual,
            run_highlight=run_highlight,
            manual_pass_profile=manual_pass_profile,
            runner=background_action_runner,
        )
        flash(f"Started seed workflow: {run.title}.")
        return redirect(_add_application_return_path(request.form.get("return_to")))

    @app.post("/applications/add/linkedin")
    def add_linkedin_application():
        run_v2 = bool(request.form.get("run_v2"))
        run_manual = bool(request.form.get("run_manual"))
        highlight_with_codex = bool(request.form.get("highlight_with_codex"))
        validation_error = _add_url_workflow_validation_error(
            run_v2=run_v2,
            run_manual=run_manual,
        )
        if validation_error:
            flash(f"LinkedIn add failed: {validation_error}")
            return redirect(_add_application_return_path(request.form.get("return_to")))
        try:
            manual_pass_profile = (
                _manual_pass_profile_key(request.form.get("manual_pass_profile"))
                if run_manual
                else DEFAULT_MANUAL_PASS_PROFILE
            )
        except ValueError as exc:
            flash(f"LinkedIn add failed: {exc}")
            return redirect(_add_application_return_path(request.form.get("return_to")))
        try:
            urls = _parse_job_url_list(request.form.get("linkedin_url", ""))
        except ValueError as exc:
            flash(f"LinkedIn add failed: {exc}")
            return redirect(_add_application_return_path(request.form.get("return_to")))

        job_ids: list[str] = []
        failures: list[tuple[str, str]] = []
        for url in urls:
            try:
                job_ids.append(
                    add_linkedin_application_from_url(
                        database_path=database_path,
                        linkedin_url=url,
                    )
                )
            except (ValueError, LinkedInCareerMcpError) as exc:
                failures.append((url, str(exc)))
        job_ids = _selected_job_ids(job_ids)
        if not job_ids:
            flash(f"LinkedIn add failed: {_format_url_add_failures(failures)}")
            return redirect(_add_application_return_path(request.form.get("return_to")))

        run = start_background_action(
            regenerate_mode=_add_url_regenerate_mode(run_v2=run_v2),
            job_ids=job_ids,
            highlight_with_codex=highlight_with_codex,
            run_manual=run_manual,
            manual_pass_profile=manual_pass_profile,
            runner=background_action_runner,
        )
        message = (
            f"Added {len(job_ids)} LinkedIn job(s). "
            f"Started background action: {run.title}."
        )
        if failures:
            message = (
                f"{message} Failed {len(failures)} URL(s): "
                f"{_format_url_add_failures(failures)}."
            )
        flash(message)
        return redirect(_add_application_return_path(request.form.get("return_to")))

    @app.post("/applications/add/other")
    def add_other_application():
        run_v2 = bool(request.form.get("run_v2"))
        run_manual = bool(request.form.get("run_manual"))
        highlight_with_codex = bool(request.form.get("highlight_with_codex"))
        validation_error = _add_url_workflow_validation_error(
            run_v2=run_v2,
            run_manual=run_manual,
        )
        if validation_error:
            flash(f"Other URL add failed: {validation_error}")
            return redirect(_add_application_return_path(request.form.get("return_to")))
        try:
            manual_pass_profile = (
                _manual_pass_profile_key(request.form.get("manual_pass_profile"))
                if run_manual
                else DEFAULT_MANUAL_PASS_PROFILE
            )
        except ValueError as exc:
            flash(f"Other URL add failed: {exc}")
            return redirect(_add_application_return_path(request.form.get("return_to")))
        try:
            urls = _parse_job_url_list(request.form.get("other_url", ""))
        except ValueError as exc:
            flash(f"Other URL add failed: {exc}")
            return redirect(_add_application_return_path(request.form.get("return_to")))

        job_ids: list[str] = []
        failures: list[tuple[str, str]] = []
        for url in urls:
            try:
                job_ids.append(
                    add_generic_application_from_url(
                        database_path=database_path,
                        job_url=url,
                    )
                )
            except (ValueError, LinkedInCareerMcpError) as exc:
                failures.append((url, str(exc)))
        job_ids = _selected_job_ids(job_ids)
        if not job_ids:
            flash(f"Other URL add failed: {_format_url_add_failures(failures)}")
            return redirect(_add_application_return_path(request.form.get("return_to")))

        run = start_background_action(
            regenerate_mode=_add_url_regenerate_mode(run_v2=run_v2),
            job_ids=job_ids,
            highlight_with_codex=highlight_with_codex,
            run_manual=run_manual,
            manual_pass_profile=manual_pass_profile,
            runner=background_action_runner,
        )
        message = (
            f"Added {len(job_ids)} job URL(s). "
            f"Started background action: {run.title}."
        )
        if failures:
            message = (
                f"{message} Failed {len(failures)} URL(s): "
                f"{_format_url_add_failures(failures)}."
            )
        flash(message)
        return redirect(_add_application_return_path(request.form.get("return_to")))

    @app.post("/actions/run")
    def run_actions():
        regenerate_mode = str(request.form.get("regenerate_mode") or "").strip()
        regenerate_requested = bool(regenerate_mode)
        highlight_with_codex = bool(request.form.get("highlight_with_codex"))
        job_ids = _selected_job_ids(request.form.getlist("job_id"))
        normalized_regenerate_mode, run_manual = _normalize_action_mode(regenerate_mode)
        if not regenerate_requested:
            flash("Choose at least one action to run.")
            return redirect_to_index_state()
        if regenerate_requested and not job_ids:
            flash("Select at least one job before regenerating documents.")
            return redirect_to_index_state()
        if regenerate_requested and regenerate_mode not in (
            set(REGENERATE_ACTION_TARGETS) | COMPOSITE_ACTION_MODES
        ):
            flash("Choose a valid regeneration option.")
            return redirect_to_index_state()
        if highlight_with_codex and regenerate_mode not in CHAINABLE_HIGHLIGHT_MODES:
            flash(
                "Codex highlighting can only be chained after v1, v1+v2, "
                "or v1+v2+manual workflows."
            )
            return redirect_to_index_state()
        manual_stage_requested = (
            normalized_regenerate_mode == "manual_pass" or run_manual
        )
        try:
            manual_pass_profile = (
                _manual_pass_profile_key(request.form.get("manual_pass_profile"))
                if manual_stage_requested
                else DEFAULT_MANUAL_PASS_PROFILE
            )
        except ValueError as exc:
            flash(f"Background action failed: {exc}")
            return redirect_to_index_state()

        run = start_background_action(
            regenerate_mode=normalized_regenerate_mode if regenerate_requested else "",
            job_ids=job_ids,
            highlight_with_codex=highlight_with_codex,
            run_manual=run_manual,
            manual_pass_profile=manual_pass_profile,
            runner=background_action_runner,
        )
        flash(f"Started background action: {run.title}.")
        return redirect_to_index_state()

    @app.get("/actions/status")
    def action_status():
        return jsonify({"runs": background_action_snapshots()})

    @app.post("/applications/delete")
    def bulk_delete_applications():
        job_ids = request.form.getlist("job_id")
        deleted_count = delete_applications(
            database_path=database_path,
            job_ids=job_ids,
        )
        flash(f"Deleted {deleted_count} application rows.")
        return redirect_to_index_state()

    @app.post("/applications/archive")
    def bulk_archive_applications():
        job_ids = request.form.getlist("job_id")
        archived_count = archive_applications(
            database_path=database_path,
            job_ids=job_ids,
        )
        flash(f"Archived {archived_count} application rows.")
        return redirect_to_index_state()

    @app.post("/applications/unarchive")
    def bulk_unarchive_applications():
        job_ids = request.form.getlist("job_id")
        restored_count = unarchive_applications(
            database_path=database_path,
            job_ids=job_ids,
        )
        flash(f"Restored {restored_count} application rows.")
        return redirect_to_index_state()

    @app.get("/linkedin/<job_id>")
    def open_linkedin_job(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or not row["linkedin_url"]:
            abort(404)
        _open_url_in_chromium(str(row["linkedin_url"]))
        flash("Opened job URL in Chromium.")
        return redirect_to_index_state()

    @app.get("/resumes/<job_id>")
    def resume(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or row["resume_content"] is None:
            return Response("Resume was not found in the database.", status=404)
        return send_file(
            BytesIO(row["resume_content"]),
            mimetype=row["resume_mime_type"],
            download_name=row["resume_filename"],
            as_attachment=False,
        )

    @app.get("/resume-html/<job_id>")
    def resume_html(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or not row["resume_html_content"]:
            return Response("Resume HTML was not found in the database.", status=404)
        return Response(
            row["resume_html_content"],
            mimetype=row["resume_html_mime_type"] or "text/html; charset=utf-8",
        )

    @app.get("/resumes/<job_id>/variants")
    def resume_variants_review(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None:
            abort(404)
        variants = fetch_application_resume_variants(
            database_path=database_path,
            job_id=job_id,
        )
        if not variants:
            return Response("Resume variants were not found in the database.", status=404)
        v1_variant = fetch_application_resume_variant(
            database_path=database_path,
            job_id=job_id,
            variant_key=DEFAULT_RESUME_VARIANT,
        )
        v2_variant = fetch_application_resume_variant(
            database_path=database_path,
            job_id=job_id,
            variant_key=SECOND_PASS_RESUME_VARIANT,
        )
        return render_template_string(
            RESUME_VARIANTS_TEMPLATE,
            row=row,
            variants=variants,
            review_details=_resume_variant_review_details(variants),
            diff_text=_resume_variant_unified_diff(v1_variant, v2_variant),
            return_to=_safe_index_return_path(request.args.get("return_to")),
        )

    @app.get("/resumes/<job_id>/variants/<variant_key>")
    def resume_variant_pdf(job_id: str, variant_key: str):
        variant = fetch_application_resume_variant(
            database_path=database_path,
            job_id=job_id,
            variant_key=variant_key,
        )
        if variant is None or variant["resume_content"] is None:
            return Response("Resume variant PDF was not found in the database.", status=404)
        return send_file(
            BytesIO(variant["resume_content"]),
            mimetype=variant["resume_mime_type"] or "application/pdf",
            download_name=variant["resume_filename"] or f"{job_id}_{variant_key}.pdf",
            as_attachment=False,
        )

    @app.get("/resume-html/<job_id>/variants/<variant_key>")
    def resume_variant_html(job_id: str, variant_key: str):
        variant = fetch_application_resume_variant(
            database_path=database_path,
            job_id=job_id,
            variant_key=variant_key,
        )
        if variant is None or not variant["resume_html_content"]:
            return Response("Resume variant HTML was not found in the database.", status=404)
        return Response(
            variant["resume_html_content"],
            mimetype=variant["resume_html_mime_type"] or "text/html; charset=utf-8",
        )

    @app.post("/resumes/<job_id>/variants/<variant_key>/use")
    def resume_variant_use(job_id: str, variant_key: str):
        try:
            selected = select_application_resume_variant(
                database_path=database_path,
                job_id=job_id,
                variant_key=variant_key,
            )
        except ValueError as exc:
            flash(str(exc))
        else:
            flash(f"Using {selected['variant_label']} for resume HTML/PDF links.")
        return redirect(
            _resume_variant_review_return_path(job_id, request.form.get("return_to"))
        )

    @app.get("/resumes/<job_id>/edit")
    def resume_edit(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or not row["application_resume_object"]:
            return Response("Application resume object was not found.", status=404)
        try:
            resume = _parse_application_resume_yaml(row["application_resume_object"])
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            return Response(f"Application resume object could not be parsed: {exc}", status=500)
        return_to = _safe_index_return_path(request.args.get("return_to"))
        return render_template_string(
            RESUME_EDIT_TEMPLATE,
            row=row,
            resume=resume,
            return_to=return_to,
            variants_return_path=_resume_variant_review_return_path(job_id, return_to),
        )

    @app.post("/resumes/<job_id>/edit")
    def resume_edit_save(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or not row["application_resume_object"]:
            return Response("Application resume object was not found.", status=404)
        try:
            resume = _parse_application_resume_yaml(row["application_resume_object"])
            updated_resume = _apply_resume_editor_form(resume, request.form)
            updated_aro = _dump_application_resume_yaml(updated_resume)
            score = save_application_resume_edit(
                database_path=database_path,
                job_id=job_id,
                application_resume_object=updated_aro,
            )
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            flash(f"Resume save failed: {exc}")
            return redirect(_resume_edit_return_path(job_id, request.form.get("return_to")))
        score_text = f" ATS: {score.overall_score}/100." if score is not None else ""
        flash(f"Saved resume edits and regenerated artifacts.{score_text}")
        return redirect(_resume_edit_return_path(job_id, request.form.get("return_to")))

    @app.post("/resumes/<job_id>/edit/revert")
    def resume_edit_revert(job_id: str):
        try:
            score = revert_application_resume_edit(
                database_path=database_path,
                job_id=job_id,
            )
        except ValueError as exc:
            flash(str(exc))
        else:
            score_text = f" ATS: {score.overall_score}/100." if score is not None else ""
            flash(f"Reverted resume to the prior manual backup.{score_text}")
        return redirect(_resume_edit_return_path(job_id, request.form.get("return_to")))

    @app.post("/resumes/<job_id>/sync")
    def resume_sync_to_aro(job_id: str):
        try:
            score = sync_application_resume_to_draft(
                database_path=database_path,
                job_id=job_id,
            )
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            flash(f"Draft sync failed: {exc}")
        else:
            score_text = f" ATS: {score.overall_score}/100." if score is not None else ""
            flash(f"Synced draft resume to the current ARO.{score_text}")
        return redirect(_resume_edit_return_path(job_id, request.form.get("return_to")))

    @app.get("/resumes/<job_id>/download")
    def resume_download(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or row["resume_content"] is None:
            return Response("Resume was not found in the database.", status=404)
        return send_file(
            BytesIO(row["resume_content"]),
            mimetype=row["resume_mime_type"],
            download_name=row["resume_filename"],
            as_attachment=True,
        )

    @app.post("/resumes/<job_id>/copy-to-downloads")
    def resume_copy_to_downloads(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or row["resume_content"] is None:
            abort(404)
        try:
            destination = copy_application_artifact_to_downloads(
                row=row,
                artifact_kind="resume",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            flash(f"Resume copy failed: {exc}")
        else:
            flash(f"Copied resume to {_download_display_path(destination)}.")
        return redirect_to_index_state()

    @app.get("/cover-letters/<job_id>/edit")
    def cover_letter_edit(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None:
            abort(404)
        try:
            cover_letter_object = _parse_cover_letter_object(row["cover_letter_object"])
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            return Response(f"Cover letter object could not be parsed: {exc}", status=500)
        return render_template_string(
            COVER_LETTER_EDIT_TEMPLATE,
            row=row,
            cover_letter_object=cover_letter_object,
            return_to=_safe_index_return_path(request.args.get("return_to")),
        )

    @app.post("/cover-letters/<job_id>/edit")
    def cover_letter_edit_save(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None:
            abort(404)
        try:
            save_cover_letter_edit(
                database_path=database_path,
                job_id=job_id,
                body_html=request.form.get("body_html", ""),
            )
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            flash(f"Cover letter save failed: {exc}")
        else:
            flash("Saved cover letter and rendered PDF.")
        return redirect(
            _cover_letter_edit_return_path(job_id, request.form.get("return_to"))
        )

    @app.get("/cover-letters/<job_id>")
    def cover_letter(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or row["cover_letter_content"] is None:
            return Response("Cover letter was not found in the database.", status=404)
        return send_file(
            BytesIO(row["cover_letter_content"]),
            mimetype=row["cover_letter_mime_type"],
            download_name=row["cover_letter_filename"],
            as_attachment=False,
        )

    @app.get("/cover-letters/<job_id>/download")
    def cover_letter_download(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or row["cover_letter_content"] is None:
            return Response("Cover letter was not found in the database.", status=404)
        return send_file(
            BytesIO(row["cover_letter_content"]),
            mimetype=row["cover_letter_mime_type"],
            download_name=row["cover_letter_filename"],
            as_attachment=True,
        )

    @app.post("/cover-letters/<job_id>/copy-to-downloads")
    def cover_letter_copy_to_downloads(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None or row["cover_letter_content"] is None:
            abort(404)
        try:
            destination = copy_application_artifact_to_downloads(
                row=row,
                artifact_kind="cover_letter",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            flash(f"Cover letter copy failed: {exc}")
        else:
            flash(f"Copied cover letter to {_download_display_path(destination)}.")
        return redirect_to_index_state()

    @app.get("/descriptions/<job_id>")
    def compare_descriptions(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None:
            abort(404)
        return render_template_string(
            DESCRIPTION_COMPARE_TEMPLATE,
            row=row,
            return_to=_safe_index_return_path(request.args.get("return_to")),
            removed_text=_description_removed_text(
                row["job_description"],
                row["prompt_job_description"],
            ),
            diff_rows=_description_diff_rows(
                row["job_description"],
                row["prompt_job_description"],
            ),
        )

    @app.post("/descriptions/<job_id>")
    def compare_descriptions_save(job_id: str):
        row = _fetch_application(database_path, job_id)
        if row is None:
            abort(404)
        try:
            score = save_description_edit(
                database_path=database_path,
                job_id=job_id,
                job_description=request.form.get("job_description", ""),
                prompt_job_description=request.form.get("prompt_job_description", ""),
            )
        except ValueError as exc:
            flash(f"Description save failed: {exc}")
        else:
            score_text = f" ATS: {score.overall_score}/100." if score is not None else ""
            flash(f"Saved descriptions and refreshed ATS fields.{score_text}")
        return redirect(
            _description_edit_return_path(job_id, request.form.get("return_to"))
        )

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the local LinkedIn application tracker.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--database", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    database_path = Path(args.database) if args.database else output_dir / DEFAULT_DATABASE
    app = create_app(database_path=database_path, output_dir=output_dir)
    if args.open_browser:
        _schedule_browser_open(host=args.host, port=args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


def _fetch_applications(
    database_path: Path,
    *,
    archive_filter: str = "active",
) -> list[sqlite3.Row]:
    archive_clause = {
        "active": "WHERE archived_at IS NULL",
        "archived": "WHERE archived_at IS NOT NULL",
        "all": "",
    }.get(archive_filter, "WHERE archived_at IS NULL")
    with connect_database(database_path) as connection:
        return list(
            connection.execute(
                f"""
                SELECT {_application_select_columns()}
                FROM applications
                {archive_clause}
                ORDER BY
                    CASE WHEN archived_at IS NULL THEN 0 ELSE 1 END,
                    CASE applied_to
                        WHEN 'No' THEN 0
                        WHEN 'Accepted for interview' THEN 1
                        WHEN 'N/A' THEN 2
                        WHEN 'Rejected' THEN 3
                        WHEN 'Yes' THEN 4
                        ELSE 5
                    END,
                    company COLLATE NOCASE ASC,
                    job_title COLLATE NOCASE ASC
                """
            )
        )


def _application_archive_counts(database_path: Path) -> dict[str, int]:
    with connect_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN archived_at IS NULL THEN 1 ELSE 0 END) AS active_count,
                SUM(CASE WHEN archived_at IS NOT NULL THEN 1 ELSE 0 END) AS archived_count
            FROM applications
            """
        ).fetchone()
    return {
        "active": int(row["active_count"] or 0),
        "archived": int(row["archived_count"] or 0),
    }


def _fetch_application(database_path: Path, job_id: str) -> sqlite3.Row | None:
    with connect_database(database_path) as connection:
        return connection.execute(
            f"SELECT {_application_select_columns()} FROM applications WHERE job_id = ?",
            (job_id,),
        ).fetchone()


def _application_select_columns() -> str:
    sync_status = """
        CASE
            WHEN COALESCE(NULLIF(applications.application_resume_object, ''), '') = ''
            THEN 'No'
            WHEN COALESCE(NULLIF(applications.application_resume_updated_at, ''), '') = ''
            THEN 'No'
            WHEN applications.resume_content IS NULL
              OR applications.resume_html_content IS NULL
            THEN 'No'
            WHEN COALESCE(NULLIF(applications.resume_updated_at, ''), '') = ''
              OR COALESCE(NULLIF(applications.resume_html_updated_at, ''), '') = ''
            THEN 'No'
            WHEN applications.resume_updated_at >= applications.application_resume_updated_at
              AND applications.resume_html_updated_at >= applications.application_resume_updated_at
            THEN 'Yes'
            ELSE 'No'
        END
    """
    manual_passthrough_status = """
        CASE
            WHEN applications.selected_resume_variant = 'manual'
              OR EXISTS (
                  SELECT 1
                  FROM application_resume_variants AS manual_variant
                  WHERE manual_variant.job_id = applications.job_id
                    AND manual_variant.variant_key = 'manual'
              )
              OR lower(COALESCE(applications.notes, '')) LIKE '%manual second pass%'
              OR lower(COALESCE(applications.notes, '')) LIKE '%manual passthrough%'
            THEN 'Yes'
            ELSE 'No'
        END
    """
    has_v1_resume_variant = """
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM application_resume_variants AS v1_variant
                WHERE v1_variant.job_id = applications.job_id
                  AND v1_variant.variant_key = 'v1'
            )
            THEN 1
            ELSE 0
        END
    """
    has_v2_resume_variant = """
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM application_resume_variants AS v2_variant
                WHERE v2_variant.job_id = applications.job_id
                  AND v2_variant.variant_key = 'v2'
            )
            THEN 1
            ELSE 0
        END
    """
    has_manual_resume_variant = """
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM application_resume_variants AS manual_variant
                WHERE manual_variant.job_id = applications.job_id
                  AND manual_variant.variant_key = 'manual'
            )
            THEN 1
            ELSE 0
        END
    """
    return f"""
        applications.*,
        {sync_status} AS aro_resume_sync_status,
        CASE WHEN {sync_status} = 'Yes' THEN 0 ELSE 1 END AS aro_resume_out_of_sync,
        {manual_passthrough_status} AS manual_passthrough_status,
        {has_v1_resume_variant} AS has_v1_resume_variant,
        {has_v2_resume_variant} AS has_v2_resume_variant,
        {has_manual_resume_variant} AS has_manual_resume_variant
    """


def _calculate_ats_score(
    *,
    resume_content: bytes | None,
    job_description: str | None,
) -> AtsProxyScore | None:
    if not resume_content or not job_description:
        return None
    return calculate_ats_proxy_score(
        resume_pdf=resume_content,
        job_description=job_description,
    )


def _format_missing_terms(score: AtsProxyScore) -> str:
    return ", ".join(score.missing_high_value_terms)


def _parse_application_resume_yaml(value: str) -> dict[str, Any]:
    parsed = yaml.safe_load(value)
    if not isinstance(parsed, dict):
        raise ValueError("Application resume object must be a YAML mapping.")
    return parsed


def _dump_application_resume_yaml(resume: dict[str, Any]) -> str:
    return yaml.safe_dump(resume, sort_keys=False, allow_unicode=False)


def _parse_cover_letter_object(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {
            "schema_version": COVER_LETTER_OBJECT_SCHEMA_VERSION,
            "source": "manual",
            "body_html": "",
            "body_text": "",
        }
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("Cover letter object must be a YAML mapping.")
    parsed.setdefault("schema_version", COVER_LETTER_OBJECT_SCHEMA_VERSION)
    parsed.setdefault("source", "manual")
    parsed["body_html"] = _sanitize_cover_letter_body_html(parsed.get("body_html"))
    parsed["body_text"] = _cover_letter_plain_text(parsed["body_html"])
    return parsed


def _dump_cover_letter_object(value: dict[str, Any]) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=False)


def _sanitize_cover_letter_body_html(value: Any) -> str:
    soup = BeautifulSoup(str(value or ""), "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    for tag in list(soup.find_all(True)):
        name = (tag.name or "").lower()
        if name in {"strong", "b"}:
            tag.name = "b"
            tag.attrs = {}
        elif name in {"em", "i"}:
            tag.name = "i"
            tag.attrs = {}
        elif name in {"p", "div", "br"}:
            tag.attrs = {}
        elif name == "a":
            href = str(tag.get("href") or "").strip()
            if href.startswith(("http://", "https://", "mailto:")):
                tag.attrs = {"href": href}
            else:
                tag.unwrap()
        else:
            tag.unwrap()
    return str(soup).strip()


def _cover_letter_plain_text(body_html: str) -> str:
    soup = BeautifulSoup(body_html or "", "html.parser")
    return "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    ).strip()


def _cover_letter_paragraph_markup(body_html: str) -> list[str]:
    soup = BeautifulSoup(body_html or "", "html.parser")
    paragraphs: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        value = "".join(current).strip()
        if value:
            paragraphs.append(value)
        current.clear()

    for child in soup.contents:
        if isinstance(child, NavigableString):
            text = str(child)
            if text.strip():
                current.append(html_escape(text))
        elif isinstance(child, Tag) and child.name in {"p", "div"}:
            flush_current()
            markup = _cover_letter_node_markup(child).strip()
            if markup:
                paragraphs.append(markup)
        elif isinstance(child, Tag) and child.name == "br":
            flush_current()
        else:
            current.append(_cover_letter_node_markup(child))
    flush_current()
    return paragraphs


def _cover_letter_node_markup(node: Any) -> str:
    if isinstance(node, NavigableString):
        return html_escape(str(node))
    if not isinstance(node, Tag):
        return ""
    name = (node.name or "").lower()
    if name == "br":
        return "<br/>"
    inner = "".join(_cover_letter_node_markup(child) for child in node.children)
    if name == "b":
        return f"<b>{inner}</b>"
    if name == "i":
        return f"<i>{inner}</i>"
    if name == "a":
        href = str(node.get("href") or "").strip()
        if href.startswith(("http://", "https://", "mailto:")):
            return f'<a href="{html_escape(href, quote=True)}" color="blue">{inner}</a>'
    return inner


def _description_tokens(value: Any) -> list[str]:
    return str(value or "").split()


def _join_description_tokens(tokens: list[str]) -> str:
    return " ".join(tokens).strip()


def _description_diff_rows(
    job_description: Any,
    prompt_job_description: Any,
) -> list[DescriptionDiffRow]:
    left_tokens = _description_tokens(job_description)
    right_tokens = _description_tokens(prompt_job_description)
    matcher = difflib.SequenceMatcher(
        a=left_tokens,
        b=right_tokens,
        autojunk=False,
    )
    rows: list[DescriptionDiffRow] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        left_text = _join_description_tokens(left_tokens[left_start:left_end])
        right_text = _join_description_tokens(right_tokens[right_start:right_end])
        rows.append(
            DescriptionDiffRow(
                status=tag,
                left_line_no=left_start + 1 if left_text else None,
                right_line_no=right_start + 1 if right_text else None,
                left_text=left_text,
                right_text=right_text,
            )
        )
    return rows


def _description_removed_text(
    job_description: Any,
    prompt_job_description: Any,
) -> str:
    removed_lines: list[str] = []
    for row in _description_diff_rows(job_description, prompt_job_description):
        if row.status in {"delete", "replace"} and row.left_text.strip():
            removed_lines.append(row.left_text)
    return "\n".join(removed_lines)


def _resume_template_path(template_path: Path | None = None) -> Path:
    if template_path is not None:
        return template_path
    if DEFAULT_RESUME_TEMPLATE.is_file():
        return DEFAULT_RESUME_TEMPLATE
    return _project_root() / DEFAULT_RESUME_TEMPLATE


def _apply_resume_editor_form(resume: dict[str, Any], form: Any) -> dict[str, Any]:
    header = _ensure_mapping(resume, "header_top")
    header["line_1_name_header_text"] = _form_text(form, "header_name")
    header["line_2_header_text"] = _form_rich_text(form, "header_line_2")
    header["line_3_applicant_info_text"] = _form_rich_text(form, "header_info")
    header.pop("line_2_applicant_info_text", None)
    header["contact_items"] = _form_lines(form, "header_contact_items")

    summary = _ensure_mapping(resume, "professional_summary")
    summary["render"] = _form_bool(form, "summary_render")
    summary["header_text"] = _form_text(form, "summary_header_text")
    summary["paragraph"] = _form_rich_text(form, "summary_paragraph")
    summary["summary_note"] = _form_rich_text(form, "summary_note")

    skills = _ensure_mapping(resume, "core_technical_skills")
    skills["render"] = _form_bool(form, "skills_render")
    skills["header_text"] = _form_text(form, "skills_header_text")
    for index, bullet in enumerate(_mapping_list(skills.get("bullet_points"))):
        bullet["category"] = _form_text_or_existing(
            form,
            f"skill_{index}_category",
            bullet.get("category"),
        )
        items = bullet.get("items")
        if not isinstance(items, dict):
            items = {}
            bullet["items"] = items
        items["primary"] = _form_lines_or_existing(
            form,
            f"skill_{index}_primary",
            items.get("primary"),
        )
        items["additional"] = _form_lines_or_existing(
            form,
            f"skill_{index}_additional",
            items.get("additional"),
        )
        bullet["jod_matched_items"] = _form_lines_or_existing(
            form,
            f"skill_{index}_jod_matched",
            bullet.get("jod_matched_items"),
        )

    experience = _ensure_mapping(resume, "professional_experience")
    experience["render"] = _form_bool(form, "experience_render")
    experience["header_text"] = _form_text(form, "experience_header_text")
    for job_index, job in enumerate(_mapping_list(experience.get("jobs"))):
        job["render"] = _form_bool(form, f"job_{job_index}_render")
        line_1 = _ensure_mapping(job, "line_1")
        line_1["company_name_text"] = _form_text(form, f"job_{job_index}_company")
        line_1["position_name_text"] = _form_text(form, f"job_{job_index}_position")
        line_1["position_dates_text"] = _form_text(form, f"job_{job_index}_dates")
        line_2 = _ensure_mapping(job, "line_2")
        line_2["position_intro_text"] = _form_rich_text(form, f"job_{job_index}_intro")
        _apply_job_bullet_edits(job=job, job_index=job_index, form=form)

    education = _ensure_mapping(resume, "education")
    education["render"] = _form_bool(form, "education_render")
    education["header_text"] = _form_text(form, "education_header_text")
    for entry_index, entry in enumerate(_mapping_list(education.get("entries"))):
        entry["render"] = _form_bool(form, f"education_{entry_index}_render")
        line_1 = _ensure_mapping(entry, "line_1")
        line_1["institution_name_text"] = _form_text(
            form,
            f"education_{entry_index}_institution",
        )
        line_2 = _ensure_mapping(entry, "line_2")
        line_2["degree_name_text"] = _form_text(form, f"education_{entry_index}_degree")
        line_2["degree_dates_text"] = _form_text(form, f"education_{entry_index}_dates")
        for bullet_index, bullet in enumerate(_mapping_list(entry.get("bullet_points"))):
            bullet["render"] = _form_bool(
                form,
                f"education_{entry_index}_bullet_{bullet_index}_render",
            )
            bullet["text"] = _form_rich_text(
                form,
                f"education_{entry_index}_bullet_{bullet_index}_text",
            )

    certifications = _ensure_mapping(resume, "certifications")
    certifications["render"] = _form_bool(form, "certifications_render")
    certifications["header_text"] = _form_text(form, "certifications_header_text")
    for index, bullet in enumerate(_mapping_list(certifications.get("bullet_points"))):
        bullet["render"] = _form_bool(form, f"certification_{index}_render")
        bullet["text"] = _form_rich_text(form, f"certification_{index}_text")

    portfolio = _ensure_mapping(resume, "portfolio")
    portfolio["render"] = _form_bool(form, "portfolio_render")
    portfolio["header_text"] = _form_text(form, "portfolio_header_text")
    for index, project in enumerate(_mapping_list(portfolio.get("projects"))):
        project["render"] = _form_bool(form, f"portfolio_{index}_render")
        project["title_text"] = _form_text(form, f"portfolio_{index}_title")
        project["url"] = _form_text(form, f"portfolio_{index}_url")
        project["description_text"] = _form_rich_text(form, f"portfolio_{index}_description")

    return resume


def _apply_job_bullet_edits(*, job: dict[str, Any], job_index: int, form: Any) -> None:
    bullets = _normalizable_bullet_list(job)
    bulk_name = f"job_{job_index}_bulk"
    bulk_original_name = f"job_{job_index}_bulk_original"
    bulk_text = _form_text(form, bulk_name)
    bulk_original = _form_text(form, bulk_original_name)
    if bulk_text != bulk_original:
        lines = _textarea_lines(bulk_text)
        for index, line in enumerate(lines):
            if index < len(bullets):
                bullet = bullets[index]
            else:
                bullet = {"order": _next_bullet_order(bullets), "render": True}
                bullets.append(bullet)
            bullet["text"] = sanitize_resume_rich_text(line)
            bullet["render"] = True
        for bullet in bullets[len(lines) :]:
            bullet["render"] = False
        job["bullet_points"] = bullets
        return

    for index, bullet in enumerate(bullets):
        bullet["render"] = _form_bool(form, f"job_{job_index}_bullet_{index}_render")
        bullet["text"] = _form_rich_text(form, f"job_{job_index}_bullet_{index}_text")
    job["bullet_points"] = bullets


def _normalizable_bullet_list(job: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, bullet in enumerate(_list_value(job.get("bullet_points"))):
        if isinstance(bullet, dict):
            normalized.append(bullet)
        elif isinstance(bullet, str):
            normalized.append({"order": index + 1, "text": bullet, "render": True})
    return normalized


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    parent[key] = value
    return value


def _mapping_list(value: object) -> list[dict[str, Any]]:
    return [item for item in _list_value(value) if isinstance(item, dict)]


def _list_value(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _next_bullet_order(bullets: list[dict[str, Any]]) -> int:
    orders = [
        int(bullet.get("order", 0))
        for bullet in bullets
        if str(bullet.get("order", "")).isdigit()
    ]
    return (max(orders) if orders else len(bullets)) + 1


def _form_text(form: Any, name: str) -> str:
    return str(form.get(name) or "").strip()


def _form_text_or_existing(form: Any, name: str, existing: Any) -> str:
    if name not in form:
        return str(existing or "").strip()
    return _form_text(form, name)


def _form_rich_text(form: Any, name: str) -> str:
    return sanitize_resume_rich_text(form.get(name))


def _form_bool(form: Any, name: str) -> bool:
    return form.get(name) == "1"


def _form_lines(form: Any, name: str) -> list[str]:
    return _textarea_lines(_form_text(form, name))


def _form_lines_or_existing(form: Any, name: str, existing: Any) -> list[str]:
    existing_items = _string_list(existing)
    if name in form:
        submitted_items = _form_lines(form, name)
        return submitted_items or existing_items
    return existing_items


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _textarea_lines(value: str) -> list[str]:
    lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        while line[:1] in {"-", "*", "\u2022"}:
            line = line[1:].strip()
        if line:
            lines.append(line)
    return lines


def _job_bulk_bullet_text(job: dict[str, Any]) -> str:
    return "\n".join(
        _bullet_text_for_editing(bullet)
        for bullet in _list_value(job.get("bullet_points"))
    )


def _bullet_text_for_editing(bullet: object) -> str:
    if isinstance(bullet, str):
        return bullet
    if not isinstance(bullet, dict):
        return ""
    for key in ("text", "rendered_text"):
        value = bullet.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _date_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    return text or None


def _artifact_timestamp(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(modified_at, UTC).isoformat(timespec="seconds")


def _resume_html_filename(row: sqlite3.Row) -> str:
    return f"mp_resume_{_filename_part(str(row['job_title'] or row['job_id']))}.html"


def _resume_pdf_filename(row: sqlite3.Row) -> str:
    return f"mp_resume_{_filename_part(str(row['job_title'] or row['job_id']))}.pdf"


def _variant_resume_html_filename(row: sqlite3.Row, variant_key: str) -> str:
    filename = _resume_html_filename(row)
    return filename if variant_key == DEFAULT_RESUME_VARIANT else _filename_with_suffix(
        filename, variant_key
    )


def _variant_resume_pdf_filename(row: sqlite3.Row, variant_key: str) -> str:
    filename = _resume_pdf_filename(row)
    return filename if variant_key == DEFAULT_RESUME_VARIANT else _filename_with_suffix(
        filename, variant_key
    )


def _filename_with_suffix(filename: str, suffix: str) -> str:
    path = Path(filename)
    return f"{path.stem}_{_filename_part(suffix)}{path.suffix}"


def _cover_letter_pdf_filename(row: sqlite3.Row) -> str:
    return f"mp_cover_letter_{_filename_part(str(row['job_title'] or row['job_id']))}.pdf"


def _filename_part(value: str) -> str:
    characters = [character.lower() if character.isalnum() else "_" for character in value]
    compact = "_".join(part for part in "".join(characters).split("_") if part)
    return compact or "resume"


def _normalize_applied_to(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in APPLICATION_STATUSES else "No"


def _application_status_row_class(value: Any) -> str:
    return {
        "Yes": "is-applied",
        "Accepted for interview": "is-interview",
        "N/A": "is-not-applicable",
    }.get(str(value or ""), "")


def _view_state_from_args(args: Any) -> dict[str, str]:
    status = str(args.get("status") or "all").strip()
    if status not in APPLICATION_STATUS_FILTERS:
        status = "all"

    archive = str(args.get("archive") or "active").strip()
    if archive not in APPLICATION_ARCHIVE_FILTERS:
        archive = "active"

    sort = str(args.get("sort") or "").strip()
    if sort not in VIEW_STATE_SORTS:
        sort = ""

    direction = str(args.get("direction") or "").strip()
    if direction not in VIEW_STATE_DIRECTIONS:
        direction = ""
    if sort and not direction:
        direction = "asc"
    if direction and not sort:
        direction = ""

    return {
        "search": str(args.get("q") or "").strip(),
        "status": status,
        "archive": archive,
        "sort": sort,
        "direction": direction,
    }


def _safe_index_return_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "/"
    parts = urlsplit(text)
    if parts.scheme or parts.netloc or parts.path not in {"", "/"}:
        return "/"

    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key in VIEW_STATE_QUERY_KEYS and value
    ]
    query = urlencode(query_items)
    return f"/?{query}" if query else "/"


def _resume_edit_return_path(job_id: str, return_to: Any) -> str:
    query = urlencode({"return_to": _safe_index_return_path(return_to)})
    return f"/resumes/{job_id}/edit?{query}"


def _resume_variant_review_return_path(job_id: str, return_to: Any) -> str:
    query = urlencode({"return_to": _safe_index_return_path(return_to)})
    return f"/resumes/{job_id}/variants?{query}"


def _cover_letter_edit_return_path(job_id: str, return_to: Any) -> str:
    query = urlencode({"return_to": _safe_index_return_path(return_to)})
    return f"/cover-letters/{job_id}/edit?{query}"


def _description_edit_return_path(job_id: str, return_to: Any) -> str:
    query = urlencode({"return_to": _safe_index_return_path(return_to)})
    return f"/descriptions/{job_id}?{query}"


def _add_application_return_path(return_to: Any) -> str:
    query = urlencode({"return_to": _safe_index_return_path(return_to)})
    return f"/applications/add?{query}"


def cleanup_downloaded_application_pdfs(download_dir: Path | None = None) -> int:
    target_dir = download_dir or Path.home() / "Downloads"
    if not target_dir.is_dir():
        return 0

    deleted_count = 0
    for path in target_dir.glob("mp_*.pdf"):
        if not path.is_file():
            continue
        try:
            path.unlink()
        except OSError:
            continue
        deleted_count += 1
    return deleted_count


def copy_application_artifact_to_downloads(
    *,
    row: sqlite3.Row,
    artifact_kind: str,
    download_dir: Path | None = None,
) -> Path:
    if artifact_kind == "resume":
        filename_column = "resume_filename"
        content_column = "resume_content"
        default_prefix = "mp_resume"
    elif artifact_kind == "cover_letter":
        filename_column = "cover_letter_filename"
        content_column = "cover_letter_content"
        default_prefix = "mp_cover_letter"
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact_kind}")

    target_dir = download_dir or Path.home() / "Downloads"
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(str(row[filename_column] or "")).name
    if not filename:
        filename = f"{default_prefix}_{row['job_id']}.pdf"
    destination = target_dir / filename

    content = row[content_column]
    if content is None:
        raise FileNotFoundError(f"No local {artifact_kind.replace('_', ' ')} artifact was found.")
    destination.write_bytes(content)
    return destination


def _download_display_path(path: Path) -> str:
    downloads_dir = Path.home() / "Downloads"
    if path.parent == downloads_dir:
        return f"~/Downloads/{path.name}"
    return str(path)


def _display_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        return text.split("T", 1)[0]
    return text


def _display_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return parsed.strftime("%Y-%m-%d %H:%M")


def _schedule_browser_open(*, host: str, port: int) -> None:
    url = _browser_url(host=host, port=port)

    def open_url() -> None:
        import webbrowser

        webbrowser.open(url)

    timer = threading.Timer(1.0, open_url)
    timer.daemon = True
    timer.start()


def _browser_url(*, host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}"


def _open_url_in_chromium(url: str) -> None:
    try:
        subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-c",
                _PLAYWRIGHT_CHROMIUM_SCRIPT,
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        _open_url_in_default_browser(url)


def _open_url_in_default_browser(url: str) -> None:
    import webbrowser

    webbrowser.open(url)


_PLAYWRIGHT_CHROMIUM_SCRIPT = r"""
import sys

url = sys.argv[1]

try:
    from playwright.sync_api import sync_playwright
except Exception:
    import webbrowser

    webbrowser.open(url)
    raise SystemExit(0)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_event("close", timeout=0)
except Exception:
    import webbrowser

    webbrowser.open(url)
"""


INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LinkedIn Applications</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #626a73;
      --line: #d9dee5;
      --surface: #ffffff;
      --band: #f4f6f8;
      --accent: #0b6e69;
      --accent-strong: #074f4b;
      --warn: #915c00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--band);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; }
    .stats { display: flex; gap: 14px; color: var(--muted); white-space: nowrap; }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    input[type="search"], input[type="date"], textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    input[type="search"] { max-width: 420px; }
    button, .link-button {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 600;
      padding: 8px 10px;
      text-decoration: none;
      white-space: nowrap;
    }
    button:hover, .link-button:hover { background: var(--accent-strong); }
    .sort-button {
      align-items: center;
      background: transparent;
      border: 0;
      color: inherit;
      display: inline-flex;
      font: inherit;
      font-weight: 700;
      gap: 4px;
      padding: 0;
      text-transform: uppercase;
    }
    .sort-button:hover, .sort-button:focus {
      background: transparent;
      color: var(--accent);
      outline: none;
    }
    .sort-indicator {
      color: var(--accent);
      font-size: 11px;
      min-width: 14px;
    }
    .ghost {
      background: #fff;
      color: var(--accent);
    }
    .danger {
      background: #9d2f2f;
      border-color: #9d2f2f;
    }
    .danger:hover {
      background: #7b2424;
      border-color: #7b2424;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: .45;
    }
    .toolbar form { margin: 0; }
    .bulk-applications-form {
      display: flex;
      gap: 8px;
    }
    .selected-count { color: var(--muted); white-space: nowrap; }
    .actions-menu {
      position: relative;
    }
    .actions-menu summary {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: #fff;
      color: var(--accent);
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      list-style: none;
      padding: 8px 10px;
      white-space: nowrap;
    }
    .actions-menu summary::-webkit-details-marker { display: none; }
    .actions-menu[open] summary {
      background: #eef8f6;
      color: var(--accent-strong);
    }
    .actions-panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 36px rgba(20, 33, 45, .18);
      min-width: 280px;
      padding: 12px;
      position: absolute;
      right: 0;
      top: 42px;
      z-index: 5;
    }
    .action-choice {
      align-items: center;
      display: flex;
      gap: 8px;
      font-weight: 650;
      margin-bottom: 8px;
    }
    .regenerate-options {
      border: 1px solid var(--line);
      border-radius: 6px;
      margin: 8px 0 10px;
      padding: 8px 10px 10px;
    }
    .regenerate-options legend {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      padding: 0 4px;
    }
    .regenerate-options label {
      align-items: center;
      display: flex;
      gap: 7px;
      margin-top: 6px;
    }
    .manual-pass-profile-control {
      display: grid;
      gap: 4px;
      margin: 0 0 10px;
      max-width: 260px;
    }
    .manual-pass-profile-control[hidden] { display: none; }
    .manual-pass-profile-control span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .manual-pass-profile-control select {
      padding: 6px 8px;
    }
    .actions-menu-footer {
      align-items: center;
      display: flex;
      gap: 10px;
      justify-content: space-between;
    }
    .actions-selected-summary {
      color: var(--muted);
      font-size: 12px;
    }
    .action-status {
      background: var(--surface);
      border: 1px solid #f1d48a;
      border-radius: 8px;
      bottom: 14px;
      box-shadow: 0 16px 44px rgba(20, 33, 45, .22);
      color: #5c4100;
      left: 24px;
      margin: 0 auto;
      max-width: 920px;
      overflow: hidden;
      padding: 0;
      position: fixed;
      right: 24px;
      z-index: 20;
    }
    .action-status[hidden] { display: none; }
    .action-status.is-running {
      border-color: #c8dfdc;
      color: var(--accent-strong);
    }
    .action-status.is-failed {
      border-color: #e3b6b6;
      color: #7b2424;
    }
    .action-status.is-collapsed .action-status-body {
      display: none;
    }
    .action-status-header {
      align-items: center;
      display: flex;
      gap: 10px;
      justify-content: space-between;
      min-height: 48px;
      padding: 9px 12px;
    }
    .action-status-summary {
      display: grid;
      gap: 2px;
      min-width: 0;
    }
    .action-status-title {
      font-weight: 750;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .action-status-latest {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .action-status-controls {
      align-items: center;
      display: flex;
      flex: 0 0 auto;
      gap: 6px;
    }
    .action-status-state {
      border: 1px solid currentColor;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      padding: 2px 8px;
      text-transform: uppercase;
    }
    .action-status-icon-button {
      align-items: center;
      background: transparent;
      border: 1px solid currentColor;
      border-radius: 999px;
      color: inherit;
      display: inline-flex;
      height: 26px;
      justify-content: center;
      padding: 0;
      width: 26px;
    }
    .action-status-icon-button:hover {
      background: rgba(11, 110, 105, .08);
    }
    .action-status-body {
      background: #fffdf6;
      border-bottom: 1px solid #f1d48a;
      max-height: 150px;
      overflow: auto;
      padding: 10px 14px;
    }
    .action-status.is-running .action-status-body {
      background: #f5fbfa;
      border-bottom-color: #c8dfdc;
    }
    .action-status-progress {
      background: #e5eceb;
      height: 4px;
      overflow: hidden;
    }
    .action-status-progress-fill {
      background: var(--accent);
      display: block;
      height: 100%;
      transition: width .2s ease;
      width: 100%;
    }
    .action-status.is-running .action-status-progress-fill {
      animation: action-status-progress 1.35s linear infinite;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
      width: 42%;
    }
    .action-status.is-failed .action-status-progress-fill {
      background: #9d2f2f;
    }
    .action-status-messages {
      margin: 0;
      padding-left: 18px;
    }
    .action-status-messages li {
      margin: 2px 0;
    }
    @keyframes action-status-progress {
      from { transform: translateX(-120%); }
      to { transform: translateX(260%); }
    }
    main { padding: 20px 24px 32px; }
    .flash {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #c8dfdc;
      border-radius: 6px;
      background: #eef8f6;
      color: var(--accent-strong);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .02em;
      background: #fafbfc;
      position: sticky;
      top: 51px;
      z-index: 1;
    }
    tr.is-applied { background: #f3faf8; }
    tr.is-interview { background: #fff8d7; }
    tr.is-not-applicable { background: #f3f4f6; }
    .select-col { width: 42px; text-align: center; }
    .company { min-width: 160px; font-weight: 650; }
    .job { min-width: 260px; }
    .job-id { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }
    .job-badges {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .variant-badge {
      border-radius: 999px;
      display: inline-flex;
      font-size: 11px;
      font-weight: 800;
      line-height: 1;
      padding: 4px 7px;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .variant-badge.is-v1 {
      background: #eef8f6;
      border: 1px solid #b9d8d3;
      color: var(--accent-strong);
    }
    .variant-badge.is-v2 {
      background: #eef2ff;
      border: 1px solid #c7d2fe;
      color: #3730a3;
    }
    .variant-badge.is-manual {
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #9a3412;
    }
    .variant-badge.is-other {
      background: #f4f4f5;
      border: 1px solid #d4d4d8;
      color: #3f3f46;
    }
    .date-col { min-width: 104px; white-space: nowrap; }
    .experience-col { min-width: 126px; white-space: nowrap; }
    .score-col { min-width: 104px; position: relative; }
    .score-details { position: relative; }
    .score-details summary {
      cursor: pointer;
      list-style: none;
      outline: none;
    }
    .score-details summary::-webkit-details-marker { display: none; }
    .score-badge {
      border: 1px solid #b9d8d3;
      border-radius: 999px;
      color: var(--accent-strong);
      display: inline-flex;
      font-size: 12px;
      font-weight: 700;
      padding: 3px 8px;
      white-space: nowrap;
    }
    .score-popover {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(20, 33, 45, .14);
      color: var(--ink);
      left: 0;
      line-height: 1.5;
      min-width: 230px;
      padding: 10px 12px;
      position: absolute;
      top: 28px;
      z-index: 4;
    }
    .score-row {
      display: flex;
      gap: 16px;
      justify-content: space-between;
      white-space: nowrap;
    }
    .score-row.is-stacked {
      display: block;
      margin-top: 6px;
      white-space: normal;
    }
    .score-row.is-stacked strong {
      display: block;
      margin-top: 2px;
    }
    .score-row strong { font-weight: 700; }
    .sync-col {
      min-width: 112px;
      text-align: center;
    }
    .sync-status {
      border-radius: 999px;
      display: inline-block;
      font-size: 12px;
      font-weight: 800;
      min-width: 42px;
      padding: 3px 8px;
      text-transform: uppercase;
    }
    .sync-status.is-synced {
      background: #e6f5ef;
      color: #047857;
    }
    .sync-status.is-stale {
      background: #fff3d8;
      color: #8a5a00;
    }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; min-width: 170px; }
    .actions form { margin: 0; }
    .artifact-cell { min-width: 132px; }
    .artifact-timestamp {
      color: var(--muted);
      display: block;
      font-size: 12px;
      margin-top: 3px;
      white-space: nowrap;
    }
    .text-link-button {
      background: transparent;
      border: 0;
      border-radius: 0;
      color: var(--accent);
      font: inherit;
      font-weight: 600;
      padding: 0;
      text-decoration: underline;
    }
    .text-link-button:hover {
      background: transparent;
      color: var(--accent-strong);
    }
    .muted { color: var(--muted); }
    .apply-form {
      display: grid;
      grid-template-columns: 96px 150px minmax(180px, 1fr) auto;
      gap: 8px;
      min-width: 540px;
    }
    textarea { min-height: 38px; resize: vertical; }
    a { color: var(--accent); font-weight: 600; }
    .empty { color: var(--muted); padding: 28px; text-align: center; }
    @media (max-width: 900px) {
      header, .toolbar { align-items: flex-start; flex-direction: column; }
      .stats { flex-wrap: wrap; }
      main { padding: 12px; overflow-x: auto; }
      th { top: 104px; }
      .action-status {
        bottom: 8px;
        left: 8px;
        right: 8px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>LinkedIn Applications</h1>
    <div class="stats">
      <span>Total: {{ stats.total }}</span>
      <span>Active: {{ stats.active }}</span>
      <span>Archived: {{ stats.archived }}</span>
      <span>Applied: {{ stats.applied }}</span>
      <span>Pending: {{ stats.pending }}</span>
      <span>Interview: {{ stats.interview }}</span>
      <span>Rejected: {{ stats.rejected }}</span>
      <span>N/A: {{ stats.not_applicable }}</span>
    </div>
  </header>
  <div class="toolbar">
    <input
      id="search"
      type="search"
      placeholder="Search company, title, or job id"
      value="{{ view_state.search }}"
    >
    <select id="status-filter" aria-label="Filter status">
      <option value="all" {{ 'selected' if view_state.status == 'all' else '' }}>
        All statuses
      </option>
      <option value="No" {{ 'selected' if view_state.status == 'No' else '' }}>
        Pending
      </option>
      <option value="Yes" {{ 'selected' if view_state.status == 'Yes' else '' }}>
        Applied
      </option>
      <option
        value="Accepted for interview"
        {{ 'selected' if view_state.status == 'Accepted for interview' else '' }}
      >
        Accepted for interview
      </option>
      <option value="Rejected" {{ 'selected' if view_state.status == 'Rejected' else '' }}>
        Rejected
      </option>
      <option value="N/A" {{ 'selected' if view_state.status == 'N/A' else '' }}>
        N/A
      </option>
    </select>
    <select id="archive-filter" aria-label="Filter archive">
      <option value="active" {{ 'selected' if view_state.archive == 'active' else '' }}>
        Active postings
      </option>
      <option value="archived" {{ 'selected' if view_state.archive == 'archived' else '' }}>
        Archived
      </option>
      <option value="all" {{ 'selected' if view_state.archive == 'all' else '' }}>
        All postings
      </option>
    </select>
    <form id="bulk-applications-form" class="bulk-applications-form" method="post">
      <input type="hidden" class="return-to-state" name="return_to" value="{{ current_path }}">
      {% if view_state.archive != 'archived' %}
        <button
          id="archive-selected"
          type="submit"
          formaction="/applications/archive"
          disabled
        >
          Archive selected
        </button>
      {% endif %}
      {% if view_state.archive != 'active' %}
        <button
          id="unarchive-selected"
          type="submit"
          formaction="/applications/unarchive"
          class="ghost"
          disabled
        >
          Restore selected
        </button>
      {% endif %}
      <button
        id="delete-selected"
        type="submit"
        formaction="/applications/delete"
        class="danger"
        disabled
      >
        Delete selected
      </button>
    </form>
    <button id="add-application" type="button" class="ghost">Add</button>
    <details class="actions-menu">
      <summary>Actions</summary>
      <div class="actions-panel">
        <form id="actions-form" method="post" action="/actions/run">
          <input
            type="hidden"
            class="return-to-state"
            name="return_to"
            value="{{ current_path }}"
          >
          <fieldset class="regenerate-options" id="regenerate-options">
            <legend>Resume actions</legend>
            <label>
              <input type="radio" name="regenerate_mode" value="aro_objects">
              <span>Regenerate ARO Objects</span>
            </label>
            <label>
              <input type="radio" name="regenerate_mode" value="draft_resumes">
              <span>Regenerate Draft v1 Only</span>
            </label>
            <label>
              <input type="radio" name="regenerate_mode" value="resume_variants">
              <span>Run v1 + v2 Resume Workflow</span>
            </label>
            <label>
              <input
                type="radio"
                name="regenerate_mode"
                value="resume_variants_manual_pass"
              >
              <span>Run v1 + v2 + Codex Manual Pass</span>
            </label>
            <label>
              <input type="radio" name="regenerate_mode" value="refine_drafts">
              <span>Run v2 Refinement</span>
            </label>
            <label>
              <input type="radio" name="regenerate_mode" value="highlight_drafts">
              <span>Codex Highlight Selected Resume</span>
            </label>
            <label>
              <input type="radio" name="regenerate_mode" value="manual_pass">
              <span>Codex Manual Pass Variant</span>
            </label>
            <label>
              <input type="radio" name="regenerate_mode" value="sync_draft_to_aro">
              <span>Sync Draft to ARO</span>
            </label>
          </fieldset>
          <label
            id="actions-manual-pass-profile-control"
            class="manual-pass-profile-control"
            hidden
          >
            <span>Manual pass profile</span>
            <select
              id="actions-manual-pass-profile"
              name="manual_pass_profile"
              disabled
            >
              {% for value, label in manual_pass_profile_options %}
                <option
                  value="{{ value }}"
                  {% if value == default_manual_pass_profile %}selected{% endif %}
                >{{ label }}</option>
              {% endfor %}
            </select>
          </label>
          <label class="action-choice">
            <input type="checkbox" name="highlight_with_codex" value="1">
            <span>Run Codex highlighting after draft generation</span>
          </label>
          <div class="actions-menu-footer">
            <span id="actions-selected-summary" class="actions-selected-summary">
              0 selected
            </span>
            <button id="run-actions" type="submit" disabled>Run</button>
          </div>
        </form>
      </div>
    </details>
    <span id="selected-count" class="selected-count">0 selected</span>
  </div>
  <section id="action-status" class="action-status" hidden aria-live="polite">
    <div id="action-status-body" class="action-status-body">
      <ol id="action-status-messages" class="action-status-messages"></ol>
    </div>
    <div class="action-status-progress" aria-hidden="true">
      <span id="action-status-progress-fill" class="action-status-progress-fill"></span>
    </div>
    <div class="action-status-header">
      <div class="action-status-summary">
        <span id="action-status-title" class="action-status-title">Background action</span>
        <span id="action-status-latest" class="action-status-latest"></span>
      </div>
      <div class="action-status-controls">
        <span id="action-status-state" class="action-status-state">Running</span>
        <button
          id="action-status-toggle"
          class="action-status-icon-button"
          type="button"
          aria-expanded="true"
          aria-label="Collapse background status"
        >v</button>
        <button
          id="action-status-close"
          class="action-status-icon-button"
          type="button"
          aria-label="Close background status"
        >x</button>
      </div>
    </div>
  </section>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% if rows %}
      <table id="applications">
        <thead>
          <tr>
            <th class="select-col">
              <input id="select-all" type="checkbox" aria-label="Select visible rows">
            </th>
            <th id="company-header" aria-sort="none">
              <button
                id="company-sort"
                class="sort-button"
                type="button"
                aria-label="Sort by company"
              >
                Company
                <span id="company-sort-indicator" class="sort-indicator" aria-hidden="true">
                  ↑↓
                </span>
              </button>
            </th>
            <th>Job</th>
            <th>Posted</th>
            <th id="matched-header" aria-sort="none">
              <button
                id="matched-sort"
                class="sort-button"
                type="button"
                aria-label="Sort by matched date"
              >
                Matched
                <span id="matched-sort-indicator" class="sort-indicator" aria-hidden="true">
                  ↑↓
                </span>
              </button>
            </th>
            <th>Experience</th>
            <th id="ats-header" aria-sort="none">
              <button
                id="ats-sort"
                class="sort-button"
                type="button"
                aria-label="Sort by ATS proxy score"
              >
                ATS
                <span id="ats-sort-indicator" class="sort-indicator" aria-hidden="true">
                  ↑↓
                </span>
              </button>
            </th>
            <th>Job Links</th>
            <th id="resume-header" aria-sort="none">
              <button
                id="resume-sort"
                class="sort-button"
                type="button"
                aria-label="Sort by resume timestamp"
              >
                Resume
                <span id="resume-sort-indicator" class="sort-indicator" aria-hidden="true">
                  ↑↓
                </span>
              </button>
            </th>
            <th>ARO/Resume Sync</th>
            <th id="cover-letter-header" aria-sort="none">
              <button
                id="cover-letter-sort"
                class="sort-button"
                type="button"
                aria-label="Sort by cover letter timestamp"
              >
                Cover Letter
                <span id="cover-letter-sort-indicator" class="sort-indicator" aria-hidden="true">
                  ↑↓
                </span>
              </button>
            </th>
            <th>Application</th>
          </tr>
        </thead>
        <tbody>
          {% for row in rows %}
            {% set search_text %}
              {{ row.company }} {{ row.job_title }} {{ row.job_id }}
              {{ row.experience_level or '' }}
            {% endset %}
            <tr class="{{ application_status_row_class(row.applied_to) }}"
                data-status="{{ row.applied_to }}"
                data-company-sort="{{ row.company }}"
                data-matched-sort="{{ row.date_matched or '' }}"
                data-ats-sort="{{ row.ats_score if row.ats_score is not none else '' }}"
                data-resume-sort="{{ row.resume_updated_at or '' }}"
                data-cover-letter-sort="{{ row.cover_letter_updated_at or '' }}"
                data-search="{{ search_text|lower }}">
              <td class="select-col">
                <input
                  class="row-selector"
                  type="checkbox"
                  name="job_id"
                  value="{{ row.job_id }}"
                  form="bulk-applications-form"
                  aria-label="Select {{ row.company }} {{ row.job_title }}"
                >
              </td>
              <td class="company">{{ row.company }}</td>
              <td class="job">
                {{ row.job_title }}
                <span class="job-id">{{ row.job_id }}</span>
                {% if row.has_v1_resume_variant
                      or row.has_v2_resume_variant
                      or row.has_manual_resume_variant %}
                  {% set selected_variant = row.selected_resume_variant or 'v1' %}
                  <span class="job-badges">
                    {% if row.has_v1_resume_variant %}
                      {% set v1_badge_state = 'Available' %}
                      {% if selected_variant == 'v1' %}
                        {% set v1_badge_state = 'Selected' %}
                      {% endif %}
                      <span
                        class="variant-badge {{ resume_variant_badge_class('v1') }}"
                        title="{{ v1_badge_state }} resume variant"
                      >
                        {{ resume_variant_label('v1') }}
                      </span>
                    {% endif %}
                    {% if row.has_v2_resume_variant %}
                      {% set v2_badge_state = 'Available' %}
                      {% if selected_variant == 'v2' %}
                        {% set v2_badge_state = 'Selected' %}
                      {% endif %}
                      <span
                        class="variant-badge {{ resume_variant_badge_class('v2') }}"
                        title="{{ v2_badge_state }} resume variant"
                      >
                        {{ resume_variant_label('v2') }}
                      </span>
                    {% endif %}
                    {% if row.has_manual_resume_variant %}
                      {% set manual_badge_state = 'Available' %}
                      {% if selected_variant == 'manual' %}
                        {% set manual_badge_state = 'Selected' %}
                      {% endif %}
                      <span
                        class="variant-badge {{ resume_variant_badge_class('manual') }}"
                        title="{{ manual_badge_state }} resume variant"
                      >
                        {{ resume_variant_label('manual') }}
                      </span>
                    {% endif %}
                  </span>
                {% endif %}
              </td>
              <td class="date-col">
                {{ row.date_posted|display_date if row.date_posted else '' }}
                {% if not row.date_posted %}<span class="muted">-</span>{% endif %}
              </td>
              <td class="date-col">
                {{ row.date_matched|display_date if row.date_matched else '' }}
                {% if not row.date_matched %}<span class="muted">-</span>{% endif %}
              </td>
              <td class="experience-col">
                {{ row.experience_level or '' }}
                {% if not row.experience_level %}<span class="muted">-</span>{% endif %}
              </td>
              <td class="score-col">
                {% if row.ats_score is not none %}
                  <details class="score-details">
                    <summary aria-label="Show ATS proxy score details">
                      <span class="score-badge">{{ row.ats_score }}/100</span>
                    </summary>
                    <div class="score-popover">
                      <div class="score-row">
                        <span>ATS proxy score:</span>
                        <strong>{{ row.ats_score }}/100</strong>
                      </div>
                      <div class="score-row">
                        <span>Parsing:</span>
                        <strong>{{ row.ats_parsing_score }}/100</strong>
                      </div>
                      <div class="score-row">
                        <span>Keyword match:</span>
                        <strong>{{ row.ats_keyword_score }}/100</strong>
                      </div>
                      <div class="score-row">
                        <span>Semantic match:</span>
                        <strong>{{ row.ats_semantic_score }}/100</strong>
                      </div>
                      <div class="score-row">
                        <span>Formatting risk:</span>
                        <strong>{{ row.ats_formatting_risk }}</strong>
                      </div>
                      <div class="score-row is-stacked">
                        <span>Missing/high-value terms:</span>
                        <strong>{{ row.ats_missing_terms or "None detected" }}</strong>
                      </div>
                    </div>
                  </details>
                {% else %}
                  <span class="muted">-</span>
                {% endif %}
              </td>
              <td>
                <div class="actions">
                  <a
                    href="{{ row.linkedin_url }}"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Job URL
                  </a>
                  <a
                    class="preserve-state-link"
                    href="/descriptions/{{ row.job_id }}"
                  >
                    Compare descriptions
                  </a>
                </div>
              </td>
              <td>
                <div class="artifact-cell">
                  <div class="actions">
                  {% if row.resume_content %}
                    <a href="/resumes/{{ row.job_id }}" target="_blank" rel="noreferrer">
                      Resume
                    </a>
                    {% if row.resume_html_content %}
                      <a href="/resume-html/{{ row.job_id }}" target="_blank" rel="noreferrer">
                        HTML
                      </a>
                    {% endif %}
                    {% if row.application_resume_object %}
                      <a
                        class="preserve-state-link"
                        href="/resumes/{{ row.job_id }}/edit"
                      >
                        Edit
                      </a>
                      <a
                        class="preserve-state-link"
                        href="/resumes/{{ row.job_id }}/variants"
                      >
                        Review
                      </a>
                    {% endif %}
                    <form method="post" action="/resumes/{{ row.job_id }}/copy-to-downloads">
                      <input
                        type="hidden"
                        class="return-to-state"
                        name="return_to"
                        value="{{ current_path }}"
                      >
                      <button class="text-link-button" type="submit">Download</button>
                    </form>
                  {% else %}
                    <span class="muted">Missing</span>
                  {% endif %}
                  </div>
                  {% if row.resume_updated_at %}
                    <span class="artifact-timestamp">
                      {{ row.resume_updated_at|display_timestamp }}
                    </span>
                  {% endif %}
                </div>
              </td>
              <td class="sync-col">
                {% if row.aro_resume_sync_status == 'Yes' %}
                  {% set sync_class = 'is-synced' %}
                {% else %}
                  {% set sync_class = 'is-stale' %}
                {% endif %}
                <span class="sync-status {{ sync_class }}">
                  {{ row.aro_resume_sync_status }}
                </span>
              </td>
              <td>
                <div class="artifact-cell">
                  <div class="actions">
                    <a
                      class="preserve-state-link"
                      href="/cover-letters/{{ row.job_id }}/edit"
                    >
                      Edit
                    </a>
                  {% if row.cover_letter_content %}
                    <a href="/cover-letters/{{ row.job_id }}" target="_blank" rel="noreferrer">
                      Cover Letter
                    </a>
                    <form
                      method="post"
                      action="/cover-letters/{{ row.job_id }}/copy-to-downloads"
                    >
                      <input
                        type="hidden"
                        class="return-to-state"
                        name="return_to"
                        value="{{ current_path }}"
                      >
                      <button class="text-link-button" type="submit">Download</button>
                    </form>
                  {% else %}
                    <span class="muted">Missing</span>
                  {% endif %}
                  </div>
                  {% if row.cover_letter_updated_at %}
                    <span class="artifact-timestamp">
                      {{ row.cover_letter_updated_at|display_timestamp }}
                    </span>
                  {% endif %}
                </div>
              </td>
              <td>
                <form class="apply-form" method="post" action="/applications/{{ row.job_id }}">
                  <input
                    type="hidden"
                    class="return-to-state"
                    name="return_to"
                    value="{{ current_path }}"
                  >
                  <select name="applied_to" aria-label="Applied status">
                    <option
                      value="No"
                      {{ 'selected' if row.applied_to == 'No' else '' }}
                    >No</option>
                    <option
                      value="Yes"
                      {{ 'selected' if row.applied_to == 'Yes' else '' }}
                    >Yes</option>
                    <option
                      value="N/A"
                      {{ 'selected' if row.applied_to == 'N/A' else '' }}
                    >N/A</option>
                    <option
                      value="Rejected"
                      {{ 'selected' if row.applied_to == 'Rejected' else '' }}
                    >Rejected</option>
                    <option
                      value="Accepted for interview"
                      {{ 'selected' if row.applied_to == 'Accepted for interview' else '' }}
                    >Accepted for interview</option>
                  </select>
                  <input type="date" name="date_applied" value="{{ row.date_applied or '' }}">
                  <textarea name="notes" placeholder="Notes">{{ row.notes }}</textarea>
                  <button type="submit">Save</button>
                </form>
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <div class="empty">No applications imported yet.</div>
    {% endif %}
  </main>
  <script>
    const search = document.querySelector("#search");
    const statusFilter = document.querySelector("#status-filter");
    const archiveFilter = document.querySelector("#archive-filter");
    const selectAll = document.querySelector("#select-all");
    const archiveButton = document.querySelector("#archive-selected");
    const unarchiveButton = document.querySelector("#unarchive-selected");
    const deleteButton = document.querySelector("#delete-selected");
    const addApplicationButton = document.querySelector("#add-application");
    const selectedCount = document.querySelector("#selected-count");
    const bulkApplicationsForm = document.querySelector("#bulk-applications-form");
    const companySortButton = document.querySelector("#company-sort");
    const companySortIndicator = document.querySelector("#company-sort-indicator");
    const companyHeader = document.querySelector("#company-header");
    const matchedSortButton = document.querySelector("#matched-sort");
    const matchedSortIndicator = document.querySelector("#matched-sort-indicator");
    const matchedHeader = document.querySelector("#matched-header");
    const atsSortButton = document.querySelector("#ats-sort");
    const atsSortIndicator = document.querySelector("#ats-sort-indicator");
    const atsHeader = document.querySelector("#ats-header");
    const resumeSortButton = document.querySelector("#resume-sort");
    const resumeSortIndicator = document.querySelector("#resume-sort-indicator");
    const resumeHeader = document.querySelector("#resume-header");
    const coverLetterSortButton = document.querySelector("#cover-letter-sort");
    const coverLetterSortIndicator = document.querySelector("#cover-letter-sort-indicator");
    const coverLetterHeader = document.querySelector("#cover-letter-header");
    const tableBody = document.querySelector("#applications tbody");
    const rows = [...document.querySelectorAll("#applications tbody tr")];
    const rowSelectors = [...document.querySelectorAll(".row-selector")];
    const returnToFields = [...document.querySelectorAll(".return-to-state")];
    const preserveStateLinks = [...document.querySelectorAll(".preserve-state-link")];
    const actionsForm = document.querySelector("#actions-form");
    const regenerateModeInputs = [...document.querySelectorAll("input[name='regenerate_mode']")];
    const highlightWithCodexInput = document.querySelector("input[name='highlight_with_codex']");
    const actionsManualPassProfileControl =
      document.querySelector("#actions-manual-pass-profile-control");
    const actionsManualPassProfile =
      document.querySelector("#actions-manual-pass-profile");
    const runActionsButton = document.querySelector("#run-actions");
    const actionsSelectedSummary = document.querySelector("#actions-selected-summary");
    const actionStatus = document.querySelector("#action-status");
    const actionStatusTitle = document.querySelector("#action-status-title");
    const actionStatusLatest = document.querySelector("#action-status-latest");
    const actionStatusState = document.querySelector("#action-status-state");
    const actionStatusMessages = document.querySelector("#action-status-messages");
    const actionStatusProgressFill = document.querySelector("#action-status-progress-fill");
    const actionStatusToggle = document.querySelector("#action-status-toggle");
    const actionStatusClose = document.querySelector("#action-status-close");
    const initialSort = {{ view_state.sort|tojson }};
    const initialDirection = {{ view_state.direction|tojson }};
    let companySortDirection = null;
    let matchedSortDirection = null;
    let atsSortDirection = null;
    let resumeSortDirection = null;
    let coverLetterSortDirection = null;
    let dismissedActionRunId =
      window.sessionStorage.getItem("actionStatusDismissedRunId") || "";
    let actionStatusCollapsed =
      window.sessionStorage.getItem("actionStatusCollapsed") === "1";
    let observedRunningRunId = null;
    const reloadedActionRunIds = new Set(
      JSON.parse(window.sessionStorage.getItem("actionStatusReloadedRunIds") || "[]"),
    );
    function activeSortState() {
      if (companySortDirection) {
        return { sort: "company", direction: companySortDirection };
      }
      if (matchedSortDirection) {
        return { sort: "matched", direction: matchedSortDirection };
      }
      if (atsSortDirection) {
        return { sort: "ats", direction: atsSortDirection };
      }
      if (resumeSortDirection) {
        return { sort: "resume", direction: resumeSortDirection };
      }
      if (coverLetterSortDirection) {
        return { sort: "cover_letter", direction: coverLetterSortDirection };
      }
      return { sort: "", direction: "" };
    }
    function currentReturnPath() {
      return `${window.location.pathname}${window.location.search}`;
    }
    function syncReturnState() {
      const value = currentReturnPath();
      returnToFields.forEach((field) => {
        field.value = value;
      });
    }
    function updateViewStateUrl() {
      const params = new URLSearchParams();
      const term = search.value.trim();
      if (term) {
        params.set("q", term);
      }
      if (statusFilter.value && statusFilter.value !== "all") {
        params.set("status", statusFilter.value);
      }
      if (archiveFilter && archiveFilter.value && archiveFilter.value !== "active") {
        params.set("archive", archiveFilter.value);
      }
      const sortState = activeSortState();
      if (sortState.sort && sortState.direction) {
        params.set("sort", sortState.sort);
        params.set("direction", sortState.direction);
      }
      const query = params.toString();
      const nextPath = query ? `${window.location.pathname}?${query}` : window.location.pathname;
      window.history.replaceState(null, "", nextPath);
      syncReturnState();
    }
    function navigateToArchiveView() {
      updateViewStateUrl();
      window.location.assign(currentReturnPath());
    }
    function applyFilters() {
      const term = search.value.trim().toLowerCase();
      const status = statusFilter.value;
      rows.forEach((row) => {
        const matchesText = !term || row.dataset.search.includes(term);
        const matchesStatus = status === "all" || row.dataset.status === status;
        row.hidden = !(matchesText && matchesStatus);
      });
      updateViewStateUrl();
      updateSelectionState();
    }
    function matchedTimestamp(row) {
      return sortableTimestamp(row.dataset.matchedSort);
    }
    function resumeTimestamp(row) {
      return sortableTimestamp(row.dataset.resumeSort);
    }
    function coverLetterTimestamp(row) {
      return sortableTimestamp(row.dataset.coverLetterSort);
    }
    function sortableTimestamp(value) {
      if (!value) {
        return null;
      }
      const parsed = Date.parse(value);
      return Number.isNaN(parsed) ? null : parsed;
    }
    function atsScore(row) {
      const value = Number.parseInt(row.dataset.atsSort, 10);
      return Number.isNaN(value) ? null : value;
    }
    function resetSortIndicator(header, indicator) {
      if (header) {
        header.setAttribute("aria-sort", "none");
      }
      if (indicator) {
        indicator.textContent = "↑↓";
      }
    }
    function sortRowsByCompany(nextDirection = null) {
      if (!tableBody) {
        return;
      }
      companySortDirection = nextDirection || (companySortDirection === "asc" ? "desc" : "asc");
      matchedSortDirection = null;
      atsSortDirection = null;
      resumeSortDirection = null;
      coverLetterSortDirection = null;
      resetSortIndicator(matchedHeader, matchedSortIndicator);
      resetSortIndicator(atsHeader, atsSortIndicator);
      resetSortIndicator(resumeHeader, resumeSortIndicator);
      resetSortIndicator(coverLetterHeader, coverLetterSortIndicator);
      const direction = companySortDirection === "asc" ? 1 : -1;
      const originalIndex = new Map(rows.map((row, index) => [row, index]));
      const sortedRows = [...rows].sort((left, right) => {
        const leftValue = (left.dataset.companySort || "").trim();
        const rightValue = (right.dataset.companySort || "").trim();
        const comparison = leftValue.localeCompare(
          rightValue,
          undefined,
          { sensitivity: "base" },
        );
        if (comparison === 0) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        return comparison * direction;
      });
      sortedRows.forEach((row) => tableBody.appendChild(row));
      if (companyHeader) {
        companyHeader.setAttribute(
          "aria-sort",
          companySortDirection === "asc" ? "ascending" : "descending",
        );
      }
      companySortIndicator.textContent = companySortDirection === "asc" ? "↑" : "↓";
      applyFilters();
    }
    function sortRowsByMatched(nextDirection = null) {
      if (!tableBody) {
        return;
      }
      matchedSortDirection = nextDirection || (matchedSortDirection === "asc" ? "desc" : "asc");
      companySortDirection = null;
      atsSortDirection = null;
      resumeSortDirection = null;
      coverLetterSortDirection = null;
      resetSortIndicator(companyHeader, companySortIndicator);
      resetSortIndicator(atsHeader, atsSortIndicator);
      resetSortIndicator(resumeHeader, resumeSortIndicator);
      resetSortIndicator(coverLetterHeader, coverLetterSortIndicator);
      const direction = matchedSortDirection === "asc" ? 1 : -1;
      const originalIndex = new Map(rows.map((row, index) => [row, index]));
      const sortedRows = [...rows].sort((left, right) => {
        const leftValue = matchedTimestamp(left);
        const rightValue = matchedTimestamp(right);
        if (leftValue === null && rightValue === null) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        if (leftValue === null) {
          return 1;
        }
        if (rightValue === null) {
          return -1;
        }
        if (leftValue === rightValue) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        return (leftValue - rightValue) * direction;
      });
      sortedRows.forEach((row) => tableBody.appendChild(row));
      if (matchedHeader) {
        matchedHeader.setAttribute(
          "aria-sort",
          matchedSortDirection === "asc" ? "ascending" : "descending",
        );
      }
      matchedSortIndicator.textContent = matchedSortDirection === "asc" ? "↑" : "↓";
      applyFilters();
    }
    function sortRowsByAts(nextDirection = null) {
      if (!tableBody) {
        return;
      }
      atsSortDirection = nextDirection || (atsSortDirection === "asc" ? "desc" : "asc");
      companySortDirection = null;
      matchedSortDirection = null;
      resumeSortDirection = null;
      coverLetterSortDirection = null;
      resetSortIndicator(companyHeader, companySortIndicator);
      resetSortIndicator(matchedHeader, matchedSortIndicator);
      resetSortIndicator(resumeHeader, resumeSortIndicator);
      resetSortIndicator(coverLetterHeader, coverLetterSortIndicator);
      const direction = atsSortDirection === "asc" ? 1 : -1;
      const originalIndex = new Map(rows.map((row, index) => [row, index]));
      const sortedRows = [...rows].sort((left, right) => {
        const leftValue = atsScore(left);
        const rightValue = atsScore(right);
        if (leftValue === null && rightValue === null) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        if (leftValue === null) {
          return 1;
        }
        if (rightValue === null) {
          return -1;
        }
        if (leftValue === rightValue) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        return (leftValue - rightValue) * direction;
      });
      sortedRows.forEach((row) => tableBody.appendChild(row));
      if (atsHeader) {
        atsHeader.setAttribute(
          "aria-sort",
          atsSortDirection === "asc" ? "ascending" : "descending",
        );
      }
      atsSortIndicator.textContent = atsSortDirection === "asc" ? "↑" : "↓";
      applyFilters();
    }
    function sortRowsByResume(nextDirection = null) {
      if (!tableBody) {
        return;
      }
      resumeSortDirection = nextDirection || (resumeSortDirection === "asc" ? "desc" : "asc");
      companySortDirection = null;
      matchedSortDirection = null;
      atsSortDirection = null;
      coverLetterSortDirection = null;
      resetSortIndicator(companyHeader, companySortIndicator);
      resetSortIndicator(matchedHeader, matchedSortIndicator);
      resetSortIndicator(atsHeader, atsSortIndicator);
      resetSortIndicator(coverLetterHeader, coverLetterSortIndicator);
      const direction = resumeSortDirection === "asc" ? 1 : -1;
      const originalIndex = new Map(rows.map((row, index) => [row, index]));
      const sortedRows = [...rows].sort((left, right) => {
        const leftValue = resumeTimestamp(left);
        const rightValue = resumeTimestamp(right);
        if (leftValue === null && rightValue === null) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        if (leftValue === null) {
          return 1;
        }
        if (rightValue === null) {
          return -1;
        }
        if (leftValue === rightValue) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        return (leftValue - rightValue) * direction;
      });
      sortedRows.forEach((row) => tableBody.appendChild(row));
      if (resumeHeader) {
        resumeHeader.setAttribute(
          "aria-sort",
          resumeSortDirection === "asc" ? "ascending" : "descending",
        );
      }
      resumeSortIndicator.textContent = resumeSortDirection === "asc" ? "↑" : "↓";
      applyFilters();
    }
    function sortRowsByCoverLetter(nextDirection = null) {
      if (!tableBody) {
        return;
      }
      coverLetterSortDirection =
        nextDirection || (coverLetterSortDirection === "asc" ? "desc" : "asc");
      companySortDirection = null;
      matchedSortDirection = null;
      atsSortDirection = null;
      resumeSortDirection = null;
      resetSortIndicator(companyHeader, companySortIndicator);
      resetSortIndicator(matchedHeader, matchedSortIndicator);
      resetSortIndicator(atsHeader, atsSortIndicator);
      resetSortIndicator(resumeHeader, resumeSortIndicator);
      const direction = coverLetterSortDirection === "asc" ? 1 : -1;
      const originalIndex = new Map(rows.map((row, index) => [row, index]));
      const sortedRows = [...rows].sort((left, right) => {
        const leftValue = coverLetterTimestamp(left);
        const rightValue = coverLetterTimestamp(right);
        if (leftValue === null && rightValue === null) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        if (leftValue === null) {
          return 1;
        }
        if (rightValue === null) {
          return -1;
        }
        if (leftValue === rightValue) {
          return originalIndex.get(left) - originalIndex.get(right);
        }
        return (leftValue - rightValue) * direction;
      });
      sortedRows.forEach((row) => tableBody.appendChild(row));
      if (coverLetterHeader) {
        coverLetterHeader.setAttribute(
          "aria-sort",
          coverLetterSortDirection === "asc" ? "ascending" : "descending",
        );
      }
      coverLetterSortIndicator.textContent =
        coverLetterSortDirection === "asc" ? "↑" : "↓";
      applyFilters();
    }
    function visibleSelectors() {
      return rowSelectors.filter((checkbox) => !checkbox.closest("tr").hidden);
    }
    function selectedJobIds() {
      return rowSelectors
        .filter((checkbox) => checkbox.checked)
        .map((checkbox) => checkbox.value);
    }
    function selectedRegenerateMode() {
      const selected = regenerateModeInputs.find((input) => input.checked);
      return selected ? selected.value : "";
    }
    function updateActionsState() {
      const selected = selectedJobIds();
      const mode = selectedRegenerateMode();
      const manualPassActive = [
        "resume_variants_manual_pass",
        "manual_pass",
      ].includes(mode);
      if (actionsSelectedSummary) {
        actionsSelectedSummary.textContent = `${selected.length} selected`;
      }
      if (actionsManualPassProfileControl && actionsManualPassProfile) {
        actionsManualPassProfileControl.hidden = !manualPassActive;
        actionsManualPassProfile.disabled = !manualPassActive;
      }
      if (highlightWithCodexInput) {
        const canChainHighlight = [
          "draft_resumes",
          "resume_variants",
          "resume_variants_manual_pass",
          "manual_pass",
        ].includes(mode);
        highlightWithCodexInput.disabled = !canChainHighlight;
        if (!canChainHighlight) {
          highlightWithCodexInput.checked = false;
        }
      }
      if (runActionsButton) {
        const hasResumeAction = Boolean(mode);
        runActionsButton.disabled = !hasResumeAction || selected.length === 0;
      }
    }
    function syncActionsFormJobIds() {
      if (!actionsForm) {
        return;
      }
      actionsForm.querySelectorAll(".actions-job-id").forEach((input) => input.remove());
      selectedJobIds().forEach((jobId) => {
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = "job_id";
        input.value = jobId;
        input.className = "actions-job-id";
        actionsForm.appendChild(input);
      });
    }
    function persistReloadedActionRunIds() {
      window.sessionStorage.setItem(
        "actionStatusReloadedRunIds",
        JSON.stringify([...reloadedActionRunIds].slice(-12)),
      );
    }
    function applyActionStatusCollapsedState() {
      if (!actionStatus || !actionStatusToggle) {
        return;
      }
      actionStatus.classList.toggle("is-collapsed", actionStatusCollapsed);
      actionStatusToggle.textContent = actionStatusCollapsed ? "^" : "v";
      actionStatusToggle.setAttribute(
        "aria-expanded",
        actionStatusCollapsed ? "false" : "true",
      );
      actionStatusToggle.setAttribute(
        "aria-label",
        actionStatusCollapsed
          ? "Expand background status"
          : "Collapse background status",
      );
    }
    function latestActionMessage(run) {
      const messages = run.messages || [];
      return messages.length ? messages[messages.length - 1] : "";
    }
    function visibleActionRun(runs) {
      const visibleRuns = (runs || []).filter((run) => run.id !== dismissedActionRunId);
      return visibleRuns.find((run) => run.status === "running") || visibleRuns[0] || null;
    }
    function maybeReloadAfterCompletedAction(run) {
      if (!run || !run.id) {
        return;
      }
      if (run.status === "running") {
        observedRunningRunId = run.id;
        return;
      }
      if (
        run.status === "completed"
        && observedRunningRunId === run.id
        && dismissedActionRunId !== run.id
        && !reloadedActionRunIds.has(run.id)
      ) {
        reloadedActionRunIds.add(run.id);
        persistReloadedActionRunIds();
        window.setTimeout(() => window.location.reload(), 1200);
      }
    }
    function renderActionStatus(run) {
      if (
        !actionStatus
        || !actionStatusTitle
        || !actionStatusLatest
        || !actionStatusState
        || !actionStatusMessages
        || !actionStatusProgressFill
      ) {
        return;
      }
      if (!run || (run.id && dismissedActionRunId === run.id)) {
        actionStatus.hidden = true;
        return;
      }
      actionStatus.hidden = false;
      actionStatus.dataset.runId = run.id || "";
      actionStatus.classList.toggle("is-running", run.status === "running");
      actionStatus.classList.toggle("is-failed", run.status === "failed");
      actionStatusTitle.textContent = run.title || "Background action";
      actionStatusState.textContent = run.status || "running";
      actionStatusLatest.textContent = latestActionMessage(run);
      actionStatusProgressFill.style.width = run.status === "running" ? "" : "100%";
      actionStatusMessages.replaceChildren();
      (run.messages || []).slice(-8).forEach((message) => {
        const item = document.createElement("li");
        item.textContent = message;
        actionStatusMessages.appendChild(item);
      });
      applyActionStatusCollapsedState();
      maybeReloadAfterCompletedAction(run);
    }
    async function refreshActionStatus() {
      try {
        const response = await fetch("/actions/status", {
          headers: { "Accept": "application/json" },
        });
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        renderActionStatus(visibleActionRun(payload.runs));
      } catch {
        // Background status is a convenience layer; table controls should keep working.
      }
    }
    function updateSelectionState() {
      const selected = rowSelectors.filter((checkbox) => checkbox.checked);
      const visible = visibleSelectors();
      if (archiveButton) {
        archiveButton.disabled = selected.length === 0;
      }
      if (unarchiveButton) {
        unarchiveButton.disabled = selected.length === 0;
      }
      if (deleteButton) {
        deleteButton.disabled = selected.length === 0;
      }
      selectedCount.textContent = `${selected.length} selected`;
      updateActionsState();
      if (!selectAll) {
        return;
      }
      selectAll.checked = visible.length > 0 && visible.every((checkbox) => checkbox.checked);
      selectAll.indeterminate = !selectAll.checked
        && visible.some((checkbox) => checkbox.checked);
    }
    search.addEventListener("input", applyFilters);
    statusFilter.addEventListener("change", applyFilters);
    if (archiveFilter) {
      archiveFilter.addEventListener("change", navigateToArchiveView);
    }
    if (companySortButton) {
      companySortButton.addEventListener("click", () => sortRowsByCompany());
    }
    if (matchedSortButton) {
      matchedSortButton.addEventListener("click", () => sortRowsByMatched());
    }
    if (atsSortButton) {
      atsSortButton.addEventListener("click", () => sortRowsByAts());
    }
    if (resumeSortButton) {
      resumeSortButton.addEventListener("click", () => sortRowsByResume());
    }
    if (coverLetterSortButton) {
      coverLetterSortButton.addEventListener("click", () => sortRowsByCoverLetter());
    }
    document.querySelectorAll("form").forEach((form) => {
      form.addEventListener("submit", syncReturnState);
    });
    regenerateModeInputs.forEach((input) => {
      input.addEventListener("change", updateActionsState);
    });
    if (actionsForm) {
      actionsForm.addEventListener("submit", syncActionsFormJobIds);
    }
    if (actionStatusToggle) {
      actionStatusToggle.addEventListener("click", () => {
        actionStatusCollapsed = !actionStatusCollapsed;
        window.sessionStorage.setItem(
          "actionStatusCollapsed",
          actionStatusCollapsed ? "1" : "0",
        );
        applyActionStatusCollapsedState();
      });
    }
    if (actionStatusClose && actionStatus) {
      actionStatusClose.addEventListener("click", () => {
        dismissedActionRunId = actionStatus.dataset.runId || "";
        if (dismissedActionRunId) {
          window.sessionStorage.setItem(
            "actionStatusDismissedRunId",
            dismissedActionRunId,
          );
        }
        actionStatus.hidden = true;
      });
    }
    if (addApplicationButton) {
      addApplicationButton.addEventListener("click", () => {
        const url = new URL("/applications/add", window.location.origin);
        url.searchParams.set("return_to", currentReturnPath());
        window.open(
          url.toString(),
          "add-application",
          "popup,width=560,height=650,noopener,noreferrer",
        );
      });
    }
    preserveStateLinks.forEach((link) => {
      link.addEventListener("click", () => {
        const url = new URL(link.href, window.location.origin);
        url.searchParams.set("return_to", currentReturnPath());
        link.href = `${url.pathname}${url.search}`;
      });
    });
    if (selectAll) {
      selectAll.addEventListener("change", () => {
        visibleSelectors().forEach((checkbox) => {
          checkbox.checked = selectAll.checked;
        });
        updateSelectionState();
      });
      rowSelectors.forEach((checkbox) => {
        checkbox.addEventListener("change", updateSelectionState);
      });
    }
    if (bulkApplicationsForm) {
      bulkApplicationsForm.addEventListener("submit", (event) => {
        const selected = rowSelectors.filter((checkbox) => checkbox.checked);
        if (selected.length === 0) {
          event.preventDefault();
          return;
        }
        if (
          event.submitter
          && event.submitter.id === "delete-selected"
          && !confirm(`Delete ${selected.length} selected application row(s)?`)
        ) {
          event.preventDefault();
        }
      });
    }
    if (initialSort === "company") {
      sortRowsByCompany(initialDirection === "desc" ? "desc" : "asc");
    } else if (initialSort === "matched") {
      sortRowsByMatched(initialDirection === "desc" ? "desc" : "asc");
    } else if (initialSort === "ats") {
      sortRowsByAts(initialDirection === "desc" ? "desc" : "asc");
    } else if (initialSort === "resume") {
      sortRowsByResume(initialDirection === "desc" ? "desc" : "asc");
    } else if (initialSort === "cover_letter") {
      sortRowsByCoverLetter(initialDirection === "desc" ? "desc" : "asc");
    } else {
      applyFilters();
    }
    refreshActionStatus();
    window.setInterval(refreshActionStatus, 2500);
  </script>
</body>
</html>
"""


ADD_APPLICATION_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Add Application</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #626a73;
      --line: #d9dee5;
      --surface: #ffffff;
      --band: #f4f6f8;
      --accent: #0b6e69;
      --accent-strong: #074f4b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--band);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0 0 4px;
      font-size: 20px;
      font-weight: 650;
    }
    main {
      display: grid;
      gap: 14px;
      padding: 16px 20px 22px;
    }
    form, .field-group {
      display: grid;
      gap: 10px;
      padding: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    form h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 750;
    }
    fieldset {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      border: 0;
    }
    legend {
      margin: 0 0 2px;
      padding: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .option-row {
      align-items: center;
      display: flex;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 650;
      text-transform: none;
    }
    .option-row input { margin: 0; }
    input[type="url"], input[type="number"], textarea,
    .manual-pass-profile-control select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--ink);
      font: inherit;
      text-transform: none;
    }
    .manual-pass-profile-control {
      display: grid;
      gap: 5px;
      margin-left: 24px;
      max-width: 300px;
    }
    .manual-pass-profile-control[hidden] { display: none; }
    textarea {
      min-height: 74px;
      resize: vertical;
    }
    button, .button-link {
      justify-self: start;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 650;
      padding: 8px 10px;
      text-decoration: none;
      white-space: nowrap;
    }
    button:hover, .button-link:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: not-allowed;
      opacity: .5;
    }
    .flash {
      margin: 0;
      padding: 10px 12px;
      border: 1px solid #b9d8d5;
      background: #eef8f6;
      color: var(--accent-strong);
    }
    .muted { color: var(--muted); }
    .top-links {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    a { color: var(--accent); font-weight: 650; }
  </style>
</head>
<body>
  <header>
    <div class="top-links">
      <a href="{{ return_to }}">Back to tracker</a>
    </div>
    <h1>Add Application</h1>
    <div class="muted">Load a public job posting into the tracker.</div>
  </header>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <form method="post" action="/applications/add/seed">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <h2>Seed jobs</h2>
      <label>
        Jobs
        <input
          type="number"
          name="max_jobs"
          value="{{ default_seed_max_jobs }}"
          min="1"
          max="{{ max_seed_jobs }}"
          required
        >
      </label>
      <fieldset>
        <legend>Posted</legend>
        {% for value, label in seed_date_posted_options %}
          <label class="option-row">
            <input
              type="radio"
              name="date_posted"
              value="{{ value }}"
              {% if value == default_seed_date_posted %}checked{% endif %}
            >
            <span>{{ label }}</span>
          </label>
        {% endfor %}
      </fieldset>
      <fieldset>
        <legend>Workflow</legend>
        <label class="option-row">
          <input type="checkbox" name="run_v1" value="1" checked>
          <span>Run v1 draft</span>
        </label>
        <label class="option-row">
          <input type="checkbox" name="run_v2" value="1" checked>
          <span>Run v2 refinement</span>
        </label>
        <label class="option-row">
          <input type="checkbox" name="run_manual" value="1">
          <span>Run Codex manual pass</span>
        </label>
        <label
          class="manual-pass-profile-control"
          data-manual-pass-profile-control
          hidden
        >
          <span>Manual pass profile</span>
          <select name="manual_pass_profile" disabled>
            {% for value, label in manual_pass_profile_options %}
              <option
                value="{{ value }}"
                {% if value == default_manual_pass_profile %}selected{% endif %}
              >{{ label }}</option>
            {% endfor %}
          </select>
        </label>
        <label class="option-row">
          <input type="checkbox" name="run_highlight" value="1">
          <span>Run Codex highlighting</span>
        </label>
      </fieldset>
      <button type="submit">Load</button>
    </form>

    <form method="post" action="/applications/add/linkedin">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <label>
        LinkedIn URLs
        <textarea
          name="linkedin_url"
          placeholder="https://www.linkedin.com/jobs/view/1234567890, https://www.linkedin.com/jobs/view/2345678901"
          required
        ></textarea>
      </label>
      <label class="option-row">
        <input type="checkbox" name="run_v2" value="1">
        <span>Run v2 refinement</span>
      </label>
      <label class="option-row">
        <input type="checkbox" name="run_manual" value="1" disabled>
        <span>Run Codex manual pass</span>
      </label>
      <label
        class="manual-pass-profile-control"
        data-manual-pass-profile-control
        hidden
      >
        <span>Manual pass profile</span>
        <select name="manual_pass_profile" disabled>
          {% for value, label in manual_pass_profile_options %}
            <option
              value="{{ value }}"
              {% if value == default_manual_pass_profile %}selected{% endif %}
            >{{ label }}</option>
          {% endfor %}
        </select>
      </label>
      <label class="option-row">
        <input type="checkbox" name="highlight_with_codex" value="1" checked>
        <span>Run Codex bullet highlighting after resume generation</span>
      </label>
      <button type="submit">Load</button>
    </form>

    <form method="post" action="/applications/add/other">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <label>
        Other URLs
        <textarea
          name="other_url"
          placeholder="https://company.example/jobs/software-engineer, https://jobs.example.com/platform-engineer"
          required
        ></textarea>
      </label>
      <label class="option-row">
        <input type="checkbox" name="run_v2" value="1">
        <span>Run v2 refinement</span>
      </label>
      <label class="option-row">
        <input type="checkbox" name="run_manual" value="1" disabled>
        <span>Run Codex manual pass</span>
      </label>
      <label
        class="manual-pass-profile-control"
        data-manual-pass-profile-control
        hidden
      >
        <span>Manual pass profile</span>
        <select name="manual_pass_profile" disabled>
          {% for value, label in manual_pass_profile_options %}
            <option
              value="{{ value }}"
              {% if value == default_manual_pass_profile %}selected{% endif %}
            >{{ label }}</option>
          {% endfor %}
        </select>
      </label>
      <label class="option-row">
        <input type="checkbox" name="highlight_with_codex" value="1" checked>
        <span>Run Codex bullet highlighting after resume generation</span>
      </label>
      <button type="submit">Load</button>
    </form>
  </main>
  <script>
    document.querySelectorAll("form").forEach((form) => {
      const runV1 = form.querySelector("input[name='run_v1']");
      const runV2 = form.querySelector("input[name='run_v2']");
      const runManual = form.querySelector("input[name='run_manual']");
      const manualPassProfileControl = form.querySelector(
        "[data-manual-pass-profile-control]",
      );
      const manualPassProfile = form.querySelector(
        "select[name='manual_pass_profile']",
      );
      if (!runV2 || !runManual) {
        return;
      }
      function syncManualDependency() {
        const dependenciesSelected = runV2.checked && (!runV1 || runV1.checked);
        runManual.disabled = !dependenciesSelected;
        if (!dependenciesSelected) {
          runManual.checked = false;
        }
        const manualPassActive = dependenciesSelected && runManual.checked;
        if (manualPassProfileControl && manualPassProfile) {
          manualPassProfileControl.hidden = !manualPassActive;
          manualPassProfile.disabled = !manualPassActive;
        }
      }
      if (runV1) {
        runV1.addEventListener("change", syncManualDependency);
      }
      runV2.addEventListener("change", syncManualDependency);
      runManual.addEventListener("change", syncManualDependency);
      syncManualDependency();
    });
  </script>
</body>
</html>
"""


RESUME_VARIANTS_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resume Variants</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --surface: #fff;
      --ink: #14212d;
      --muted: #617080;
      --line: #d9e0e4;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --added: #eef8f1;
      --removed: #fff1f1;
      --changed: #fff8dd;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 22px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 4px 0 2px; font-size: 20px; }
    h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      font-weight: 750;
    }
    main { padding: 18px 22px 30px; }
    section {
      margin-bottom: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    a { color: var(--accent); font-weight: 650; }
    button, .button-link {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      font: inherit;
      font-weight: 700;
      padding: 7px 10px;
      text-decoration: none;
      white-space: nowrap;
    }
    button:hover, .button-link:hover { background: var(--accent-strong); }
    button[disabled] {
      background: #e5e7eb;
      border-color: #d1d5db;
      color: #4b5563;
      cursor: default;
    }
    .button-link.secondary {
      background: #fff;
      color: var(--accent);
    }
    .button-link.secondary:hover {
      background: #eff7f6;
      color: var(--accent-strong);
    }
    .top-links, .actions-inline {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .meta, .muted { color: var(--muted); }
    .flash {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #c8dfdc;
      border-radius: 6px;
      background: #eef8f6;
      color: var(--accent-strong);
    }
    .variant-badge {
      border-radius: 999px;
      display: inline-flex;
      font-size: 11px;
      font-weight: 850;
      line-height: 1;
      padding: 4px 7px;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .variant-badge.is-v1 {
      background: #eef8f6;
      border: 1px solid #b9d8d3;
      color: var(--accent-strong);
    }
    .variant-badge.is-v2 {
      background: #eef2ff;
      border: 1px solid #c7d2fe;
      color: #3730a3;
    }
    .variant-badge.is-manual {
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #9a3412;
    }
    .variant-badge.is-other {
      background: #f4f4f5;
      border: 1px solid #d4d4d8;
      color: #3f3f46;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #fafbfc;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .score-grid {
      display: grid;
      grid-template-columns: auto auto;
      gap: 2px 12px;
      margin-top: 4px;
      max-width: 260px;
      font-size: 12px;
    }
    .delta-positive { color: #047857; font-weight: 800; }
    .delta-negative { color: #b91c1c; font-weight: 800; }
    .delta-neutral { color: var(--muted); font-weight: 800; }
    pre {
      margin: 0;
      overflow: auto;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff;
      color: var(--ink);
      font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .diff-pre { max-height: 520px; }
    .review-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .review-list {
      margin: 0;
      padding: 12px 18px 14px 28px;
    }
    .review-list li { margin-bottom: 8px; }
    .issue {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }
    @media (max-width: 980px) {
      header { flex-direction: column; }
      main { padding: 14px; overflow-x: auto; }
      .review-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="top-links">
        <a href="{{ return_to }}">Back to tracker</a>
        {% if row.linkedin_url %}
          <a href="{{ row.linkedin_url }}" target="_blank" rel="noreferrer">Job URL</a>
        {% endif %}
        {% if row.resume_content %}
          <a href="/resumes/{{ row.job_id }}" target="_blank" rel="noreferrer">
            Current PDF
          </a>
        {% endif %}
        {% if row.resume_html_content %}
          <a href="/resume-html/{{ row.job_id }}" target="_blank" rel="noreferrer">
            Current HTML
          </a>
        {% endif %}
      </div>
      <h1>Resume Variants</h1>
      <div class="meta">{{ row.company }} - {{ row.job_title }} - {{ row.job_id }}</div>
    </div>
    {% set selected_variant = row.selected_resume_variant or 'v1' %}
    <span class="variant-badge {{ resume_variant_badge_class(selected_variant) }}">
      {{ resume_variant_label(selected_variant) }}
    </span>
  </header>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <section>
      <h2>Variants</h2>
      <table>
        <thead>
          <tr>
            <th>Variant</th>
            <th>ATS</th>
            <th>Artifacts</th>
            <th>Actions</th>
            <th>Model / Updated</th>
          </tr>
        </thead>
        <tbody>
          {% for variant in variants %}
            <tr>
              <td>
                <span class="variant-badge {{ variant.badge_class }}">
                  {{ variant.variant_label }}
                </span>
                {% if variant.is_selected %}
                  <div class="muted">Selected</div>
                {% endif %}
              </td>
              <td>
                <strong>
                  {{ variant.ats_score if variant.ats_score is not none else '-' }}/100
                </strong>
                {% if variant.ats_delta is not none %}
                  {% if variant.ats_delta > 0 %}
                    <span class="delta-positive">+{{ variant.ats_delta }}</span>
                  {% elif variant.ats_delta < 0 %}
                    <span class="delta-negative">{{ variant.ats_delta }}</span>
                  {% else %}
                    <span class="delta-neutral">0</span>
                  {% endif %}
                {% endif %}
                {% set parsing_score = variant.ats_parsing_score %}
                {% set keyword_score = variant.ats_keyword_score %}
                {% set semantic_score = variant.ats_semantic_score %}
                <div class="score-grid">
                  <span>Parsing</span>
                  <strong>{{ parsing_score if parsing_score is not none else '-' }}/100</strong>
                  <span>Keyword</span>
                  <strong>{{ keyword_score if keyword_score is not none else '-' }}/100</strong>
                  <span>Semantic</span>
                  <strong>{{ semantic_score if semantic_score is not none else '-' }}/100</strong>
                  <span>Risk</span>
                  <strong>{{ variant.ats_formatting_risk or '-' }}</strong>
                </div>
                {% if variant.ats_missing_terms %}
                  <div class="muted">{{ variant.ats_missing_terms }}</div>
                {% endif %}
              </td>
              <td>
                <div class="actions-inline">
                  {% if variant.has_pdf %}
                    <a
                      href="/resumes/{{ row.job_id }}/variants/{{ variant.variant_key }}"
                      target="_blank"
                      rel="noreferrer"
                    >
                      PDF
                    </a>
                  {% endif %}
                  {% if variant.has_html %}
                    <a
                      href="/resume-html/{{ row.job_id }}/variants/{{ variant.variant_key }}"
                      target="_blank"
                      rel="noreferrer"
                    >
                      HTML
                    </a>
                  {% endif %}
                </div>
              </td>
              <td>
                {% if variant.is_selected %}
                  <button type="button" disabled>Selected</button>
                {% else %}
                  <form
                    method="post"
                    action="/resumes/{{ row.job_id }}/variants/{{ variant.variant_key }}/use"
                  >
                    <input type="hidden" name="return_to" value="{{ return_to }}">
                    <button type="submit">{{ variant.action_label }}</button>
                  </form>
                {% endif %}
              </td>
              <td>
                <div>{{ variant.model or '-' }}</div>
                <div class="muted">{{ variant.updated_at|display_timestamp }}</div>
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section>
      <h2>v1 / v2 ARO Diff</h2>
      {% if diff_text %}
        <pre class="diff-pre">{{ diff_text }}</pre>
      {% else %}
        <pre class="diff-pre">No v1/v2 differences detected.</pre>
      {% endif %}
    </section>

    <div class="review-grid">
      <section>
        <h2>Accepted Changes</h2>
        {% if review_details.accepted_changes %}
          <ol class="review-list">
            {% for change in review_details.accepted_changes %}
              <li>
                <strong>{{ change.change_id }}</strong>
                {% if change.rationale %}
                  <span class="issue">{{ change.rationale }}</span>
                {% endif %}
              </li>
            {% endfor %}
          </ol>
        {% else %}
          <pre>No accepted changes recorded.</pre>
        {% endif %}
      </section>

      <section>
        <h2>Rejected Changes</h2>
        {% if review_details.rejected_changes %}
          <ol class="review-list">
            {% for rejected in review_details.rejected_changes %}
              <li>
                <strong>{{ rejected.change_id }}</strong>
                {% for issue in rejected.issues or [] %}
                  <span class="issue">
                    {{ issue.reason or 'issue' }}
                    {% if issue.message %}: {{ issue.message }}{% endif %}
                  </span>
                {% endfor %}
              </li>
            {% endfor %}
          </ol>
        {% else %}
          <pre>No rejected changes recorded.</pre>
        {% endif %}
      </section>

      <section>
        <h2>Unsupported Terms</h2>
        {% if review_details.unsupported_claims %}
          <ol class="review-list">
            {% for claim in review_details.unsupported_claims %}
              <li>{{ claim }}</li>
            {% endfor %}
          </ol>
        {% else %}
          <pre>No unsupported terms recorded.</pre>
        {% endif %}
      </section>
    </div>
  </main>
</body>
</html>
"""


COVER_LETTER_EDIT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edit Cover Letter - {{ row.company }}</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #626a73;
      --line: #d9dee5;
      --surface: #ffffff;
      --band: #f4f6f8;
      --accent: #0b6e69;
      --accent-strong: #074f4b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--band);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 3px 0 2px; font-size: 20px; font-weight: 650; }
    a { color: var(--accent); font-weight: 650; }
    .muted { color: var(--muted); }
    .top-links { display: flex; gap: 10px; flex-wrap: wrap; }
    main { padding: 18px 24px 40px; }
    .flash {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #b9d8d5;
      background: #eef8f6;
    }
    .sync-warning {
      align-items: center;
      background: #fff8e7;
      border: 1px solid #f2d28b;
      color: #5c4100;
      display: flex;
      gap: 12px;
      justify-content: space-between;
      margin: 0 0 14px;
      padding: 12px;
    }
    .sync-warning p {
      margin: 3px 0 0;
    }
    .sync-warning form {
      margin: 0;
    }
    .save-bar {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: -18px -24px 18px;
      padding: 12px 24px;
      background: rgba(255, 255, 255, .96);
      border-bottom: 1px solid var(--line);
    }
    button, .button-link {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 650;
      padding: 8px 10px;
      text-decoration: none;
      white-space: nowrap;
    }
    button:hover, .button-link:hover { background: var(--accent-strong); }
    button.secondary, .button-link.secondary {
      background: #fff;
      color: var(--accent);
    }
    button.secondary:hover, .button-link.secondary:hover {
      background: #eff7f6;
      color: var(--accent-strong);
    }
    .editor-shell {
      background: var(--surface);
      border: 1px solid var(--line);
    }
    .editor-toolbar {
      display: flex;
      gap: 6px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      background: #fafbfc;
    }
    .editor-toolbar button {
      min-width: 38px;
      padding: 6px 8px;
    }
    .cover-editor {
      min-height: 620px;
      padding: 28px 32px;
      background: #fff;
      color: var(--ink);
      font: 16px/1.58 Georgia, "Times New Roman", serif;
      outline: none;
      white-space: normal;
    }
    .cover-editor:focus {
      box-shadow: inset 0 0 0 2px #b9d8d5;
    }
    .cover-editor p, .cover-editor div {
      margin: 0 0 14px;
    }
    @media (max-width: 900px) {
      header, .save-bar { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .save-bar { margin: -14px -14px 14px; padding: 12px 14px; }
      .cover-editor { min-height: 480px; padding: 20px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="top-links">
        <a href="{{ return_to }}">Back to tracker</a>
        {% if row.cover_letter_content %}
          <a
            href="/cover-letters/{{ row.job_id }}"
            target="_blank"
            rel="noreferrer"
          >
            Cover Letter PDF
          </a>
        {% endif %}
      </div>
      <h1>Edit Cover Letter</h1>
      <div class="muted">{{ row.company }} - {{ row.job_title }} - {{ row.job_id }}</div>
    </div>
    {% if row.cover_letter_object_updated_at %}
      <div class="muted">{{ row.cover_letter_object_updated_at|display_timestamp }}</div>
    {% endif %}
  </header>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <form id="cover-letter-form" method="post" action="/cover-letters/{{ row.job_id }}/edit">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <input id="body-html" type="hidden" name="body_html">
      <div class="save-bar">
        <div>
          <button type="submit">Save PDF</button>
          <a class="button-link secondary" href="{{ return_to }}">Close</a>
        </div>
        <span class="muted">{{ row.cover_letter_filename or 'No cover letter' }}</span>
      </div>
      <div class="editor-shell">
        <div class="editor-toolbar" aria-label="Formatting">
          <button type="button" data-command="bold"><strong>B</strong></button>
          <button type="button" data-command="italic"><em>I</em></button>
        </div>
        <div id="cover-editor" class="cover-editor" contenteditable="true">
          {{ cover_letter_object.body_html | safe }}
        </div>
      </div>
    </form>
  </main>
  <script>
    const form = document.querySelector("#cover-letter-form");
    const editor = document.querySelector("#cover-editor");
    const bodyHtml = document.querySelector("#body-html");
    document.querySelectorAll("[data-command]").forEach((button) => {
      button.addEventListener("click", () => {
        editor.focus();
        document.execCommand(button.dataset.command, false, null);
      });
    });
    form.addEventListener("submit", () => {
      bodyHtml.value = editor.innerHTML;
    });
  </script>
</body>
</html>
"""


RESUME_EDIT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edit Resume - {{ row.company }}</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #626a73;
      --line: #d9dee5;
      --surface: #ffffff;
      --band: #f4f6f8;
      --accent: #0b6e69;
      --accent-strong: #074f4b;
      --danger: #a13d2d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--band);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 3px 0 2px; font-size: 20px; font-weight: 650; }
    h2 { margin: 0; font-size: 15px; }
    h3 { margin: 0 0 8px; font-size: 14px; }
    a { color: var(--accent); font-weight: 650; }
    .muted { color: var(--muted); }
    .top-links { display: flex; gap: 10px; flex-wrap: wrap; }
    .score-panel {
      min-width: 280px;
      border: 1px solid var(--line);
      background: var(--band);
      padding: 10px 12px;
    }
    .score-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 3px 14px;
      margin-top: 6px;
      font-size: 13px;
    }
    main { padding: 18px 24px 40px; }
    .flash {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #b9d8d5;
      background: #eef8f6;
    }
    .save-bar {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: -18px -24px 18px;
      padding: 12px 24px;
      background: rgba(255, 255, 255, .96);
      border-bottom: 1px solid var(--line);
    }
    button, .button-link {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 650;
      padding: 8px 10px;
      text-decoration: none;
      white-space: nowrap;
    }
    button:hover, .button-link:hover { background: var(--accent-strong); }
    button.secondary, .button-link.secondary {
      background: #fff;
      color: var(--accent);
    }
    button.secondary:hover, .button-link.secondary:hover {
      background: #eff7f6;
      color: var(--accent-strong);
    }
    button.danger {
      border-color: var(--danger);
      background: #fff;
      color: var(--danger);
    }
    button.danger:hover {
      background: #fff3f0;
      color: #7f2b1d;
    }
    .section {
      margin: 0 0 14px;
      padding: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .section-title label, .render-toggle {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-weight: 650;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label.field {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    input[type="text"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      text-transform: none;
    }
    textarea {
      min-height: 76px;
      resize: vertical;
      line-height: 1.4;
    }
    textarea.tall { min-height: 126px; }
    textarea.bulk {
      min-height: 160px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .rich-field {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
    }
    .rich-field .editor-toolbar {
      display: flex;
      gap: 5px;
      padding: 6px;
      border-bottom: 1px solid var(--line);
      background: #fafbfc;
    }
    .rich-field .editor-toolbar button {
      min-width: 32px;
      padding: 4px 7px;
    }
    .rich-editor {
      min-height: 39px;
      padding: 8px 9px;
      outline: none;
      white-space: pre-wrap;
    }
    .rich-editor.tall { min-height: 126px; }
    .rich-editor:focus {
      box-shadow: inset 0 0 0 2px #b9d8d5;
    }
    .job-card, .nested-card {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #fbfcfd;
    }
    .bullet-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 9px;
      align-items: start;
      margin-top: 8px;
    }
    .bullet-row textarea { min-height: 54px; }
    .compact-list {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }
    .actions-inline {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    @media (max-width: 900px) {
      header, .save-bar, .section-title { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .save-bar { margin: -14px -14px 14px; padding: 12px 14px; }
      .grid { grid-template-columns: 1fr; }
      .score-panel { min-width: 0; width: 100%; }
    }
  </style>
</head>
<body>
  {% set header_top = resume.header_top | default({}) %}
  {% set summary = resume.professional_summary | default({}) %}
  {% set skills = resume.core_technical_skills | default({}) %}
  {% set experience = resume.professional_experience | default({}) %}
  {% set education = resume.education | default({}) %}
  {% set certifications = resume.certifications | default({}) %}
  {% set portfolio = resume.portfolio | default({}) %}
  {% macro rich_editor(name, value, tall=False) -%}
    <div class="rich-field">
      <div class="editor-toolbar" aria-label="Formatting">
        <button type="button" data-rich-command="bold"><strong>B</strong></button>
        <button type="button" data-rich-command="italic"><em>I</em></button>
      </div>
      <input
        type="hidden"
        name="{{ name }}"
        data-rich-input="{{ name }}"
        value="{{ value | default('', true) }}"
      >
      <div
        class="rich-editor {{ 'tall' if tall else '' }}"
        contenteditable="true"
        data-rich-target="{{ name }}"
      >{{ value | default('', true) | rich_text }}</div>
    </div>
  {%- endmacro %}
  <header>
    <div>
      <div class="top-links">
        <a href="{{ return_to }}">Back to tracker</a>
        {% if row.resume_content %}
          <a href="/resumes/{{ row.job_id }}" target="_blank" rel="noreferrer">
            Resume PDF
          </a>
        {% endif %}
        {% if row.resume_html_content %}
          <a href="/resume-html/{{ row.job_id }}" target="_blank" rel="noreferrer">
            Resume HTML
          </a>
        {% endif %}
        <a href="{{ variants_return_path }}">
          Review variants
        </a>
      </div>
      <h1>Edit Resume</h1>
      <div class="muted">{{ row.company }} - {{ row.job_title }} - {{ row.job_id }}</div>
    </div>
    <div class="score-panel">
      <strong>ATS proxy score</strong>
      <div class="score-grid">
        <span>Overall</span>
        <strong>{{ row.ats_score if row.ats_score is not none else '-' }}/100</strong>
        <span>Parsing</span>
        <strong>
          {{ row.ats_parsing_score if row.ats_parsing_score is not none else '-' }}/100
        </strong>
        <span>Keyword</span>
        <strong>
          {{ row.ats_keyword_score if row.ats_keyword_score is not none else '-' }}/100
        </strong>
        <span>Semantic</span>
        <strong>
          {{ row.ats_semantic_score if row.ats_semantic_score is not none else '-' }}/100
        </strong>
        <span>Risk</span><strong>{{ row.ats_formatting_risk or '-' }}</strong>
      </div>
      {% if row.ats_missing_terms %}
        <p class="muted">{{ row.ats_missing_terms }}</p>
      {% endif %}
    </div>
  </header>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% if row.aro_resume_out_of_sync %}
      <section class="sync-warning">
        <div>
          <strong>ARO and draft resume are out of sync.</strong>
          <p>
            The ARO changed after the rendered draft resume. Sync before editing if
            you want the draft HTML/PDF/ATS to reflect the current ARO.
          </p>
        </div>
        <form method="post" action="/resumes/{{ row.job_id }}/sync">
          <input type="hidden" name="return_to" value="{{ return_to }}">
          <button type="submit">Sync Draft to ARO</button>
        </form>
      </section>
    {% endif %}
    <form id="resume-edit-form" method="post" action="/resumes/{{ row.job_id }}/edit">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <div class="save-bar">
        <div class="actions-inline">
          <button type="submit">Save, Render, Rescore</button>
          <a class="button-link secondary" href="{{ return_to }}">Close</a>
        </div>
        {% if row.application_resume_backup_object %}
          <span class="muted">
            Backup: {{ row.application_resume_backup_created_at|display_timestamp }}
          </span>
        {% else %}
          <span class="muted">Backup: none</span>
        {% endif %}
      </div>

      <section class="section">
        <div class="section-title">
          <h2>Header</h2>
        </div>
        <div class="grid">
          <label class="field">Name
            <input
              type="text"
              name="header_name"
              value="{{ header_top.line_1_name_header_text | default('', true) }}"
            >
          </label>
          <label class="field">Header Line 2
            {{ rich_editor('header_line_2', header_top.line_2_header_text | default('', true)) }}
          </label>
          <label class="field">Fallback Contact Line
            {% set header_info = header_top.line_3_applicant_info_text | default('', true) %}
            {% if not header_info %}
              {% set header_info = header_top.line_2_applicant_info_text | default('', true) %}
            {% endif %}
            {{ rich_editor('header_info', header_info) }}
          </label>
        </div>
        <label class="field">Contact Items
          <textarea name="header_contact_items">
            {{- (header_top.contact_items | default([])) | join('\n') -}}
          </textarea>
        </label>
      </section>

      <section class="section">
        <div class="section-title">
          <label>
            <input
              type="checkbox"
              name="summary_render"
              value="1"
              {{ 'checked' if summary.render | default(true) else '' }}
            >
            Professional Summary
          </label>
        </div>
        <div class="grid">
          <label class="field">Heading
            <input
              type="text"
              name="summary_header_text"
              value="{{ summary.header_text | default('Professional Summary', true) }}"
            >
          </label>
          <label class="field">Note
            {{ rich_editor('summary_note', summary.summary_note | default('', true)) }}
          </label>
        </div>
        <label class="field">Paragraph
          {{ rich_editor('summary_paragraph', summary.paragraph | default('', true), tall=True) }}
        </label>
      </section>

      <section class="section">
        <div class="section-title">
          <label>
            <input
              type="checkbox"
              name="skills_render"
              value="1"
              {{ 'checked' if skills.render | default(true) else '' }}
            >
            Core Technical Skills
          </label>
        </div>
        <label class="field">Heading
          <input
            type="text"
            name="skills_header_text"
            value="{{ skills.header_text | default('Core Technical Skills', true) }}"
          >
        </label>
        {% for bullet in skills.bullet_points | default([]) %}
          {% set skill_index = loop.index0 %}
          {% set skill_items = bullet["items"] | default({}) %}
          <div class="nested-card">
            <label class="field">Category
              <input
                type="text"
                name="skill_{{ skill_index }}_category"
                value="{{ bullet.category | default('', true) }}"
              >
            </label>
            <div class="grid">
              <label class="field">Primary Items
                <textarea name="skill_{{ skill_index }}_primary">
                  {{- (skill_items.primary | default([])) | join('\n') -}}
                </textarea>
              </label>
              <label class="field">Additional Items
                <textarea name="skill_{{ skill_index }}_additional">
                  {{- (skill_items.additional | default([])) | join('\n') -}}
                </textarea>
              </label>
            </div>
            <label class="field">JOD Matched Items
              <textarea name="skill_{{ skill_index }}_jod_matched">
                {{- (bullet.jod_matched_items | default([])) | join('\n') -}}
              </textarea>
            </label>
          </div>
        {% endfor %}
      </section>

      <section class="section">
        <div class="section-title">
          <label>
            <input
              type="checkbox"
              name="experience_render"
              value="1"
              {{ 'checked' if experience.render | default(true) else '' }}
            >
            Professional Experience
          </label>
        </div>
        <label class="field">Heading
          <input
            type="text"
            name="experience_header_text"
            value="{{ experience.header_text | default('Professional Experience', true) }}"
          >
        </label>
        {% for job in experience.jobs | default([]) %}
          {% set job_index = loop.index0 %}
          {% set line_1 = job.line_1 | default({}) %}
          {% set line_2 = job.line_2 | default({}) %}
          {% set bulk_text = job_bulk_bullet_text(job) %}
          <div class="job-card">
            <div class="section-title">
              <h3>Job {{ loop.index }}</h3>
              <label class="render-toggle">
                <input
                  type="checkbox"
                  name="job_{{ job_index }}_render"
                  value="1"
                  {{ 'checked' if job.render | default(true) else '' }}
                >
                Render job
              </label>
            </div>
            <div class="grid">
              <label class="field">Company
                <input
                  type="text"
                  name="job_{{ job_index }}_company"
                  value="{{ line_1.company_name_text | default('', true) }}"
                >
              </label>
              <label class="field">Position
                <input
                  type="text"
                  name="job_{{ job_index }}_position"
                  value="{{ line_1.position_name_text | default('', true) }}"
                >
              </label>
              <label class="field">Dates
                <input
                  type="text"
                  name="job_{{ job_index }}_dates"
                  value="{{ line_1.position_dates_text | default('', true) }}"
                >
              </label>
              <label class="field">Intro
                {{ rich_editor(
                  'job_' ~ job_index ~ '_intro',
                  line_2.position_intro_text | default('', true)
                ) }}
              </label>
            </div>
            <label class="field">All Bullet Points
              <textarea class="bulk" name="job_{{ job_index }}_bulk">{{ bulk_text }}</textarea>
              <textarea hidden name="job_{{ job_index }}_bulk_original">{{ bulk_text }}</textarea>
            </label>
            <div class="compact-list">
              {% for bullet in job.bullet_points | default([]) %}
                {% set bullet_index = loop.index0 %}
                <div class="bullet-row">
                  <input
                    type="checkbox"
                    name="job_{{ job_index }}_bullet_{{ bullet_index }}_render"
                    value="1"
                    aria-label="Render bullet {{ bullet_index + 1 }}"
                    {{ 'checked' if bullet is string or (bullet.render | default(true)) else '' }}
                  >
                  {{ rich_editor(
                    'job_' ~ job_index ~ '_bullet_' ~ bullet_index ~ '_text',
                    bullet_text_for_editing(bullet)
                  ) }}
                </div>
              {% endfor %}
            </div>
          </div>
        {% endfor %}
      </section>

      <section class="section">
        <div class="section-title">
          <label>
            <input
              type="checkbox"
              name="education_render"
              value="1"
              {{ 'checked' if education.render | default(true) else '' }}
            >
            Education
          </label>
        </div>
        <label class="field">Heading
          <input
            type="text"
            name="education_header_text"
            value="{{ education.header_text | default('Education', true) }}"
          >
        </label>
        {% for entry in education.entries | default([]) %}
          {% set entry_index = loop.index0 %}
          {% set line_1 = entry.line_1 | default({}) %}
          {% set line_2 = entry.line_2 | default({}) %}
          <div class="nested-card">
            <label class="render-toggle">
              <input
                type="checkbox"
                name="education_{{ entry_index }}_render"
                value="1"
                {{ 'checked' if entry.render | default(true) else '' }}
              >
              Render entry
            </label>
            <div class="grid">
              <label class="field">Institution
                <input
                  type="text"
                  name="education_{{ entry_index }}_institution"
                  value="{{ line_1.institution_name_text | default('', true) }}"
                >
              </label>
              <label class="field">Degree
                <input
                  type="text"
                  name="education_{{ entry_index }}_degree"
                  value="{{ line_2.degree_name_text | default('', true) }}"
                >
              </label>
              <label class="field">Dates
                <input
                  type="text"
                  name="education_{{ entry_index }}_dates"
                  value="{{ line_2.degree_dates_text | default('', true) }}"
                >
              </label>
            </div>
            {% for bullet in entry.bullet_points | default([]) %}
              {% set bullet_index = loop.index0 %}
              <div class="bullet-row">
                <input
                  type="checkbox"
                  name="education_{{ entry_index }}_bullet_{{ bullet_index }}_render"
                  value="1"
                  {{ 'checked' if bullet.render | default(true) else '' }}
                >
                {{ rich_editor(
                  'education_' ~ entry_index ~ '_bullet_' ~ bullet_index ~ '_text',
                  bullet.text | default('', true)
                ) }}
              </div>
            {% endfor %}
          </div>
        {% endfor %}
      </section>

      <section class="section">
        <div class="section-title">
          <label>
            <input
              type="checkbox"
              name="certifications_render"
              value="1"
              {{ 'checked' if certifications.render | default(true) else '' }}
            >
            Certifications
          </label>
        </div>
        <label class="field">Heading
          <input
            type="text"
            name="certifications_header_text"
            value="{{ certifications.header_text | default('Certifications', true) }}"
          >
        </label>
        {% for bullet in certifications.bullet_points | default([]) %}
          {% set bullet_index = loop.index0 %}
          <div class="bullet-row">
            <input
              type="checkbox"
              name="certification_{{ bullet_index }}_render"
              value="1"
              {{ 'checked' if bullet.render | default(true) else '' }}
            >
            {{ rich_editor(
              'certification_' ~ bullet_index ~ '_text',
              bullet.text | default('', true)
            ) }}
          </div>
        {% endfor %}
      </section>

      <section class="section">
        <div class="section-title">
          <label>
            <input
              type="checkbox"
              name="portfolio_render"
              value="1"
              {{ 'checked' if portfolio.render | default(true) else '' }}
            >
            Portfolio
          </label>
        </div>
        <label class="field">Heading
          <input
            type="text"
            name="portfolio_header_text"
            value="{{ portfolio.header_text | default('Portfolio', true) }}"
          >
        </label>
        {% for project in portfolio.projects | default([]) %}
          {% set project_index = loop.index0 %}
          <div class="nested-card">
            <label class="render-toggle">
              <input
                type="checkbox"
                name="portfolio_{{ project_index }}_render"
                value="1"
                {{ 'checked' if project.render | default(true) else '' }}
              >
              Render project
            </label>
            <div class="grid">
              <label class="field">Title
                <input
                  type="text"
                  name="portfolio_{{ project_index }}_title"
                  value="{{ project.title_text | default('', true) }}"
                >
              </label>
              <label class="field">URL
                <input
                  type="text"
                  name="portfolio_{{ project_index }}_url"
                  value="{{ project.url | default('', true) }}"
                >
              </label>
            </div>
            <label class="field">Description
              {{ rich_editor(
                'portfolio_' ~ project_index ~ '_description',
                project.description_text | default('', true)
              ) }}
            </label>
          </div>
        {% endfor %}
      </section>
    </form>

    <form method="post" action="/resumes/{{ row.job_id }}/edit/revert">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <button
        class="danger"
        type="submit"
        {{ 'disabled' if not row.application_resume_backup_object else '' }}
      >
        Revert To Backup
      </button>
    </form>
  </main>
  <script>
    const resumeForm = document.querySelector("#resume-edit-form");
    document.querySelectorAll("[data-rich-command]").forEach((button) => {
      button.addEventListener("click", () => {
        const editor = button.closest(".rich-field").querySelector(".rich-editor");
        editor.focus();
        document.execCommand(button.dataset.richCommand, false, null);
      });
    });
    if (resumeForm) {
      resumeForm.addEventListener("submit", () => {
        document.querySelectorAll(".rich-editor[data-rich-target]").forEach((editor) => {
          const target = editor.dataset.richTarget;
          const input = document.querySelector(`[data-rich-input="${target}"]`);
          if (input) {
            input.value = editor.innerHTML;
          }
        });
      });
    }
  </script>
</body>
</html>
"""


DESCRIPTION_COMPARE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edit Descriptions - {{ row.company }} - {{ row.job_title }}</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #626a73;
      --line: #d9dee5;
      --surface: #ffffff;
      --band: #f4f6f8;
      --accent: #0b6e69;
      --accent-strong: #074f4b;
      --removed: #fff1f1;
      --added: #edf8f0;
      --changed: #fff8e6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--band);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 3px 0 2px; font-size: 20px; font-weight: 650; }
    .meta, .muted { color: var(--muted); }
    a { color: var(--accent); font-weight: 650; }
    .top-links { display: flex; gap: 10px; flex-wrap: wrap; }
    main {
      padding: 18px 24px 40px;
    }
    .flash {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #b9d8d5;
      background: #eef8f6;
    }
    .save-bar {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: -18px -24px 18px;
      padding: 12px 24px;
      background: rgba(255, 255, 255, .96);
      border-bottom: 1px solid var(--line);
    }
    button, .button-link {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 650;
      padding: 8px 10px;
      text-decoration: none;
      white-space: nowrap;
    }
    button:hover, .button-link:hover { background: var(--accent-strong); }
    .button-link.secondary {
      background: #fff;
      color: var(--accent);
    }
    .button-link.secondary:hover {
      background: #eff7f6;
      color: var(--accent-strong);
    }
    .editor-grid, .diff-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      margin-bottom: 14px;
    }
    section {
      min-width: 0;
      background: var(--surface);
      border: 1px solid var(--line);
    }
    .wide { margin-bottom: 14px; }
    h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      font-weight: 700;
    }
    textarea {
      display: block;
      width: 100%;
      min-height: calc(100vh - 270px);
      max-height: 72vh;
      margin: 0;
      overflow: auto;
      resize: vertical;
      border: 0;
      border-radius: 0;
      padding: 14px;
      background: #fff;
      color: var(--ink);
      font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    textarea:focus {
      outline: 2px solid #b9d8d5;
      outline-offset: -2px;
    }
    pre {
      margin: 0;
      overflow: auto;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff;
      color: var(--ink);
      font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .removed-text { max-height: 360px; background: var(--removed); }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--line);
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
      text-align: left;
    }
    th {
      background: #fafbfc;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    td.line-no {
      width: 52px;
      color: var(--muted);
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-align: right;
    }
    td.status {
      width: 86px;
      font-weight: 700;
    }
    td pre { padding: 0; background: transparent; }
    tr.delete { background: var(--removed); }
    tr.insert { background: var(--added); }
    tr.replace { background: var(--changed); }
    .score-panel {
      min-width: 230px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: #fbfcfd;
    }
    .score-grid {
      display: grid;
      grid-template-columns: auto auto;
      gap: 2px 12px;
      margin-top: 6px;
      font-size: 12px;
    }
    @media (max-width: 900px) {
      header, .save-bar { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .save-bar { margin: -14px -14px 14px; padding: 12px 14px; }
      .editor-grid, .diff-grid { grid-template-columns: 1fr; }
      textarea {
        min-height: 46vh;
        max-height: 56vh;
      }
      .score-panel { min-width: 0; width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="top-links">
        <a href="{{ return_to }}">Back to tracker</a>
        {% if row.linkedin_url %}
          <a href="{{ row.linkedin_url }}" target="_blank" rel="noreferrer">Job URL</a>
        {% endif %}
      </div>
      <h1>Edit Job Descriptions</h1>
      <div class="meta">{{ row.company }} - {{ row.job_title }} - {{ row.job_id }}</div>
    </div>
    <div class="score-panel">
      <strong>ATS proxy score</strong>
      <div class="score-grid">
        <span>Overall</span>
        <strong>{{ row.ats_score if row.ats_score is not none else '-' }}/100</strong>
        <span>Keyword</span>
        <strong>
          {{ row.ats_keyword_score if row.ats_keyword_score is not none else '-' }}/100
        </strong>
        <span>Semantic</span>
        <strong>
          {{ row.ats_semantic_score if row.ats_semantic_score is not none else '-' }}/100
        </strong>
        <span>Risk</span><strong>{{ row.ats_formatting_risk or '-' }}</strong>
      </div>
      {% if row.ats_missing_terms %}
        <div class="muted">{{ row.ats_missing_terms }}</div>
      {% endif %}
    </div>
  </header>
  <main>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <form method="post" action="/descriptions/{{ row.job_id }}">
      <input type="hidden" name="return_to" value="{{ return_to }}">
      <div class="save-bar">
        <div>
          <button type="submit">Save And Rescore</button>
          <a class="button-link secondary" href="{{ return_to }}">Close</a>
        </div>
        <span class="muted">ATS source: Prompt Job Description when present</span>
      </div>
      <div class="editor-grid">
        <section>
          <h2>Parsed Job Description</h2>
          <textarea name="job_description">{{ row.job_description or "" }}</textarea>
        </section>
        <section>
          <h2>Prompt Job Description</h2>
          <textarea name="prompt_job_description">{{ row.prompt_job_description or "" }}</textarea>
        </section>
      </div>
    </form>

    <section class="wide">
      <h2>Removed By Trimming</h2>
      {% if removed_text %}
        <pre class="removed-text">{{ removed_text }}</pre>
      {% else %}
        <pre class="removed-text">No removed text detected.</pre>
      {% endif %}
    </section>

    <section class="wide">
      <h2>Description Diff</h2>
      {% if diff_rows %}
        <table>
          <thead>
            <tr>
              <th class="status">Type</th>
              <th>Parsed Token</th>
              <th>Parsed Text</th>
              <th>Prompt Token</th>
              <th>Prompt Text</th>
            </tr>
          </thead>
          <tbody>
            {% for diff in diff_rows %}
              <tr class="{{ diff.status }}">
                <td class="status">
                  {% if diff.status == "delete" %}Removed
                  {% elif diff.status == "insert" %}Added
                  {% elif diff.status == "replace" %}Changed
                  {% else %}{{ diff.status }}
                  {% endif %}
                </td>
                <td class="line-no">{{ diff.left_line_no or "" }}</td>
                <td><pre>{{ diff.left_text }}</pre></td>
                <td class="line-no">{{ diff.right_line_no or "" }}</td>
                <td><pre>{{ diff.right_text }}</pre></td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <pre>No line-level differences detected.</pre>
      {% endif %}
    </section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
