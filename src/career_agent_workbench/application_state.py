"""Caller-injected, side-effect-free SQLite application state.

The module deliberately contains no workspace discovery, artifact generation,
or workflow execution.  A database is created or migrated only by an explicit
call to :meth:`ApplicationStateStore.initialize`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import quote

import yaml

from career_agent_workbench.config import WorkspaceMember, WorkspacePaths
from career_agent_workbench.errors import ProviderError, WorkflowError
from career_agent_workbench.generic_job_scraper import normalize_job_url
from career_agent_workbench.query_optimizer import StoredQueryOutcome

SCHEMA_VERSION = 1
DEFAULT_BUSY_TIMEOUT_SECONDS = 1.0

MAX_IDENTIFIER_CHARS = 128
MAX_LABEL_CHARS = 1_024
MAX_URL_CHARS = 4_096
MAX_JOD_SOURCE_CHARS = 500_000
MAX_JOD_PROMPT_CHARS = 12_000
MAX_ARO_YAML_BYTES = 2_000_000
MAX_HTML_CHARS = 2_000_000
MAX_PDF_BYTES = 20_000_000
MAX_JSON_CHARS = 2_000_000
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
MAX_NOTES_CHARS = 100_000
MAX_METADATA_TEXT_CHARS = 500_000
MAX_BULK_IDENTIFIERS = 500
MAX_QUERY_RESULTS = 1_000
MAX_QUERY_HISTORY_RESULTS = 500
MAX_QUERY_HISTORY_WRITES = 200

APPLICATION_STATUSES = frozenset(
    {"No", "Yes", "N/A", "Rejected", "Accepted for interview"}
)
VARIANT_KEYS = ("v1", "v2", "manual")
VARIANT_PREFERENCE = ("manual", "v2", "v1")
SELECTION_MODES = frozenset({"auto", "manual"})

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MIME_HTML = "text/html; charset=utf-8"
_MIME_PDF = "application/pdf"
_MISSING = object()

_CONFIGURATION_ERROR = "Application state database is not configured."
_NOT_INITIALIZED_ERROR = "Application state is not initialized."
_VALIDATION_ERROR = "Application state input is invalid."
_NOT_FOUND_ERROR = "Application state record was not found."
_CORRUPTION_ERROR = "Application state data is invalid."
_BUSY_ERROR = "Application state is temporarily unavailable."
_INITIALIZATION_ERROR = "Application state initialization failed."
_OPERATION_ERROR = "Application state operation failed."
_CONFLICT_ERROR = "Application state changed."


class ApplicationStateError(WorkflowError):
    """Base class for content-free application-state failures."""

    __slots__ = ()


class ApplicationStateConfigurationError(ApplicationStateError):
    """Raised when a caller did not supply an exact database path."""

    __slots__ = ()


class ApplicationStateNotInitializedError(ApplicationStateError):
    """Raised when an ordinary operation precedes explicit initialization."""

    __slots__ = ()


class ApplicationStateValidationError(ApplicationStateError):
    """Raised when caller-supplied state is outside the bounded contract."""

    __slots__ = ()


class ApplicationStateNotFoundError(ApplicationStateError):
    """Raised when an explicitly addressed state record does not exist."""

    __slots__ = ()


class ApplicationStateCorruptionError(ApplicationStateError):
    """Raised when persisted structured state is malformed or oversized."""

    __slots__ = ()


class ApplicationStateBusyError(ApplicationStateError):
    """Raised when SQLite cannot acquire its finite operation-local lock."""

    __slots__ = ()


class ApplicationStateInitializationError(ApplicationStateError):
    """Raised when a schema cannot be initialized conservatively."""

    __slots__ = ()


class ApplicationStateConflictError(ApplicationStateError):
    """Raised when a conditional workflow write observes changed state."""

    __slots__ = ()


class ApplicationScope(StrEnum):
    """Explicit application-list selection."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    ALL = "all"


class ResumeVariantKey(StrEnum):
    """Bounded resume-variant identities."""

    V1 = "v1"
    V2 = "v2"
    MANUAL = "manual"


class ResumeSelectionMode(StrEnum):
    """Automatic preference or an explicit human pin."""

    AUTO = "auto"
    MANUAL = "manual"


class ArtifactKind(StrEnum):
    """Artifact metadata kinds supported by this state-only layer."""

    RESUME_HTML = "resume_html"
    RESUME_PDF = "resume_pdf"
    COVER_LETTER_PDF = "cover_letter_pdf"


@dataclass(frozen=True, slots=True)
class AtsFields:
    """Already-calculated ATS values supplied by a caller."""

    score: int | None = None
    parsing_score: int | None = None
    keyword_score: int | None = None
    semantic_score: int | None = None
    formatting_risk: str | None = None
    missing_terms: str | None = None
    diagnostics: Mapping[str, Any] | None = field(default=None, repr=False)
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ApplicationMetadata:
    """Provider-neutral metadata used to create or refresh an application."""

    job_id: str
    company: str
    job_title: str
    job_url: str
    source: str = ""
    date_matched: str | None = None
    date_posted: str | None = None
    experience_level: str | None = None


@dataclass(frozen=True, slots=True)
class ResumeVariantWrite:
    """One complete, caller-produced resume variant."""

    variant_key: str
    variant_label: str
    source: str
    application_resume_yaml: str = field(repr=False)
    parent_variant_key: str | None = None
    resume_html: str | None = field(default=None, repr=False)
    resume_pdf: bytes | None = field(default=None, repr=False)
    source_resume_html_path: str = field(default="", repr=False)
    source_resume_path: str = field(default="", repr=False)
    ats: AtsFields = field(default_factory=AtsFields, repr=False)
    ats_diagnostics: Mapping[str, Any] | None = field(default=None, repr=False)
    evidence_packet: Mapping[str, Any] | None = field(default=None, repr=False)
    external_critique: Mapping[str, Any] | None = field(default=None, repr=False)
    critique_prompt: str | None = field(default=None, repr=False)
    critique_response: str | None = field(default=None, repr=False)
    critique: Mapping[str, Any] | None = field(default=None, repr=False)
    validation: Mapping[str, Any] | None = field(default=None, repr=False)
    model_metadata: Mapping[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ResumeVariantRecord:
    """Immutable resume-variant snapshot."""

    job_id: str
    variant_key: str
    variant_label: str
    source: str
    parent_variant_key: str | None
    application_resume: Mapping[str, Any] = field(repr=False)
    resume_html_filename: str
    resume_html: str | None = field(repr=False)
    resume_html_mime_type: str
    source_resume_html_path: str = field(repr=False)
    resume_html_updated_at: str | None
    resume_pdf_filename: str
    resume_pdf: bytes | None = field(repr=False)
    resume_pdf_mime_type: str
    source_resume_path: str = field(repr=False)
    resume_pdf_updated_at: str | None
    ats: AtsFields = field(repr=False)
    ats_diagnostics: Mapping[str, Any] | None = field(repr=False)
    evidence_packet: Mapping[str, Any] | None = field(repr=False)
    external_critique: Mapping[str, Any] | None = field(repr=False)
    critique_prompt: str | None = field(repr=False)
    critique_response: str | None = field(repr=False)
    critique: Mapping[str, Any] | None = field(repr=False)
    validation: Mapping[str, Any] | None = field(repr=False)
    model_metadata: Mapping[str, Any] | None = field(repr=False)
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ApplicationRecord:
    """Immutable provider-neutral application snapshot."""

    job_id: str
    company: str
    job_title: str
    job_url: str
    source: str
    job_description: str | None = field(repr=False)
    prompt_job_description: str | None = field(repr=False)
    application_resume: Mapping[str, Any] | None = field(repr=False)
    application_resume_backup: Mapping[str, Any] | None = field(repr=False)
    application_resume_backup_target: str | None
    selected_resume_variant: str | None
    resume_variant_selection_mode: str
    selected_variant: ResumeVariantRecord | None = field(repr=False)
    resume_html_filename: str
    resume_html: str | None = field(repr=False)
    resume_html_mime_type: str
    source_resume_html_path: str = field(repr=False)
    resume_html_updated_at: str | None
    resume_pdf_filename: str
    resume_pdf: bytes | None = field(repr=False)
    resume_pdf_mime_type: str
    source_resume_path: str = field(repr=False)
    resume_pdf_updated_at: str | None
    ats: AtsFields = field(repr=False)
    cover_letter: Mapping[str, Any] | None = field(repr=False)
    cover_letter_filename: str
    cover_letter_pdf: bytes | None = field(repr=False)
    cover_letter_mime_type: str
    source_cover_letter_path: str = field(repr=False)
    cover_letter_updated_at: str | None
    date_matched: str | None
    date_posted: str | None
    experience_level: str | None
    applied_to: str
    date_applied: str | None
    notes: str = field(repr=False)
    archived_at: str | None
    imported_at: str
    updated_at: str


@dataclass(frozen=True, slots=True, repr=False)
class ApplicationSeedOutcome:
    """Immutable transaction-confirmed application seed outcome."""

    application: ApplicationRecord = field(repr=False)
    created: bool

    def __post_init__(self) -> None:
        if (
            type(self.application) is not ApplicationRecord
            or type(self.created) is not bool
        ):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)

    def __repr__(self) -> str:
        return f"ApplicationSeedOutcome(created={self.created}, content_hidden=True)"


@dataclass(frozen=True, slots=True, repr=False)
class QueryOutcomeWrite:
    """One bounded public-search outcome without job or profile content."""

    keywords: str = field(repr=False)
    location: str = field(repr=False)
    date_posted: str
    workplace_type: str | None
    experience_level: str | None
    job_type: str | None
    sort_by: str
    limit: int
    page: int
    profile_match: float
    query_score: float
    results_returned: int
    fresh_jobs_accepted: int
    skipped_existing: int = 0
    skipped_blacklisted: int = 0
    skipped_workplace_type: int = 0
    skipped_experience_level: int = 0

    def __repr__(self) -> str:
        return "QueryOutcomeWrite(configured=True, content_hidden=True)"


@dataclass(frozen=True, slots=True, repr=False)
class ApplicationWorkflowRevision:
    """Opaque fixed-length revision for one coherent workflow snapshot."""

    _value: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self._value) is not bytes
            or len(self._value) != hashlib.sha256().digest_size
        ):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)

    def __repr__(self) -> str:
        return "ApplicationWorkflowRevision(hidden=True)"

    __str__ = __repr__


@dataclass(frozen=True, slots=True, repr=False)
class ApplicationWorkflowSnapshot:
    """One coherent application, variant set, and opaque workflow revision."""

    application: ApplicationRecord = field(repr=False)
    variants: tuple[ResumeVariantRecord, ...] = field(repr=False)
    revision: ApplicationWorkflowRevision = field(repr=False)
    edit_revision: ApplicationWorkflowRevision | None = field(default=None, repr=False)
    active_resume_yaml: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return "ApplicationWorkflowSnapshot(hidden=True)"


_APPLICATION_COLUMN_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("company", "TEXT NOT NULL DEFAULT ''"),
    ("job_title", "TEXT NOT NULL DEFAULT ''"),
    ("job_url", "TEXT NOT NULL DEFAULT ''"),
    ("linkedin_url", "TEXT NOT NULL DEFAULT ''"),
    ("source", "TEXT NOT NULL DEFAULT ''"),
    ("job_description", "TEXT"),
    ("prompt_job_description", "TEXT"),
    ("aro_yaml", "TEXT"),
    ("aro_backup_yaml", "TEXT"),
    ("application_resume_object", "TEXT"),
    ("application_resume_updated_at", "TEXT"),
    ("application_resume_backup_object", "TEXT"),
    ("application_resume_backup_created_at", "TEXT"),
    ("application_resume_backup_target", "TEXT"),
    ("resume_html_filename", "TEXT NOT NULL DEFAULT ''"),
    ("resume_html_content", "TEXT"),
    ("resume_html_mime_type", f"TEXT NOT NULL DEFAULT '{_MIME_HTML}'"),
    ("source_resume_html_path", "TEXT NOT NULL DEFAULT ''"),
    ("resume_html_updated_at", "TEXT"),
    ("resume_filename", "TEXT NOT NULL DEFAULT ''"),
    ("resume_content", "BLOB"),
    ("resume_mime_type", f"TEXT NOT NULL DEFAULT '{_MIME_PDF}'"),
    ("source_resume_path", "TEXT NOT NULL DEFAULT ''"),
    ("resume_updated_at", "TEXT"),
    ("cover_letter_object", "TEXT"),
    ("cover_letter_object_updated_at", "TEXT"),
    ("cover_letter_filename", "TEXT NOT NULL DEFAULT ''"),
    ("cover_letter_content", "BLOB"),
    ("cover_letter_mime_type", f"TEXT NOT NULL DEFAULT '{_MIME_PDF}'"),
    ("source_cover_letter_path", "TEXT NOT NULL DEFAULT ''"),
    ("cover_letter_updated_at", "TEXT"),
    ("date_matched", "TEXT"),
    ("date_posted", "TEXT"),
    ("experience_level", "TEXT"),
    ("ats_score", "INTEGER"),
    ("ats_parsing_score", "INTEGER"),
    ("ats_keyword_score", "INTEGER"),
    ("ats_semantic_score", "INTEGER"),
    ("ats_formatting_risk", "TEXT"),
    ("ats_missing_terms", "TEXT"),
    ("ats_updated_at", "TEXT"),
    ("ats_diagnostics_json", "TEXT"),
    ("evidence_packet_json", "TEXT"),
    ("external_critique_json", "TEXT"),
    ("critique_prompt", "TEXT"),
    ("critique_response", "TEXT"),
    ("critique_json", "TEXT"),
    ("validation_json", "TEXT"),
    ("model_metadata_json", "TEXT"),
    ("selected_variant_created_at", "TEXT"),
    ("selected_variant_updated_at", "TEXT"),
    ("selected_resume_variant", "TEXT NOT NULL DEFAULT ''"),
    ("resume_variant_selection_mode", "TEXT NOT NULL DEFAULT 'auto'"),
    ("applied_to", "TEXT NOT NULL DEFAULT 'No'"),
    ("date_applied", "TEXT"),
    ("notes", "TEXT NOT NULL DEFAULT ''"),
    ("archived_at", "TEXT"),
    ("imported_at", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "TEXT NOT NULL DEFAULT ''"),
)

_VARIANT_COLUMN_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("variant_label", "TEXT NOT NULL DEFAULT ''"),
    ("source", "TEXT NOT NULL DEFAULT ''"),
    ("parent_variant_key", "TEXT"),
    ("application_resume_object", "TEXT NOT NULL DEFAULT ''"),
    ("resume_html_filename", "TEXT NOT NULL DEFAULT ''"),
    ("resume_html_content", "TEXT"),
    ("resume_html_mime_type", f"TEXT NOT NULL DEFAULT '{_MIME_HTML}'"),
    ("source_resume_html_path", "TEXT NOT NULL DEFAULT ''"),
    ("resume_html_updated_at", "TEXT"),
    ("resume_filename", "TEXT NOT NULL DEFAULT ''"),
    ("resume_content", "BLOB"),
    ("resume_mime_type", f"TEXT NOT NULL DEFAULT '{_MIME_PDF}'"),
    ("source_resume_path", "TEXT NOT NULL DEFAULT ''"),
    ("resume_updated_at", "TEXT"),
    ("ats_score", "INTEGER"),
    ("ats_parsing_score", "INTEGER"),
    ("ats_keyword_score", "INTEGER"),
    ("ats_semantic_score", "INTEGER"),
    ("ats_formatting_risk", "TEXT"),
    ("ats_missing_terms", "TEXT"),
    ("ats_updated_at", "TEXT"),
    ("ats_diagnostics_json", "TEXT"),
    ("evidence_packet_json", "TEXT"),
    ("external_critique_json", "TEXT"),
    ("critique_prompt", "TEXT"),
    ("critique_response", "TEXT"),
    ("critique_json", "TEXT"),
    ("validation_json", "TEXT"),
    ("model_metadata_json", "TEXT"),
    ("created_at", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "TEXT NOT NULL DEFAULT ''"),
)

_APPLICATION_WORKFLOW_REVISION_COLUMNS = (
    "job_id",
    "company",
    "job_title",
    "job_url",
    "linkedin_url",
    "source",
    "job_description",
    "prompt_job_description",
    "date_matched",
    "date_posted",
    "experience_level",
    "imported_at",
)

_APPLICATION_RESUME_EDIT_REVISION_COLUMNS = (
    *_APPLICATION_WORKFLOW_REVISION_COLUMNS,
    "aro_yaml",
    "aro_backup_yaml",
    "application_resume_object",
    "application_resume_backup_object",
    "application_resume_backup_target",
    "resume_html_content",
    "resume_content",
    "ats_score",
    "ats_parsing_score",
    "ats_keyword_score",
    "ats_semantic_score",
    "ats_formatting_risk",
    "ats_missing_terms",
    "ats_diagnostics_json",
    "selected_resume_variant",
    "resume_variant_selection_mode",
)

_VARIANT_WORKFLOW_REVISION_COLUMNS = (
    "job_id",
    "variant_key",
    *tuple(name for name, _definition in _VARIANT_COLUMN_DEFINITIONS),
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ApplicationStateStore:
    """Bounded SQLite repository rooted only in caller-supplied paths."""

    __slots__ = (
        "_bound_database_path",
        "_bound_output_dir",
        "_busy_timeout_seconds",
        "_database_path",
        "_failure_injector",
        "_operation_hook",
        "_output_dir",
        "_utc_clock",
    )

    def __init__(
        self,
        paths: WorkspacePaths,
        *,
        utc_clock: Callable[[], datetime] = _utc_now,
        busy_timeout_seconds: float = DEFAULT_BUSY_TIMEOUT_SECONDS,
        failure_injector: Callable[[str], None] | None = None,
        operation_hook: Callable[[str], None] | None = None,
    ) -> None:
        if type(paths) is not WorkspacePaths:
            raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)
        database_path: object = None
        resolution_failed = False
        try:
            database_path = paths.require(WorkspaceMember.DATABASE)
        except Exception:  # noqa: BLE001 - sanitize a configured path boundary.
            resolution_failed = True
        if (
            resolution_failed
            or type(database_path) is not type(Path())
            or not database_path.is_absolute()
        ):
            raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)
        if paths.output_dir is not None and type(paths.output_dir) is not type(Path()):
            raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)
        if type(busy_timeout_seconds) is int:
            if not 1 <= busy_timeout_seconds <= 5:
                raise ApplicationStateValidationError(_VALIDATION_ERROR)
        elif type(busy_timeout_seconds) is float:
            if (
                not math.isfinite(busy_timeout_seconds)
                or not 0.01 <= busy_timeout_seconds <= 5.0
            ):
                raise ApplicationStateValidationError(_VALIDATION_ERROR)
        else:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        timeout = float(busy_timeout_seconds)
        if not callable(utc_clock):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        if failure_injector is not None and not callable(failure_injector):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        if operation_hook is not None and not callable(operation_hook):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)

        self._database_path = database_path
        self._output_dir = paths.output_dir
        self._bound_database_path = _normalized_binding_path(database_path)
        self._bound_output_dir = (
            None
            if paths.output_dir is None or not paths.output_dir.is_absolute()
            else _normalized_binding_path(paths.output_dir)
        )
        self._utc_clock = utc_clock
        self._busy_timeout_seconds = timeout
        self._failure_injector = failure_injector
        self._operation_hook = operation_hook

    @property
    def output_dir(self) -> Path | None:
        """Return the uninspected output-directory snapshot."""

        return self._output_dir

    def assert_workspace_binding(self, paths: WorkspacePaths) -> None:
        """Require exact caller paths without inspecting either filesystem path."""

        if type(paths) is not WorkspacePaths:
            raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)
        database = paths.database
        output = paths.output_dir
        if (
            type(database) is not type(Path())
            or type(output) is not type(Path())
            or not database.is_absolute()
            or not output.is_absolute()
        ):
            raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)
        if (
            _normalized_binding_path(database) != self._bound_database_path
            or _normalized_binding_path(output) != self._bound_output_dir
        ):
            raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)

    def initialize(self) -> None:
        """Explicitly create or conservatively migrate the configured database."""

        parent_error = False
        try:
            if "\x00" in str(self._database_path):
                parent_error = True
            else:
                self._database_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 - sanitize filesystem implementation errors.
            parent_error = True
        if parent_error:
            raise ApplicationStateInitializationError(_INITIALIZATION_ERROR)

        connection: sqlite3.Connection | None = None
        failure: Literal["busy", "initialization"] | None = None
        try:
            connection = self._connect(create=True)
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise _UnsafeSchema
            self._ensure_applications_table(connection)
            self._inject_failure("initialize_after_first_statement")
            self._ensure_variants_table(connection)
            self._ensure_query_outcomes_table(connection)
            self._migrate_legacy_state(connection)
            if not _schema_is_initialized(connection):
                raise _UnsafeSchema
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._inject_failure("initialize_after_user_version")
            connection.commit()
        except _UnsafeSchema:
            failure = "initialization"
            _rollback_quietly(connection)
        except sqlite3.Error as error:
            failure = "busy" if _sqlite_is_busy(error) else "initialization"
            _rollback_quietly(connection)
        except Exception:  # noqa: BLE001 - contain injected migration failures.
            failure = "initialization"
            _rollback_quietly(connection)
        finally:
            _close_quietly(connection)

        if failure == "busy":
            raise ApplicationStateBusyError(_BUSY_ERROR)
        if failure is not None:
            raise ApplicationStateInitializationError(_INITIALIZATION_ERROR)

    def upsert_application(self, metadata: ApplicationMetadata) -> ApplicationRecord:
        """Create or refresh only provider-neutral application metadata."""

        values = _application_metadata_values(metadata)
        now = self._timestamp()
        _materialize_application_metadata(values, now)

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            self._upsert_application_metadata(connection, values, now)
            return self._load_application(connection, values["job_id"])

        return self._write(write)

    def seed_application(
        self,
        metadata: ApplicationMetadata,
        *,
        source_text: str,
        prompt_text: str,
    ) -> ApplicationRecord:
        """Atomically refresh metadata and persist both bounded JOD forms."""

        return self.seed_application_with_outcome(
            metadata,
            source_text=source_text,
            prompt_text=prompt_text,
        ).application

    def seed_application_with_outcome(
        self,
        metadata: ApplicationMetadata,
        *,
        source_text: str,
        prompt_text: str,
    ) -> ApplicationSeedOutcome:
        """Seed one complete application and report its in-transaction existence."""

        values = _application_metadata_values(metadata)
        source = _text(source_text, MAX_JOD_SOURCE_CHARS, required=False)
        prompt = _text(prompt_text, MAX_JOD_PROMPT_CHARS, required=False)
        now = self._timestamp()
        _materialize_application_metadata(values, now)

        def write(connection: sqlite3.Connection) -> ApplicationSeedOutcome:
            self._inject_failure("seed_before_metadata")
            created = self._upsert_application_metadata(connection, values, now)
            self._inject_failure("seed_after_metadata")
            connection.execute(
                """
                UPDATE applications
                SET job_description = ?, prompt_job_description = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (source, prompt, now, values["job_id"]),
            )
            self._inject_failure("seed_after_jod")
            return ApplicationSeedOutcome(
                application=self._load_application(connection, values["job_id"]),
                created=created,
            )

        return self._write(write)

    def get_application(self, job_id: str) -> ApplicationRecord:
        """Fetch one immutable application snapshot."""

        validated = _identifier(job_id)

        def read(connection: sqlite3.Connection) -> ApplicationRecord:
            return self._load_application(
                connection,
                validated,
                checkpoint_after_row=True,
            )

        return self._read(read)

    def get_workflow_snapshot(self, job_id: str) -> ApplicationWorkflowSnapshot:
        """Read one coherent application and all same-job variants."""

        validated = _identifier(job_id)

        def read(connection: sqlite3.Connection) -> ApplicationWorkflowSnapshot:
            application_row = connection.execute(
                "SELECT * FROM applications WHERE job_id = ?",
                (validated,),
            ).fetchone()
            if application_row is None:
                raise _MissingRecord
            self._call_operation_hook("workflow_application_row_fetched")
            variant_rows = self._workflow_variant_rows(connection, validated)
            application = self._application_record(connection, application_row)
            variants = tuple(_variant_record(connection, row) for row in variant_rows)
            revision = ApplicationWorkflowRevision(
                _workflow_revision_digest(application_row, variant_rows)
            )
            return ApplicationWorkflowSnapshot(
                application=application,
                variants=variants,
                revision=revision,
                edit_revision=ApplicationWorkflowRevision(
                    _resume_edit_revision_digest(application_row, variant_rows)
                ),
                active_resume_yaml=_snapshot_active_resume_yaml(
                    application_row, variant_rows
                ),
            )

        return self._read(read)

    def list_applications(
        self,
        scope: ApplicationScope | str = ApplicationScope.ACTIVE,
        *,
        limit: int = MAX_QUERY_RESULTS,
    ) -> tuple[ApplicationRecord, ...]:
        """List applications deterministically using an explicit archive scope."""

        selected_scope = _scope(scope)
        result_limit = _result_limit(limit)
        where = {
            ApplicationScope.ACTIVE: "WHERE archived_at IS NULL",
            ApplicationScope.ARCHIVED: "WHERE archived_at IS NOT NULL",
            ApplicationScope.ALL: "",
        }[selected_scope]

        def read(connection: sqlite3.Connection) -> tuple[ApplicationRecord, ...]:
            rows = connection.execute(
                f"""
                SELECT * FROM applications
                {where}
                ORDER BY job_id COLLATE BINARY ASC
                LIMIT ?
                """,
                (result_limit,),
            ).fetchall()
            return tuple(self._application_record(connection, row) for row in rows)

        return self._read(read)

    def fetch_job_records(
        self,
        job_ids: Sequence[str],
    ) -> tuple[ApplicationRecord, ...]:
        """Fetch existing records in the caller's explicit identifier order."""

        identifiers = _bulk_identifiers(job_ids, allow_empty=True)
        if not identifiers:
            return self._read(lambda _connection: ())

        def read(connection: sqlite3.Connection) -> tuple[ApplicationRecord, ...]:
            placeholders = ", ".join("?" for _ in identifiers)
            rows = connection.execute(
                f"SELECT * FROM applications WHERE job_id IN ({placeholders})",
                identifiers,
            ).fetchall()
            by_id = {str(row["job_id"]): row for row in rows}
            return tuple(
                self._application_record(connection, by_id[job_id])
                for job_id in identifiers
                if job_id in by_id
            )

        return self._read(read)

    def archive(self, job_ids: Sequence[str]) -> int:
        """Archive an exact bounded set atomically."""

        return self._set_archive_state(job_ids, archived=True)

    def unarchive(self, job_ids: Sequence[str]) -> int:
        """Unarchive an exact bounded set atomically."""

        return self._set_archive_state(job_ids, archived=False)

    def delete(self, job_ids: Sequence[str]) -> int:
        """Delete exact applications and children in one transaction."""

        identifiers = _bulk_identifiers(job_ids)

        def write(connection: sqlite3.Connection) -> int:
            self._require_applications(connection, identifiers)
            placeholders = ", ".join("?" for _ in identifiers)
            connection.execute(
                f"""
                DELETE FROM application_resume_variants
                WHERE job_id IN ({placeholders})
                """,
                identifiers,
            )
            cursor = connection.execute(
                f"DELETE FROM applications WHERE job_id IN ({placeholders})",
                identifiers,
            )
            return int(cursor.rowcount)

        return self._write(write)

    def update_application_status(
        self,
        job_id: str,
        *,
        applied_to: str,
        date_applied: str | None = None,
        notes: str | None = None,
    ) -> ApplicationRecord:
        """Update explicit application status fields without touching artifacts."""

        validated_id = _identifier(job_id)
        status = _status(applied_to)
        applied_date = _optional_text(date_applied, 128)
        validated_notes = (
            None if notes is None else _text(notes, MAX_NOTES_CHARS, required=False)
        )
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            self._require_application(connection, validated_id)
            if validated_notes is None:
                connection.execute(
                    """
                    UPDATE applications
                    SET applied_to = ?, date_applied = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, applied_date, now, validated_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE applications
                    SET applied_to = ?, date_applied = ?, notes = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, applied_date, validated_notes, now, validated_id),
                )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def store_jod(
        self,
        job_id: str,
        *,
        source_text: str,
        prompt_text: str,
        ats: AtsFields | None = None,
    ) -> ApplicationRecord:
        """Persist caller-supplied source and prompt-trimmed JOD text."""

        validated_id = _identifier(job_id)
        source = _text(source_text, MAX_JOD_SOURCE_CHARS, required=False)
        prompt = _text(prompt_text, MAX_JOD_PROMPT_CHARS, required=False)
        ats_values = None if ats is None else _ats(ats)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            application = connection.execute(
                "SELECT * FROM applications WHERE job_id = ?", (validated_id,)
            ).fetchone()
            if application is None:
                raise _MissingRecord
            connection.execute(
                """
                UPDATE applications
                SET job_description = ?, prompt_job_description = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (source, prompt, now, validated_id),
            )
            if ats_values is not None:
                selected = _stored_variant_key(application["selected_resume_variant"])
                self._write_ats_projection(
                    connection,
                    validated_id,
                    selected,
                    ats_values,
                    timestamp=now,
                )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def store_ats(self, job_id: str, ats: AtsFields) -> ApplicationRecord:
        """Persist only caller-supplied ATS fields for the active projection."""

        validated_id = _identifier(job_id)
        ats_values = _ats(ats)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            application = connection.execute(
                "SELECT * FROM applications WHERE job_id = ?", (validated_id,)
            ).fetchone()
            if application is None:
                raise _MissingRecord
            selected = _stored_variant_key(application["selected_resume_variant"])
            self._write_ats_projection(
                connection,
                validated_id,
                selected,
                ats_values,
                timestamp=now,
            )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def store_active_resume_if_revision(
        self,
        job_id: str,
        *,
        yaml_text: str,
        resume_html: str,
        resume_pdf: bytes,
        ats: AtsFields | None,
        expected_revision: ApplicationWorkflowRevision,
        backup_current: bool,
    ) -> ApplicationRecord:
        """Atomically render into the exact active target under CAS control."""

        validated_id = _identifier(job_id)
        _validate_yaml_text(yaml_text)
        html = _text(resume_html, MAX_HTML_CHARS, required=True)
        pdf = _optional_bytes(resume_pdf, MAX_PDF_BYTES)
        if pdf is None or not pdf or type(backup_current) is not bool:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        ats_values = None if ats is None else _ats(ats)
        expected = _expected_workflow_revision(expected_revision)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            application = connection.execute(
                "SELECT * FROM applications WHERE job_id = ?", (validated_id,)
            ).fetchone()
            if application is None:
                raise _MissingRecord
            variant_rows = self._workflow_variant_rows(connection, validated_id)
            if _resume_edit_revision_digest(application, variant_rows) != expected:
                raise _RevisionConflict
            target = _active_resume_target(application)
            current_yaml = self._active_resume_yaml(
                connection, validated_id, application, target
            )
            if backup_current:
                connection.execute(
                    """
                    UPDATE applications
                    SET aro_backup_yaml = ?, application_resume_backup_object = ?,
                        application_resume_backup_created_at = ?,
                        application_resume_backup_target = ?
                    WHERE job_id = ?
                    """,
                    (current_yaml, current_yaml, now, target, validated_id),
                )
            self._write_active_resume(
                connection,
                validated_id,
                application,
                target,
                yaml_text=yaml_text,
                resume_html=html,
                resume_pdf=pdf,
                ats_values=ats_values,
                timestamp=now,
            )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def revert_active_resume_if_revision(
        self,
        job_id: str,
        *,
        resume_html: str,
        resume_pdf: bytes,
        ats: AtsFields | None,
        expected_revision: ApplicationWorkflowRevision,
    ) -> ApplicationRecord:
        """Swap the active YAML with its target-bound one-level backup."""

        validated_id = _identifier(job_id)
        html = _text(resume_html, MAX_HTML_CHARS, required=True)
        pdf = _optional_bytes(resume_pdf, MAX_PDF_BYTES)
        if pdf is None or not pdf:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        ats_values = None if ats is None else _ats(ats)
        expected = _expected_workflow_revision(expected_revision)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            application = connection.execute(
                "SELECT * FROM applications WHERE job_id = ?", (validated_id,)
            ).fetchone()
            if application is None:
                raise _MissingRecord
            variant_rows = self._workflow_variant_rows(connection, validated_id)
            if _resume_edit_revision_digest(application, variant_rows) != expected:
                raise _RevisionConflict
            target = _active_resume_target(application)
            backup_target = _stored_backup_target(
                application["application_resume_backup_target"]
            )
            backup_yaml = _stored_optional_text(
                application["aro_backup_yaml"], MAX_ARO_YAML_BYTES
            )
            if backup_target != target or backup_yaml is None:
                raise _InvalidWrite
            _validate_yaml_text(backup_yaml)
            current_yaml = self._active_resume_yaml(
                connection, validated_id, application, target
            )
            connection.execute(
                """
                UPDATE applications
                SET aro_backup_yaml = ?, application_resume_backup_object = ?,
                    application_resume_backup_created_at = ?,
                    application_resume_backup_target = ?
                WHERE job_id = ?
                """,
                (current_yaml, current_yaml, now, target, validated_id),
            )
            self._write_active_resume(
                connection,
                validated_id,
                application,
                target,
                yaml_text=backup_yaml,
                resume_html=html,
                resume_pdf=pdf,
                ats_values=ats_values,
                timestamp=now,
            )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def load_query_outcomes(
        self,
        *,
        limit: int = MAX_QUERY_HISTORY_RESULTS,
    ) -> tuple[StoredQueryOutcome, ...]:
        """Load a bounded newest-first history for deterministic reuse."""

        if type(limit) is not int or not 1 <= limit <= MAX_QUERY_HISTORY_RESULTS:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)

        def read(connection: sqlite3.Connection) -> tuple[StoredQueryOutcome, ...]:
            rows = connection.execute(
                """
                SELECT keywords, location, date_posted, workplace_type,
                       experience_level, job_type, sort_by, limit_value,
                       profile_match, query_score, results_returned,
                       fresh_jobs_accepted, skipped_existing,
                       skipped_blacklisted, skipped_workplace_type,
                       skipped_experience_level, resumes_generated,
                       average_ats_score
                FROM search_query_outcomes
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(_stored_query_outcome(row) for row in rows)

        return self._read(read)

    def record_query_outcomes(self, outcomes: Sequence[QueryOutcomeWrite]) -> int:
        """Persist one bounded run atomically without returning content."""

        if (
            type(outcomes) not in {list, tuple}
            or len(outcomes) > MAX_QUERY_HISTORY_WRITES
        ):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        values = tuple(_query_outcome_values(value) for value in outcomes)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> int:
            connection.executemany(
                """
                INSERT INTO search_query_outcomes (
                    created_at, keywords, location, date_posted, workplace_type,
                    experience_level, job_type, sort_by, limit_value, page,
                    profile_match, query_score, results_returned,
                    fresh_jobs_accepted, skipped_existing, skipped_blacklisted,
                    skipped_workplace_type, skipped_experience_level,
                    resumes_generated, average_ats_score, artifact_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ((now, *value) for value in values),
            )
            connection.execute(
                """
                DELETE FROM search_query_outcomes
                WHERE id NOT IN (
                    SELECT id FROM search_query_outcomes
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (MAX_QUERY_HISTORY_RESULTS,),
            )
            return len(values)

        return self._write(write)

    def store_aro(
        self,
        job_id: str,
        *,
        yaml_text: str,
        backup_yaml_text: str | None = None,
    ) -> Mapping[str, Any]:
        """Persist an inert caller-supplied ARO YAML mapping."""

        validated_id = _identifier(job_id)
        aro = _validate_yaml_text(yaml_text)
        if backup_yaml_text is not None:
            _validate_yaml_text(backup_yaml_text)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> None:
            application = self._require_application(connection, validated_id)
            if _stored_variant_key(application["selected_resume_variant"]) is not None:
                raise _InvalidWrite
            connection.execute(
                """
                UPDATE applications
                SET aro_yaml = ?,
                    aro_backup_yaml = CASE
                        WHEN ? IS NULL THEN aro_backup_yaml ELSE ? END,
                    application_resume_object = ?,
                    application_resume_updated_at = ?,
                    application_resume_backup_object = CASE
                        WHEN ? IS NULL
                        THEN application_resume_backup_object ELSE ? END,
                    application_resume_backup_created_at = CASE
                        WHEN ? IS NULL
                        THEN application_resume_backup_created_at ELSE ? END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    yaml_text,
                    backup_yaml_text,
                    backup_yaml_text,
                    yaml_text,
                    now,
                    backup_yaml_text,
                    backup_yaml_text,
                    backup_yaml_text,
                    now if backup_yaml_text is not None else None,
                    now,
                    validated_id,
                ),
            )

        self._write(write)
        return _freeze_mapping(aro)

    def get_aro(self, job_id: str) -> Mapping[str, Any] | None:
        """Return a defensive immutable application-level ARO mapping."""

        return self.get_application(job_id).application_resume

    def store_clo(
        self,
        job_id: str,
        *,
        value: dict[str, Any],
        pdf_content: bytes | None = None,
        source_path: str = "",
    ) -> Mapping[str, Any]:
        """Persist inert CLO data and optional caller-supplied PDF bytes."""

        validated_id = _identifier(job_id)
        clo_json, normalized = _encode_json_mapping(value)
        pdf = _optional_bytes(pdf_content, MAX_PDF_BYTES)
        provenance = _provenance(source_path)
        filename = (
            artifact_filename(validated_id, ArtifactKind.COVER_LETTER_PDF)
            if pdf is not None
            else ""
        )
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> None:
            self._require_application(connection, validated_id)
            connection.execute(
                """
                UPDATE applications
                SET cover_letter_object = ?,
                    cover_letter_object_updated_at = ?,
                    cover_letter_filename = CASE
                        WHEN ? IS NULL THEN cover_letter_filename ELSE ? END,
                    cover_letter_content = CASE
                        WHEN ? IS NULL THEN cover_letter_content ELSE ? END,
                    cover_letter_mime_type = ?,
                    source_cover_letter_path = CASE
                        WHEN ? IS NULL THEN source_cover_letter_path ELSE ? END,
                    cover_letter_updated_at = CASE
                        WHEN ? IS NULL THEN cover_letter_updated_at ELSE ? END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    clo_json,
                    now,
                    pdf,
                    filename,
                    pdf,
                    pdf,
                    _MIME_PDF,
                    pdf,
                    provenance,
                    pdf,
                    now if pdf is not None else None,
                    now,
                    validated_id,
                ),
            )

        self._write(write)
        return _freeze_mapping(normalized)

    def get_clo(self, job_id: str) -> Mapping[str, Any] | None:
        """Return a defensive immutable CLO mapping."""

        return self.get_application(job_id).cover_letter

    def store_application_artifacts(
        self,
        job_id: str,
        *,
        resume_html: str | None = None,
        resume_pdf: bytes | None = None,
        cover_letter_pdf: bytes | None = None,
        source_resume_html_path: str = "",
        source_resume_path: str = "",
        source_cover_letter_path: str = "",
        ats: AtsFields | None = None,
    ) -> ApplicationRecord:
        """Store caller-produced artifacts and already-calculated ATS fields."""

        validated_id = _identifier(job_id)
        html = _optional_text(resume_html, MAX_HTML_CHARS)
        pdf = _optional_bytes(resume_pdf, MAX_PDF_BYTES)
        cover_pdf = _optional_bytes(cover_letter_pdf, MAX_PDF_BYTES)
        html_source = _provenance(source_resume_html_path)
        pdf_source = _provenance(source_resume_path)
        cover_source = _provenance(source_cover_letter_path)
        ats_values = _ats(ats if ats is not None else AtsFields())
        if html is None and pdf is None and cover_pdf is None and ats is None:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        now = self._timestamp()
        html_filename = (
            artifact_filename(validated_id, ArtifactKind.RESUME_HTML)
            if html is not None
            else ""
        )
        pdf_filename = (
            artifact_filename(validated_id, ArtifactKind.RESUME_PDF)
            if pdf is not None
            else ""
        )
        cover_filename = (
            artifact_filename(validated_id, ArtifactKind.COVER_LETTER_PDF)
            if cover_pdf is not None
            else ""
        )

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            application = self._require_application(connection, validated_id)
            if _stored_variant_key(
                application["selected_resume_variant"]
            ) is not None and (html is not None or pdf is not None or ats is not None):
                raise _InvalidWrite
            assignments: list[str] = []
            parameters: list[Any] = []
            if html is not None:
                assignments.extend(
                    (
                        "resume_html_filename = ?",
                        "resume_html_content = ?",
                        "resume_html_mime_type = ?",
                        "source_resume_html_path = ?",
                        "resume_html_updated_at = ?",
                    )
                )
                parameters.extend((html_filename, html, _MIME_HTML, html_source, now))
            if pdf is not None:
                assignments.extend(
                    (
                        "resume_filename = ?",
                        "resume_content = ?",
                        "resume_mime_type = ?",
                        "source_resume_path = ?",
                        "resume_updated_at = ?",
                    )
                )
                parameters.extend((pdf_filename, pdf, _MIME_PDF, pdf_source, now))
            if cover_pdf is not None:
                assignments.extend(
                    (
                        "cover_letter_filename = ?",
                        "cover_letter_content = ?",
                        "cover_letter_mime_type = ?",
                        "source_cover_letter_path = ?",
                        "cover_letter_updated_at = ?",
                    )
                )
                parameters.extend(
                    (cover_filename, cover_pdf, _MIME_PDF, cover_source, now)
                )
            if ats is not None:
                assignments.extend(
                    (
                        "ats_score = ?",
                        "ats_parsing_score = ?",
                        "ats_keyword_score = ?",
                        "ats_semantic_score = ?",
                        "ats_formatting_risk = ?",
                        "ats_missing_terms = ?",
                        "ats_updated_at = ?",
                        "ats_diagnostics_json = ?",
                    )
                )
                parameters.extend(ats_values)
            assignments.append("updated_at = ?")
            parameters.extend((now, validated_id))
            connection.execute(
                f"""
                UPDATE applications
                SET {", ".join(assignments)}
                WHERE job_id = ?
                """,
                parameters,
            )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def upsert_resume_variant(
        self,
        job_id: str,
        variant: ResumeVariantWrite,
    ) -> ResumeVariantRecord:
        """Insert or update one distinct variant and reconcile selection."""

        validated_id = _identifier(job_id)
        values = _variant_write(variant, validated_id)
        self._call_operation_hook("variant_validated")
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ResumeVariantRecord:
            return self._upsert_resume_variant(
                connection,
                validated_id,
                values,
                timestamp=now,
                advance_existing_timestamp=False,
            )

        return self._write(write)

    def upsert_resume_variant_if_revision(
        self,
        job_id: str,
        variant: ResumeVariantWrite,
        *,
        expected_revision: ApplicationWorkflowRevision,
    ) -> ResumeVariantRecord:
        """Conditionally write a complete variant against one workflow revision."""

        validated_id = _identifier(job_id)
        values = _variant_write(variant, validated_id)
        expected = _expected_workflow_revision(expected_revision)
        self._call_operation_hook("variant_validated")
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ResumeVariantRecord:
            application_row = connection.execute(
                "SELECT * FROM applications WHERE job_id = ?",
                (validated_id,),
            ).fetchone()
            if application_row is None:
                raise _MissingRecord
            variant_rows = self._workflow_variant_rows(connection, validated_id)
            current = _workflow_revision_digest(application_row, variant_rows)
            if current != expected:
                raise _RevisionConflict
            self._call_operation_hook("variant_revision_checked")
            return self._upsert_resume_variant(
                connection,
                validated_id,
                values,
                timestamp=now,
                advance_existing_timestamp=True,
            )

        return self._write(write)

    def get_resume_variant(
        self,
        job_id: str,
        variant_key: ResumeVariantKey | str,
    ) -> ResumeVariantRecord:
        """Fetch one same-application resume variant."""

        validated_id = _identifier(job_id)
        key = _variant_key(variant_key)

        def read(connection: sqlite3.Connection) -> ResumeVariantRecord:
            return self._load_variant(connection, validated_id, key)

        return self._read(read)

    def list_resume_variants(
        self,
        job_id: str,
    ) -> tuple[ResumeVariantRecord, ...]:
        """List the same-job variants in canonical order."""

        validated_id = _identifier(job_id)

        def read(connection: sqlite3.Connection) -> tuple[ResumeVariantRecord, ...]:
            self._require_application(connection, validated_id)
            rows = connection.execute(
                """
                SELECT * FROM application_resume_variants
                WHERE job_id = ?
                ORDER BY CASE variant_key
                    WHEN 'v1' THEN 1 WHEN 'v2' THEN 2 WHEN 'manual' THEN 3
                    ELSE 4 END
                LIMIT 4
                """,
                (validated_id,),
            ).fetchall()
            if len(rows) > len(VARIANT_KEYS):
                raise _CorruptRecord
            return tuple(_variant_record(connection, row) for row in rows)

        return self._read(read)

    def select_resume_variant(
        self,
        job_id: str,
        variant_key: ResumeVariantKey | str,
    ) -> ApplicationRecord:
        """Pin one existing same-job variant as an explicit human selection."""

        validated_id = _identifier(job_id)
        key = _variant_key(variant_key)
        self._call_operation_hook("selection_validated")
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            self._require_application(connection, validated_id)
            row = connection.execute(
                """
                SELECT 1 FROM application_resume_variants
                WHERE job_id = ? AND variant_key = ?
                """,
                (validated_id, key),
            ).fetchone()
            if row is None:
                raise _MissingRecord
            self._project_variant(
                connection,
                validated_id,
                key,
                ResumeSelectionMode.MANUAL,
                touch_updated_at=now,
            )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def reset_resume_variant_selection(self, job_id: str) -> ApplicationRecord:
        """Return one application to canonical automatic preference."""

        validated_id = _identifier(job_id)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> ApplicationRecord:
            self._require_application(connection, validated_id)
            preferred = self._preferred_variant(connection, validated_id)
            self._project_variant(
                connection,
                validated_id,
                preferred,
                ResumeSelectionMode.AUTO,
                touch_updated_at=now,
            )
            return self._load_application(connection, validated_id)

        return self._write(write)

    def _set_archive_state(
        self,
        job_ids: Sequence[str],
        *,
        archived: bool,
    ) -> int:
        identifiers = _bulk_identifiers(job_ids)
        now = self._timestamp()

        def write(connection: sqlite3.Connection) -> int:
            self._require_applications(connection, identifiers)
            placeholders = ", ".join("?" for _ in identifiers)
            target = now if archived else None
            predicate = "archived_at IS NULL" if archived else "archived_at IS NOT NULL"
            cursor = connection.execute(
                f"""
                UPDATE applications
                SET archived_at = ?, updated_at = ?
                WHERE job_id IN ({placeholders}) AND {predicate}
                """,
                (target, now, *identifiers),
            )
            return int(cursor.rowcount)

        return self._write(write)

    def _load_application(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        *,
        checkpoint_after_row: bool = False,
    ) -> ApplicationRecord:
        row = connection.execute(
            "SELECT * FROM applications WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise _MissingRecord
        if checkpoint_after_row:
            self._call_operation_hook("application_row_fetched")
        return self._application_record(connection, row)

    def _load_variant(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        variant_key: str,
    ) -> ResumeVariantRecord:
        row = connection.execute(
            """
            SELECT * FROM application_resume_variants
            WHERE job_id = ? AND variant_key = ?
            """,
            (job_id, variant_key),
        ).fetchone()
        if row is None:
            raise _MissingRecord
        return _variant_record(connection, row)

    def _application_record(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> ApplicationRecord:
        job_id = _stored_identifier(row["job_id"])
        selected_key = _stored_variant_key(row["selected_resume_variant"])
        selected: ResumeVariantRecord | None = None
        if selected_key is not None:
            variant = connection.execute(
                """
                SELECT * FROM application_resume_variants
                WHERE job_id = ? AND variant_key = ?
                """,
                (job_id, selected_key),
            ).fetchone()
            if variant is None:
                raise _CorruptRecord
            selected = _variant_record(connection, variant)
        application_resume = _decode_optional_yaml(row["aro_yaml"])
        backup = _decode_optional_yaml(row["aro_backup_yaml"])
        cover_letter = _decode_optional_structured_mapping(row["cover_letter_object"])
        mode = _stored_mode(row["resume_variant_selection_mode"])
        status = _stored_text(row["applied_to"], MAX_LABEL_CHARS)
        if status not in APPLICATION_STATUSES:
            raise _CorruptRecord
        resume_html = _stored_optional_text(row["resume_html_content"], MAX_HTML_CHARS)
        resume_pdf = _stored_optional_bytes(row["resume_content"], MAX_PDF_BYTES)
        cover_letter_pdf = _stored_optional_bytes(
            row["cover_letter_content"], MAX_PDF_BYTES
        )
        ats = AtsFields(
            score=_stored_score(row["ats_score"]),
            parsing_score=_stored_score(row["ats_parsing_score"]),
            keyword_score=_stored_score(row["ats_keyword_score"]),
            semantic_score=_stored_score(row["ats_semantic_score"]),
            formatting_risk=_stored_optional_text(
                row["ats_formatting_risk"], MAX_LABEL_CHARS
            ),
            missing_terms=_stored_optional_text(
                row["ats_missing_terms"], MAX_METADATA_TEXT_CHARS
            ),
            diagnostics=_decode_optional_json(row["ats_diagnostics_json"]),
            updated_at=_stored_optional_text(row["ats_updated_at"], 128),
        )
        return ApplicationRecord(
            job_id=job_id,
            company=_stored_text(row["company"], MAX_LABEL_CHARS),
            job_title=_stored_text(row["job_title"], MAX_LABEL_CHARS),
            job_url=_stored_job_url(row["job_url"]),
            source=_stored_text(row["source"], MAX_LABEL_CHARS),
            job_description=_stored_optional_text(
                row["job_description"], MAX_JOD_SOURCE_CHARS
            ),
            prompt_job_description=_stored_optional_text(
                row["prompt_job_description"], MAX_JOD_PROMPT_CHARS
            ),
            application_resume=application_resume,
            application_resume_backup=backup,
            application_resume_backup_target=_stored_backup_target(
                row["application_resume_backup_target"]
            ),
            selected_resume_variant=selected_key,
            resume_variant_selection_mode=mode.value,
            selected_variant=selected,
            resume_html_filename=(
                artifact_filename(job_id, ArtifactKind.RESUME_HTML)
                if resume_html is not None
                else ""
            ),
            resume_html=resume_html,
            resume_html_mime_type=_stored_text(
                row["resume_html_mime_type"], MAX_LABEL_CHARS
            ),
            source_resume_html_path=_stored_text(
                row["source_resume_html_path"], MAX_URL_CHARS
            ),
            resume_html_updated_at=_stored_optional_text(
                row["resume_html_updated_at"], 128
            ),
            resume_pdf_filename=(
                artifact_filename(job_id, ArtifactKind.RESUME_PDF)
                if resume_pdf is not None
                else ""
            ),
            resume_pdf=resume_pdf,
            resume_pdf_mime_type=_stored_text(row["resume_mime_type"], MAX_LABEL_CHARS),
            source_resume_path=_stored_text(row["source_resume_path"], MAX_URL_CHARS),
            resume_pdf_updated_at=_stored_optional_text(row["resume_updated_at"], 128),
            ats=ats,
            cover_letter=cover_letter,
            cover_letter_filename=(
                artifact_filename(job_id, ArtifactKind.COVER_LETTER_PDF)
                if cover_letter_pdf is not None
                else ""
            ),
            cover_letter_pdf=cover_letter_pdf,
            cover_letter_mime_type=_stored_text(
                row["cover_letter_mime_type"], MAX_LABEL_CHARS
            ),
            source_cover_letter_path=_stored_text(
                row["source_cover_letter_path"], MAX_URL_CHARS
            ),
            cover_letter_updated_at=_stored_optional_text(
                row["cover_letter_updated_at"], 128
            ),
            date_matched=_stored_optional_text(row["date_matched"], 128),
            date_posted=_stored_optional_text(row["date_posted"], 128),
            experience_level=_stored_optional_text(
                row["experience_level"], MAX_LABEL_CHARS
            ),
            applied_to=status,
            date_applied=_stored_optional_text(row["date_applied"], 128),
            notes=_stored_text(row["notes"], MAX_NOTES_CHARS),
            archived_at=_stored_optional_text(row["archived_at"], 128),
            imported_at=_stored_text(row["imported_at"], 128),
            updated_at=_stored_text(row["updated_at"], 128),
        )

    def _upsert_application_metadata(
        self,
        connection: sqlite3.Connection,
        values: Mapping[str, Any],
        timestamp: str,
    ) -> bool:
        job_id = values["job_id"]
        existing = connection.execute(
            "SELECT date_matched FROM applications WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        effective_date_matched = values["date_matched"]
        if existing is not None:
            stored_date_matched = _stored_optional_text(
                existing["date_matched"],
                128,
            )
            if stored_date_matched is not None and stored_date_matched.strip():
                effective_date_matched = stored_date_matched
        connection.execute(
            """
            INSERT INTO applications (
                job_id, company, job_title, job_url, linkedin_url, source,
                date_matched, date_posted, experience_level,
                selected_resume_variant, resume_variant_selection_mode,
                resume_filename, source_resume_path, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', 'auto', '', '', ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                company = excluded.company,
                job_title = excluded.job_title,
                job_url = excluded.job_url,
                linkedin_url = excluded.linkedin_url,
                source = CASE
                    WHEN excluded.source = '' THEN applications.source
                    ELSE excluded.source
                END,
                date_matched = excluded.date_matched,
                date_posted = COALESCE(
                    excluded.date_posted, applications.date_posted
                ),
                experience_level = COALESCE(
                    excluded.experience_level, applications.experience_level
                ),
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                values["company"],
                values["job_title"],
                values["job_url"],
                values["job_url"],
                values["source"],
                effective_date_matched,
                values["date_posted"],
                values["experience_level"],
                timestamp,
                timestamp,
            ),
        )
        return existing is None

    def _workflow_variant_rows(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            connection.execute(
                """
                SELECT * FROM application_resume_variants
                WHERE job_id = ?
                ORDER BY CASE variant_key
                    WHEN 'v1' THEN 1 WHEN 'v2' THEN 2 WHEN 'manual' THEN 3
                    ELSE 4 END
                LIMIT 4
                """,
                (job_id,),
            ).fetchall()
        )
        if len(rows) > len(VARIANT_KEYS):
            raise _CorruptRecord
        keys = tuple(row["variant_key"] for row in rows)
        if any(type(key) is not str for key in keys):
            raise _CorruptRecord
        expected = tuple(key for key in VARIANT_KEYS if key in keys)
        if keys != expected:
            raise _CorruptRecord
        return rows

    def _upsert_resume_variant(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        values: Mapping[str, Any],
        *,
        timestamp: str,
        advance_existing_timestamp: bool,
    ) -> ResumeVariantRecord:
        application = self._require_application(connection, job_id)
        self._validate_parent(
            connection,
            job_id,
            values["variant_key"],
            values["parent_variant_key"],
        )
        existing = connection.execute(
            """
            SELECT created_at, updated_at
            FROM application_resume_variants
            WHERE job_id = ? AND variant_key = ?
            """,
            (job_id, values["variant_key"]),
        ).fetchone()
        effective_timestamp = timestamp
        if existing is not None and advance_existing_timestamp:
            effective_timestamp = _strictly_advance_timestamp(
                timestamp,
                existing["updated_at"],
            )
        created_at = (
            _stored_text(existing["created_at"], 128)
            if existing is not None
            else effective_timestamp
        )
        connection.execute(
            """
            INSERT INTO application_resume_variants (
                job_id, variant_key, variant_label, source, parent_variant_key,
                application_resume_object,
                resume_html_filename, resume_html_content,
                resume_html_mime_type, source_resume_html_path,
                resume_html_updated_at,
                resume_filename, resume_content, resume_mime_type,
                source_resume_path, resume_updated_at,
                ats_score, ats_parsing_score, ats_keyword_score,
                ats_semantic_score, ats_formatting_risk, ats_missing_terms,
                ats_updated_at, ats_diagnostics_json, evidence_packet_json,
                external_critique_json, critique_prompt, critique_response,
                critique_json, validation_json, model_metadata_json,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                job_id,
                values["variant_key"],
                values["variant_label"],
                values["source"],
                values["parent_variant_key"],
                values["application_resume_object"],
                values["resume_html_filename"],
                values["resume_html_content"],
                _MIME_HTML,
                values["source_resume_html_path"],
                (
                    effective_timestamp
                    if values["resume_html_content"] is not None
                    else None
                ),
                values["resume_filename"],
                values["resume_content"],
                _MIME_PDF,
                values["source_resume_path"],
                (effective_timestamp if values["resume_content"] is not None else None),
                values["ats_score"],
                values["ats_parsing_score"],
                values["ats_keyword_score"],
                values["ats_semantic_score"],
                values["ats_formatting_risk"],
                values["ats_missing_terms"],
                values["ats_updated_at"],
                values["ats_diagnostics_json"],
                values["evidence_packet_json"],
                values["external_critique_json"],
                values["critique_prompt"],
                values["critique_response"],
                values["critique_json"],
                values["validation_json"],
                values["model_metadata_json"],
                created_at,
                effective_timestamp,
            ),
        )
        self._inject_failure("variant_after_upsert")
        mode = _stored_mode(application["resume_variant_selection_mode"])
        selected = _stored_variant_key(application["selected_resume_variant"])
        if mode == ResumeSelectionMode.MANUAL:
            if selected == values["variant_key"]:
                self._project_variant(
                    connection,
                    job_id,
                    selected,
                    ResumeSelectionMode.MANUAL,
                    touch_updated_at=effective_timestamp,
                )
        else:
            preferred = self._preferred_variant(connection, job_id)
            self._project_variant(
                connection,
                job_id,
                preferred,
                ResumeSelectionMode.AUTO,
                touch_updated_at=effective_timestamp,
            )
        return self._load_variant(connection, job_id, values["variant_key"])

    def _validate_parent(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        variant_key: str,
        parent_key: str | None,
    ) -> None:
        if variant_key == "v1":
            if parent_key is not None:
                raise _InvalidWrite
            return
        allowed = ("v1",) if variant_key == "v2" else ("v2", "v1")
        if parent_key not in allowed or parent_key == variant_key:
            raise _InvalidWrite
        row = connection.execute(
            """
            SELECT 1 FROM application_resume_variants
            WHERE job_id = ? AND variant_key = ?
            """,
            (job_id, parent_key),
        ).fetchone()
        if row is None:
            raise _InvalidWrite

    def _preferred_variant(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT variant_key FROM application_resume_variants
            WHERE job_id = ? AND variant_key IN ('manual', 'v2', 'v1')
            ORDER BY CASE variant_key
                WHEN 'manual' THEN 1 WHEN 'v2' THEN 2 ELSE 3 END
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        return None if row is None else str(row["variant_key"])

    def _write_ats_projection(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        selected: str | None,
        ats_values: tuple[Any, ...],
        *,
        timestamp: str,
    ) -> None:
        if selected is not None:
            cursor = connection.execute(
                """
                UPDATE application_resume_variants
                SET ats_score = ?, ats_parsing_score = ?, ats_keyword_score = ?,
                    ats_semantic_score = ?, ats_formatting_risk = ?,
                    ats_missing_terms = ?, ats_updated_at = ?,
                    ats_diagnostics_json = ?, updated_at = ?
                WHERE job_id = ? AND variant_key = ?
                """,
                (*ats_values, timestamp, job_id, selected),
            )
            if cursor.rowcount != 1:
                raise _MissingRecord
        connection.execute(
            """
            UPDATE applications
            SET ats_score = ?, ats_parsing_score = ?, ats_keyword_score = ?,
                ats_semantic_score = ?, ats_formatting_risk = ?,
                ats_missing_terms = ?, ats_updated_at = ?,
                ats_diagnostics_json = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (*ats_values, timestamp, job_id),
        )

    def _active_resume_yaml(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        application: sqlite3.Row,
        target: str,
    ) -> str:
        if target == "fallback":
            value = application["aro_yaml"]
        else:
            row = connection.execute(
                """
                SELECT application_resume_object
                FROM application_resume_variants
                WHERE job_id = ? AND variant_key = ?
                """,
                (job_id, target),
            ).fetchone()
            if row is None:
                raise _MissingRecord
            value = row["application_resume_object"]
        rendered = _stored_text(value, MAX_ARO_YAML_BYTES)
        _validate_yaml_text(rendered)
        return rendered

    def _write_active_resume(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        application: sqlite3.Row,
        target: str,
        *,
        yaml_text: str,
        resume_html: str,
        resume_pdf: bytes,
        ats_values: tuple[Any, ...] | None,
        timestamp: str,
    ) -> None:
        html_filename = artifact_filename(job_id, ArtifactKind.RESUME_HTML)
        pdf_filename = artifact_filename(job_id, ArtifactKind.RESUME_PDF)
        if target == "fallback":
            connection.execute(
                """
                UPDATE applications
                SET aro_yaml = ?, application_resume_object = ?,
                    application_resume_updated_at = ?,
                    resume_html_filename = ?, resume_html_content = ?,
                    resume_html_mime_type = ?, resume_html_updated_at = ?,
                    resume_filename = ?, resume_content = ?, resume_mime_type = ?,
                    resume_updated_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    yaml_text,
                    yaml_text,
                    timestamp,
                    html_filename,
                    resume_html,
                    _MIME_HTML,
                    timestamp,
                    pdf_filename,
                    resume_pdf,
                    _MIME_PDF,
                    timestamp,
                    timestamp,
                    job_id,
                ),
            )
            if ats_values is not None:
                self._write_ats_projection(
                    connection,
                    job_id,
                    None,
                    ats_values,
                    timestamp=timestamp,
                )
            return
        cursor = connection.execute(
            """
            UPDATE application_resume_variants
            SET application_resume_object = ?,
                resume_html_filename = ?, resume_html_content = ?,
                resume_html_mime_type = ?, resume_html_updated_at = ?,
                resume_filename = ?, resume_content = ?, resume_mime_type = ?,
                resume_updated_at = ?, updated_at = ?
            WHERE job_id = ? AND variant_key = ?
            """,
            (
                yaml_text,
                html_filename,
                resume_html,
                _MIME_HTML,
                timestamp,
                pdf_filename,
                resume_pdf,
                _MIME_PDF,
                timestamp,
                timestamp,
                job_id,
                target,
            ),
        )
        if cursor.rowcount != 1:
            raise _MissingRecord
        if ats_values is not None:
            self._write_ats_projection(
                connection,
                job_id,
                target,
                ats_values,
                timestamp=timestamp,
            )
        self._project_variant(
            connection,
            job_id,
            target,
            _stored_mode(application["resume_variant_selection_mode"]),
            touch_updated_at=timestamp,
        )

    def _project_variant(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        variant_key: str | None,
        mode: ResumeSelectionMode,
        *,
        touch_updated_at: str | None,
    ) -> None:
        if variant_key is None:
            connection.execute(
                """
                UPDATE applications
                SET selected_resume_variant = '',
                    resume_variant_selection_mode = ?,
                    selected_variant_created_at = NULL,
                    selected_variant_updated_at = NULL,
                    updated_at = COALESCE(?, updated_at)
                WHERE job_id = ?
                """,
                (mode.value, touch_updated_at, job_id),
            )
            return
        row = connection.execute(
            """
            SELECT * FROM application_resume_variants
            WHERE job_id = ? AND variant_key = ?
            """,
            (job_id, variant_key),
        ).fetchone()
        if row is None:
            raise _MissingRecord
        connection.execute(
            """
            UPDATE applications
            SET selected_resume_variant = ?,
                resume_variant_selection_mode = ?,
                aro_yaml = ?,
                application_resume_object = ?,
                application_resume_updated_at = ?,
                resume_html_filename = ?, resume_html_content = ?,
                resume_html_mime_type = ?, source_resume_html_path = ?,
                resume_html_updated_at = ?,
                resume_filename = ?, resume_content = ?, resume_mime_type = ?,
                source_resume_path = ?, resume_updated_at = ?,
                ats_score = ?, ats_parsing_score = ?, ats_keyword_score = ?,
                ats_semantic_score = ?, ats_formatting_risk = ?,
                ats_missing_terms = ?, ats_updated_at = ?,
                ats_diagnostics_json = ?, evidence_packet_json = ?,
                external_critique_json = ?, critique_prompt = ?,
                critique_response = ?, critique_json = ?, validation_json = ?,
                model_metadata_json = ?,
                selected_variant_created_at = ?,
                selected_variant_updated_at = ?,
                updated_at = COALESCE(?, updated_at)
            WHERE job_id = ?
            """,
            (
                variant_key,
                mode.value,
                row["application_resume_object"],
                row["application_resume_object"],
                row["updated_at"],
                row["resume_html_filename"],
                row["resume_html_content"],
                row["resume_html_mime_type"],
                row["source_resume_html_path"],
                row["resume_html_updated_at"],
                row["resume_filename"],
                row["resume_content"],
                row["resume_mime_type"],
                row["source_resume_path"],
                row["resume_updated_at"],
                row["ats_score"],
                row["ats_parsing_score"],
                row["ats_keyword_score"],
                row["ats_semantic_score"],
                row["ats_formatting_risk"],
                row["ats_missing_terms"],
                row["ats_updated_at"],
                row["ats_diagnostics_json"],
                row["evidence_packet_json"],
                row["external_critique_json"],
                row["critique_prompt"],
                row["critique_response"],
                row["critique_json"],
                row["validation_json"],
                row["model_metadata_json"],
                row["created_at"],
                row["updated_at"],
                touch_updated_at,
                job_id,
            ),
        )

    def _ensure_applications_table(self, connection: sqlite3.Connection) -> None:
        exists = _table_exists(connection, "applications")
        if not exists:
            definitions = ",\n                    ".join(
                f"{name} {definition}"
                for name, definition in _APPLICATION_COLUMN_DEFINITIONS
            )
            connection.execute(
                f"""
                CREATE TABLE applications (
                    job_id TEXT PRIMARY KEY,
                    {definitions}
                )
                """
            )
        else:
            columns = _table_columns(connection, "applications")
            if "job_id" not in columns:
                raise _UnsafeSchema
            for name, definition in _APPLICATION_COLUMN_DEFINITIONS:
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE applications ADD COLUMN {name} {definition}"
                    )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS applications_unique_job_id
            ON applications(job_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS applications_archive_order
            ON applications(archived_at, job_id)
            """
        )

    def _ensure_variants_table(self, connection: sqlite3.Connection) -> None:
        exists = _table_exists(connection, "application_resume_variants")
        if not exists:
            definitions = ",\n                    ".join(
                f"{name} {definition}"
                for name, definition in _VARIANT_COLUMN_DEFINITIONS
            )
            connection.execute(
                f"""
                CREATE TABLE application_resume_variants (
                    job_id TEXT NOT NULL,
                    variant_key TEXT NOT NULL,
                    {definitions},
                    PRIMARY KEY (job_id, variant_key),
                    FOREIGN KEY (job_id) REFERENCES applications(job_id)
                        ON DELETE CASCADE
                )
                """
            )
        else:
            columns = _table_columns(connection, "application_resume_variants")
            if "job_id" not in columns or "variant_key" not in columns:
                raise _UnsafeSchema
            for name, definition in _VARIANT_COLUMN_DEFINITIONS:
                if name not in columns:
                    connection.execute(
                        "ALTER TABLE application_resume_variants "
                        f"ADD COLUMN {name} {definition}"
                    )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS application_resume_variants_identity
            ON application_resume_variants(job_id, variant_key)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS application_resume_variants_variant_key
            ON application_resume_variants(variant_key)
            """
        )

    def _ensure_query_outcomes_table(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS search_query_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                keywords TEXT NOT NULL,
                location TEXT NOT NULL,
                date_posted TEXT NOT NULL,
                workplace_type TEXT,
                experience_level TEXT,
                job_type TEXT,
                sort_by TEXT NOT NULL,
                limit_value INTEGER NOT NULL,
                page INTEGER NOT NULL,
                profile_match REAL NOT NULL,
                query_score REAL NOT NULL,
                results_returned INTEGER NOT NULL,
                fresh_jobs_accepted INTEGER NOT NULL,
                skipped_existing INTEGER NOT NULL DEFAULT 0,
                skipped_blacklisted INTEGER NOT NULL DEFAULT 0,
                skipped_workplace_type INTEGER NOT NULL DEFAULT 0,
                skipped_experience_level INTEGER NOT NULL DEFAULT 0,
                resumes_generated INTEGER NOT NULL DEFAULT 0,
                average_ats_score REAL,
                artifact_mode TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        required = {
            "id",
            "created_at",
            "keywords",
            "location",
            "date_posted",
            "workplace_type",
            "experience_level",
            "job_type",
            "sort_by",
            "limit_value",
            "page",
            "profile_match",
            "query_score",
            "results_returned",
            "fresh_jobs_accepted",
            "skipped_existing",
            "skipped_blacklisted",
            "skipped_workplace_type",
            "skipped_experience_level",
            "resumes_generated",
            "average_ats_score",
            "artifact_mode",
            "updated_at",
        }
        if not required <= _table_columns(connection, "search_query_outcomes"):
            raise _UnsafeSchema
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS search_query_outcomes_keywords_idx
            ON search_query_outcomes(keywords, location, workplace_type)
            """
        )

    def _migrate_legacy_state(self, connection: sqlite3.Connection) -> None:
        columns = _table_columns(connection, "applications")
        invalid_variant_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM application_resume_variants
                WHERE variant_key NOT IN ('v1', 'v2', 'manual')
                """
            ).fetchone()[0]
        )
        if invalid_variant_count:
            raise _UnsafeSchema
        orphan_variant_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM application_resume_variants AS v
                LEFT JOIN applications AS a ON a.job_id = v.job_id
                WHERE a.job_id IS NULL
                """
            ).fetchone()[0]
        )
        if orphan_variant_count:
            raise _UnsafeSchema
        if "linkedin_url" in columns:
            connection.execute(
                """
                UPDATE applications
                SET job_url = linkedin_url
                WHERE (job_url IS NULL OR TRIM(job_url) = '')
                  AND linkedin_url IS NOT NULL
                """
            )
        url_rows = connection.execute(
            """
            SELECT job_id, job_url
            FROM applications
            ORDER BY job_id COLLATE BINARY
            """
        ).fetchall()
        for row in url_rows:
            normalized_url = _normalize_migrated_url(row["job_url"])
            if normalized_url != row["job_url"]:
                connection.execute(
                    "UPDATE applications SET job_url = ? WHERE job_id = ?",
                    (normalized_url, row["job_id"]),
                )
        connection.execute(
            """
            UPDATE applications
            SET date_matched = imported_at
            WHERE (date_matched IS NULL OR TRIM(date_matched) = '')
              AND imported_at IS NOT NULL AND TRIM(imported_at) != ''
            """
        )
        connection.execute(
            """
            UPDATE applications
            SET aro_yaml = application_resume_object
            WHERE aro_yaml IS NULL AND application_resume_object IS NOT NULL
              AND TRIM(application_resume_object) != ''
            """
        )
        connection.execute(
            """
            UPDATE applications
            SET aro_backup_yaml = application_resume_backup_object
            WHERE aro_backup_yaml IS NULL
              AND application_resume_backup_object IS NOT NULL
              AND TRIM(application_resume_backup_object) != ''
            """
        )
        self._backfill_v1_variants(connection)
        self._backfill_manual_variants(connection)
        self._validate_migrated_variant_parents(connection)
        rows = connection.execute(
            """
            SELECT job_id, selected_resume_variant, resume_variant_selection_mode
            FROM applications ORDER BY job_id COLLATE BINARY
            """
        ).fetchall()
        for row in rows:
            job_id = str(row["job_id"])
            mode = _stored_mode_for_migration(row["resume_variant_selection_mode"])
            selected = _stored_variant_key_for_migration(row["selected_resume_variant"])
            if mode == ResumeSelectionMode.MANUAL:
                if selected is None or not self._variant_exists(
                    connection, job_id, selected
                ):
                    raise _UnsafeSchema
                self._project_variant(
                    connection,
                    job_id,
                    selected,
                    mode,
                    touch_updated_at=None,
                )
            else:
                preferred = self._preferred_variant(connection, job_id)
                self._project_variant(
                    connection,
                    job_id,
                    preferred,
                    ResumeSelectionMode.AUTO,
                    touch_updated_at=None,
                )

    def _validate_migrated_variant_parents(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        rows = connection.execute(
            """
            SELECT job_id, variant_key, parent_variant_key
            FROM application_resume_variants
            ORDER BY job_id COLLATE BINARY, variant_key COLLATE BINARY
            """
        ).fetchall()
        for row in rows:
            job_id = str(row["job_id"])
            key = _stored_variant_key_for_migration(row["variant_key"])
            parent = _stored_variant_key_for_migration(row["parent_variant_key"])
            if key is None:
                raise _UnsafeSchema
            if key == "v1":
                if parent is not None:
                    raise _UnsafeSchema
                continue
            allowed = {"v1"} if key == "v2" else {"v1", "v2"}
            if parent not in allowed or not self._variant_exists(
                connection,
                job_id,
                parent,
            ):
                raise _UnsafeSchema

    def _backfill_v1_variants(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT * FROM applications AS a
            WHERE a.application_resume_object IS NOT NULL
              AND TRIM(a.application_resume_object) != ''
              AND NOT EXISTS (
                  SELECT 1 FROM application_resume_variants AS v
                  WHERE v.job_id = a.job_id AND v.variant_key = 'v1'
              )
            ORDER BY a.job_id COLLATE BINARY
            """
        ).fetchall()
        for row in rows:
            _validate_migrated_yaml(row["application_resume_object"])
            timestamp = _legacy_timestamp(row, self._timestamp())
            self._insert_backfill_variant(
                connection,
                row,
                variant_key="v1",
                label="Draft v1",
                source="legacy_first_draft",
                parent=None,
                timestamp=timestamp,
            )

    def _backfill_manual_variants(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT * FROM applications AS a
            WHERE TRIM(COALESCE(a.selected_resume_variant, '')) = 'manual'
              AND a.application_resume_object IS NOT NULL
              AND TRIM(a.application_resume_object) != ''
              AND a.resume_content IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM application_resume_variants AS v
                  WHERE v.job_id = a.job_id AND v.variant_key = 'manual'
              )
            ORDER BY a.job_id COLLATE BINARY
            """
        ).fetchall()
        for row in rows:
            _validate_migrated_yaml(row["application_resume_object"])
            content = row["resume_content"]
            if type(content) is not bytes or len(content) > MAX_PDF_BYTES:
                raise _UnsafeSchema
            job_id = str(row["job_id"])
            if self._variant_exists(connection, job_id, "v2"):
                parent = "v2"
            elif self._variant_exists(connection, job_id, "v1"):
                parent = "v1"
            else:
                raise _UnsafeSchema
            timestamp = _legacy_timestamp(row, self._timestamp())
            self._insert_backfill_variant(
                connection,
                row,
                variant_key="manual",
                label="Manual pass",
                source="legacy_manual_pass",
                parent=parent,
                timestamp=timestamp,
            )

    def _insert_backfill_variant(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        variant_key: str,
        label: str,
        source: str,
        parent: str | None,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO application_resume_variants (
                job_id, variant_key, variant_label, source, parent_variant_key,
                application_resume_object,
                resume_html_filename, resume_html_content,
                resume_html_mime_type, source_resume_html_path,
                resume_html_updated_at,
                resume_filename, resume_content, resume_mime_type,
                source_resume_path, resume_updated_at,
                ats_score, ats_parsing_score, ats_keyword_score,
                ats_semantic_score, ats_formatting_risk, ats_missing_terms,
                ats_updated_at, ats_diagnostics_json, evidence_packet_json,
                external_critique_json, critique_prompt, critique_response,
                critique_json, validation_json, model_metadata_json,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                row["job_id"],
                variant_key,
                label,
                source,
                parent,
                row["application_resume_object"],
                row["resume_html_filename"],
                row["resume_html_content"],
                row["resume_html_mime_type"],
                row["source_resume_html_path"],
                row["resume_html_updated_at"],
                row["resume_filename"],
                row["resume_content"],
                row["resume_mime_type"],
                row["source_resume_path"],
                row["resume_updated_at"],
                row["ats_score"],
                row["ats_parsing_score"],
                row["ats_keyword_score"],
                row["ats_semantic_score"],
                row["ats_formatting_risk"],
                row["ats_missing_terms"],
                row["ats_updated_at"],
                row["ats_diagnostics_json"],
                row["evidence_packet_json"],
                row["external_critique_json"],
                row["critique_prompt"],
                row["critique_response"],
                row["critique_json"],
                row["validation_json"],
                row["model_metadata_json"],
                timestamp,
                timestamp,
            ),
        )

    def _variant_exists(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        variant_key: str,
    ) -> bool:
        return (
            connection.execute(
                """
                SELECT 1 FROM application_resume_variants
                WHERE job_id = ? AND variant_key = ?
                """,
                (job_id, variant_key),
            ).fetchone()
            is not None
        )

    def _require_application(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT job_id, selected_resume_variant, resume_variant_selection_mode
            FROM applications WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise _MissingRecord
        return row

    def _require_applications(
        self,
        connection: sqlite3.Connection,
        job_ids: tuple[str, ...],
    ) -> None:
        placeholders = ", ".join("?" for _ in job_ids)
        count = int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM applications
                WHERE job_id IN ({placeholders})
                """,
                job_ids,
            ).fetchone()[0]
        )
        if count != len(job_ids):
            raise _MissingRecord

    def _connect(self, *, create: bool) -> sqlite3.Connection:
        target: str
        if create:
            target = str(self._database_path)
            connection = sqlite3.connect(
                target,
                timeout=self._busy_timeout_seconds,
                isolation_level=None,
            )
        else:
            encoded = quote(str(self._database_path), safe="/")
            connection = sqlite3.connect(
                f"file:{encoded}?mode=rw",
                uri=True,
                timeout=self._busy_timeout_seconds,
                isolation_level=None,
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"PRAGMA busy_timeout = {int(self._busy_timeout_seconds * 1_000)}"
        )
        return connection

    def _open_initialized(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        failure: Literal["missing", "busy", "operation"] | None = None
        try:
            connection = self._connect(create=False)
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != SCHEMA_VERSION or not _schema_is_initialized(connection):
                failure = "missing"
        except sqlite3.Error as error:
            failure = "busy" if _sqlite_is_busy(error) else "missing"
        except Exception:  # noqa: BLE001 - normalize connection/schema failures.
            failure = "operation"
        if failure is not None:
            _close_quietly(connection)
            if failure == "busy":
                raise ApplicationStateBusyError(_BUSY_ERROR)
            if failure == "operation":
                raise ApplicationStateError(_OPERATION_ERROR)
            raise ApplicationStateNotInitializedError(_NOT_INITIALIZED_ERROR)
        if connection is None:
            raise ApplicationStateNotInitializedError(_NOT_INITIALIZED_ERROR)
        return connection

    def _assert_initialized(self) -> None:
        connection = self._open_initialized()
        _close_quietly(connection)

    def _read(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        connection: sqlite3.Connection | None = None
        result: Any = None
        read_completed = False
        failure: (
            Literal[
                "busy",
                "missing",
                "missing_record",
                "corruption",
                "validation",
                "operation",
            ]
            | None
        ) = None
        try:
            connection = self._open_initialized()
            connection.execute("BEGIN")
            result = operation(connection)
            connection.commit()
            read_completed = True
        except ApplicationStateNotInitializedError:
            failure = "missing"
        except ApplicationStateBusyError:
            failure = "busy"
        except _MissingRecord:
            failure = "missing_record"
        except _CorruptRecord:
            failure = "corruption"
        except sqlite3.Error as error:
            failure = "busy" if _sqlite_is_busy(error) else "operation"
        except ApplicationStateCorruptionError:
            failure = "corruption"
        except Exception:  # noqa: BLE001 - normalize injected read failures.
            failure = "operation"
        finally:
            if not read_completed:
                _rollback_quietly(connection)
            _close_quietly(connection)
        if failure == "busy":
            raise ApplicationStateBusyError(_BUSY_ERROR)
        if failure == "missing":
            raise ApplicationStateNotInitializedError(_NOT_INITIALIZED_ERROR)
        if failure == "missing_record":
            raise ApplicationStateNotFoundError(_NOT_FOUND_ERROR)
        if failure == "corruption":
            raise ApplicationStateCorruptionError(_CORRUPTION_ERROR)
        if failure is not None:
            raise ApplicationStateError(_OPERATION_ERROR)
        return result

    def _write(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        connection: sqlite3.Connection | None = None
        result: Any = None
        failure: (
            Literal[
                "busy",
                "not_initialized",
                "missing",
                "validation",
                "conflict",
                "corruption",
                "operation",
            ]
            | None
        ) = None
        try:
            connection = self._open_initialized()
            connection.execute("BEGIN IMMEDIATE")
            result = operation(connection)
            connection.commit()
        except ApplicationStateNotInitializedError:
            failure = "not_initialized"
            _rollback_quietly(connection)
        except ApplicationStateBusyError:
            failure = "busy"
            _rollback_quietly(connection)
        except _MissingRecord:
            failure = "missing"
            _rollback_quietly(connection)
        except _InvalidWrite:
            failure = "validation"
            _rollback_quietly(connection)
        except _RevisionConflict:
            failure = "conflict"
            _rollback_quietly(connection)
        except _CorruptRecord:
            failure = "corruption"
            _rollback_quietly(connection)
        except sqlite3.Error as error:
            failure = "busy" if _sqlite_is_busy(error) else "operation"
            _rollback_quietly(connection)
        except Exception:  # noqa: BLE001 - normalize injected write failures.
            failure = "operation"
            _rollback_quietly(connection)
        finally:
            _close_quietly(connection)
        if failure == "busy":
            raise ApplicationStateBusyError(_BUSY_ERROR)
        if failure == "not_initialized":
            raise ApplicationStateNotInitializedError(_NOT_INITIALIZED_ERROR)
        if failure == "missing":
            raise ApplicationStateNotFoundError(_NOT_FOUND_ERROR)
        if failure == "validation":
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        if failure == "conflict":
            raise ApplicationStateConflictError(_CONFLICT_ERROR)
        if failure == "corruption":
            raise ApplicationStateCorruptionError(_CORRUPTION_ERROR)
        if failure is not None:
            raise ApplicationStateError(_OPERATION_ERROR)
        return result

    def _timestamp(self) -> str:
        rendered: str | None = None
        failed = False
        try:
            value = self._utc_clock()
            if type(value) is not datetime or value.tzinfo is None:
                failed = True
            else:
                offset = value.utcoffset()
                if type(offset) is not timedelta or offset != timedelta(0):
                    failed = True
                else:
                    rendered = value.astimezone(UTC).isoformat(timespec="microseconds")
        except Exception:  # noqa: BLE001 - contain hostile clock implementations.
            failed = True
        if failed or rendered is None:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        return rendered

    def _inject_failure(self, checkpoint: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(checkpoint)

    def _call_operation_hook(self, checkpoint: str) -> None:
        failed = False
        if self._operation_hook is not None:
            try:
                self._operation_hook(checkpoint)
            except Exception:  # noqa: BLE001 - contain an injected test hook.
                failed = True
        if failed:
            raise ApplicationStateError(_OPERATION_ERROR)


def artifact_filename(
    job_id: str,
    artifact_kind: ArtifactKind | str,
) -> str:
    """Return deterministic, generic, traversal-safe artifact metadata."""

    validated_id = _identifier(job_id)
    kind: ArtifactKind | None = None
    failed = False
    if type(artifact_kind) is ArtifactKind:
        kind = artifact_kind
    elif type(artifact_kind) is str:
        try:
            kind = ArtifactKind(artifact_kind)
        except ValueError:
            failed = True
    else:
        failed = True
    if failed or kind is None:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    digest = hashlib.sha256(validated_id.encode("ascii")).hexdigest()[:24]
    suffix = {
        ArtifactKind.RESUME_HTML: "resume.html",
        ArtifactKind.RESUME_PDF: "resume.pdf",
        ArtifactKind.COVER_LETTER_PDF: "cover-letter.pdf",
    }[kind]
    return f"artifact-{digest}.{suffix}"


class _UnsafeSchema(Exception):
    __slots__ = ()


class _MissingRecord(Exception):
    __slots__ = ()


class _InvalidWrite(Exception):
    __slots__ = ()


class _RevisionConflict(Exception):
    __slots__ = ()


class _CorruptRecord(Exception):
    __slots__ = ()


def _normalized_binding_path(value: Path) -> Path:
    if type(value) is not type(Path()) or not value.is_absolute():
        raise ApplicationStateConfigurationError(_CONFIGURATION_ERROR)
    return Path(os.path.normpath(str(value)))


def _application_metadata_values(metadata: object) -> dict[str, Any]:
    if type(metadata) is not ApplicationMetadata:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    source = _label(metadata.source, required=False)
    date_matched = _optional_text(metadata.date_matched, 128)
    date_posted = _optional_text(metadata.date_posted, 128)
    experience_level = _optional_text(metadata.experience_level, MAX_LABEL_CHARS)
    return {
        "job_id": _identifier(metadata.job_id),
        "company": _label(metadata.company, required=True),
        "job_title": _label(metadata.job_title, required=True),
        "job_url": _url(metadata.job_url),
        "source": source if source.strip() else "",
        "date_matched": (
            date_matched if date_matched is not None and date_matched.strip() else None
        ),
        "date_posted": (
            date_posted if date_posted is not None and date_posted.strip() else None
        ),
        "experience_level": (
            experience_level
            if experience_level is not None and experience_level.strip()
            else None
        ),
    }


def _materialize_application_metadata(
    values: dict[str, Any],
    timestamp: str,
) -> None:
    if values["date_matched"] is None:
        values["date_matched"] = timestamp


def _expected_workflow_revision(value: object) -> bytes:
    if type(value) is not ApplicationWorkflowRevision:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    missing = False
    raw: object = None
    try:
        raw = object.__getattribute__(value, "_value")
    except (AttributeError, TypeError):
        missing = True
    if missing:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    if type(raw) is not bytes or len(raw) != hashlib.sha256().digest_size:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return raw


def workflow_revision_token(value: ApplicationWorkflowRevision) -> str:
    """Encode one opaque revision for a same-origin form round trip."""

    return _expected_workflow_revision(value).hex()


def workflow_revision_from_token(value: object) -> ApplicationWorkflowRevision:
    """Decode an exact public revision token without exposing state content."""

    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        raise ApplicationStateValidationError(_VALIDATION_ERROR) from None
    return ApplicationWorkflowRevision(raw)


def _workflow_revision_digest(
    application_row: sqlite3.Row,
    variant_rows: tuple[sqlite3.Row, ...],
) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"career-agent-workbench:application-workflow-revision:v1")
    for column in _APPLICATION_WORKFLOW_REVISION_COLUMNS:
        _update_workflow_digest(digest, column, application_row[column])
    digest.update(len(variant_rows).to_bytes(1, "big"))
    for row in variant_rows:
        digest.update(b"variant")
        for column in _VARIANT_WORKFLOW_REVISION_COLUMNS:
            _update_workflow_digest(digest, column, row[column])
    return digest.digest()


def _resume_edit_revision_digest(
    application_row: sqlite3.Row,
    variant_rows: tuple[sqlite3.Row, ...],
) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"career-agent-workbench:resume-edit-revision:v1")
    for column in _APPLICATION_RESUME_EDIT_REVISION_COLUMNS:
        _update_workflow_digest(digest, column, application_row[column])
    digest.update(len(variant_rows).to_bytes(1, "big"))
    for row in variant_rows:
        digest.update(b"variant")
        for column in _VARIANT_WORKFLOW_REVISION_COLUMNS:
            _update_workflow_digest(digest, column, row[column])
    return digest.digest()


def _snapshot_active_resume_yaml(
    application_row: sqlite3.Row,
    variant_rows: tuple[sqlite3.Row, ...],
) -> str | None:
    selected = _stored_variant_key(application_row["selected_resume_variant"])
    value: object = application_row["aro_yaml"]
    if selected is not None:
        matching = tuple(row for row in variant_rows if row["variant_key"] == selected)
        if len(matching) != 1:
            raise _CorruptRecord
        value = matching[0]["application_resume_object"]
    if value is None:
        return None
    rendered = _stored_text(value, MAX_ARO_YAML_BYTES)
    _validate_yaml_text(rendered)
    return rendered


def _update_workflow_digest(
    digest: Any,
    column: str,
    value: object,
) -> None:
    encoded_column = column.encode("ascii")
    digest.update(len(encoded_column).to_bytes(2, "big"))
    digest.update(encoded_column)
    if value is None:
        tag = b"n"
        payload = b""
    elif type(value) is str:
        tag = b"s"
        try:
            payload = value.encode("utf-8")
        except UnicodeError:
            raise _CorruptRecord from None
    elif type(value) is int:
        tag = b"i"
        payload = str(value).encode("ascii")
    elif type(value) is bytes:
        tag = b"b"
        payload = value
    else:
        raise _CorruptRecord
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _strictly_advance_timestamp(candidate: str, previous: object) -> str:
    prior_text = _stored_text(previous, 128)
    try:
        candidate_time = datetime.fromisoformat(candidate)
        prior_time = datetime.fromisoformat(prior_text)
        if (
            type(candidate_time) is not datetime
            or type(prior_time) is not datetime
            or candidate_time.tzinfo is None
            or prior_time.tzinfo is None
            or candidate_time.utcoffset() != timedelta(0)
            or prior_time.utcoffset() != timedelta(0)
        ):
            raise ValueError
        if candidate_time <= prior_time:
            candidate_time = prior_time + timedelta(microseconds=1)
    except (OverflowError, TypeError, ValueError):
        raise _CorruptRecord from None
    return candidate_time.astimezone(UTC).isoformat(timespec="microseconds")


def _identifier(value: object) -> str:
    if type(value) is not str or not _IDENTIFIER_RE.fullmatch(value):
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    if value in {".", ".."}:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return value


def _label(value: object, *, required: bool) -> str:
    return _text(value, MAX_LABEL_CHARS, required=required)


def _text(value: object, limit: int, *, required: bool) -> str:
    if type(value) is not str or len(value) > limit:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    if required and not value.strip():
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    if "\x00" in value:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return value


def _optional_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return _text(value, limit, required=False)


def _provenance(value: object) -> str:
    if type(value) is not str:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return _text(value, MAX_URL_CHARS, required=False)


def _url(value: object) -> str:
    rendered = _text(value, MAX_URL_CHARS, required=True)
    normalized: str | None = None
    failed = False
    try:
        normalized = normalize_job_url(rendered)
    except ProviderError:
        failed = True
    if failed or normalized is None or len(normalized) > MAX_URL_CHARS:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return normalized


def _normalize_migrated_url(value: object) -> str:
    failed = False
    normalized: str | None = None
    if type(value) is not str or len(value) > MAX_URL_CHARS:
        failed = True
    else:
        try:
            normalized = normalize_job_url(value)
        except ProviderError:
            failed = True
    if failed or normalized is None or len(normalized) > MAX_URL_CHARS:
        raise _UnsafeSchema
    return normalized


def _status(value: object) -> str:
    if type(value) is not str or value not in APPLICATION_STATUSES:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return value


def _variant_key(value: object) -> str:
    rendered = value.value if isinstance(value, ResumeVariantKey) else value
    if type(rendered) is not str or rendered not in VARIANT_KEYS:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return rendered


def _scope(value: object) -> ApplicationScope:
    selected: ApplicationScope | None = None
    failed = False
    if type(value) is ApplicationScope:
        selected = value
    elif type(value) is str:
        try:
            selected = ApplicationScope(value)
        except ValueError:
            failed = True
    else:
        failed = True
    if failed or selected is None:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return selected


def _result_limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_QUERY_RESULTS:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return value


def _bounded_counter(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_QUERY_RESULTS:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return value


def _unit_score(value: object) -> float:
    if type(value) not in {int, float}:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return rendered


def _query_outcome_values(value: object) -> tuple[Any, ...]:
    if type(value) is not QueryOutcomeWrite:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return (
        _text(value.keywords, MAX_LABEL_CHARS, required=True),
        _text(value.location, MAX_LABEL_CHARS, required=False),
        _text(value.date_posted, 64, required=True),
        _optional_text(value.workplace_type, 64),
        _optional_text(value.experience_level, 64),
        _optional_text(value.job_type, 64),
        _text(value.sort_by, 64, required=True),
        _result_limit(value.limit),
        _bounded_counter(value.page),
        _unit_score(value.profile_match),
        _unit_score(value.query_score),
        _bounded_counter(value.results_returned),
        _bounded_counter(value.fresh_jobs_accepted),
        _bounded_counter(value.skipped_existing),
        _bounded_counter(value.skipped_blacklisted),
        _bounded_counter(value.skipped_workplace_type),
        _bounded_counter(value.skipped_experience_level),
        0,
        None,
        "application-seed",
    )


def _bulk_identifiers(
    values: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(values) not in {list, tuple}:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    if len(values) > MAX_BULK_IDENTIFIERS or (not values and not allow_empty):
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    result = tuple(_identifier(value) for value in values)
    if len(set(result)) != len(result):
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return result


def _optional_bytes(value: object, limit: int) -> bytes | None:
    if value is None:
        return None
    if type(value) is not bytes or len(value) > limit:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return value


def _ats(value: object) -> tuple[Any, ...]:
    if type(value) is not AtsFields:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    scores = (
        value.score,
        value.parsing_score,
        value.keyword_score,
        value.semantic_score,
    )
    for score in scores:
        if score is not None and (type(score) is not int or not 0 <= score <= 100):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
    formatting = _optional_text(value.formatting_risk, MAX_LABEL_CHARS)
    missing = _optional_text(value.missing_terms, MAX_METADATA_TEXT_CHARS)
    updated = _optional_text(value.updated_at, 128)
    diagnostics_json = (
        None
        if value.diagnostics is None
        else _encode_json_mapping(value.diagnostics)[0]
    )
    return (
        *scores,
        formatting,
        missing,
        updated,
        diagnostics_json,
    )


def _variant_write(
    value: object,
    job_id: str,
) -> dict[str, Any]:
    if type(value) is not ResumeVariantWrite:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    key = _variant_key(value.variant_key)
    parent = (
        None
        if value.parent_variant_key is None
        else _variant_key(value.parent_variant_key)
    )
    label = _label(value.variant_label, required=True)
    source = _label(value.source, required=True)
    _validate_yaml_text(value.application_resume_yaml)
    html = _optional_text(value.resume_html, MAX_HTML_CHARS)
    pdf = _optional_bytes(value.resume_pdf, MAX_PDF_BYTES)
    ats_values = _ats(value.ats)
    diagnostics = value.ats_diagnostics
    if diagnostics is None and value.ats.diagnostics is not None:
        diagnostics = value.ats.diagnostics
    return {
        "variant_key": key,
        "variant_label": label,
        "source": source,
        "parent_variant_key": parent,
        "application_resume_object": value.application_resume_yaml,
        "resume_html_filename": (
            artifact_filename(job_id, ArtifactKind.RESUME_HTML)
            if html is not None
            else ""
        ),
        "resume_html_content": html,
        "source_resume_html_path": _provenance(value.source_resume_html_path),
        "resume_filename": (
            artifact_filename(job_id, ArtifactKind.RESUME_PDF)
            if pdf is not None
            else ""
        ),
        "resume_content": pdf,
        "source_resume_path": _provenance(value.source_resume_path),
        "ats_score": ats_values[0],
        "ats_parsing_score": ats_values[1],
        "ats_keyword_score": ats_values[2],
        "ats_semantic_score": ats_values[3],
        "ats_formatting_risk": ats_values[4],
        "ats_missing_terms": ats_values[5],
        "ats_updated_at": ats_values[6],
        "ats_diagnostics_json": (
            None if diagnostics is None else _encode_json_mapping(diagnostics)[0]
        ),
        "evidence_packet_json": _optional_json_mapping(value.evidence_packet),
        "external_critique_json": _optional_json_mapping(value.external_critique),
        "critique_prompt": _optional_text(
            value.critique_prompt, MAX_METADATA_TEXT_CHARS
        ),
        "critique_response": _optional_text(
            value.critique_response, MAX_METADATA_TEXT_CHARS
        ),
        "critique_json": _optional_json_mapping(value.critique),
        "validation_json": _optional_json_mapping(value.validation),
        "model_metadata_json": _optional_json_mapping(value.model_metadata),
    }


def _optional_json_mapping(value: object) -> str | None:
    if value is None:
        return None
    return _encode_json_mapping(value)[0]


def _encode_json_mapping(value: object) -> tuple[str, dict[str, Any]]:
    if type(value) is not dict:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    normalized, nodes, characters = _validate_json_tree(
        value,
        depth=0,
        active=set(),
    )
    if nodes > MAX_JSON_NODES or characters > MAX_JSON_CHARS:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    serialization: object = None
    failed = False
    try:
        serialization = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        failed = True
    if failed or type(serialization) is not str or len(serialization) > MAX_JSON_CHARS:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return serialization, normalized


def _validate_json_tree(
    value: object,
    *,
    depth: int,
    active: set[int],
) -> tuple[Any, int, int]:
    if depth > MAX_JSON_DEPTH:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    if value is None or type(value) is bool:
        return value, 1, 0
    if type(value) is int:
        rendered_integer = ""
        failed = value.bit_length() > MAX_JSON_CHARS * 4
        if not failed:
            try:
                rendered_integer = str(value)
            except ValueError:
                failed = True
        if failed or len(rendered_integer) > MAX_JSON_CHARS:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        return value, 1, len(rendered_integer)
    if type(value) is float:
        if not math.isfinite(value):
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        return value, 1, len(repr(value))
    if type(value) is str:
        if len(value) > MAX_JSON_CHARS or "\x00" in value:
            raise ApplicationStateValidationError(_VALIDATION_ERROR)
        return value, 1, len(value)
    if type(value) not in {list, dict}:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    identity = id(value)
    if identity in active:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    active.add(identity)
    try:
        if type(value) is list:
            result_list: list[Any] = []
            nodes = 1
            characters = 0
            for item in value:
                normalized, item_nodes, item_chars = _validate_json_tree(
                    item,
                    depth=depth + 1,
                    active=active,
                )
                nodes += item_nodes
                characters += item_chars
                if nodes > MAX_JSON_NODES or characters > MAX_JSON_CHARS:
                    raise ApplicationStateValidationError(_VALIDATION_ERROR)
                result_list.append(normalized)
            return result_list, nodes, characters
        result_dict: dict[str, Any] = {}
        nodes = 1
        characters = 0
        for key, item in value.items():
            if type(key) is not str or "\x00" in key:
                raise ApplicationStateValidationError(_VALIDATION_ERROR)
            normalized, item_nodes, item_chars = _validate_json_tree(
                item,
                depth=depth + 1,
                active=active,
            )
            nodes += item_nodes
            characters += len(key) + item_chars
            if nodes > MAX_JSON_NODES or characters > MAX_JSON_CHARS:
                raise ApplicationStateValidationError(_VALIDATION_ERROR)
            result_dict[key] = normalized
        return result_dict, nodes, characters
    finally:
        active.remove(identity)


def _validate_yaml_text(value: object) -> dict[str, Any]:
    if type(value) is not str:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    encoding_failed = False
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        encoding_failed = True
        size = MAX_ARO_YAML_BYTES + 1
    if encoding_failed or size > MAX_ARO_YAML_BYTES:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    parsed: object = None
    parse_failed = False
    try:
        parsed = yaml.safe_load(value)
    except (yaml.YAMLError, ValueError, OverflowError, RecursionError):
        parse_failed = True
    if parse_failed or type(parsed) is not dict:
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    normalized, nodes, characters = _validate_json_tree(
        parsed,
        depth=0,
        active=set(),
    )
    if (
        type(normalized) is not dict
        or nodes > MAX_JSON_NODES
        or characters > MAX_JSON_CHARS
    ):
        raise ApplicationStateValidationError(_VALIDATION_ERROR)
    return normalized


def _validate_migrated_yaml(value: object) -> None:
    failed = False
    try:
        _validate_yaml_text(value)
    except ApplicationStateValidationError:
        failed = True
    if failed:
        raise _UnsafeSchema


def _decode_optional_yaml(value: object) -> Mapping[str, Any] | None:
    if value is None or value == "":
        return None
    if type(value) is not str:
        raise _CorruptRecord
    parsed: object = None
    failed = False
    try:
        if len(value.encode("utf-8")) > MAX_ARO_YAML_BYTES:
            failed = True
        else:
            parsed = yaml.safe_load(value)
    except (UnicodeError, yaml.YAMLError, ValueError, OverflowError, RecursionError):
        failed = True
    if failed or type(parsed) is not dict:
        raise _CorruptRecord
    try:
        normalized, nodes, characters = _validate_json_tree(
            parsed,
            depth=0,
            active=set(),
        )
    except ApplicationStateValidationError:
        raise _CorruptRecord from None
    if (
        type(normalized) is not dict
        or nodes > MAX_JSON_NODES
        or characters > MAX_JSON_CHARS
    ):
        raise _CorruptRecord
    return _freeze_mapping(normalized)


def _decode_optional_json(value: object) -> Mapping[str, Any] | None:
    if value is None or value == "":
        return None
    if type(value) is not str or len(value) > MAX_JSON_CHARS:
        raise _CorruptRecord
    parsed: object = None
    failed = False
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        failed = True
    if failed or type(parsed) is not dict:
        raise _CorruptRecord
    try:
        normalized, nodes, characters = _validate_json_tree(
            parsed,
            depth=0,
            active=set(),
        )
    except ApplicationStateValidationError:
        raise _CorruptRecord from None
    if (
        type(normalized) is not dict
        or nodes > MAX_JSON_NODES
        or characters > MAX_JSON_CHARS
    ):
        raise _CorruptRecord
    return _freeze_mapping(normalized)


def _decode_optional_structured_mapping(
    value: object,
) -> Mapping[str, Any] | None:
    """Decode canonical JSON or the bounded legacy YAML mapping form."""

    if value is None or value == "":
        return None
    if type(value) is not str or len(value) > MAX_JSON_CHARS:
        raise _CorruptRecord
    parsed: object = None
    json_failed = False
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        json_failed = True
    if json_failed:
        yaml_failed = False
        try:
            if len(value.encode("utf-8")) > MAX_ARO_YAML_BYTES:
                yaml_failed = True
            else:
                parsed = yaml.safe_load(value)
        except (
            UnicodeError,
            yaml.YAMLError,
            ValueError,
            OverflowError,
            RecursionError,
        ):
            yaml_failed = True
        if yaml_failed:
            raise _CorruptRecord
    if type(parsed) is not dict:
        raise _CorruptRecord
    try:
        normalized, nodes, characters = _validate_json_tree(
            parsed,
            depth=0,
            active=set(),
        )
    except ApplicationStateValidationError:
        raise _CorruptRecord from None
    if (
        type(normalized) is not dict
        or nodes > MAX_JSON_NODES
        or characters > MAX_JSON_CHARS
    ):
        raise _CorruptRecord
    return _freeze_mapping(normalized)


def _freeze_mapping(value: dict[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_json(value)
    if not isinstance(frozen, Mapping):
        raise _CorruptRecord
    return frozen


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _variant_record(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> ResumeVariantRecord:
    application = connection.execute(
        "SELECT 1 FROM applications WHERE job_id = ?",
        (row["job_id"],),
    ).fetchone()
    if application is None:
        raise _CorruptRecord
    key = _stored_variant_key(row["variant_key"])
    if key is None:
        raise _CorruptRecord
    parent = _stored_variant_key(row["parent_variant_key"])
    if key == "v1" and parent is not None:
        raise _CorruptRecord
    if key == "v2" and parent != "v1":
        raise _CorruptRecord
    if key == "manual" and parent not in {"v1", "v2"}:
        raise _CorruptRecord
    if parent is not None:
        parent_row = connection.execute(
            """
            SELECT 1 FROM application_resume_variants
            WHERE job_id = ? AND variant_key = ?
            """,
            (row["job_id"], parent),
        ).fetchone()
        if parent_row is None:
            raise _CorruptRecord
    aro = _decode_optional_yaml(row["application_resume_object"])
    if aro is None:
        raise _CorruptRecord
    job_id = _stored_identifier(row["job_id"])
    resume_html = _stored_optional_text(row["resume_html_content"], MAX_HTML_CHARS)
    resume_pdf = _stored_optional_bytes(row["resume_content"], MAX_PDF_BYTES)
    ats = AtsFields(
        score=_stored_score(row["ats_score"]),
        parsing_score=_stored_score(row["ats_parsing_score"]),
        keyword_score=_stored_score(row["ats_keyword_score"]),
        semantic_score=_stored_score(row["ats_semantic_score"]),
        formatting_risk=_stored_optional_text(
            row["ats_formatting_risk"], MAX_LABEL_CHARS
        ),
        missing_terms=_stored_optional_text(
            row["ats_missing_terms"], MAX_METADATA_TEXT_CHARS
        ),
        diagnostics=_decode_optional_json(row["ats_diagnostics_json"]),
        updated_at=_stored_optional_text(row["ats_updated_at"], 128),
    )
    return ResumeVariantRecord(
        job_id=job_id,
        variant_key=key,
        variant_label=_stored_text(row["variant_label"], MAX_LABEL_CHARS),
        source=_stored_text(row["source"], MAX_LABEL_CHARS),
        parent_variant_key=parent,
        application_resume=aro,
        resume_html_filename=(
            artifact_filename(job_id, ArtifactKind.RESUME_HTML)
            if resume_html is not None
            else ""
        ),
        resume_html=resume_html,
        resume_html_mime_type=_stored_text(
            row["resume_html_mime_type"], MAX_LABEL_CHARS
        ),
        source_resume_html_path=_stored_text(
            row["source_resume_html_path"], MAX_URL_CHARS
        ),
        resume_html_updated_at=_stored_optional_text(
            row["resume_html_updated_at"], 128
        ),
        resume_pdf_filename=(
            artifact_filename(job_id, ArtifactKind.RESUME_PDF)
            if resume_pdf is not None
            else ""
        ),
        resume_pdf=resume_pdf,
        resume_pdf_mime_type=_stored_text(row["resume_mime_type"], MAX_LABEL_CHARS),
        source_resume_path=_stored_text(row["source_resume_path"], MAX_URL_CHARS),
        resume_pdf_updated_at=_stored_optional_text(row["resume_updated_at"], 128),
        ats=ats,
        ats_diagnostics=_decode_optional_json(row["ats_diagnostics_json"]),
        evidence_packet=_decode_optional_json(row["evidence_packet_json"]),
        external_critique=_decode_optional_json(row["external_critique_json"]),
        critique_prompt=_stored_optional_text(
            row["critique_prompt"], MAX_METADATA_TEXT_CHARS
        ),
        critique_response=_stored_optional_text(
            row["critique_response"], MAX_METADATA_TEXT_CHARS
        ),
        critique=_decode_optional_json(row["critique_json"]),
        validation=_decode_optional_json(row["validation_json"]),
        model_metadata=_decode_optional_json(row["model_metadata_json"]),
        created_at=_stored_text(row["created_at"], 128),
        updated_at=_stored_text(row["updated_at"], 128),
    )


def _stored_text(value: object, limit: int) -> str:
    if type(value) is not str or len(value) > limit or "\x00" in value:
        raise _CorruptRecord
    return value


def _stored_identifier(value: object) -> str:
    rendered = _stored_text(value, MAX_IDENTIFIER_CHARS)
    if not _IDENTIFIER_RE.fullmatch(rendered) or rendered in {".", ".."}:
        raise _CorruptRecord
    return rendered


def _stored_job_url(value: object) -> str:
    rendered = _stored_text(value, MAX_URL_CHARS)
    failed = False
    normalized: str | None = None
    try:
        normalized = normalize_job_url(rendered)
    except ProviderError:
        failed = True
    if failed or normalized != rendered:
        raise _CorruptRecord
    return rendered


def _stored_optional_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return _stored_text(value, limit)


def _stored_optional_bytes(value: object, limit: int) -> bytes | None:
    if value is None:
        return None
    if type(value) is not bytes or len(value) > limit:
        raise _CorruptRecord
    return value


def _stored_score(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 100:
        raise _CorruptRecord
    return value


def _stored_query_outcome(row: sqlite3.Row) -> StoredQueryOutcome:
    profile_match = row["profile_match"]
    query_score = row["query_score"]
    average_ats_score = row["average_ats_score"]
    if (
        type(profile_match) not in {int, float}
        or not math.isfinite(float(profile_match))
        or not 0.0 <= float(profile_match) <= 1.0
        or type(query_score) not in {int, float}
        or not math.isfinite(float(query_score))
        or not 0.0 <= float(query_score) <= 1.0
    ):
        raise _CorruptRecord
    if average_ats_score is not None and (
        type(average_ats_score) not in {int, float}
        or not math.isfinite(float(average_ats_score))
        or not 0.0 <= float(average_ats_score) <= 100.0
    ):
        raise _CorruptRecord

    def counter(name: str) -> int:
        value = row[name]
        if type(value) is not int or not 0 <= value <= MAX_QUERY_RESULTS:
            raise _CorruptRecord
        return value

    limit = counter("limit_value")
    if limit == 0:
        raise _CorruptRecord
    return StoredQueryOutcome(
        keywords=_stored_text(row["keywords"], MAX_LABEL_CHARS),
        location=_stored_text(row["location"], MAX_LABEL_CHARS),
        date_posted=_stored_text(row["date_posted"], 64),
        workplace_type=_stored_optional_text(row["workplace_type"], 64),
        experience_level=_stored_optional_text(row["experience_level"], 64),
        job_type=_stored_optional_text(row["job_type"], 64),
        sort_by=_stored_text(row["sort_by"], 64),
        limit=limit,
        profile_match=float(profile_match),
        query_score=float(query_score),
        results_returned=counter("results_returned"),
        fresh_jobs_accepted=counter("fresh_jobs_accepted"),
        skipped_existing=counter("skipped_existing"),
        skipped_blacklisted=counter("skipped_blacklisted"),
        skipped_workplace_type=counter("skipped_workplace_type"),
        skipped_experience_level=counter("skipped_experience_level"),
        resumes_generated=counter("resumes_generated"),
        average_ats_score=(
            None if average_ats_score is None else float(average_ats_score)
        ),
    )


def _stored_variant_key(value: object) -> str | None:
    if value is None or value == "":
        return None
    if type(value) is not str or value not in VARIANT_KEYS:
        raise _CorruptRecord
    return value


def _active_resume_target(application: sqlite3.Row) -> str:
    return _stored_variant_key(application["selected_resume_variant"]) or "fallback"


def _stored_backup_target(value: object) -> str | None:
    if value is None or value == "":
        return None
    if type(value) is not str or value not in {"fallback", *VARIANT_KEYS}:
        raise _CorruptRecord
    return value


def _stored_variant_key_for_migration(value: object) -> str | None:
    try:
        return _stored_variant_key(value)
    except _CorruptRecord:
        raise _UnsafeSchema from None


def _stored_mode(value: object) -> ResumeSelectionMode:
    try:
        return ResumeSelectionMode(value)
    except (TypeError, ValueError):
        raise _CorruptRecord from None


def _stored_mode_for_migration(value: object) -> ResumeSelectionMode:
    try:
        return _stored_mode(value)
    except _CorruptRecord:
        raise _UnsafeSchema from None


def _legacy_timestamp(row: sqlite3.Row, fallback: str) -> str:
    for column in (
        "application_resume_updated_at",
        "resume_updated_at",
        "updated_at",
        "imported_at",
    ):
        value = row[column]
        if type(value) is str and value:
            return value
    return fallback


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (name,),
        ).fetchone()
        is not None
    )


def _schema_is_initialized(connection: sqlite3.Connection) -> bool:
    if not all(
        _table_exists(connection, name)
        for name in (
            "applications",
            "application_resume_variants",
            "search_query_outcomes",
        )
    ):
        return False
    application_columns = _table_columns(connection, "applications")
    variant_columns = _table_columns(connection, "application_resume_variants")
    query_columns = _table_columns(connection, "search_query_outcomes")
    required_application_columns = {
        "job_id",
        *(name for name, _ in _APPLICATION_COLUMN_DEFINITIONS),
    }
    required_variant_columns = {
        "job_id",
        "variant_key",
        *(name for name, _ in _VARIANT_COLUMN_DEFINITIONS),
    }
    if not required_application_columns <= application_columns:
        return False
    if not required_variant_columns <= variant_columns:
        return False
    if (
        not {
            "id",
            "created_at",
            "keywords",
            "location",
            "query_score",
            "fresh_jobs_accepted",
        }
        <= query_columns
    ):
        return False
    return all(
        (
            _index_matches(
                connection,
                table="applications",
                name="applications_unique_job_id",
                unique=True,
                columns=("job_id",),
            ),
            _index_matches(
                connection,
                table="applications",
                name="applications_archive_order",
                unique=False,
                columns=("archived_at", "job_id"),
            ),
            _index_matches(
                connection,
                table="application_resume_variants",
                name="application_resume_variants_identity",
                unique=True,
                columns=("job_id", "variant_key"),
            ),
            _index_matches(
                connection,
                table="application_resume_variants",
                name="application_resume_variants_variant_key",
                unique=False,
                columns=("variant_key",),
            ),
            _index_matches(
                connection,
                table="search_query_outcomes",
                name="search_query_outcomes_keywords_idx",
                unique=False,
                columns=("keywords", "location", "workplace_type"),
            ),
        )
    )


def _index_matches(
    connection: sqlite3.Connection,
    *,
    table: str,
    name: str,
    unique: bool,
    columns: tuple[str, ...],
) -> bool:
    allowed = {
        (
            "applications",
            "applications_unique_job_id",
        ),
        ("applications", "applications_archive_order"),
        (
            "application_resume_variants",
            "application_resume_variants_identity",
        ),
        (
            "application_resume_variants",
            "application_resume_variants_variant_key",
        ),
        ("search_query_outcomes", "search_query_outcomes_keywords_idx"),
    }
    if (table, name) not in allowed:
        raise _UnsafeSchema
    row = next(
        (
            item
            for item in connection.execute(f"PRAGMA index_list({table})").fetchall()
            if item["name"] == name
        ),
        None,
    )
    if row is None or bool(row["unique"]) is not unique or bool(row["partial"]):
        return False
    actual_columns = tuple(
        str(item["name"])
        for item in sorted(
            connection.execute(f"PRAGMA index_info({name})").fetchall(),
            key=lambda item: int(item["seqno"]),
        )
    )
    return actual_columns == columns


def _table_columns(connection: sqlite3.Connection, name: str) -> set[str]:
    if name not in {
        "applications",
        "application_resume_variants",
        "search_query_outcomes",
    }:
        raise _UnsafeSchema
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({name})").fetchall()
    }


def _sqlite_is_busy(error: sqlite3.Error) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return type(code) is int and code & 0xFF in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }


def _rollback_quietly(connection: sqlite3.Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.rollback()
    except sqlite3.Error:
        pass


def _close_quietly(connection: sqlite3.Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except sqlite3.Error:
        pass


__all__ = [
    "APPLICATION_STATUSES",
    "MAX_BULK_IDENTIFIERS",
    "MAX_QUERY_RESULTS",
    "MAX_QUERY_HISTORY_RESULTS",
    "SCHEMA_VERSION",
    "ApplicationMetadata",
    "ApplicationRecord",
    "ApplicationSeedOutcome",
    "ApplicationScope",
    "ApplicationStateBusyError",
    "ApplicationStateConfigurationError",
    "ApplicationStateConflictError",
    "ApplicationStateCorruptionError",
    "ApplicationStateError",
    "ApplicationStateInitializationError",
    "ApplicationStateNotFoundError",
    "ApplicationStateNotInitializedError",
    "ApplicationStateStore",
    "ApplicationStateValidationError",
    "ApplicationWorkflowRevision",
    "ApplicationWorkflowSnapshot",
    "ArtifactKind",
    "AtsFields",
    "QueryOutcomeWrite",
    "ResumeSelectionMode",
    "ResumeVariantKey",
    "ResumeVariantRecord",
    "ResumeVariantWrite",
    "artifact_filename",
    "workflow_revision_from_token",
    "workflow_revision_token",
]
