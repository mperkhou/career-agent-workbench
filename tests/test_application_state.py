from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
import socket
import sqlite3
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, ClassVar, get_args

import httpx
import pytest

import career_agent_workbench.api_client as api_client_module
import career_agent_workbench.application_state as application_state_module
import career_agent_workbench.ats as ats_module
import career_agent_workbench.config as config_module
import career_agent_workbench.llm as llm_module
import career_agent_workbench.ollama as ollama_module
import career_agent_workbench.resume_rendering as rendering_module
from career_agent_workbench.application_state import (
    APPLICATION_STATUSES,
    MAX_ARO_YAML_BYTES,
    MAX_BULK_IDENTIFIERS,
    MAX_JSON_CHARS,
    MAX_QUERY_RESULTS,
    ApplicationMetadata,
    ApplicationSeedOutcome,
    ApplicationScope,
    ApplicationStateBusyError,
    ApplicationStateConfigurationError,
    ApplicationStateConflictError,
    ApplicationStateCorruptionError,
    ApplicationStateError,
    ApplicationStateInitializationError,
    ApplicationStateNotFoundError,
    ApplicationStateNotInitializedError,
    ApplicationStateStore,
    ApplicationStateValidationError,
    ApplicationWorkflowRevision,
    ApplicationWorkflowSnapshot,
    ArtifactKind,
    AtsFields,
    ResumeVariantWrite,
    artifact_filename,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.providers.linkedin_public import (
    LinkedInPublicJobsProvider,
)
from career_agent_workbench.workflows import (
    DEFAULT_APPLICATION_POLICY,
    ApplicationWorkflowPolicy,
    ApplicationWorkflowStage,
)

FIXED_TIME = datetime(2035, 2, 3, 4, 5, 6, tzinfo=UTC)
JOB_ONE = "job-example-001"
JOB_TWO = "job-example-002"
JOB_THREE = "job-example-003"
ARO_V1 = "profile:\n  summary: Synthetic draft one\nskills:\n  - Testing\n"
ARO_V2 = "profile:\n  summary: Synthetic draft two\nskills:\n  - Analysis\n"
ARO_MANUAL = "profile:\n  summary: Synthetic reviewed draft\nskills:\n  - Review\n"


class TickingClock:
    def __init__(self, start: datetime = FIXED_TIME) -> None:
        self._next = start
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            value = self._next
            self._next += timedelta(microseconds=1)
            return value


class ActiveValue:
    def __init__(self) -> None:
        self.touched = False

    def __iter__(self) -> Iterator[object]:
        self.touched = True
        raise AssertionError("active conversion must not run")

    def __str__(self) -> str:
        self.touched = True
        raise AssertionError("active conversion must not run")


class HostileCapability:
    __slots__ = ("calls",)

    def __init__(self) -> None:
        object.__setattr__(self, "calls", [])

    def _trip(self, name: str) -> Any:
        object.__getattribute__(self, "calls").append(name)
        raise RuntimeError("synthetic hostile capability detail")

    def __getattribute__(self, name: str) -> Any:
        if name in {"calls", "_trip"}:
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_trip")(f"get:{name}")

    def __hash__(self) -> int:
        return self._trip("hash")

    def __eq__(self, other: object) -> bool:
        return self._trip("eq")

    def __str__(self) -> str:
        return self._trip("str")

    def __repr__(self) -> str:
        return self._trip("repr")

    def __iter__(self) -> Iterator[object]:
        return self._trip("iter")


class HostileDatetime(datetime):
    calls: ClassVar[list[str]] = []

    def __getattribute__(self, name: str) -> Any:
        type(self).calls.append(name)
        raise RuntimeError("synthetic hostile datetime detail")


class HostileTimezone(tzinfo):
    def __init__(self) -> None:
        self.calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta:
        self.calls += 1
        raise RuntimeError("synthetic hostile timezone detail")


class TwoStepTimezone(tzinfo):
    def __init__(self) -> None:
        self.calls = 0

    def utcoffset(self, value: datetime | None) -> timedelta:
        self.calls += 1
        if self.calls == 1:
            return timedelta(0)
        raise RuntimeError("synthetic conversion detail")


def _store(
    tmp_path: Path,
    *,
    output_dir: Path | None = None,
    clock: Callable[[], datetime] | None = None,
    failure_injector: Callable[[str], None] | None = None,
    operation_hook: Callable[[str], None] | None = None,
    busy_timeout_seconds: float = 1.0,
) -> tuple[ApplicationStateStore, Path]:
    database = tmp_path / "database-parent" / "applications.sqlite3"
    return (
        ApplicationStateStore(
            WorkspacePaths(database=database, output_dir=output_dir),
            utc_clock=clock or TickingClock(),
            failure_injector=failure_injector,
            operation_hook=operation_hook,
            busy_timeout_seconds=busy_timeout_seconds,
        ),
        database,
    )


def _metadata(
    job_id: str = JOB_ONE,
    *,
    company: str = "Example Research Cooperative",
    title: str = "Synthetic Systems Analyst",
    url: str | None = None,
) -> ApplicationMetadata:
    return ApplicationMetadata(
        job_id=job_id,
        company=company,
        job_title=title,
        job_url=url or f"https://jobs.example.com/openings/{job_id}",
        source="public_example",
        date_matched="2035-02-01",
        date_posted="2035-01-30",
        experience_level="Mid-level",
    )


def _required_metadata(
    job_id: str = JOB_ONE,
    *,
    company: str = "Example Research Cooperative",
    title: str = "Synthetic Systems Analyst",
    url: str | None = None,
) -> ApplicationMetadata:
    return ApplicationMetadata(
        job_id=job_id,
        company=company,
        job_title=title,
        job_url=url or f"https://jobs.example.com/openings/{job_id}",
    )


def _variant(
    key: str,
    *,
    parent: str | None = None,
    yaml_text: str | None = None,
    marker: str | None = None,
) -> ResumeVariantWrite:
    label = {
        "v1": "Synthetic draft one",
        "v2": "Synthetic draft two",
        "manual": "Synthetic reviewed draft",
    }[key]
    yaml_value = yaml_text or {"v1": ARO_V1, "v2": ARO_V2, "manual": ARO_MANUAL}[key]
    token = marker or key
    return ResumeVariantWrite(
        variant_key=key,
        variant_label=label,
        source="synthetic_test",
        parent_variant_key=parent,
        application_resume_yaml=yaml_value,
        resume_html=f"<main>{token}</main>",
        resume_pdf=f"pdf-{token}".encode(),
        source_resume_html_path=f"provenance/{token}.html",
        source_resume_path=f"provenance/{token}.pdf",
        ats=AtsFields(
            score=80 + len(token) % 10,
            parsing_score=91,
            keyword_score=82,
            semantic_score=83,
            formatting_risk="low",
            missing_terms="fictional-term",
            diagnostics={"source": token, "checks": [True, 2]},
            updated_at="2035-02-03T04:05:06+00:00",
        ),
        evidence_packet={"variant": token, "items": [1, 2]},
        external_critique={"summary": f"critique-{token}"},
        critique_prompt=f"prompt-{token}",
        critique_response=f"response-{token}",
        critique={"rating": 4},
        validation={"valid": True},
        model_metadata={"model": "synthetic-local", "temperature": 0},
    )


def _initialize_with_application(
    tmp_path: Path,
    job_id: str = JOB_ONE,
) -> tuple[ApplicationStateStore, Path]:
    store, database = _store(tmp_path)
    store.initialize()
    store.upsert_application(_metadata(job_id))
    return store, database


def _database_digest(database: Path) -> str:
    return hashlib.sha256(database.read_bytes()).hexdigest()


def _schema_snapshot(database: Path) -> tuple[object, ...]:
    with sqlite3.connect(database) as connection:
        objects = tuple(
            connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                ORDER BY type, name
                """
            ).fetchall()
        )
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        applications = tuple(
            connection.execute("SELECT * FROM applications ORDER BY job_id").fetchall()
        )
        variants = (
            tuple(
                connection.execute(
                    """
                    SELECT * FROM application_resume_variants
                    ORDER BY job_id, variant_key
                    """
                ).fetchall()
            )
            if any(row[1] == "application_resume_variants" for row in objects)
            else ()
        )
    return version, objects, applications, variants


def _assert_content_free(
    error: BaseException,
    *,
    forbidden: tuple[str, ...] = (),
) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = f"{error!s}\n{error!r}"
    for value in forbidden:
        assert value not in rendered
    assert "\n" not in str(error)
    assert len(str(error)) < 100


def _assert_exact_public_error(
    error: BaseException,
    expected_type: type[BaseException],
    message: str,
    *,
    forbidden: tuple[str, ...] = (),
) -> None:
    assert type(error) is expected_type
    assert str(error) == message
    assert repr(error) == f"{expected_type.__name__}({message!r})"
    _assert_content_free(error, forbidden=forbidden)


def _assert_projection_matches_selected_variant(application: object) -> None:
    assert type(application).__name__ == "ApplicationRecord"
    selected = application.selected_variant
    assert selected is not None
    assert application.application_resume == selected.application_resume
    assert application.resume_html_filename == selected.resume_html_filename
    assert application.resume_html == selected.resume_html
    assert application.resume_html_mime_type == selected.resume_html_mime_type
    assert application.source_resume_html_path == selected.source_resume_html_path
    assert application.resume_html_updated_at == selected.resume_html_updated_at
    assert application.resume_pdf_filename == selected.resume_pdf_filename
    assert application.resume_pdf == selected.resume_pdf
    assert application.resume_pdf_mime_type == selected.resume_pdf_mime_type
    assert application.source_resume_path == selected.source_resume_path
    assert application.resume_pdf_updated_at == selected.resume_pdf_updated_at
    assert application.ats == selected.ats


def _create_minimal_legacy(
    database: Path,
    *,
    job_id: str = JOB_ONE,
    selected: str = "v1",
    mode: str = "auto",
    aro_yaml: str | None = ARO_V1,
    notes: str = "",
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE applications (
                job_id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                job_title TEXT NOT NULL,
                linkedin_url TEXT NOT NULL,
                application_resume_object TEXT,
                application_resume_updated_at TEXT,
                resume_html_filename TEXT NOT NULL DEFAULT '',
                resume_html_content TEXT,
                resume_html_mime_type TEXT NOT NULL DEFAULT
                    'text/html; charset=utf-8',
                source_resume_html_path TEXT NOT NULL DEFAULT '',
                resume_html_updated_at TEXT,
                resume_filename TEXT NOT NULL,
                resume_content BLOB,
                resume_mime_type TEXT NOT NULL DEFAULT 'application/pdf',
                source_resume_path TEXT NOT NULL,
                resume_updated_at TEXT,
                selected_resume_variant TEXT NOT NULL DEFAULT 'v1',
                resume_variant_selection_mode TEXT NOT NULL DEFAULT 'auto',
                applied_to TEXT NOT NULL DEFAULT 'No',
                notes TEXT NOT NULL DEFAULT '',
                imported_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO applications (
                job_id, company, job_title, linkedin_url,
                application_resume_object, application_resume_updated_at,
                resume_html_filename, resume_html_content,
                source_resume_html_path, resume_html_updated_at,
                resume_filename, resume_content, source_resume_path,
                resume_updated_at, selected_resume_variant,
                resume_variant_selection_mode, notes, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "Example Legacy Cooperative",
                "Synthetic Legacy Analyst",
                f"https://legacy.example.com/jobs/{job_id}",
                aro_yaml,
                "2034-01-02T03:04:05+00:00",
                "legacy-resume.html",
                "<main>legacy synthetic</main>",
                "legacy/provenance.html",
                "2034-01-02T03:04:05+00:00",
                "legacy-resume.pdf",
                b"legacy-synthetic-pdf",
                "legacy/provenance.pdf",
                "2034-01-02T03:04:05+00:00",
                selected,
                mode,
                notes,
                "2034-01-01T00:00:00+00:00",
                "2034-01-02T03:04:05+00:00",
            ),
        )


def _create_complete_cutoff(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE applications (
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
                resume_html_mime_type TEXT NOT NULL DEFAULT
                    'text/html; charset=utf-8',
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
                updated_at TEXT NOT NULL,
                selected_resume_variant TEXT NOT NULL DEFAULT 'v1',
                resume_variant_selection_mode TEXT NOT NULL DEFAULT 'auto'
            );

            CREATE TABLE application_resume_variants (
                job_id TEXT NOT NULL,
                variant_key TEXT NOT NULL,
                variant_label TEXT NOT NULL,
                source TEXT NOT NULL,
                parent_variant_key TEXT,
                application_resume_object TEXT NOT NULL,
                resume_html_filename TEXT NOT NULL DEFAULT '',
                resume_html_content TEXT,
                resume_html_mime_type TEXT NOT NULL DEFAULT
                    'text/html; charset=utf-8',
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
            );
            CREATE INDEX application_resume_variants_variant_key
            ON application_resume_variants(variant_key);
            """
        )
        connection.execute(
            """
            INSERT INTO applications (
                job_id, company, job_title, linkedin_url,
                resume_filename, source_resume_path, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, '', '', ?, ?)
            """,
            (
                "cutoff-empty",
                "Example Cutoff Cooperative",
                "Synthetic Cutoff Analyst",
                "https://cutoff.example.com/jobs/empty",
                "2033-01-01T00:00:00+00:00",
                "2033-01-01T00:00:00+00:00",
            ),
        )


def test_import_and_construction_are_side_effect_free(tmp_path: Path) -> None:
    database = tmp_path / "missing" / "state.sqlite3"
    output = tmp_path / "independent-output"

    store = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=lambda: FIXED_TIME,
    )

    assert store.output_dir == output
    assert not database.parent.exists()
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_fresh_isolated_import_performs_no_workspace_discovery(
    tmp_path: Path,
) -> None:
    module_file = application_state_module.__file__
    assert module_file is not None
    package_dir = Path(module_file).resolve().parent
    package_parent = package_dir.parent
    off_root = tmp_path / "unrelated-empty-working-directory"
    hostile_root = tmp_path / "fictional-hostile-workspace"
    off_root.mkdir()
    hostile_root.mkdir()
    residue_before = tuple(
        sorted(
            path.relative_to(package_parent).as_posix()
            for path in package_dir.rglob("*")
            if path.name == "__pycache__" or path.suffix == ".pyc"
        )
    )
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CAREER_AGENT_WORKBENCH_WORKSPACE": str(hostile_root / "workspace"),
        "CAREER_AGENT_WORKBENCH_PROFILE_DIR": str(hostile_root / "profile"),
        "CAREER_AGENT_WORKBENCH_MASTER_RESUME": str(
            hostile_root / "profile" / "master.yml"
        ),
        "CAREER_AGENT_WORKBENCH_MASTER_RESUME_TEXT": str(
            hostile_root / "profile" / "master.txt"
        ),
        "CAREER_AGENT_WORKBENCH_OUTPUT_DIR": str(hostile_root / "output"),
        "CAREER_AGENT_WORKBENCH_DATABASE": str(
            hostile_root / "output" / "tracking.sqlite3"
        ),
        "CAREER_AGENT_WORKBENCH_BLACKLIST": str(hostile_root / "blacklist.txt"),
        "CAREER_AGENT_WORKBENCH_TMP_DIR": str(hostile_root / "tmp"),
        "CAREER_AGENT_WORKBENCH_ENV_FILE": str(hostile_root / ".env"),
    }
    command = (
        "import logging, sys; "
        "records = []; "
        "handler = logging.Handler(); "
        "handler.emit = records.append; "
        "root = logging.getLogger(); "
        "root.addHandler(handler); "
        "sys.path.insert(0, sys.argv[1]); "
        "import career_agent_workbench.application_state; "
        "raise SystemExit(bool(records))"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            command,
            str(package_parent),
        ],
        cwd=off_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    residue_after = tuple(
        sorted(
            path.relative_to(package_parent).as_posix()
            for path in package_dir.rglob("*")
            if path.name == "__pycache__" or path.suffix == ".pyc"
        )
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert tuple(off_root.iterdir()) == ()
    assert tuple(hostile_root.iterdir()) == ()
    assert residue_after == residue_before


def test_database_only_configuration_and_independent_output_do_not_drift(
    tmp_path: Path,
) -> None:
    database_only, database = _store(tmp_path)
    assert database_only.output_dir is None
    database_only.initialize()
    assert database.exists()

    other_root = tmp_path / "second"
    output = tmp_path / "future-output" / "nested"
    independent, independent_database = _store(other_root, output_dir=output)
    independent.initialize()

    assert independent.output_dir == output
    assert independent_database.exists()
    assert not output.exists()
    assert independent_database.parent != output


def test_missing_or_relative_database_configuration_fails_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for paths in (WorkspacePaths(), WorkspacePaths(database=Path("relative.sqlite3"))):
        with pytest.raises(ApplicationStateConfigurationError) as caught:
            ApplicationStateStore(paths)
        _assert_content_free(caught.value, forbidden=(str(tmp_path),))
    assert list(tmp_path.iterdir()) == []


def test_reads_and_mutations_against_missing_database_do_not_create_it(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)

    operations: tuple[Callable[[], object], ...] = (
        lambda: store.get_application(JOB_ONE),
        lambda: store.list_applications(),
        lambda: store.fetch_job_records([]),
        lambda: store.archive([JOB_ONE]),
        lambda: store.store_jod(JOB_ONE, source_text="source", prompt_text="prompt"),
    )
    for operation in operations:
        with pytest.raises(ApplicationStateNotInitializedError) as caught:
            operation()
        _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database)))
        assert not database.parent.exists()


def test_existing_legacy_database_read_before_initialize_is_byte_identical(
    tmp_path: Path,
) -> None:
    _, database = _store(tmp_path)
    _create_minimal_legacy(database)
    before_digest = _database_digest(database)
    before_schema = _schema_snapshot(database)
    store = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=lambda: FIXED_TIME,
    )

    with pytest.raises(ApplicationStateNotInitializedError) as caught:
        store.get_application(JOB_ONE)

    _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database)))
    assert _database_digest(database) == before_digest
    assert _schema_snapshot(database) == before_schema


def test_incomplete_versioned_database_is_still_not_initialized(
    tmp_path: Path,
) -> None:
    database = tmp_path / "incomplete.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE applications (job_id TEXT PRIMARY KEY)")
        connection.execute(
            """
            CREATE TABLE application_resume_variants (
                job_id TEXT NOT NULL,
                variant_key TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
    before = _database_digest(database)
    store = ApplicationStateStore(WorkspacePaths(database=database))

    with pytest.raises(ApplicationStateNotInitializedError) as caught:
        store.list_applications()

    _assert_content_free(caught.value, forbidden=(str(database),))
    assert _database_digest(database) == before


def test_wrong_named_index_signature_is_not_treated_as_initialized(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    store.initialize()
    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX applications_archive_order")
        connection.execute(
            """
            CREATE INDEX applications_archive_order
            ON applications(job_id, archived_at)
            """
        )
    before = _database_digest(database)

    with pytest.raises(ApplicationStateNotInitializedError) as caught:
        store.list_applications()

    _assert_content_free(caught.value, forbidden=(str(database),))
    assert _database_digest(database) == before
    with pytest.raises(ApplicationStateInitializationError) as initialize_error:
        store.initialize()
    _assert_content_free(initialize_error.value, forbidden=(str(database),))
    assert _database_digest(database) == before


def test_fresh_initialization_is_idempotent_with_exact_schema(tmp_path: Path) -> None:
    store, database = _store(tmp_path)
    store.initialize()
    store.initialize()

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        application_indexes = {
            row["name"] for row in connection.execute("PRAGMA index_list(applications)")
        }
        variant_indexes = {
            row["name"]
            for row in connection.execute(
                "PRAGMA index_list(application_resume_variants)"
            )
        }
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(application_resume_variants)"
        ).fetchall()
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    assert version == 1
    assert tables == {"applications", "application_resume_variants"}
    assert application_indexes == {
        "applications_archive_order",
        "applications_unique_job_id",
        "sqlite_autoindex_applications_1",
    }
    assert variant_indexes == {
        "application_resume_variants_variant_key",
        "application_resume_variants_identity",
        "sqlite_autoindex_application_resume_variants_1",
    }
    assert len(foreign_keys) == 1
    foreign_key = foreign_keys[0]
    assert foreign_key["table"] == "applications"
    assert foreign_key["from"] == "job_id"
    assert foreign_key["to"] == "job_id"
    assert foreign_key["on_delete"] == "CASCADE"
    assert not database.with_name(f"{database.name}-journal").exists()


def test_minimal_legacy_migration_preserves_bytes_and_is_repeatable(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_minimal_legacy(database)

    store.initialize()
    migrated = store.get_application(JOB_ONE)
    variant = store.get_resume_variant(JOB_ONE, "v1")
    timestamps = (variant.created_at, variant.updated_at, migrated.updated_at)
    store.initialize()

    repeated = store.get_application(JOB_ONE)
    repeated_variants = store.list_resume_variants(JOB_ONE)
    assert migrated.job_url == f"https://legacy.example.com/jobs/{JOB_ONE}"
    assert migrated.date_matched == "2034-01-01T00:00:00+00:00"
    assert variant.resume_html == "<main>legacy synthetic</main>"
    assert variant.resume_pdf == b"legacy-synthetic-pdf"
    assert migrated.resume_html_filename == artifact_filename(
        JOB_ONE, ArtifactKind.RESUME_HTML
    )
    assert migrated.resume_pdf_filename == artifact_filename(
        JOB_ONE, ArtifactKind.RESUME_PDF
    )
    assert variant.resume_html_filename == migrated.resume_html_filename
    assert variant.resume_pdf_filename == migrated.resume_pdf_filename
    assert "legacy" not in variant.resume_html_filename
    assert "legacy" not in variant.resume_pdf_filename
    assert variant.application_resume["profile"]["summary"] == "Synthetic draft one"
    assert len(repeated_variants) == 1
    assert repeated_variants[0].variant_key == "v1"
    assert (
        repeated_variants[0].created_at,
        repeated_variants[0].updated_at,
        repeated.updated_at,
    ) == timestamps
    with sqlite3.connect(database) as connection:
        raw_names = connection.execute(
            """
            SELECT resume_html_filename, resume_filename
            FROM applications WHERE job_id = ?
            """,
            (JOB_ONE,),
        ).fetchone()
    assert raw_names == ("legacy-resume.html", "legacy-resume.pdf")


def test_legacy_notes_do_not_infer_manual_variant(tmp_path: Path) -> None:
    store, database = _store(tmp_path)
    _create_minimal_legacy(
        database,
        notes="Free-form note mentioning a manual review should remain inert.",
    )

    store.initialize()

    assert [item.variant_key for item in store.list_resume_variants(JOB_ONE)] == ["v1"]
    assert store.get_application(JOB_ONE).selected_resume_variant == "v1"


def test_explicit_structured_legacy_manual_projection_backfills_once(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_minimal_legacy(database, selected="manual", mode="manual")

    store.initialize()
    first = store.get_application(JOB_ONE)
    variants = store.list_resume_variants(JOB_ONE)
    store.initialize()
    repeated = store.list_resume_variants(JOB_ONE)

    assert [item.variant_key for item in variants] == ["v1", "manual"]
    assert first.selected_resume_variant == "manual"
    assert first.resume_variant_selection_mode == "manual"
    assert first.selected_variant is not None
    assert first.selected_variant.resume_pdf == b"legacy-synthetic-pdf"
    assert [(item.variant_key, item.created_at) for item in repeated] == [
        (item.variant_key, item.created_at) for item in variants
    ]


def test_structured_manual_without_artifact_rolls_back_migration(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_minimal_legacy(database, selected="manual", mode="manual")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE applications SET resume_content = NULL WHERE job_id = ?",
            (JOB_ONE,),
        )
    before = _schema_snapshot(database)

    with pytest.raises(ApplicationStateInitializationError) as caught:
        store.initialize()

    _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database)))
    assert _schema_snapshot(database) == before


@pytest.mark.parametrize(
    ("selected", "aro_yaml", "expected"),
    [
        ("v2", ARO_V1, "v1"),
        ("manual", None, None),
    ],
)
def test_automatic_dangling_migration_falls_back_or_becomes_unselected(
    tmp_path: Path,
    selected: str,
    aro_yaml: str | None,
    expected: str | None,
) -> None:
    store, database = _store(tmp_path)
    _create_minimal_legacy(
        database,
        selected=selected,
        mode="auto",
        aro_yaml=aro_yaml,
    )

    store.initialize()

    assert store.get_application(JOB_ONE).selected_resume_variant == expected


def test_unresolved_manual_migration_rolls_back_schema_and_rows(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_minimal_legacy(
        database,
        selected="v2",
        mode="manual",
        aro_yaml=None,
    )
    before = _schema_snapshot(database)

    with pytest.raises(ApplicationStateInitializationError) as caught:
        store.initialize()

    _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database), "v2"))
    assert _schema_snapshot(database) == before
    assert not database.with_name(f"{database.name}-journal").exists()


def test_fresh_application_metadata_lists_fetches_and_archive_lifecycle(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    store.initialize()
    store.upsert_application(_metadata(JOB_TWO))
    first = store.upsert_application(_metadata(JOB_ONE))
    store.upsert_application(_metadata(JOB_THREE))

    assert first.selected_resume_variant is None
    assert first.selected_variant is None
    assert first.resume_variant_selection_mode == "auto"
    assert [item.job_id for item in store.list_applications()] == [
        JOB_ONE,
        JOB_TWO,
        JOB_THREE,
    ]
    assert [
        item.job_id
        for item in store.fetch_job_records([JOB_THREE, JOB_ONE, "not-present"])
    ] == [JOB_THREE, JOB_ONE]
    assert store.archive([JOB_ONE, JOB_TWO]) == 2
    assert store.archive([JOB_ONE, JOB_TWO]) == 0
    assert [item.job_id for item in store.list_applications("archived")] == [
        JOB_ONE,
        JOB_TWO,
    ]
    assert [item.job_id for item in store.list_applications("active")] == [JOB_THREE]
    assert store.unarchive([JOB_TWO]) == 1
    assert [item.job_id for item in store.list_applications(ApplicationScope.ALL)] == [
        JOB_ONE,
        JOB_TWO,
        JOB_THREE,
    ]
    assert store.delete([JOB_ONE, JOB_THREE]) == 2
    assert [item.job_id for item in store.list_applications("all")] == [JOB_TWO]


def test_required_metadata_creation_uses_exact_injected_first_seen(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path, clock=lambda: FIXED_TIME)
    store.initialize()

    created = store.upsert_application(_required_metadata())
    blank_created = store.upsert_application(
        replace(_required_metadata(JOB_TWO), date_matched=" \t")
    )

    expected = FIXED_TIME.isoformat(timespec="microseconds")
    assert created.source == ""
    assert created.date_matched == expected
    assert created.date_posted is None
    assert created.experience_level is None
    assert created.imported_at == expected
    assert blank_created.date_matched == expected


def test_required_metadata_refresh_preserves_lifecycle_and_dedicated_state(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.store_jod(
        JOB_ONE,
        source_text="Synthetic public job description.",
        prompt_text="Synthetic bounded prompt description.",
    )
    store.store_aro(
        JOB_ONE,
        yaml_text=ARO_V1,
        backup_yaml_text="profile:\n  summary: Synthetic backup\n",
    )
    store.store_clo(
        JOB_ONE,
        value={"letter": {"paragraphs": ["Synthetic body"]}},
        pdf_content=b"synthetic-cover-letter-pdf",
        source_path="provenance/cover-letter.pdf",
    )
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.select_resume_variant(JOB_ONE, "v1")
    store.update_application_status(
        JOB_ONE,
        applied_to="Accepted for interview",
        date_applied="2035-02-04",
        notes="Synthetic reviewed status.",
    )
    store.archive([JOB_ONE])
    before = store.get_application(JOB_ONE)

    after = store.upsert_application(
        _required_metadata(
            company="Example Refreshed Cooperative",
            title="Synthetic Refreshed Analyst",
            url=f"https://jobs.example.com/refreshed/{JOB_ONE}",
        )
    )

    assert after.company == "Example Refreshed Cooperative"
    assert after.job_title == "Synthetic Refreshed Analyst"
    assert after.job_url == f"https://jobs.example.com/refreshed/{JOB_ONE}"
    assert (
        replace(
            after,
            company=before.company,
            job_title=before.job_title,
            job_url=before.job_url,
            updated_at=before.updated_at,
        )
        == before
    )


def test_blank_metadata_optionals_preserve_existing_values(tmp_path: Path) -> None:
    store, _ = _initialize_with_application(tmp_path)
    before = store.get_application(JOB_ONE)
    assert before.date_matched == "2035-02-01"

    after = store.upsert_application(
        ApplicationMetadata(
            job_id=JOB_ONE,
            company=before.company,
            job_title=before.job_title,
            job_url=before.job_url,
            source=" \t",
            date_matched="",
            date_posted="  ",
            experience_level="\t",
        )
    )

    assert (
        after.source,
        after.date_matched,
        after.date_posted,
        after.experience_level,
        after.imported_at,
    ) == (
        before.source,
        before.date_matched,
        before.date_posted,
        before.experience_level,
        before.imported_at,
    )


def test_existing_whitespace_match_date_receives_injected_first_seen(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path, clock=lambda: FIXED_TIME)
    store.initialize()
    store.upsert_application(_metadata())
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE applications SET date_matched = ? WHERE job_id = ?",
            ("\t", JOB_ONE),
        )

    refreshed = store.upsert_application(_required_metadata())

    assert refreshed.date_matched == FIXED_TIME.isoformat(timespec="microseconds")


def test_nonblank_metadata_optionals_update_except_first_seen(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    before = store.get_application(JOB_ONE)

    after = store.upsert_application(
        ApplicationMetadata(
            job_id=JOB_ONE,
            company="Example Updated Cooperative",
            job_title="Synthetic Updated Analyst",
            job_url=f"https://jobs.example.com/updated/{JOB_ONE}",
            source="public_refresh",
            date_matched="2040-01-01",
            date_posted="2035-02-02",
            experience_level="Senior",
        )
    )

    assert after.company == "Example Updated Cooperative"
    assert after.job_title == "Synthetic Updated Analyst"
    assert after.job_url == f"https://jobs.example.com/updated/{JOB_ONE}"
    assert after.source == "public_refresh"
    assert after.date_matched == before.date_matched
    assert after.date_posted == "2035-02-02"
    assert after.experience_level == "Senior"
    assert after.imported_at == before.imported_at


def test_complete_cutoff_metadata_preserves_first_seen_and_omissions(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path, clock=lambda: FIXED_TIME)
    _create_complete_cutoff(database)
    store.initialize()
    migrated = store.get_application("cutoff-empty")

    updated = store.upsert_application(
        ApplicationMetadata(
            job_id="cutoff-empty",
            company="Example Updated Cutoff Cooperative",
            job_title="Synthetic Updated Cutoff Analyst",
            job_url="https://cutoff.example.com/jobs/refreshed",
            source="public_cutoff",
            date_matched="2040-01-01",
            date_posted="2033-01-02",
            experience_level="Senior",
        )
    )
    required_refresh = store.upsert_application(
        _required_metadata(
            "cutoff-empty",
            company="Example Required Refresh Cooperative",
            title="Synthetic Required Refresh Analyst",
            url="https://cutoff.example.com/jobs/required-refresh",
        )
    )
    blank_refresh = store.upsert_application(
        ApplicationMetadata(
            job_id="cutoff-empty",
            company=required_refresh.company,
            job_title=required_refresh.job_title,
            job_url=required_refresh.job_url,
            source=" ",
            date_matched="\t",
            date_posted="",
            experience_level="  ",
        )
    )
    inserted = store.upsert_application(_required_metadata(JOB_ONE))

    assert migrated.date_matched == "2033-01-01T00:00:00+00:00"
    assert updated.date_matched == migrated.date_matched
    assert (
        blank_refresh.source,
        blank_refresh.date_matched,
        blank_refresh.date_posted,
        blank_refresh.experience_level,
    ) == (
        "public_cutoff",
        migrated.date_matched,
        "2033-01-02",
        "Senior",
    )
    assert inserted.date_matched == FIXED_TIME.isoformat(timespec="microseconds")
    with sqlite3.connect(database) as connection:
        compatibility = connection.execute(
            """
            SELECT linkedin_url, resume_filename, source_resume_path,
                   selected_resume_variant
            FROM applications WHERE job_id = ?
            """,
            (JOB_ONE,),
        ).fetchone()
    assert compatibility == (
        f"https://jobs.example.com/openings/{JOB_ONE}",
        "",
        "",
        "",
    )


def test_jod_aro_clo_application_artifacts_and_ats_round_trip(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    source = "Synthetic public job description."
    prompt = "Synthetic trimmed description."
    aro = store.store_aro(
        JOB_ONE,
        yaml_text=ARO_V1,
        backup_yaml_text="profile:\n  summary: Synthetic backup\n",
    )
    clo = store.store_clo(
        JOB_ONE,
        value={"letter": {"paragraphs": ["Synthetic greeting", "Synthetic close"]}},
        pdf_content=b"synthetic-cover-pdf",
        source_path="provenance/cover.pdf",
    )
    ats = AtsFields(
        score=88,
        parsing_score=91,
        keyword_score=82,
        semantic_score=87,
        formatting_risk="low",
        missing_terms="fictional-term",
        diagnostics={"checks": [True, False]},
        updated_at="2035-02-03T04:05:06+00:00",
    )

    store.store_jod(JOB_ONE, source_text=source, prompt_text=prompt)
    result = store.store_application_artifacts(
        JOB_ONE,
        resume_html="<main>Synthetic résumé</main>",
        resume_pdf=b"synthetic-resume-pdf",
        cover_letter_pdf=b"synthetic-cover-pdf-updated",
        source_resume_html_path="provenance/resume.html",
        source_resume_path="provenance/resume.pdf",
        source_cover_letter_path="provenance/cover-updated.pdf",
        ats=ats,
    )

    assert aro["profile"]["summary"] == "Synthetic draft one"
    assert aro["skills"] == ("Testing",)
    assert dict(clo)["letter"]["paragraphs"] == (
        "Synthetic greeting",
        "Synthetic close",
    )
    assert store.get_aro(JOB_ONE) == aro
    assert store.get_clo(JOB_ONE) == clo
    assert result.job_description == source
    assert result.prompt_job_description == prompt
    assert result.cover_letter_pdf == b"synthetic-cover-pdf-updated"
    assert result.resume_html == "<main>Synthetic résumé</main>"
    assert result.resume_pdf == b"synthetic-resume-pdf"
    assert (
        result.ats.score,
        result.ats.parsing_score,
        result.ats.keyword_score,
        result.ats.semantic_score,
        result.ats.formatting_risk,
        result.ats.missing_terms,
        result.ats.updated_at,
    ) == (
        ats.score,
        ats.parsing_score,
        ats.keyword_score,
        ats.semantic_score,
        ats.formatting_risk,
        ats.missing_terms,
        ats.updated_at,
    )
    assert result.ats.diagnostics["checks"] == (True, False)


def test_metadata_refresh_preserves_dedicated_state(tmp_path: Path) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.store_jod(JOB_ONE, source_text="source", prompt_text="prompt")
    store.store_aro(JOB_ONE, yaml_text=ARO_V1)
    store.store_clo(JOB_ONE, value={"letter": "synthetic"}, pdf_content=b"cover")
    store.store_application_artifacts(
        JOB_ONE,
        resume_html="<main>one</main>",
        resume_pdf=b"resume",
        ats=AtsFields(score=77),
    )
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.update_application_status(
        JOB_ONE,
        applied_to="Yes",
        date_applied="2035-02-04",
        notes="Synthetic note",
    )
    store.archive([JOB_ONE])
    before = store.get_application(JOB_ONE)

    after = store.upsert_application(
        _metadata(
            JOB_ONE,
            company="Example Updated Cooperative",
            title="Updated Synthetic Analyst",
        )
    )

    assert after.company == "Example Updated Cooperative"
    assert after.job_title == "Updated Synthetic Analyst"
    for field in (
        "job_description",
        "prompt_job_description",
        "application_resume",
        "application_resume_backup",
        "selected_resume_variant",
        "resume_variant_selection_mode",
        "selected_variant",
        "cover_letter",
        "cover_letter_filename",
        "cover_letter_pdf",
        "source_cover_letter_path",
        "applied_to",
        "date_applied",
        "notes",
        "archived_at",
    ):
        assert getattr(after, field) == getattr(before, field)


def test_application_status_is_exact_and_atomic(tmp_path: Path) -> None:
    store, _ = _initialize_with_application(tmp_path)
    for status in sorted(APPLICATION_STATUSES):
        result = store.update_application_status(
            JOB_ONE,
            applied_to=status,
            date_applied="2035-02-05",
            notes=f"Synthetic status: {status}",
        )
        assert result.applied_to == status
        assert result.notes == f"Synthetic status: {status}"
    before = store.get_application(JOB_ONE)

    for invalid in ("yes", "Pending", "", 1, None):
        with pytest.raises(ApplicationStateValidationError):
            store.update_application_status(JOB_ONE, applied_to=invalid)  # type: ignore[arg-type]
    assert store.get_application(JOB_ONE) == before


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://127.0.0.1/opening",
        "http://2130706433/opening",
        "http://127\u30020\u30020\u30021/opening",
        "https://example.com%2f.evil.test/opening",
        "https://example.com:\x01/opening",
        "https://example.com:99999/opening",
    ],
)
def test_metadata_reuses_public_url_safety_without_networking(
    tmp_path: Path,
    unsafe_url: str,
) -> None:
    store, _ = _store(tmp_path)
    store.initialize()

    with pytest.raises(ApplicationStateValidationError) as caught:
        store.upsert_application(_metadata(url=unsafe_url))

    _assert_content_free(caught.value, forbidden=(unsafe_url,))
    assert store.list_applications("all") == ()


def test_auto_variant_preference_and_complete_projection(tmp_path: Path) -> None:
    store, _ = _initialize_with_application(tmp_path)

    for key, parent in (("v1", None), ("v2", "v1"), ("manual", "v2")):
        written = store.upsert_resume_variant(
            JOB_ONE,
            _variant(key, parent=parent),
        )
        application = store.get_application(JOB_ONE)
        assert application.selected_resume_variant == key
        assert application.resume_variant_selection_mode == "auto"
        assert application.selected_variant == written
        assert application.selected_variant is not None
        assert application.selected_variant.resume_html == f"<main>{key}</main>"
        assert application.selected_variant.resume_pdf == f"pdf-{key}".encode()
        assert application.selected_variant.ats == written.ats
        assert application.selected_variant.ats_diagnostics["source"] == key
        assert application.selected_variant.evidence_packet["variant"] == key
        assert application.selected_variant.external_critique["summary"] == (
            f"critique-{key}"
        )
        assert application.selected_variant.critique == {"rating": 4}
        assert application.selected_variant.validation == {"valid": True}
        assert application.selected_variant.model_metadata == {
            "model": "synthetic-local",
            "temperature": 0,
        }

    assert [item.variant_key for item in store.list_resume_variants(JOB_ONE)] == [
        "v1",
        "v2",
        "manual",
    ]


@pytest.mark.parametrize("pinned", ["v1", "v2"])
def test_explicit_pin_survives_manual_upserts_and_initialize(
    tmp_path: Path,
    pinned: str,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    v1 = store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    v2 = store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    expected = {"v1": v1, "v2": v2}[pinned]

    selected = store.select_resume_variant(JOB_ONE, pinned)
    store.upsert_resume_variant(JOB_ONE, _variant("manual", parent="v2"))
    store.upsert_resume_variant(
        JOB_ONE,
        _variant("manual", parent="v2", marker="manual-updated"),
    )
    store.initialize()
    final = store.get_application(JOB_ONE)

    assert selected.selected_resume_variant == pinned
    assert selected.resume_variant_selection_mode == "manual"
    assert final.selected_resume_variant == pinned
    assert final.resume_variant_selection_mode == "manual"
    assert final.selected_variant == expected


def test_updating_selected_variant_refreshes_only_selected_projection(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    store.select_resume_variant(JOB_ONE, "v1")
    old_v2 = store.get_resume_variant(JOB_ONE, "v2")

    updated = store.upsert_resume_variant(
        JOB_ONE,
        _variant("v1", marker="v1-updated"),
    )
    application = store.get_application(JOB_ONE)

    assert application.selected_resume_variant == "v1"
    assert application.resume_variant_selection_mode == "manual"
    assert application.selected_variant == updated
    assert application.selected_variant.resume_pdf == b"pdf-v1-updated"
    assert store.get_resume_variant(JOB_ONE, "v2") == old_v2


def test_direct_resume_projection_writes_cannot_diverge_selected_variant(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    selected = store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    before = store.get_application(JOB_ONE)

    invalid_writes: tuple[Callable[[], object], ...] = (
        lambda: store.store_aro(JOB_ONE, yaml_text=ARO_V2),
        lambda: store.store_application_artifacts(
            JOB_ONE,
            resume_html="<main>divergent</main>",
        ),
        lambda: store.store_application_artifacts(
            JOB_ONE,
            resume_pdf=b"divergent",
        ),
        lambda: store.store_application_artifacts(
            JOB_ONE,
            ats=AtsFields(score=1),
        ),
    )
    for operation in invalid_writes:
        with pytest.raises(ApplicationStateValidationError):
            operation()
        assert store.get_application(JOB_ONE) == before

    store.store_jod(JOB_ONE, source_text="updated source", prompt_text="updated")
    store.store_clo(JOB_ONE, value={"letter": "updated"})
    after_cover = store.store_application_artifacts(
        JOB_ONE,
        cover_letter_pdf=b"independent-cover",
    )
    assert after_cover.selected_variant == selected
    assert after_cover.application_resume == selected.application_resume
    assert after_cover.resume_html == selected.resume_html
    assert after_cover.resume_pdf == selected.resume_pdf
    assert after_cover.ats == selected.ats
    assert after_cover.cover_letter_pdf == b"independent-cover"


def test_reset_to_auto_selects_highest_available_variant(tmp_path: Path) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("manual", parent="v2"))
    store.select_resume_variant(JOB_ONE, "v1")

    result = store.reset_resume_variant_selection(JOB_ONE)

    assert result.selected_resume_variant == "manual"
    assert result.resume_variant_selection_mode == "auto"
    assert result.selected_variant == store.get_resume_variant(JOB_ONE, "manual")


@pytest.mark.parametrize(
    ("key", "parent"),
    [
        ("v1", "v1"),
        ("v1", "v2"),
        ("v2", None),
        ("v2", "v2"),
        ("v2", "manual"),
        ("manual", None),
        ("manual", "manual"),
    ],
)
def test_variant_parent_provenance_rejects_invalid_or_missing_without_write(
    tmp_path: Path,
    key: str,
    parent: str | None,
) -> None:
    store, _ = _initialize_with_application(tmp_path)

    with pytest.raises(ApplicationStateValidationError):
        store.upsert_resume_variant(JOB_ONE, _variant(key, parent=parent))

    assert store.list_resume_variants(JOB_ONE) == ()
    assert store.get_application(JOB_ONE).selected_resume_variant is None


def test_parent_provenance_is_same_job_and_accepts_allowed_chain(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.upsert_application(_metadata(JOB_TWO))
    store.upsert_resume_variant(JOB_TWO, _variant("v1"))

    with pytest.raises(ApplicationStateValidationError):
        store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    assert store.list_resume_variants(JOB_ONE) == ()

    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("manual", parent="v1"))
    assert [
        item.parent_variant_key for item in store.list_resume_variants(JOB_ONE)
    ] == [
        None,
        "v1",
        "v1",
    ]


def test_missing_and_cross_job_selection_roll_back_exactly(tmp_path: Path) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.upsert_application(_metadata(JOB_TWO))
    v1 = store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    before_one = store.get_application(JOB_ONE)
    before_two = store.get_application(JOB_TWO)

    with pytest.raises(ApplicationStateNotFoundError):
        store.select_resume_variant(JOB_TWO, "v1")
    with pytest.raises(ApplicationStateNotFoundError):
        store.select_resume_variant(JOB_ONE, "v2")
    with pytest.raises(ApplicationStateNotFoundError):
        store.select_resume_variant("not-present", "v1")

    assert store.get_application(JOB_ONE) == before_one
    assert store.get_application(JOB_TWO) == before_two
    assert store.get_resume_variant(JOB_ONE, "v1") == v1


def test_delete_removes_children_for_fresh_schema(tmp_path: Path) -> None:
    store, database = _initialize_with_application(tmp_path)
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))

    assert store.delete([JOB_ONE]) == 1

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM application_resume_variants"
            ).fetchone()[0]
            == 0
        )


def test_complete_cutoff_shape_supports_metadata_first_variant_and_delete(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_complete_cutoff(database)
    store.initialize()

    empty = store.get_application("cutoff-empty")
    assert empty.job_url == "https://cutoff.example.com/jobs/empty"
    assert empty.selected_resume_variant is None
    inserted = store.upsert_application(_metadata(JOB_ONE))
    assert inserted.selected_resume_variant is None
    updated = store.upsert_application(
        _metadata(JOB_ONE, company="Example Updated Cutoff Cooperative")
    )
    assert updated.company == "Example Updated Cutoff Cooperative"
    assert updated.selected_resume_variant is None
    with sqlite3.connect(database) as connection:
        compatibility_insert = connection.execute(
            """
            SELECT linkedin_url, resume_filename, source_resume_path,
                   selected_resume_variant
            FROM applications WHERE job_id = ?
            """,
            (JOB_ONE,),
        ).fetchone()
    assert compatibility_insert == (
        f"https://jobs.example.com/openings/{JOB_ONE}",
        "",
        "",
        "",
    )
    first = store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    assert store.get_application(JOB_ONE).selected_variant == first
    assert store.archive([JOB_ONE]) == 1
    assert store.unarchive([JOB_ONE]) == 1
    assert store.delete([JOB_ONE]) == 1

    with sqlite3.connect(database) as connection:
        mirrored = connection.execute(
            """
            SELECT linkedin_url, resume_filename, source_resume_path,
                   selected_resume_variant
            FROM applications WHERE job_id = ?
            """,
            ("cutoff-empty",),
        ).fetchone()
        orphans = connection.execute(
            """
            SELECT COUNT(*) FROM application_resume_variants AS v
            LEFT JOIN applications AS a ON a.job_id = v.job_id
            WHERE a.job_id IS NULL
            """
        ).fetchone()[0]
    assert mirrored == (
        "https://cutoff.example.com/jobs/empty",
        "",
        "",
        "",
    )
    assert orphans == 0


def test_legacy_yaml_cover_letter_mapping_remains_readable(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_complete_cutoff(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE applications
            SET cover_letter_object = ?
            WHERE job_id = 'cutoff-empty'
            """,
            ("body:\n  greeting: Synthetic hello\nmetadata:\n  reviewed: true\n",),
        )

    store.initialize()
    record = store.get_application("cutoff-empty")

    assert record.cover_letter is not None
    assert record.cover_letter["body"]["greeting"] == "Synthetic hello"
    assert record.cover_letter["metadata"]["reviewed"] is True


@pytest.mark.parametrize("defect", ["orphan", "missing_parent"])
def test_legacy_orphan_or_missing_parent_variant_rolls_back_initialization(
    tmp_path: Path,
    defect: str,
) -> None:
    store, database = _store(tmp_path)
    _create_complete_cutoff(database)
    if defect == "orphan":
        job_id, key, parent = "orphan-example-001", "v1", None
    else:
        job_id, key, parent = "cutoff-empty", "v2", "v1"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO application_resume_variants (
                job_id, variant_key, variant_label, source,
                parent_variant_key, application_resume_object,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'synthetic_fixture', ?, ?, ?, ?)
            """,
            (
                job_id,
                key,
                "Synthetic legacy variant",
                parent,
                ARO_V2,
                "2033-01-01T00:00:00+00:00",
                "2033-01-01T00:00:00+00:00",
            ),
        )
    before = _schema_snapshot(database)

    with pytest.raises(ApplicationStateInitializationError) as caught:
        store.initialize()

    _assert_content_free(caught.value, forbidden=(job_id, str(database)))
    assert _schema_snapshot(database) == before


def test_legacy_variant_table_without_foreign_key_is_explicitly_cleaned(
    tmp_path: Path,
) -> None:
    store, database = _store(tmp_path)
    _create_complete_cutoff(database)
    store.initialize()
    store.upsert_application(_metadata(JOB_ONE))
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "PRAGMA foreign_key_list(application_resume_variants)"
            ).fetchall()
            == []
        )

    store.delete([JOB_ONE])
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                """
                SELECT COUNT(*) FROM application_resume_variants
                WHERE job_id = ?
                """,
                (JOB_ONE,),
            ).fetchone()[0]
            == 0
        )


def test_json_yaml_binary_and_path_like_validation_is_inert_and_atomic(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    before = store.get_application(JOB_ONE)
    recursive: dict[str, Any] = {}
    recursive["self"] = recursive
    active = ActiveValue()

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: store.store_aro(JOB_ONE, yaml_text="not-a-mapping"),
        lambda: store.store_aro(
            JOB_ONE,
            yaml_text="value: !!python/object:example.ActiveValue {}",
        ),
        lambda: store.store_aro(JOB_ONE, yaml_text="x: " + "a" * MAX_ARO_YAML_BYTES),
        lambda: store.store_clo(JOB_ONE, value=recursive),
        lambda: store.store_clo(JOB_ONE, value={"active": active}),
        lambda: store.store_clo(  # type: ignore[arg-type]
            JOB_ONE,
            value={"safe": True},
            source_path=Path("provenance.pdf"),
        ),
        lambda: store.store_application_artifacts(
            JOB_ONE,
            resume_pdf=bytearray(b"not-bytes"),  # type: ignore[arg-type]
        ),
        lambda: store.store_application_artifacts(
            JOB_ONE,
            ats=AtsFields(score=101),
        ),
    )
    for operation in invalid_calls:
        with pytest.raises(ApplicationStateValidationError) as caught:
            operation()
        _assert_content_free(caught.value, forbidden=(JOB_ONE, "ActiveValue"))
        assert store.get_application(JOB_ONE) == before
    assert not active.touched


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("aro_yaml", "not: [valid"),
        ("aro_yaml", "x: " + "a" * MAX_ARO_YAML_BYTES),
        ("cover_letter_object", "{not-json"),
        ("cover_letter_object", json.dumps({"x": "a" * MAX_JSON_CHARS})),
    ],
)
def test_malformed_or_oversized_stored_application_data_is_corruption(
    tmp_path: Path,
    column: str,
    value: str,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"UPDATE applications SET {column} = ? WHERE job_id = ?",
            (value, JOB_ONE),
        )
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ApplicationStateCorruptionError) as caught:
        store.get_application(JOB_ONE)

    _assert_content_free(caught.value, forbidden=(JOB_ONE, value[:20], str(database)))
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("company", "x" * 1_025),
        ("notes", "x" * 100_001),
        ("resume_content", b"x" * (20_000_000 + 1)),
    ],
)
def test_oversized_stored_scalar_or_blob_is_corruption(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"UPDATE applications SET {column} = ? WHERE job_id = ?",
            (value, JOB_ONE),
        )

    with pytest.raises(ApplicationStateCorruptionError) as caught:
        store.get_application(JOB_ONE)

    _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database)))


def test_returning_mutation_rolls_back_if_stored_projection_is_corrupt(
    tmp_path: Path,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE applications
            SET cover_letter_object = ?
            WHERE job_id = ?
            """,
            ("{malformed", JOB_ONE),
        )
    before = _schema_snapshot(database)

    with pytest.raises(ApplicationStateCorruptionError) as caught:
        store.update_application_status(
            JOB_ONE,
            applied_to="Rejected",
            notes="Synthetic status must roll back",
        )

    _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database)))
    assert _schema_snapshot(database) == before


def test_malformed_stored_variant_metadata_is_corruption(tmp_path: Path) -> None:
    store, database = _initialize_with_application(tmp_path)
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE application_resume_variants
            SET evidence_packet_json = ?
            WHERE job_id = ? AND variant_key = 'v1'
            """,
            ("[not-a-mapping]", JOB_ONE),
        )

    with pytest.raises(ApplicationStateCorruptionError) as caught:
        store.get_resume_variant(JOB_ONE, "v1")

    _assert_content_free(caught.value, forbidden=(JOB_ONE, "not-a-mapping"))


def test_strict_scope_variant_key_selection_mode_and_limits_reject_atomically(
    tmp_path: Path,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    before = _schema_snapshot(database)

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: store.list_applications("unknown"),
        lambda: store.list_applications(limit=0),
        lambda: store.list_applications(limit=MAX_QUERY_RESULTS + 1),
        lambda: store.fetch_job_records([JOB_ONE] * 2),
        lambda: store.fetch_job_records(
            [f"job-{index}" for index in range(MAX_BULK_IDENTIFIERS + 1)]
        ),
        lambda: store.get_resume_variant(JOB_ONE, "draft"),
        lambda: ApplicationStateStore(
            WorkspacePaths(database=database),
            busy_timeout_seconds=0,
        ),
        lambda: ApplicationStateStore(
            WorkspacePaths(database=database),
            busy_timeout_seconds=float("inf"),
        ),
    )
    for operation in invalid_calls:
        with pytest.raises(ApplicationStateValidationError):
            operation()
        assert _schema_snapshot(database) == before

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE applications
            SET resume_variant_selection_mode = 'unexpected'
            WHERE job_id = ?
            """,
            (JOB_ONE,),
        )
    with pytest.raises(ApplicationStateCorruptionError):
        store.get_application(JOB_ONE)


def test_timeout_validation_contains_huge_and_unsupported_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    database = tmp_path / "missing" / "applications.sqlite3"
    output = tmp_path / "future-output"
    hostile = HostileCapability()
    invalid_values: tuple[object, ...] = (
        10**400,
        -(10**400),
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        False,
        hostile,
    )

    for value in invalid_values:
        with pytest.raises(ApplicationStateValidationError) as caught:
            ApplicationStateStore(
                WorkspacePaths(database=database, output_dir=output),
                busy_timeout_seconds=value,  # type: ignore[arg-type]
            )
        _assert_exact_public_error(
            caught.value,
            ApplicationStateValidationError,
            "Application state input is invalid.",
            forbidden=(str(database),),
        )

    assert object.__getattribute__(hostile, "calls") == []
    assert tuple(tmp_path.iterdir()) == ()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_scope_and_artifact_reject_capabilities_without_invocation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    class DerivedString(str):
        pass

    database = tmp_path / "missing" / "applications.sqlite3"
    output = tmp_path / "future-output"
    store = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=lambda: FIXED_TIME,
    )
    hostile_scope = HostileCapability()
    hostile_kind = HostileCapability()
    operations: tuple[Callable[[], object], ...] = (
        lambda: store.list_applications(hostile_scope),  # type: ignore[arg-type]
        lambda: artifact_filename(JOB_ONE, hostile_kind),  # type: ignore[arg-type]
        lambda: store.list_applications(DerivedString("active")),
        lambda: artifact_filename(JOB_ONE, DerivedString("resume_pdf")),
    )

    for operation in operations:
        with pytest.raises(ApplicationStateValidationError) as caught:
            operation()
        _assert_exact_public_error(
            caught.value,
            ApplicationStateValidationError,
            "Application state input is invalid.",
            forbidden=(str(database),),
        )

    assert object.__getattribute__(hostile_scope, "calls") == []
    assert object.__getattribute__(hostile_kind, "calls") == []
    assert tuple(tmp_path.iterdir()) == ()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_clock_validation_contains_callback_timezone_and_conversion_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    hostile_timezone = HostileTimezone()
    two_step_timezone = TwoStepTimezone()
    HostileDatetime.calls = []
    hostile_datetime = HostileDatetime(2035, 2, 3, 4, 5, 6, tzinfo=UTC)

    def raising_clock() -> datetime:
        raise RuntimeError("synthetic clock callback detail")

    clocks: tuple[Callable[[], object], ...] = (
        raising_clock,
        lambda: object(),
        lambda: hostile_datetime,
        lambda: datetime(2035, 2, 3, 4, 5, 6),  # noqa: DTZ001 - invalid fixture.
        lambda: datetime(
            2035,
            2,
            3,
            4,
            5,
            6,
            tzinfo=timezone(timedelta(hours=1)),
        ),
        lambda: datetime(2035, 2, 3, 4, 5, 6, tzinfo=hostile_timezone),
        lambda: datetime(2035, 2, 3, 4, 5, 6, tzinfo=two_step_timezone),
    )

    for index, clock in enumerate(clocks):
        database = tmp_path / f"missing-{index}" / "applications.sqlite3"
        output = tmp_path / f"future-output-{index}"
        store = ApplicationStateStore(
            WorkspacePaths(database=database, output_dir=output),
            utc_clock=clock,  # type: ignore[arg-type]
        )
        with pytest.raises(ApplicationStateValidationError) as caught:
            store.upsert_application(_required_metadata())
        _assert_exact_public_error(
            caught.value,
            ApplicationStateValidationError,
            "Application state input is invalid.",
            forbidden=(
                "synthetic clock callback detail",
                "synthetic hostile timezone detail",
                "synthetic conversion detail",
                str(database),
            ),
        )

    assert HostileDatetime.calls == []
    assert hostile_timezone.calls == 1
    assert two_step_timezone.calls >= 2
    assert tuple(tmp_path.iterdir()) == ()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_nul_parent_initialization_is_sanitized_without_partial_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    database = tmp_path / "synthetic\x00parent" / "applications.sqlite3"
    output = tmp_path / "future-output"
    store = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=lambda: FIXED_TIME,
    )

    with pytest.raises(ApplicationStateInitializationError) as caught:
        store.initialize()

    _assert_exact_public_error(
        caught.value,
        ApplicationStateInitializationError,
        "Application state initialization failed.",
        forbidden=(str(database),),
    )
    assert tuple(tmp_path.iterdir()) == ()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


@pytest.mark.parametrize(
    "failure_type",
    [OSError, ValueError, RuntimeError, RecursionError],
)
def test_parent_directory_failures_are_sanitized_without_partial_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    failure_type: type[Exception],
) -> None:
    caplog.set_level(logging.DEBUG)
    database = tmp_path / "missing-parent" / "applications.sqlite3"
    output = tmp_path / "future-output"
    store = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=lambda: FIXED_TIME,
    )

    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise failure_type("synthetic parent creation detail")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "mkdir", fail_mkdir)
        with pytest.raises(ApplicationStateInitializationError) as caught:
            store.initialize()

    _assert_exact_public_error(
        caught.value,
        ApplicationStateInitializationError,
        "Application state initialization failed.",
        forbidden=("synthetic parent creation detail", str(database)),
    )
    assert tuple(tmp_path.iterdir()) == ()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_initializer_failure_after_first_statement_rolls_back_schema_version(
    tmp_path: Path,
) -> None:
    def fail(checkpoint: str) -> None:
        if checkpoint == "initialize_after_first_statement":
            raise RuntimeError("synthetic-sensitive-initialization-detail")

    store, database = _store(tmp_path, failure_injector=fail)

    with pytest.raises(ApplicationStateError) as caught:
        store.initialize()

    _assert_content_free(
        caught.value,
        forbidden=("synthetic-sensitive-initialization-detail", str(database)),
    )
    with sqlite3.connect(database) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 0
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            ).fetchall()
            == []
        )
    assert not database.with_name(f"{database.name}-journal").exists()


def test_post_user_version_failure_rolls_back_fresh_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def fail(checkpoint: str) -> None:
        if checkpoint == "initialize_after_user_version":
            raise RuntimeError("synthetic post-version detail")

    store, database = _store(tmp_path, failure_injector=fail)

    with pytest.raises(ApplicationStateInitializationError) as caught:
        store.initialize()

    _assert_exact_public_error(
        caught.value,
        ApplicationStateInitializationError,
        "Application state initialization failed.",
        forbidden=("synthetic post-version detail", str(database)),
    )
    with sqlite3.connect(database) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 0
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            ).fetchall()
            == []
        )
    assert not database.with_name(f"{database.name}-journal").exists()

    recovered = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=lambda: FIXED_TIME,
    )
    recovered.initialize()
    with sqlite3.connect(database) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 1
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_post_user_version_failure_restores_legacy_database_exactly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    store, database = _store(tmp_path)
    _create_complete_cutoff(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE applications
            SET resume_filename = ?,
                resume_content = ?,
                source_resume_path = ?,
                resume_updated_at = ?
            WHERE job_id = 'cutoff-empty'
            """,
            (
                "synthetic-cutoff.pdf",
                b"synthetic-cutoff-pdf-bytes",
                "provenance/synthetic-cutoff.pdf",
                "2033-01-02T03:04:05+00:00",
            ),
        )
    before_digest = _database_digest(database)
    before_snapshot = _schema_snapshot(database)

    def fail(checkpoint: str) -> None:
        if checkpoint == "initialize_after_user_version":
            raise RuntimeError("synthetic legacy post-version detail")

    failing = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=lambda: FIXED_TIME,
        failure_injector=fail,
    )
    with pytest.raises(ApplicationStateInitializationError) as caught:
        failing.initialize()

    _assert_exact_public_error(
        caught.value,
        ApplicationStateInitializationError,
        "Application state initialization failed.",
        forbidden=("synthetic legacy post-version detail", str(database)),
    )
    assert _database_digest(database) == before_digest
    assert _schema_snapshot(database) == before_snapshot
    assert not database.with_name(f"{database.name}-journal").exists()

    store.initialize()
    migrated = store.get_application("cutoff-empty")
    assert migrated.resume_pdf == b"synthetic-cutoff-pdf-bytes"
    assert migrated.resume_pdf_updated_at == "2033-01-02T03:04:05+00:00"
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_variant_failure_after_upsert_rolls_back_variant_and_projection(
    tmp_path: Path,
) -> None:
    base, database = _initialize_with_application(tmp_path)
    first = base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    before = base.get_application(JOB_ONE)

    def fail(checkpoint: str) -> None:
        if checkpoint == "variant_after_upsert":
            raise RuntimeError("synthetic-sensitive-variant-detail")

    failing = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=TickingClock(FIXED_TIME + timedelta(days=1)),
        failure_injector=fail,
    )
    with pytest.raises(ApplicationStateError) as caught:
        failing.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))

    _assert_content_free(
        caught.value,
        forbidden=(JOB_ONE, "synthetic-sensitive-variant-detail", str(database)),
    )
    assert base.get_application(JOB_ONE) == before
    assert base.get_resume_variant(JOB_ONE, "v1") == first
    with pytest.raises(ApplicationStateNotFoundError):
        base.get_resume_variant(JOB_ONE, "v2")


def test_two_store_instances_are_operation_local_and_sequentially_consistent(
    tmp_path: Path,
) -> None:
    first, database = _initialize_with_application(tmp_path)
    second = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=TickingClock(FIXED_TIME + timedelta(days=1)),
    )

    first.upsert_resume_variant(JOB_ONE, _variant("v1"))
    second.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    selected = first.select_resume_variant(JOB_ONE, "v1")
    observed = second.get_application(JOB_ONE)

    assert selected == observed
    assert observed.selected_resume_variant == "v1"
    assert observed.resume_variant_selection_mode == "manual"


def test_event_gated_same_job_upsert_cannot_clobber_explicit_pin(
    tmp_path: Path,
) -> None:
    base, database = _initialize_with_application(tmp_path)
    v1 = base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    base.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    variant_validated = threading.Event()
    allow_variant_transaction = threading.Event()
    worker_done = threading.Event()
    worker_errors: list[BaseException] = []

    def hook(checkpoint: str) -> None:
        if checkpoint == "variant_validated":
            variant_validated.set()
            if not allow_variant_transaction.wait(timeout=3):
                raise RuntimeError("bounded synchronization failed")

    upserter = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=TickingClock(FIXED_TIME + timedelta(days=1)),
        operation_hook=hook,
    )
    selector = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=TickingClock(FIXED_TIME + timedelta(days=2)),
    )

    def run_upsert() -> None:
        try:
            upserter.upsert_resume_variant(
                JOB_ONE,
                _variant("manual", parent="v2"),
            )
        except BaseException as error:  # noqa: BLE001 - capture thread failures.
            worker_errors.append(error)
        finally:
            worker_done.set()

    thread = threading.Thread(target=run_upsert, name="synthetic-variant-upsert")
    thread.start()
    assert variant_validated.wait(timeout=3)
    pinned = selector.select_resume_variant(JOB_ONE, "v1")
    allow_variant_transaction.set()
    assert worker_done.wait(timeout=3)
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert worker_errors == []
    final = base.get_application(JOB_ONE)
    assert pinned.selected_resume_variant == "v1"
    assert final.selected_resume_variant == "v1"
    assert final.resume_variant_selection_mode == "manual"
    assert final.selected_variant == v1
    assert final.selected_variant is not None
    assert final.selected_variant.resume_html == "<main>v1</main>"
    assert final.selected_variant.resume_pdf == b"pdf-v1"
    assert final.selected_variant.ats == v1.ats
    assert final.selected_variant.updated_at == v1.updated_at
    assert final.selected_variant.model_metadata == v1.model_metadata


def test_wal_application_read_is_one_coherent_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    base, database = _initialize_with_application(tmp_path)
    base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    with sqlite3.connect(database) as connection:
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    assert str(mode).casefold() == "wal"
    output = tmp_path / "future-output"

    for iteration in range(3):
        expected_old = base.get_application(JOB_ONE)
        expected_old_variant = expected_old.selected_variant
        assert expected_old_variant is not None
        _assert_projection_matches_selected_variant(expected_old)
        row_fetched = threading.Event()
        release_reader = threading.Event()
        reader_done = threading.Event()
        reader_results: list[object] = []
        reader_errors: list[BaseException] = []

        def hook(checkpoint: str) -> None:
            if checkpoint == "application_row_fetched":
                row_fetched.set()  # noqa: B023 - joined before the next iteration.
                if not release_reader.wait(  # noqa: B023 - same bounded iteration.
                    timeout=3
                ):
                    raise RuntimeError("bounded snapshot synchronization failed")

        reader = ApplicationStateStore(
            WorkspacePaths(database=database, output_dir=output),
            utc_clock=TickingClock(FIXED_TIME + timedelta(days=10 + iteration)),
            operation_hook=hook,
        )
        writer = ApplicationStateStore(
            WorkspacePaths(database=database, output_dir=output),
            utc_clock=TickingClock(FIXED_TIME + timedelta(days=20 + iteration)),
        )

        def run_read() -> None:
            try:
                reader_results.append(  # noqa: B023 - same bounded iteration.
                    reader.get_application(JOB_ONE)  # noqa: B023
                )
            except BaseException as error:  # noqa: BLE001 - capture thread failures.
                reader_errors.append(error)  # noqa: B023 - same bounded iteration.
            finally:
                reader_done.set()  # noqa: B023 - same bounded iteration.

        thread = threading.Thread(
            target=run_read,
            name=f"synthetic-snapshot-reader-{iteration}",
        )
        thread.start()
        try:
            assert row_fetched.wait(timeout=3)
            marker = f"snapshot-generation-{iteration}"
            new_ats = AtsFields(
                score=95 + iteration,
                parsing_score=94 + iteration,
                keyword_score=93 + iteration,
                semantic_score=92 + iteration,
                formatting_risk="none",
                missing_terms="",
                diagnostics={"source": marker, "checks": [iteration]},
                updated_at=f"2035-02-0{4 + iteration}T00:00:00+00:00",
            )
            replacement = replace(
                _variant(
                    "v1",
                    yaml_text=(
                        "profile:\n"
                        f"  summary: Synthetic snapshot {iteration}\n"
                        "skills:\n"
                        "  - Snapshot testing\n"
                    ),
                    marker=marker,
                ),
                ats=new_ats,
                critique={"rating": iteration},
                validation={"valid": True, "generation": iteration},
                model_metadata={
                    "model": "synthetic-local",
                    "generation": iteration,
                },
            )
            updated_variant = writer.upsert_resume_variant(JOB_ONE, replacement)
        finally:
            release_reader.set()
            thread.join(timeout=3)

        assert not thread.is_alive()
        assert reader_done.is_set()
        assert reader_errors == []
        assert len(reader_results) == 1
        observed = reader_results[0]
        assert observed == expected_old
        assert observed.selected_variant == expected_old_variant
        _assert_projection_matches_selected_variant(observed)

        current = writer.get_application(JOB_ONE)
        assert current.selected_variant == updated_variant
        _assert_projection_matches_selected_variant(current)
        assert current.application_resume != expected_old.application_resume
        assert current.resume_html != expected_old.resume_html
        assert current.resume_pdf != expected_old.resume_pdf
        assert current.ats != expected_old.ats
        assert current.resume_html_updated_at != expected_old.resume_html_updated_at
        assert current.resume_pdf_updated_at != expected_old.resume_pdf_updated_at
        assert (
            current.selected_variant.evidence_packet
            != expected_old_variant.evidence_packet
        )
        assert (
            current.selected_variant.external_critique
            != expected_old_variant.external_critique
        )
        assert current.selected_variant.critique != expected_old_variant.critique
        assert current.selected_variant.validation != expected_old_variant.validation
        assert (
            current.selected_variant.model_metadata
            != expected_old_variant.model_metadata
        )
        assert current.selected_variant.updated_at != expected_old_variant.updated_at
        base = writer
        with sqlite3.connect(database) as connection:
            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
        assert checkpoint[0] == 0

    assert not output.exists()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_failed_snapshot_read_rolls_back_and_releases_connection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    base, database = _initialize_with_application(tmp_path)
    base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode = WAL")

    def fail(checkpoint: str) -> None:
        if checkpoint == "application_row_fetched":
            raise RuntimeError("synthetic snapshot hook detail")

    output = tmp_path / "future-output"
    failing = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=lambda: FIXED_TIME,
        operation_hook=fail,
    )
    with pytest.raises(ApplicationStateError) as caught:
        failing.get_application(JOB_ONE)

    _assert_exact_public_error(
        caught.value,
        ApplicationStateError,
        "Application state operation failed.",
        forbidden=("synthetic snapshot hook detail", JOB_ONE, str(database)),
    )
    refreshed = base.upsert_application(
        ApplicationMetadata(
            job_id=JOB_ONE,
            company="Example Post-Failure Cooperative",
            job_title="Synthetic Post-Failure Analyst",
            job_url=f"https://jobs.example.com/post-failure/{JOB_ONE}",
            source="public_after_failure",
        )
    )
    assert refreshed.company == "Example Post-Failure Cooperative"
    with sqlite3.connect(database) as connection:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    assert checkpoint[0] == 0
    assert not output.exists()
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_finite_lock_timeout_is_atomic_and_content_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    constrained = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=lambda: FIXED_TIME,
        busy_timeout_seconds=0.01,
    )
    lock = sqlite3.connect(database, timeout=0, isolation_level=None)
    lock.execute("BEGIN EXCLUSIVE")
    caplog.set_level(logging.DEBUG)
    try:
        with pytest.raises(ApplicationStateBusyError) as caught:
            constrained.update_application_status(
                JOB_ONE,
                applied_to="Rejected",
                notes="must roll back",
            )
    finally:
        lock.rollback()
        lock.close()

    _assert_content_free(caught.value, forbidden=(JOB_ONE, str(database), "locked"))
    assert store.get_application(JOB_ONE).applied_to == "No"
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_generic_artifact_filenames_are_safe_deterministic_and_neutral() -> None:
    names = {
        kind: artifact_filename(JOB_ONE, kind)
        for kind in (
            ArtifactKind.RESUME_HTML,
            ArtifactKind.RESUME_PDF,
            ArtifactKind.COVER_LETTER_PDF,
        )
    }

    assert names == {kind: artifact_filename(JOB_ONE, kind.value) for kind in names}
    assert names[ArtifactKind.RESUME_HTML].endswith(".resume.html")
    assert names[ArtifactKind.RESUME_PDF].endswith(".resume.pdf")
    assert names[ArtifactKind.COVER_LETTER_PDF].endswith(".cover-letter.pdf")
    assert len(set(names.values())) == 3
    for name in names.values():
        assert len(name) < 80
        assert "/" not in name
        assert "\\" not in name
        assert ".." not in name
        assert JOB_ONE not in name
        assert "example" not in name.casefold()
        assert "analyst" not in name.casefold()
        assert "legacy" not in name.casefold()
        assert Path(name).name == name

    for invalid in ("../job", "/job", ".", "..", "job/name", "job name", ""):
        with pytest.raises(ApplicationStateValidationError):
            artifact_filename(invalid, ArtifactKind.RESUME_PDF)
    with pytest.raises(ApplicationStateValidationError):
        artifact_filename(JOB_ONE, "resume.docx")


def test_workflow_surface_is_immutable_and_preserves_human_review() -> None:
    assert {
        "ApplicationStateConflictError",
        "ApplicationWorkflowRevision",
        "ApplicationWorkflowSnapshot",
    }.issubset(application_state_module.__all__)
    assert get_args(ApplicationWorkflowStage) == (
        "not_started",
        "draft_ready",
        "awaiting_user_review",
        "approved_for_submission",
        "submitted",
    )
    assert DEFAULT_APPLICATION_POLICY == ApplicationWorkflowPolicy()
    assert DEFAULT_APPLICATION_POLICY.require_user_approval_before_submit is True
    assert DEFAULT_APPLICATION_POLICY.record_audit_events is True
    with pytest.raises(FrozenInstanceError):
        DEFAULT_APPLICATION_POLICY.require_user_approval_before_submit = False  # type: ignore[misc]


def test_workflow_snapshot_is_coherent_canonical_immutable_and_content_hidden(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    v1 = store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    v2 = store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))

    snapshot = store.get_workflow_snapshot(JOB_ONE)

    assert type(snapshot) is ApplicationWorkflowSnapshot
    assert snapshot.application == store.get_application(JOB_ONE)
    assert snapshot.variants == (v1, v2)
    assert type(snapshot.variants) is tuple
    assert type(snapshot.revision) is ApplicationWorkflowRevision
    rendered = f"{snapshot!r}\n{snapshot.revision!r}\n{snapshot.revision!s}"
    for forbidden in (
        JOB_ONE,
        snapshot.application.company,
        snapshot.application.job_url,
        snapshot.application.prompt_job_description or "",
        snapshot.application.resume_html or "",
        object.__getattribute__(snapshot.revision, "_value").hex(),
    ):
        assert not forbidden or forbidden not in rendered
    with pytest.raises(FrozenInstanceError):
        snapshot.variants = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.revision = snapshot.revision  # type: ignore[misc]


def test_workflow_revision_excludes_human_state_and_includes_prompt_state(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    store.store_jod(JOB_ONE, source_text="source-one", prompt_text="prompt-one")
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    initial = store.get_workflow_snapshot(JOB_ONE)

    store.select_resume_variant(JOB_ONE, "v1")
    store.update_application_status(
        JOB_ONE,
        applied_to="Rejected",
        date_applied="2035-02-05",
        notes="Synthetic human-only note",
    )
    store.archive([JOB_ONE])
    human_only = store.get_workflow_snapshot(JOB_ONE)
    assert human_only.revision == initial.revision

    store.store_jod(JOB_ONE, source_text="source-two", prompt_text="prompt-two")
    jod_changed = store.get_workflow_snapshot(JOB_ONE)
    assert jod_changed.revision != human_only.revision

    store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    sibling_changed = store.get_workflow_snapshot(JOB_ONE)
    assert sibling_changed.revision != jod_changed.revision

    store.upsert_application(
        _metadata(
            company="Example Prompt Metadata Cooperative",
            title="Synthetic Prompt Metadata Analyst",
        )
    )
    metadata_changed = store.get_workflow_snapshot(JOB_ONE)
    assert metadata_changed.revision != sibling_changed.revision


def test_conditional_variant_write_allows_pin_status_note_and_archive_drift(
    tmp_path: Path,
) -> None:
    store, _ = _initialize_with_application(tmp_path)
    v1 = store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    expected = store.get_workflow_snapshot(JOB_ONE).revision

    store.select_resume_variant(JOB_ONE, "v1")
    store.update_application_status(
        JOB_ONE,
        applied_to="Accepted for interview",
        notes="Synthetic human checkpoint",
    )
    store.archive([JOB_ONE])
    written = store.upsert_resume_variant_if_revision(
        JOB_ONE,
        _variant("v2", parent="v1", marker="conditional-v2"),
        expected_revision=expected,
    )

    application = store.get_application(JOB_ONE)
    assert written.resume_pdf == b"pdf-conditional-v2"
    assert application.selected_resume_variant == "v1"
    assert application.resume_variant_selection_mode == "manual"
    assert application.selected_variant == v1
    assert application.applied_to == "Accepted for interview"
    assert application.notes == "Synthetic human checkpoint"
    assert application.archived_at is not None


def test_conditional_variant_write_rejects_stale_and_active_tokens_atomically(
    tmp_path: Path,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    snapshot = store.get_workflow_snapshot(JOB_ONE)
    store.store_jod(JOB_ONE, source_text="changed-source", prompt_text="changed-prompt")
    before = _schema_snapshot(database)

    with pytest.raises(ApplicationStateConflictError) as caught:
        store.upsert_resume_variant_if_revision(
            JOB_ONE,
            _variant("v2", parent="v1"),
            expected_revision=snapshot.revision,
        )
    _assert_exact_public_error(
        caught.value,
        ApplicationStateConflictError,
        "Application state changed.",
        forbidden=(JOB_ONE, "changed-prompt", str(database)),
    )
    assert _schema_snapshot(database) == before

    class DerivedRevision(ApplicationWorkflowRevision):
        pass

    hostile = HostileCapability()
    malformed = object.__new__(ApplicationWorkflowRevision)
    object.__setattr__(malformed, "_value", hostile)
    invalid: tuple[object, ...] = (
        b"x" * 32,
        DerivedRevision(b"x" * 32),
        malformed,
        hostile,
    )
    for value in invalid:
        with pytest.raises(ApplicationStateValidationError):
            store.upsert_resume_variant_if_revision(
                JOB_ONE,
                _variant("v2", parent="v1"),
                expected_revision=value,  # type: ignore[arg-type]
            )
    assert object.__getattribute__(hostile, "calls") == []
    assert _schema_snapshot(database) == before


def test_malformed_exact_revision_slots_reject_before_any_state_capability(
    tmp_path: Path,
) -> None:
    base, database = _initialize_with_application(tmp_path)
    base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    before = _schema_snapshot(database)
    clock_calls: list[str] = []
    hook_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("clock")
        raise AssertionError("clock must not run for a malformed revision")

    def hook(checkpoint: str) -> None:
        hook_calls.append(checkpoint)
        raise AssertionError("operation hook must not run for a malformed revision")

    guarded = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=clock,
        operation_hook=hook,
    )
    missing = object.__new__(ApplicationWorkflowRevision)
    deleted = ApplicationWorkflowRevision(b"d" * hashlib.sha256().digest_size)
    object.__delattr__(deleted, "_value")
    undersized = object.__new__(ApplicationWorkflowRevision)
    object.__setattr__(undersized, "_value", b"u" * 31)
    oversized = ApplicationWorkflowRevision(b"o" * hashlib.sha256().digest_size)
    object.__setattr__(oversized, "_value", b"o" * 33)
    hostile = HostileCapability()
    active = object.__new__(ApplicationWorkflowRevision)
    object.__setattr__(active, "_value", hostile)

    for revision in (missing, deleted, undersized, oversized, active):
        with pytest.raises(ApplicationStateValidationError) as caught:
            guarded.upsert_resume_variant_if_revision(
                JOB_ONE,
                _variant("v2", parent="v1"),
                expected_revision=revision,
            )
        _assert_exact_public_error(
            caught.value,
            ApplicationStateValidationError,
            "Application state input is invalid.",
            forbidden=(
                JOB_ONE,
                str(database),
                (b"d" * hashlib.sha256().digest_size).hex(),
                (b"o" * 33).hex(),
            ),
        )

    assert object.__getattribute__(hostile, "calls") == []
    assert clock_calls == []
    assert hook_calls == []
    assert _schema_snapshot(database) == before


@pytest.mark.parametrize(
    "changed_component",
    ("target", "parent", "sibling", "metadata"),
)
def test_conditional_variant_write_rejects_all_prompt_state_drift_atomically(
    tmp_path: Path,
    changed_component: str,
) -> None:
    store, database = _initialize_with_application(tmp_path / changed_component)
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("v2", parent="v1"))
    store.upsert_resume_variant(JOB_ONE, _variant("manual", parent="v2"))
    snapshot = store.get_workflow_snapshot(JOB_ONE)

    if changed_component == "target":
        store.upsert_resume_variant(
            JOB_ONE,
            _variant("v2", parent="v1", marker="changed-target"),
        )
    elif changed_component == "parent":
        store.upsert_resume_variant(
            JOB_ONE,
            _variant("v1", marker="changed-parent"),
        )
    elif changed_component == "sibling":
        store.upsert_resume_variant(
            JOB_ONE,
            _variant("manual", parent="v2", marker="changed-sibling"),
        )
    else:
        store.upsert_application(
            _metadata(
                company="Example Changed Metadata Cooperative",
                title="Synthetic Changed Metadata Analyst",
            )
        )
    before = _schema_snapshot(database)

    with pytest.raises(ApplicationStateConflictError):
        store.upsert_resume_variant_if_revision(
            JOB_ONE,
            _variant("v2", parent="v1", marker="rejected-write"),
            expected_revision=snapshot.revision,
        )
    assert _schema_snapshot(database) == before


def test_conditional_identical_writers_linearize_with_frozen_reversing_clocks(
    tmp_path: Path,
) -> None:
    base, database = _store(tmp_path, clock=lambda: FIXED_TIME)
    base.initialize()
    base.upsert_application(_metadata())
    prior = base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    snapshot = base.get_workflow_snapshot(JOB_ONE)
    start = threading.Barrier(3)
    results: list[object] = []
    result_lock = threading.Lock()

    def run(store: ApplicationStateStore) -> None:
        start.wait(timeout=3)
        try:
            outcome: object = store.upsert_resume_variant_if_revision(
                JOB_ONE,
                _variant("v1"),
                expected_revision=snapshot.revision,
            )
        except Exception as error:  # noqa: BLE001 - compare both worker outcomes.
            outcome = error
        with result_lock:
            results.append(outcome)

    stores = (
        ApplicationStateStore(
            WorkspacePaths(database=database),
            utc_clock=lambda: FIXED_TIME,
        ),
        ApplicationStateStore(
            WorkspacePaths(database=database),
            utc_clock=lambda: FIXED_TIME - timedelta(days=30),
        ),
    )
    threads = tuple(threading.Thread(target=run, args=(store,)) for store in stores)
    for thread in threads:
        thread.start()
    start.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert sum(type(item) is not ApplicationStateConflictError for item in results) == 1
    assert sum(type(item) is ApplicationStateConflictError for item in results) == 1
    successful = next(
        item for item in results if type(item) is not ApplicationStateConflictError
    )
    assert type(successful).__name__ == "ResumeVariantRecord"
    assert datetime.fromisoformat(successful.updated_at) > datetime.fromisoformat(
        prior.updated_at
    )
    assert base.get_workflow_snapshot(JOB_ONE).revision != snapshot.revision


def test_wal_workflow_snapshot_hydrates_application_and_variants_together(
    tmp_path: Path,
) -> None:
    base, database = _initialize_with_application(tmp_path)
    base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    with sqlite3.connect(database) as connection:
        assert (
            str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]) == "wal"
        )
    expected = base.get_workflow_snapshot(JOB_ONE)
    application_fetched = threading.Event()
    release_reader = threading.Event()
    observed: list[ApplicationWorkflowSnapshot] = []
    errors: list[BaseException] = []

    def hook(checkpoint: str) -> None:
        if checkpoint == "workflow_application_row_fetched":
            application_fetched.set()
            if not release_reader.wait(timeout=3):
                raise RuntimeError("bounded workflow snapshot synchronization failed")

    reader = ApplicationStateStore(
        WorkspacePaths(database=database),
        operation_hook=hook,
    )
    writer = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=TickingClock(FIXED_TIME + timedelta(days=1)),
    )

    def read_snapshot() -> None:
        try:
            observed.append(reader.get_workflow_snapshot(JOB_ONE))
        except Exception as error:  # noqa: BLE001 - capture reader thread failures.
            errors.append(error)

    thread = threading.Thread(target=read_snapshot)
    thread.start()
    try:
        assert application_fetched.wait(timeout=3)
        writer.upsert_resume_variant(
            JOB_ONE,
            _variant("v1", marker="later-generation"),
        )
    finally:
        release_reader.set()
        thread.join(timeout=3)

    assert not thread.is_alive()
    assert errors == []
    assert observed == [expected]
    current = writer.get_workflow_snapshot(JOB_ONE)
    assert current.revision != expected.revision
    assert current.application.selected_variant == current.variants[0]


def test_workspace_binding_is_normalized_hidden_and_checked_before_access(
    tmp_path: Path,
) -> None:
    database = tmp_path / "database-parent" / ".." / "database-parent" / "state.db"
    output = tmp_path / "output-parent" / ".." / "output-parent" / "artifacts"
    store = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=lambda: FIXED_TIME,
    )
    store.assert_workspace_binding(
        WorkspacePaths(
            database=tmp_path / "database-parent" / "state.db",
            output_dir=tmp_path / "output-parent" / "artifacts",
        )
    )
    wrong = WorkspacePaths(
        database=tmp_path / "elsewhere" / "state.db",
        output_dir=tmp_path / "other-output",
    )
    with pytest.raises(ApplicationStateConfigurationError) as caught:
        store.assert_workspace_binding(wrong)
    _assert_exact_public_error(
        caught.value,
        ApplicationStateConfigurationError,
        "Application state database is not configured.",
        forbidden=(
            str(database),
            str(output),
            str(wrong.database),
            str(wrong.output_dir),
        ),
    )
    assert not database.parent.exists()
    assert not output.parent.exists()

    hostile = HostileCapability()
    with pytest.raises(ApplicationStateConfigurationError):
        store.assert_workspace_binding(
            WorkspacePaths(
                database=hostile,  # type: ignore[arg-type]
                output_dir=hostile,  # type: ignore[arg-type]
            )
        )
    assert object.__getattribute__(hostile, "calls") == []


@pytest.mark.parametrize("checkpoint", ("seed_after_metadata", "seed_after_jod"))
def test_atomic_application_seed_never_leaves_metadata_only_row(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    def fail(observed: str) -> None:
        if observed == checkpoint:
            raise RuntimeError("synthetic fresh seed detail")

    store, _ = _store(
        tmp_path,
        failure_injector=fail,
    )
    store.initialize()

    with pytest.raises(ApplicationStateError):
        store.seed_application(
            _metadata(),
            source_text="Synthetic full source JOD",
            prompt_text="Synthetic bounded prompt JOD",
        )

    assert store.list_applications(ApplicationScope.ALL) == ()


@pytest.mark.parametrize("checkpoint", ("seed_after_metadata", "seed_after_jod"))
def test_atomic_application_seed_rolls_back_every_injected_partial_write(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    base, database = _initialize_with_application(tmp_path)
    base.store_jod(JOB_ONE, source_text="old-source", prompt_text="old-prompt")
    base.upsert_resume_variant(JOB_ONE, _variant("v1"))
    base.update_application_status(
        JOB_ONE,
        applied_to="Rejected",
        notes="Synthetic state to preserve",
    )
    base.archive([JOB_ONE])
    before = _schema_snapshot(database)

    def fail(observed: str) -> None:
        if observed == checkpoint:
            raise RuntimeError("synthetic seed rollback detail")

    failing = ApplicationStateStore(
        WorkspacePaths(database=database),
        utc_clock=TickingClock(FIXED_TIME + timedelta(days=1)),
        failure_injector=fail,
    )
    with pytest.raises(ApplicationStateError) as caught:
        failing.seed_application(
            _metadata(
                company="Example Replacement Cooperative",
                title="Synthetic Replacement Analyst",
            ),
            source_text="new-source",
            prompt_text="new-prompt",
        )
    _assert_content_free(
        caught.value,
        forbidden=("synthetic seed rollback detail", "new-prompt", str(database)),
    )
    assert _schema_snapshot(database) == before


def test_atomic_application_seed_creates_complete_row_and_preserves_existing_state(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    store.initialize()
    created = store.seed_application(
        _metadata(),
        source_text="Synthetic full source JOD",
        prompt_text="Synthetic bounded prompt JOD",
    )
    assert created.job_description == "Synthetic full source JOD"
    assert created.prompt_job_description == "Synthetic bounded prompt JOD"

    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.select_resume_variant(JOB_ONE, "v1")
    store.update_application_status(
        JOB_ONE,
        applied_to="Rejected",
        notes="Synthetic preserved note",
    )
    store.archive([JOB_ONE])
    before = store.get_application(JOB_ONE)
    variants = store.list_resume_variants(JOB_ONE)
    refreshed = store.seed_application(
        _metadata(
            company="Example Refreshed Cooperative",
            title="Synthetic Refreshed Analyst",
        ),
        source_text="Synthetic refreshed full source JOD",
        prompt_text="Synthetic refreshed bounded prompt JOD",
    )

    assert refreshed.company == "Example Refreshed Cooperative"
    assert refreshed.job_title == "Synthetic Refreshed Analyst"
    assert refreshed.job_description == "Synthetic refreshed full source JOD"
    assert refreshed.prompt_job_description == "Synthetic refreshed bounded prompt JOD"
    assert refreshed.selected_resume_variant == before.selected_resume_variant
    assert (
        refreshed.resume_variant_selection_mode == before.resume_variant_selection_mode
    )
    assert refreshed.selected_variant == before.selected_variant
    assert refreshed.applied_to == before.applied_to
    assert refreshed.notes == before.notes
    assert refreshed.archived_at == before.archived_at
    assert refreshed.imported_at == before.imported_at
    assert store.list_resume_variants(JOB_ONE) == variants


def test_seed_outcome_is_immutable_content_hidden_and_compatibility_is_preserved(
    tmp_path: Path,
) -> None:
    store, _ = _store(tmp_path)
    store.initialize()

    created = store.seed_application_with_outcome(
        _metadata(),
        source_text="Synthetic first complete source",
        prompt_text="Synthetic first bounded prompt",
    )
    assert type(created) is ApplicationSeedOutcome
    assert created.created is True
    assert created.application == store.get_application(JOB_ONE)
    rendered = f"{created!r}\n{created!s}"
    assert rendered == (
        "ApplicationSeedOutcome(created=True, content_hidden=True)\n"
        "ApplicationSeedOutcome(created=True, content_hidden=True)"
    )
    for forbidden in (
        JOB_ONE,
        created.application.company,
        created.application.job_url,
        created.application.job_description or "",
        created.application.prompt_job_description or "",
    ):
        assert forbidden not in rendered
    with pytest.raises(FrozenInstanceError):
        created.created = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        created.application = created.application  # type: ignore[misc]

    compatible = store.seed_application(
        _metadata(company="Example Compatibility Cooperative"),
        source_text="Synthetic compatibility source",
        prompt_text="Synthetic compatibility prompt",
    )
    assert type(compatible).__name__ == "ApplicationRecord"
    assert compatible.company == "Example Compatibility Cooperative"

    refreshed = store.seed_application_with_outcome(
        _metadata(company="Example Refreshed Outcome Cooperative"),
        source_text="Synthetic refreshed outcome source",
        prompt_text="Synthetic refreshed outcome prompt",
    )
    assert type(refreshed) is ApplicationSeedOutcome
    assert refreshed.created is False
    assert refreshed.application.company == "Example Refreshed Outcome Cooperative"


def test_seed_outcome_is_authoritative_inside_concurrent_seed_transactions(
    tmp_path: Path,
) -> None:
    base, database = _store(tmp_path)
    base.initialize()
    start = threading.Barrier(3)
    outcomes: list[tuple[str, ApplicationSeedOutcome]] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def seed(marker: str) -> None:
        store = ApplicationStateStore(
            WorkspacePaths(database=database),
            utc_clock=lambda: FIXED_TIME,
        )
        try:
            start.wait(timeout=5)
            outcome = store.seed_application_with_outcome(
                _metadata(company=f"Example Concurrent {marker} Cooperative"),
                source_text=f"Synthetic concurrent source {marker}",
                prompt_text=f"Synthetic concurrent prompt {marker}",
            )
            with result_lock:
                outcomes.append((marker, outcome))
        except BaseException as error:  # noqa: BLE001 - capture worker failures.
            with result_lock:
                errors.append(error)

    threads = tuple(
        threading.Thread(target=seed, args=(marker,)) for marker in ("Alpha", "Beta")
    )
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(outcomes) == 2
    assert sum(outcome.created is True for _, outcome in outcomes) == 1
    assert sum(outcome.created is False for _, outcome in outcomes) == 1
    for marker, outcome in outcomes:
        assert type(outcome) is ApplicationSeedOutcome
        assert outcome.application.job_description == (
            f"Synthetic concurrent source {marker}"
        )
        assert outcome.application.prompt_job_description == (
            f"Synthetic concurrent prompt {marker}"
        )
    stored = base.get_application(JOB_ONE)
    assert stored.company in {
        "Example Concurrent Alpha Cooperative",
        "Example Concurrent Beta Cooperative",
    }
    assert len(base.list_applications(ApplicationScope.ALL)) == 1


def test_state_lifecycle_uses_no_network_process_or_unexpected_file_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "future-output"
    database_parent = tmp_path / "database-parent"
    database_parent.mkdir()
    database = database_parent / "applications.sqlite3"
    calls: list[str] = []
    original_iterdir = Path.iterdir
    original_mkdir = Path.mkdir
    original_stat = Path.stat

    def forbidden(*args: object, **kwargs: object) -> Any:
        calls.append("forbidden")
        raise AssertionError("unauthorized integration entry point")

    def guarded_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path == output or output in path.parents:
            forbidden()
        original_mkdir(path, *args, **kwargs)

    def guarded_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == output or output in path.parents:
            return forbidden()
        return original_stat(path, *args, **kwargs)

    for target, name in (
        (socket, "getaddrinfo"),
        (socket, "create_connection"),
        (subprocess, "run"),
        (subprocess, "Popen"),
        (urllib.request, "urlopen"),
        (httpx.Client, "send"),
        (httpx.AsyncClient, "send"),
        (config_module, "load_runtime_config"),
        (config_module, "load_settings"),
        (config_module, "dotenv_values"),
        (ats_module, "calculate_ats_proxy_score"),
        (ats_module, "calculate_ats_diagnostics"),
        (rendering_module, "render_resume_html_from_mapping"),
        (rendering_module, "render_resume_pdf_from_html"),
        (rendering_module, "_render_pdf_with_playwright"),
        (llm_module, "build_llm_client"),
        (api_client_module.ApiLlmClient, "generate_text"),
        (api_client_module.ApiLlmClient, "generate_json"),
        (ollama_module.OllamaClient, "generate_text"),
        (ollama_module.OllamaClient, "generate_json"),
        (LinkedInPublicJobsProvider, "search_jobs"),
        (LinkedInPublicJobsProvider, "get_job_details"),
        (LinkedInPublicJobsProvider, "get_job_raw_payload"),
        (importlib.resources, "files"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    monkeypatch.setattr(Path, "stat", guarded_stat)

    store = ApplicationStateStore(
        WorkspacePaths(database=database, output_dir=output),
        utc_clock=TickingClock(),
    )
    store.initialize()
    store.upsert_application(_metadata())
    store.store_jod(JOB_ONE, source_text="source", prompt_text="prompt")
    store.store_aro(JOB_ONE, yaml_text=ARO_V1)
    store.store_clo(JOB_ONE, value={"letter": "synthetic"})
    store.upsert_resume_variant(JOB_ONE, _variant("v1"))
    store.select_resume_variant(JOB_ONE, "v1")
    store.list_applications()
    store.delete([JOB_ONE])

    assert calls == []
    with pytest.raises(FileNotFoundError):
        original_stat(output)
    assert {path.name for path in original_iterdir(database_parent)} == {
        "applications.sqlite3"
    }


def test_all_public_failures_are_sanitized_without_output_or_logging(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, database = _initialize_with_application(tmp_path)
    caplog.set_level(logging.DEBUG)
    cases: tuple[tuple[type[BaseException], Callable[[], object]], ...] = (
        (
            ApplicationStateValidationError,
            lambda: store.get_application("invalid/id"),
        ),
        (
            ApplicationStateNotFoundError,
            lambda: store.get_application("synthetic-secret-id"),
        ),
        (
            ApplicationStateConfigurationError,
            lambda: ApplicationStateStore(WorkspacePaths()),
        ),
    )

    for expected, operation in cases:
        with pytest.raises(expected) as caught:
            operation()
        _assert_content_free(
            caught.value,
            forbidden=("synthetic-secret-id", str(database), "invalid/id"),
        )
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []
