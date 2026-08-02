"""Synthetic guarded-ops migration regressions."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import career_agent_workbench.ops_migration as ops_migration

from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateStore,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.ops_migration import (
    OpsMigrationError,
    backup_sqlite_database,
    cleanup_disposable_workspace,
    copy_workspace_members,
    inspect_database,
    materialize_workspace,
    validate_disposable_database_copy,
)

PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_database(tmp_path: Path) -> tuple[ApplicationStateStore, Path]:
    workspace = tmp_path / "source-workspace"
    output = workspace / "output"
    output.mkdir(parents=True)
    database = workspace / "state.sqlite3"
    store = ApplicationStateStore(
        WorkspacePaths(root=workspace, database=database, output_dir=output)
    )
    store.initialize()
    store.upsert_application(
        ApplicationMetadata(
            job_id="fictional-job",
            company="Example Cooperative",
            job_title="Reliability Engineer",
            job_url="https://jobs.example.com/fictional-job",
            source="example_public",
        )
    )
    store.store_jod(
        "fictional-job",
        source_text="Synthetic source description.",
        prompt_text="Synthetic prompt description.",
    )
    store.store_aro(
        "fictional-job",
        yaml_text="profile:\n  summary: Synthetic candidate\n",
    )
    store.store_clo(
        "fictional-job",
        value={"letter": "Synthetic cover letter."},
        pdf_content=b"synthetic-cover-pdf",
    )
    store.store_application_artifacts(
        "fictional-job",
        resume_html="<main>Synthetic resume</main>",
        resume_pdf=b"synthetic-resume-pdf",
        ats=AtsFields(score=80),
    )
    store.upsert_resume_variant(
        "fictional-job",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Synthetic first draft",
            source="synthetic",
            application_resume_yaml="profile:\n  summary: Synthetic candidate\n",
            resume_html="<main>Synthetic resume</main>",
            resume_pdf=b"synthetic-resume-pdf",
            ats=AtsFields(score=80),
        ),
    )
    return store, database


def _artifact_database(tmp_path: Path, *, with_variants: bool) -> Path:
    workspace = tmp_path / (
        "variant-workspace" if with_variants else "fallback-workspace"
    )
    output = workspace / "output"
    output.mkdir(parents=True)
    database = workspace / "state.sqlite3"
    store = ApplicationStateStore(
        WorkspacePaths(root=workspace, database=database, output_dir=output)
    )
    store.initialize()
    store.upsert_application(
        ApplicationMetadata(
            job_id="fictional-canonical-artifact",
            company="Example Cooperative",
            job_title="Reliability Engineer",
            job_url="https://jobs.example.com/fictional-canonical-artifact",
            source="example_public",
        )
    )
    if with_variants:
        store.upsert_resume_variant(
            "fictional-canonical-artifact",
            ResumeVariantWrite(
                variant_key="v1",
                variant_label="Synthetic first draft",
                source="synthetic",
                application_resume_yaml="profile:\n  summary: Synthetic v1\n",
                resume_html="<main>Synthetic v1</main>",
                resume_pdf=b"synthetic-v1-pdf",
            ),
        )
        store.upsert_resume_variant(
            "fictional-canonical-artifact",
            ResumeVariantWrite(
                variant_key="v2",
                variant_label="Synthetic second draft",
                source="synthetic",
                parent_variant_key="v1",
                application_resume_yaml="profile:\n  summary: Synthetic v2\n",
                resume_html="<main>Synthetic v2</main>",
                resume_pdf=b"synthetic-v2-pdf",
            ),
        )
        store.select_resume_variant("fictional-canonical-artifact", "v2")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE applications SET application_resume_object = ?, "
            "resume_html_content = ?, resume_content = ?, "
            "application_resume_backup_object = ?, cover_letter_object = ?, "
            "cover_letter_content = ?",
            (
                "profile:\n  summary: Synthetic stale projection\n",
                "<main>Synthetic stale projection</main>",
                b"synthetic-stale-projection-pdf",
                "profile:\n  summary: Synthetic backup\n",
                '{"letter":"Synthetic cover letter"}',
                b"synthetic-cover-letter-pdf",
            ),
        )
    return database


def test_materialize_and_copy_explicit_members_without_following_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "profile").mkdir(parents=True)
    (source / "profile" / "MASTER-RESUME.yml").write_text(
        "name: Fictional Candidate\n",
        encoding="utf-8",
    )
    (source / ".blacklist").write_text(
        "Blocked Example*\n",
        encoding="utf-8",
    )
    destination = tmp_path / "ops"

    layout = materialize_workspace(
        destination,
        public_repository=PUBLIC_ROOT,
        directories=(Path("state"), Path("output"), Path("tmp")),
    )
    copied = copy_workspace_members(
        source,
        destination,
        members=(Path("profile"), Path(".blacklist")),
        public_repository=PUBLIC_ROOT,
    )

    assert layout.directory_count == 4
    assert layout.mode_is_user_only is True
    assert copied.member_count == 2
    assert copied.file_count == 2
    assert copied.digest_equal is True
    assert (destination / "profile" / "MASTER-RESUME.yml").is_file()
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o700
        assert (destination / ".blacklist").stat().st_mode & 0o777 == 0o600


def test_guarded_paths_reject_existing_public_and_symlink_destinations(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    public_destination = PUBLIC_ROOT / "synthetic-ops-must-not-exist"

    for destination in (existing, public_destination):
        with pytest.raises(OpsMigrationError) as caught:
            materialize_workspace(
                destination,
                public_repository=PUBLIC_ROOT,
                directories=(),
            )
        assert str(destination) not in str(caught.value)
    assert not public_destination.exists()

    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("synthetic", encoding="utf-8")
    (source / "linked.txt").symlink_to(outside)
    destination = tmp_path / "ops"
    materialize_workspace(
        destination,
        public_repository=PUBLIC_ROOT,
        directories=(),
    )
    with pytest.raises(OpsMigrationError):
        copy_workspace_members(
            source,
            destination,
            members=(Path("linked.txt"),),
            public_repository=PUBLIC_ROOT,
        )
    assert not (destination / "linked.txt").exists()


def test_sqlite_online_backup_is_logically_equal_and_source_immutable(
    tmp_path: Path,
) -> None:
    _store, source = _synthetic_database(tmp_path)
    destination_root = tmp_path / "backup"
    destination_root.mkdir()
    destination = destination_root / "snapshot.sqlite3"
    before = source.read_bytes()

    with sqlite3.connect(source) as live_connection:
        assert (
            live_connection.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
            == 1
        )
        result = backup_sqlite_database(
            source,
            destination,
            public_repository=PUBLIC_ROOT,
        )

    assert result.source_unchanged is True
    assert result.destination_created is True
    assert result.logical_equal is True
    assert result.table_count == 3
    assert source.read_bytes() == before
    assert inspect_database(destination).application_count == 1
    if os.name == "posix":
        assert destination.stat().st_mode & 0o777 == 0o600

    existing = destination_root / "existing.sqlite3"
    existing.write_bytes(b"synthetic retained bytes")
    with pytest.raises(OpsMigrationError):
        backup_sqlite_database(
            source,
            existing,
            public_repository=PUBLIC_ROOT,
        )
    assert existing.read_bytes() == b"synthetic retained bytes"


def test_disposable_initialization_preserves_counts_content_and_lineage(
    tmp_path: Path,
) -> None:
    _store, source = _synthetic_database(tmp_path)
    disposable = tmp_path / "disposable"
    before = source.read_bytes()

    result = validate_disposable_database_copy(
        source,
        disposable,
        database_relative=Path("state/copied.sqlite3"),
        public_repository=PUBLIC_ROOT,
    )

    assert result.passed is True
    assert result.source_unchanged is True
    assert result.disposable_retained is True
    assert result.schema_compatible is True
    assert result.application_count_equal is True
    assert result.query_outcome_count_equal is True
    assert result.existing_table_row_counts_equal is True
    assert result.selection_equal is True
    assert result.invalid_auto_selection_normalized_count == 0
    assert result.lineage_equal is True
    assert result.lineage_valid is True
    assert result.jod_presence_equal is True
    assert result.aro_presence_equal is True
    assert result.clo_presence_equal is True
    assert result.artifact_count_equal is True
    assert result.artifact_digest_equal is True
    assert source.read_bytes() == before
    assert cleanup_disposable_workspace(
        disposable,
        public_repository=PUBLIC_ROOT,
    )
    assert not disposable.exists()


def test_canonical_artifact_exact_equality_passes(tmp_path: Path) -> None:
    source = _artifact_database(tmp_path, with_variants=True)
    evidence = ops_migration._inspect(source)

    comparison = ops_migration._compare(
        evidence,
        evidence,
        source_unchanged=True,
    )

    assert comparison["passed"] is True
    assert comparison["artifact_count_equal"] is True
    assert comparison["artifact_digest_equal"] is True


def test_stale_manual_v2_projection_may_resynchronize(tmp_path: Path) -> None:
    source = _artifact_database(tmp_path, with_variants=True)
    disposable = tmp_path / "stale-projection-disposable"

    result = validate_disposable_database_copy(
        source,
        disposable,
        database_relative=Path("state/copied.sqlite3"),
        public_repository=PUBLIC_ROOT,
    )

    assert result.passed is True
    assert result.selection_equal is True
    assert result.artifact_count_equal is True
    assert result.artifact_digest_equal is True
    with sqlite3.connect(disposable / "state/copied.sqlite3") as connection:
        application = connection.execute(
            "SELECT application_resume_object, resume_html_content, resume_content, "
            "selected_resume_variant, resume_variant_selection_mode "
            "FROM applications"
        ).fetchone()
        selected_v2 = connection.execute(
            "SELECT application_resume_object, resume_html_content, resume_content "
            "FROM application_resume_variants WHERE variant_key = 'v2'"
        ).fetchone()
    assert application[:3] == selected_v2
    assert application[3:] == ("v2", "manual")
    rendered = repr(result)
    assert "fictional-canonical-artifact" not in rendered
    assert "Synthetic stale projection" not in rendered
    assert cleanup_disposable_workspace(
        disposable,
        public_repository=PUBLIC_ROOT,
    )


@pytest.mark.parametrize(
    "field",
    ("application_resume_object", "resume_html_content", "resume_content"),
)
def test_variant_artifact_mutation_fails(tmp_path: Path, field: str) -> None:
    source = _artifact_database(tmp_path, with_variants=True)
    baseline = ops_migration._inspect(source)
    value: str | bytes = (
        b"mutated-synthetic-pdf" if field == "resume_content" else "mutated synthetic"
    )
    with sqlite3.connect(source) as connection:
        connection.execute(
            f"UPDATE application_resume_variants SET {field} = ? WHERE variant_key = 'v2'",
            (value,),
        )
    migrated = ops_migration._inspect(source)

    comparison = ops_migration._compare(
        baseline,
        migrated,
        source_unchanged=True,
    )

    assert comparison["passed"] is False
    assert comparison["artifact_count_equal"] is True
    assert comparison["artifact_digest_equal"] is False


@pytest.mark.parametrize(
    "field",
    ("application_resume_object", "resume_html_content", "resume_content"),
)
def test_no_variant_fallback_artifact_mutation_fails(
    tmp_path: Path,
    field: str,
) -> None:
    source = _artifact_database(tmp_path, with_variants=False)
    baseline = ops_migration._inspect(source)
    value: str | bytes = (
        b"mutated-synthetic-pdf" if field == "resume_content" else "mutated synthetic"
    )
    with sqlite3.connect(source) as connection:
        connection.execute(f"UPDATE applications SET {field} = ?", (value,))
    migrated = ops_migration._inspect(source)

    comparison = ops_migration._compare(
        baseline,
        migrated,
        source_unchanged=True,
    )

    assert comparison["passed"] is False
    assert comparison["artifact_count_equal"] is True
    assert comparison["artifact_digest_equal"] is False


@pytest.mark.parametrize(
    "field",
    (
        "application_resume_backup_object",
        "cover_letter_object",
        "cover_letter_content",
    ),
)
def test_backup_and_cover_letter_artifact_mutation_fails(
    tmp_path: Path,
    field: str,
) -> None:
    source = _artifact_database(tmp_path, with_variants=True)
    baseline = ops_migration._inspect(source)
    value: str | bytes = (
        b"mutated-synthetic-pdf"
        if field == "cover_letter_content"
        else "mutated synthetic"
    )
    with sqlite3.connect(source) as connection:
        connection.execute(f"UPDATE applications SET {field} = ?", (value,))
    migrated = ops_migration._inspect(source)

    comparison = ops_migration._compare(
        baseline,
        migrated,
        source_unchanged=True,
    )

    assert comparison["passed"] is False
    assert comparison["artifact_count_equal"] is True
    assert comparison["artifact_digest_equal"] is False


def test_disposable_initialization_accepts_only_invalid_automatic_clear(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "invalid-auto-workspace"
    output = workspace / "output"
    output.mkdir(parents=True)
    source = workspace / "state.sqlite3"
    store = ApplicationStateStore(
        WorkspacePaths(root=workspace, database=source, output_dir=output)
    )
    store.initialize()
    store.upsert_application(
        ApplicationMetadata(
            job_id="fictional-invalid-auto",
            company="Example Cooperative",
            job_title="Reliability Engineer",
            job_url="https://jobs.example.com/fictional-invalid-auto",
            source="example_public",
        )
    )
    with sqlite3.connect(source) as connection:
        connection.execute(
            "UPDATE applications SET selected_resume_variant = 'v2', "
            "resume_variant_selection_mode = 'auto'"
        )

    disposable = tmp_path / "invalid-auto-disposable"
    result = validate_disposable_database_copy(
        source,
        disposable,
        database_relative=Path("state/copied.sqlite3"),
        public_repository=PUBLIC_ROOT,
    )

    assert result.passed is True
    assert result.selection_equal is True
    assert result.invalid_auto_selection_normalized_count == 1
    assert result.baseline_application_count == result.migrated_application_count == 1
    assert "fictional-invalid-auto" not in repr(result)
    assert cleanup_disposable_workspace(
        disposable,
        public_repository=PUBLIC_ROOT,
    )


def _selection_evidence(
    rows: tuple[tuple[object, object, object, bool], ...],
) -> ops_migration._DatabaseEvidence:
    counts = ops_migration.DatabaseCounts(
        table_count=2,
        total_row_count=len(rows),
        application_count=len(rows),
        variant_count=0,
        query_outcome_count=0,
        selected_count=sum(ops_migration._selection_nonempty(row[1]) for row in rows),
        lineage_valid=True,
        jod_source_count=0,
        jod_prompt_count=0,
        aro_count=0,
        clo_count=0,
        artifact_count=0,
    )
    return ops_migration._DatabaseEvidence(
        counts=counts,
        schema={},
        row_counts={},
        selection_rows=rows,
        selection_digest=b"synthetic-selection",
        lineage_digest=b"synthetic-lineage",
        artifact_digest=b"synthetic-artifact",
        selection_supported=True,
        lineage_supported=True,
    )


def test_selection_comparison_accepts_exact_equality() -> None:
    rows = (("fictional-a", "v1", "auto", True),)
    evidence = _selection_evidence(rows)

    assert ops_migration._compare_selection_rows(
        evidence,
        evidence,
        application_level_migration=False,
    ) == (True, 0)


@pytest.mark.parametrize(
    ("source_rows", "migrated_rows"),
    (
        (
            (("fictional-a", "missing", "manual", False),),
            (("fictional-a", "", "manual", False),),
        ),
        (
            (("fictional-a", "v1", "auto", True),),
            (("fictional-a", "", "auto", False),),
        ),
        (
            (("fictional-a", "missing", "auto", False),),
            (("fictional-a", "", "manual", False),),
        ),
        (
            (
                ("fictional-a", "missing", "auto", False),
                ("fictional-b", "missing", "auto", False),
            ),
            (
                ("fictional-a", "", "auto", False),
                ("fictional-b", "missing", "auto", False),
            ),
        ),
        (
            (("fictional-a", "missing", "auto", False),),
            (("fictional-a", "v1", "auto", True),),
        ),
        (
            (("fictional-a", "missing", "auto", False),),
            (("fictional-b", "", "auto", False),),
        ),
    ),
)
def test_selection_comparison_rejects_unbounded_changes(
    source_rows: tuple[tuple[object, object, object, bool], ...],
    migrated_rows: tuple[tuple[object, object, object, bool], ...],
) -> None:
    accepted, _count = ops_migration._compare_selection_rows(
        _selection_evidence(source_rows),
        _selection_evidence(migrated_rows),
        application_level_migration=False,
    )

    assert accepted is False


def test_comparison_still_rejects_lineage_and_artifact_changes(
    tmp_path: Path,
) -> None:
    _store, source = _synthetic_database(tmp_path)
    baseline = ops_migration._inspect(source)

    lineage = ops_migration._compare(
        baseline,
        replace(baseline, lineage_digest=b"changed-lineage"),
        source_unchanged=True,
    )
    artifact = ops_migration._compare(
        baseline,
        replace(baseline, artifact_digest=b"changed-artifact"),
        source_unchanged=True,
    )

    assert lineage["passed"] is False
    assert lineage["lineage_equal"] is False
    assert artifact["passed"] is False
    assert artifact["artifact_digest_equal"] is False


def test_failed_disposable_validation_cleans_copy_and_keeps_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "not-a-database.sqlite3"
    source.write_text("synthetic invalid database", encoding="utf-8")
    disposable = tmp_path / "disposable"
    before = source.read_bytes()

    with pytest.raises(OpsMigrationError) as caught:
        validate_disposable_database_copy(
            source,
            disposable,
            database_relative=Path("state/copied.sqlite3"),
            public_repository=PUBLIC_ROOT,
        )

    assert str(source) not in str(caught.value)
    assert source.read_bytes() == before
    assert not disposable.exists()


def test_disposable_cleanup_requires_helper_marker(tmp_path: Path) -> None:
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    retained = unmarked / "retained.txt"
    retained.write_text("synthetic retained content", encoding="utf-8")

    with pytest.raises(OpsMigrationError):
        cleanup_disposable_workspace(
            unmarked,
            public_repository=PUBLIC_ROOT,
        )

    assert retained.read_text(encoding="utf-8") == "synthetic retained content"
