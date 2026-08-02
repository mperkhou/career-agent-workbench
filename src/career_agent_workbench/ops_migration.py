"""Public-safe primitives for synthetic-first private workspace migration."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.config import WorkspacePaths

MAX_COPY_MEMBERS = 64
MAX_COPY_FILES = 10_000
MAX_COPY_BYTES = 2_000_000_000
_DISPOSABLE_MARKER = ".career-agent-workbench-disposable"
_MARKER_CONTENT = "career-agent-workbench disposable migration v1\n"
_ERROR = "Guarded ops migration failed."


class OpsMigrationError(Exception):
    """Stable content-free migration failure."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class WorkspaceMaterializationResult:
    directory_count: int
    mode_is_user_only: bool


@dataclass(frozen=True, slots=True)
class WorkspaceCopyResult:
    member_count: int
    file_count: int
    directory_count: int
    byte_count: int
    digest_equal: bool


@dataclass(frozen=True, slots=True)
class DatabaseCounts:
    table_count: int
    total_row_count: int
    application_count: int
    variant_count: int
    query_outcome_count: int
    selected_count: int
    lineage_valid: bool
    jod_source_count: int
    jod_prompt_count: int
    aro_count: int
    clo_count: int
    artifact_count: int


@dataclass(frozen=True, slots=True)
class SqliteBackupResult:
    source_unchanged: bool
    destination_created: bool
    logical_equal: bool
    table_count: int
    total_row_count: int
    destination_mode_is_user_only: bool


@dataclass(frozen=True, slots=True)
class MigrationValidationResult:
    passed: bool
    source_unchanged: bool
    disposable_retained: bool
    schema_compatible: bool
    baseline_table_count: int
    migrated_table_count: int
    baseline_total_row_count: int
    migrated_total_row_count: int
    application_count_equal: bool
    baseline_application_count: int
    migrated_application_count: int
    baseline_variant_count: int
    migrated_variant_count: int
    query_outcome_count_equal: bool
    baseline_query_outcome_count: int
    migrated_query_outcome_count: int
    existing_table_row_counts_equal: bool
    selection_equal: bool
    invalid_auto_selection_normalized_count: int
    lineage_equal: bool
    lineage_valid: bool
    jod_presence_equal: bool
    aro_presence_equal: bool
    clo_presence_equal: bool
    artifact_count_equal: bool
    artifact_digest_equal: bool


@dataclass(frozen=True, slots=True, repr=False)
class _DatabaseEvidence:
    counts: DatabaseCounts
    schema: Mapping[str, frozenset[str]] = field(repr=False)
    row_counts: Mapping[str, int] = field(repr=False)
    selection_rows: tuple[tuple[Any, Any, Any, bool], ...] = field(repr=False)
    selection_digest: bytes = field(repr=False)
    lineage_digest: bytes = field(repr=False)
    artifact_digest: bytes = field(repr=False)
    selection_supported: bool = field(repr=False)
    lineage_supported: bool = field(repr=False)


def materialize_workspace(
    destination: Path,
    *,
    public_repository: Path,
    directories: Sequence[Path],
) -> WorkspaceMaterializationResult:
    """Create one new user-only root with an explicit relative layout."""

    created = False
    try:
        root = _new_root(destination, public_repository)
        members = _relative_members(directories)
        root.mkdir(mode=0o700)
        created = True
        made: set[Path] = set()
        for member in members:
            current = root
            for part in member.parts:
                current /= part
                if current not in made:
                    current.mkdir(mode=0o700)
                    made.add(current)
                    _private_mode(current, directory=True)
        _private_mode(root, directory=True)
        mode_is_user_only = _mode_is(root, 0o700)
        if not mode_is_user_only:
            raise ValueError
        return WorkspaceMaterializationResult(1 + len(made), mode_is_user_only)
    except Exception:
        if created:
            _remove_tree(destination)
        raise OpsMigrationError(_ERROR) from None


def copy_workspace_members(
    source_root: Path,
    destination_root: Path,
    *,
    members: Sequence[Path],
    public_repository: Path,
) -> WorkspaceCopyResult:
    """Copy only explicit members without links, escapes, or overwrites."""

    created: list[Path] = []
    try:
        source = _existing_directory(source_root)
        destination = _private_root(destination_root, public_repository)
        selected = _relative_members(members)
        source_digest = hashlib.sha256()
        destination_digest = hashlib.sha256()
        files = directories = byte_count = 0
        for member in selected:
            source_member = source / member
            target_member = destination / member
            _beneath(source_member, source)
            _beneath(target_member, destination)
            _reject_links(source_member, stop=source)
            _reject_links(target_member, stop=destination)
            if target_member.exists() or target_member.is_symlink():
                raise ValueError
            cleanup_target = target_member
            while (
                cleanup_target.parent != destination
                and not cleanup_target.parent.exists()
            ):
                cleanup_target = cleanup_target.parent
            if cleanup_target not in created:
                created.append(cleanup_target)
            if source_member.is_file():
                target_member.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                size = _copy_file(
                    source_member,
                    target_member,
                    member,
                    source_digest,
                    destination_digest,
                )
                files += 1
                byte_count += size
            elif source_member.is_dir():
                target_member.mkdir(parents=True, mode=0o700)
                copied = _copy_directory(
                    source_member,
                    target_member,
                    member,
                    source_digest,
                    destination_digest,
                )
                files += copied[0]
                directories += copied[1] + 1
                byte_count += copied[2]
            else:
                raise ValueError
            if files > MAX_COPY_FILES or byte_count > MAX_COPY_BYTES:
                raise ValueError
        digest_equal = source_digest.digest() == destination_digest.digest()
        if not digest_equal:
            raise ValueError
        return WorkspaceCopyResult(
            len(selected),
            files,
            directories,
            byte_count,
            digest_equal,
        )
    except Exception:
        for target in reversed(created):
            _remove_path(target)
        raise OpsMigrationError(_ERROR) from None


def backup_sqlite_database(
    source_database: Path,
    destination_database: Path,
    *,
    public_repository: Path,
) -> SqliteBackupResult:
    """Create and validate an online SQLite backup from an explicit source."""

    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    cleanup_allowed = False
    try:
        source = _existing_file(source_database)
        destination = _new_file(destination_database, public_repository)
        source_digest = _file_digest(source)
        baseline = _inspect(source)
        source_connection = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
        descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        cleanup_allowed = True
        os.close(descriptor)
        target_connection = sqlite3.connect(destination)
        source_connection.execute("PRAGMA query_only = ON")
        source_connection.backup(target_connection)
        if target_connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise sqlite3.DatabaseError
        target_connection.close()
        target_connection = None
        source_connection.close()
        source_connection = None
        _private_mode(destination)
        copied = _inspect(destination)
        source_unchanged = source_digest == _file_digest(source)
        logical_equal = _evidence_equal(baseline, copied)
        mode_is_user_only = _mode_is(destination, 0o600)
        if not source_unchanged or not logical_equal or not mode_is_user_only:
            raise ValueError
        return SqliteBackupResult(
            source_unchanged=source_unchanged,
            destination_created=True,
            logical_equal=logical_equal,
            table_count=copied.counts.table_count,
            total_row_count=copied.counts.total_row_count,
            destination_mode_is_user_only=mode_is_user_only,
        )
    except Exception:
        if cleanup_allowed:
            _remove_database_files(destination_database)
        raise OpsMigrationError(_ERROR) from None
    finally:
        if target_connection is not None:
            target_connection.close()
        if source_connection is not None:
            source_connection.close()


def inspect_database(database: Path) -> DatabaseCounts:
    """Return counts and invariant booleans without rows, values, or paths."""

    return _inspect(database).counts


def validate_disposable_database_copy(
    source_database: Path,
    disposable_workspace: Path,
    *,
    database_relative: Path,
    public_repository: Path,
) -> MigrationValidationResult:
    """Initialize only a second copy, compare it, and retain only success."""

    created = False
    try:
        source = _existing_file(source_database)
        root = _new_root(disposable_workspace, public_repository)
        relative = _relative_member(database_relative)
        root.mkdir(mode=0o700)
        created = True
        marker = root / _DISPOSABLE_MARKER
        marker.write_text(_MARKER_CONTENT, encoding="utf-8")
        _private_mode(marker)
        output = root / "output"
        output.mkdir(mode=0o700)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        baseline_digest = _file_digest(source)
        baseline = _inspect(source)
        backup = backup_sqlite_database(
            source, destination, public_repository=public_repository
        )
        ApplicationStateStore(
            WorkspacePaths(root=root, database=destination, output_dir=output)
        ).initialize()
        migrated = _inspect(destination)
        comparison = _compare(
            baseline,
            migrated,
            source_unchanged=(
                baseline_digest == _file_digest(source) and backup.source_unchanged
            ),
        )
        result = MigrationValidationResult(
            **comparison,
            disposable_retained=bool(comparison["passed"]),
        )
        if not result.passed:
            cleanup_disposable_workspace(root, public_repository=public_repository)
        return result
    except Exception:
        if created:
            _remove_tree(disposable_workspace)
        raise OpsMigrationError(_ERROR) from None


def cleanup_disposable_workspace(
    workspace: Path,
    *,
    public_repository: Path,
) -> bool:
    """Remove only a helper-marked disposable root outside public state."""

    try:
        root = _private_root(workspace, public_repository)
        marker = root / _DISPOSABLE_MARKER
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf-8") != _MARKER_CONTENT
        ):
            raise ValueError
        shutil.rmtree(root)
        return not root.exists()
    except Exception:
        raise OpsMigrationError(_ERROR) from None


def _inspect(database: Path) -> _DatabaseEvidence:
    connection: sqlite3.Connection | None = None
    try:
        selected = _existing_file(database)
        connection = sqlite3.connect(f"{selected.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError
        tables = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE BINARY"
            )
        )
        schema: dict[str, frozenset[str]] = {}
        row_counts: dict[str, int] = {}
        for table in tables:
            quoted = _quote(table)
            schema[table] = frozenset(
                str(row["name"])
                for row in connection.execute(f"PRAGMA table_info({quoted})")
            )
            row_counts[table] = int(
                connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            )
        columns = schema.get("applications", frozenset())
        selection_supported = {
            "job_id",
            "selected_resume_variant",
            "resume_variant_selection_mode",
        } <= columns
        variant_columns = schema.get("application_resume_variants", frozenset())
        selection_variant_supported = {"job_id", "variant_key"} <= variant_columns
        if selection_supported:
            selected_variant_exists = (
                "EXISTS (SELECT 1 FROM application_resume_variants AS variant "
                "WHERE variant.job_id = app.job_id "
                "AND variant.variant_key = app.selected_resume_variant)"
                if selection_variant_supported
                else "0"
            )
            raw_selection_rows = connection.execute(
                "SELECT app.job_id, app.selected_resume_variant, "
                "app.resume_variant_selection_mode, "
                f"{selected_variant_exists} AS selected_variant_exists "
                "FROM applications AS app ORDER BY app.job_id COLLATE BINARY"
            ).fetchall()
            selection_rows = tuple(
                (
                    row["job_id"],
                    row["selected_resume_variant"],
                    row["resume_variant_selection_mode"],
                    bool(row["selected_variant_exists"]),
                )
                for row in raw_selection_rows
            )
            selections = tuple(
                row for row in selection_rows if _selection_nonempty(row[1])
            )
        else:
            selection_rows = ()
            selections = ()
        lineage_supported = {
            "job_id",
            "variant_key",
            "parent_variant_key",
        } <= variant_columns
        lineage = (
            connection.execute(
                "SELECT job_id, variant_key, parent_variant_key "
                "FROM application_resume_variants "
                "ORDER BY job_id COLLATE BINARY, variant_key COLLATE BINARY"
            ).fetchall()
            if lineage_supported
            else ()
        )
        artifacts = _artifact_rows(connection, columns)
        artifact_count, artifact_digest = _artifact_evidence(artifacts)
        counts = DatabaseCounts(
            table_count=len(tables),
            total_row_count=sum(row_counts.values()),
            application_count=row_counts.get("applications", 0),
            variant_count=row_counts.get("application_resume_variants", 0),
            query_outcome_count=row_counts.get("search_query_outcomes", 0),
            selected_count=len(selections),
            lineage_valid=_lineage_valid(
                connection,
                selection_supported=selection_supported,
                lineage_supported=lineage_supported,
            ),
            jod_source_count=_presence(connection, columns, ("job_description",)),
            jod_prompt_count=_presence(
                connection, columns, ("prompt_job_description",)
            ),
            aro_count=_presence(
                connection, columns, ("aro_yaml", "application_resume_object")
            ),
            clo_count=_presence(connection, columns, ("cover_letter_object",)),
            artifact_count=artifact_count,
        )
        return _DatabaseEvidence(
            counts=counts,
            schema=schema,
            row_counts=row_counts,
            selection_rows=selection_rows,
            selection_digest=_rows_digest(selections),
            lineage_digest=_rows_digest(lineage),
            artifact_digest=artifact_digest,
            selection_supported=selection_supported,
            lineage_supported=lineage_supported,
        )
    except Exception:
        raise OpsMigrationError(_ERROR) from None
    finally:
        if connection is not None:
            connection.close()


def _compare(
    baseline: _DatabaseEvidence,
    migrated: _DatabaseEvidence,
    *,
    source_unchanged: bool,
) -> dict[str, Any]:
    left, right = baseline.counts, migrated.counts
    application_level_migration = left.variant_count == 0 and left.aro_count > 0
    schema_compatible = all(
        table in migrated.schema and columns <= migrated.schema[table]
        for table, columns in baseline.schema.items()
    )
    row_counts_equal = all(
        (
            table == "application_resume_variants"
            and application_level_migration
            and right.variant_count <= left.application_count
        )
        or migrated.row_counts.get(table) == count
        for table, count in baseline.row_counts.items()
    )
    selection_equal, invalid_auto_normalized = _compare_selection_rows(
        baseline,
        migrated,
        application_level_migration=application_level_migration,
    )
    lineage_equal = (
        not baseline.lineage_supported
        or (
            left.variant_count == right.variant_count
            and baseline.lineage_digest == migrated.lineage_digest
        )
        or (
            application_level_migration
            and 0 < right.variant_count <= left.application_count
            and right.lineage_valid
        )
    )
    values = {
        "source_unchanged": source_unchanged,
        "schema_compatible": schema_compatible,
        "application_count_equal": left.application_count == right.application_count,
        "query_outcome_count_equal": (
            left.query_outcome_count == right.query_outcome_count
        ),
        "existing_table_row_counts_equal": row_counts_equal,
        "selection_equal": selection_equal,
        "lineage_equal": lineage_equal,
        "lineage_valid": right.lineage_valid,
        "jod_presence_equal": (
            left.jod_source_count == right.jod_source_count
            and left.jod_prompt_count == right.jod_prompt_count
        ),
        "aro_presence_equal": left.aro_count == right.aro_count,
        "clo_presence_equal": left.clo_count == right.clo_count,
        "artifact_count_equal": left.artifact_count == right.artifact_count,
        "artifact_digest_equal": baseline.artifact_digest == migrated.artifact_digest,
    }
    return {
        "passed": all(values.values()),
        **values,
        "baseline_table_count": left.table_count,
        "migrated_table_count": right.table_count,
        "baseline_total_row_count": left.total_row_count,
        "migrated_total_row_count": right.total_row_count,
        "baseline_application_count": left.application_count,
        "migrated_application_count": right.application_count,
        "baseline_variant_count": left.variant_count,
        "migrated_variant_count": right.variant_count,
        "baseline_query_outcome_count": left.query_outcome_count,
        "migrated_query_outcome_count": right.query_outcome_count,
        "invalid_auto_selection_normalized_count": invalid_auto_normalized,
    }


def _compare_selection_rows(
    baseline: _DatabaseEvidence,
    migrated: _DatabaseEvidence,
    *,
    application_level_migration: bool,
) -> tuple[bool, int]:
    if not baseline.selection_supported:
        return True, 0
    if baseline.selection_rows == migrated.selection_rows:
        return True, 0
    if (
        application_level_migration
        and baseline.counts.selected_count == 0
        and migrated.counts.selected_count == migrated.counts.variant_count
    ):
        return True, 0
    if not migrated.selection_supported:
        return False, 0

    source_rows = baseline.selection_rows
    migrated_rows = migrated.selection_rows
    if len(source_rows) != len(migrated_rows):
        return False, 0

    invalid_source_count = sum(_invalid_automatic_selection(row) for row in source_rows)
    normalized_count = 0
    for source_row, migrated_row in zip(source_rows, migrated_rows, strict=True):
        if source_row[:3] == migrated_row[:3]:
            continue
        if (
            source_row[0] != migrated_row[0]
            or not _invalid_automatic_selection(source_row)
            or migrated_row[2] != "auto"
            or not _selection_empty(migrated_row[1])
        ):
            return False, normalized_count
        normalized_count += 1
    return normalized_count == invalid_source_count, normalized_count


def _invalid_automatic_selection(row: tuple[Any, Any, Any, bool]) -> bool:
    return row[2] == "auto" and _selection_nonempty(row[1]) and not row[3]


def _selection_nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _selection_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _evidence_equal(left: _DatabaseEvidence, right: _DatabaseEvidence) -> bool:
    return all(
        (
            left.counts == right.counts,
            left.schema == right.schema,
            left.row_counts == right.row_counts,
            left.selection_rows == right.selection_rows,
            left.selection_digest == right.selection_digest,
            left.lineage_digest == right.lineage_digest,
            left.artifact_digest == right.artifact_digest,
        )
    )


def _lineage_valid(
    connection: sqlite3.Connection,
    *,
    selection_supported: bool,
    lineage_supported: bool,
) -> bool:
    if not lineage_supported:
        return True
    invalid = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM application_resume_variants AS child
            LEFT JOIN applications AS app ON app.job_id = child.job_id
            LEFT JOIN application_resume_variants AS parent
              ON parent.job_id = child.job_id
             AND parent.variant_key = child.parent_variant_key
            WHERE app.job_id IS NULL
               OR child.variant_key NOT IN ('v1', 'v2', 'manual')
               OR (child.variant_key = 'v1' AND child.parent_variant_key IS NOT NULL)
               OR (child.variant_key = 'v2' AND (
                    child.parent_variant_key != 'v1' OR parent.variant_key IS NULL))
               OR (child.variant_key = 'manual' AND (
                    child.parent_variant_key NOT IN ('v1', 'v2')
                    OR parent.variant_key IS NULL))
            """
        ).fetchone()[0]
    )
    if selection_supported:
        invalid += int(
            connection.execute(
                """
                SELECT COUNT(*) FROM applications AS app
                LEFT JOIN application_resume_variants AS selected
                  ON selected.job_id = app.job_id
                 AND selected.variant_key = app.selected_resume_variant
                WHERE app.selected_resume_variant IS NOT NULL
                  AND TRIM(app.selected_resume_variant) != ''
                  AND selected.variant_key IS NULL
                """
            ).fetchone()[0]
        )
    return invalid == 0


def _artifact_rows(
    connection: sqlite3.Connection,
    columns: frozenset[str],
) -> Iterable[sqlite3.Row]:
    selected = tuple(
        name
        for name in (
            "application_resume_object",
            "application_resume_backup_object",
            "resume_html_content",
            "resume_content",
            "cover_letter_object",
            "cover_letter_content",
        )
        if name in columns
    )
    if "job_id" not in columns or not selected:
        return ()
    names = ", ".join(_quote(name) for name in selected)
    return connection.execute(
        f"SELECT job_id, {names} FROM applications ORDER BY job_id COLLATE BINARY"
    )


def _artifact_evidence(rows: Iterable[sqlite3.Row]) -> tuple[int, bytes]:
    count = 0
    byte_count = 0
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            encoded = (
                b""
                if value is None
                else (value if type(value) is bytes else str(value).encode())
            )
            digest.update(type(value).__name__.encode())
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            byte_count += len(encoded)
        count += sum(1 for value in row[1:] if value not in (None, "", b""))
        if count > MAX_COPY_FILES or byte_count > MAX_COPY_BYTES:
            raise ValueError
    return count, digest.digest()


def _presence(
    connection: sqlite3.Connection,
    columns: frozenset[str],
    candidates: tuple[str, ...],
) -> int:
    selected = tuple(name for name in candidates if name in columns)
    if not selected:
        return 0
    clauses = " OR ".join(
        f"({_quote(name)} IS NOT NULL AND LENGTH({_quote(name)}) > 0)"
        for name in selected
    )
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM applications WHERE {clauses}"
        ).fetchone()[0]
    )


def _rows_digest(rows: Sequence[Sequence[Any]]) -> bytes:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            encoded = (
                b""
                if value is None
                else (value if type(value) is bytes else str(value).encode())
            )
            digest.update(type(value).__name__.encode())
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.digest()


def _copy_directory(
    source: Path,
    destination: Path,
    member: Path,
    source_digest: Any,
    destination_digest: Any,
) -> tuple[int, int, int]:
    files = directories = byte_count = 0
    for current_text, directory_names, file_names in os.walk(source, followlinks=False):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        relative = current.relative_to(source)
        target_current = destination / relative
        for name in directory_names:
            child = current / name
            if child.is_symlink():
                raise ValueError
            (target_current / name).mkdir(mode=0o700)
            _digest_label(source_digest, member / relative / name, b"directory")
            _digest_label(destination_digest, member / relative / name, b"directory")
            directories += 1
        for name in file_names:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise ValueError
            size = _copy_file(
                child,
                target_current / name,
                member / relative / name,
                source_digest,
                destination_digest,
            )
            files += 1
            byte_count += size
            if files > MAX_COPY_FILES or byte_count > MAX_COPY_BYTES:
                raise ValueError
    return files, directories, byte_count


def _copy_file(
    source: Path,
    destination: Path,
    relative: Path,
    source_digest: Any,
    destination_digest: Any,
) -> int:
    size = source.stat(follow_symlinks=False).st_size
    if size > MAX_COPY_BYTES:
        raise ValueError
    shutil.copyfile(source, destination, follow_symlinks=False)
    _private_mode(destination)
    _digest_file(source_digest, relative, source, size)
    _digest_file(destination_digest, relative, destination, size)
    return size


def _digest_label(digest: Any, relative: Path, content: bytes) -> None:
    label = relative.as_posix().encode()
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)


def _digest_file(digest: Any, relative: Path, path: Path, size: int) -> None:
    label = relative.as_posix().encode()
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)


def _relative_members(values: Sequence[Path]) -> tuple[Path, ...]:
    if type(values) not in {list, tuple} or len(values) > MAX_COPY_MEMBERS:
        raise ValueError
    result = tuple(_relative_member(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError
    return result


def _relative_member(value: Path) -> Path:
    if (
        type(value) is not type(Path())
        or value.is_absolute()
        or value in {Path(), Path(".")}
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise ValueError
    return value


def _new_root(path: Path, public: Path) -> Path:
    selected = _absolute(path)
    public_root = _existing_directory(public)
    if selected.exists() or selected.is_symlink():
        raise ValueError
    _reject_links(selected.parent)
    _reject_within(selected, public_root)
    return selected


def _private_root(path: Path, public: Path) -> Path:
    selected = _existing_directory(path)
    _reject_within(selected, _existing_directory(public))
    return selected


def _new_file(path: Path, public: Path) -> Path:
    selected = _absolute(path)
    if selected.exists() or selected.is_symlink() or not selected.parent.is_dir():
        raise ValueError
    _reject_links(selected.parent)
    _reject_within(selected, _existing_directory(public))
    return selected


def _existing_directory(path: Path) -> Path:
    selected = _absolute(path)
    _reject_links(selected)
    if not selected.is_dir():
        raise ValueError
    return selected.resolve(strict=True)


def _existing_file(path: Path) -> Path:
    selected = _absolute(path)
    _reject_links(selected)
    if not selected.is_file():
        raise ValueError
    return selected.resolve(strict=True)


def _absolute(path: Path) -> Path:
    if type(path) is not type(Path()) or not path.is_absolute() or "\x00" in str(path):
        raise ValueError
    return path


def _reject_links(path: Path, *, stop: Path | None = None) -> None:
    current = path
    boundary = stop.parent if stop is not None else None
    while True:
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise ValueError
        except FileNotFoundError:
            pass
        if current == current.parent or current == boundary:
            return
        current = current.parent


def _beneath(path: Path, root: Path) -> None:
    path.resolve(strict=False).relative_to(root.resolve(strict=True))


def _reject_within(path: Path, root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError:
        return
    raise ValueError


def _quote(value: str) -> str:
    if type(value) is not str or "\x00" in value:
        raise ValueError
    return '"' + value.replace('"', '""') + '"'


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.digest()


def _private_mode(path: Path, *, directory: bool = False) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600)


def _mode_is(path: Path, expected: int) -> bool:
    return os.name != "posix" or stat.S_IMODE(path.stat().st_mode) == expected


def _remove_database_files(database: object) -> None:
    if type(database) is not type(Path()):
        return
    for candidate in (
        database,
        database.with_name(f"{database.name}-journal"),
        database.with_name(f"{database.name}-wal"),
        database.with_name(f"{database.name}-shm"),
    ):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def _remove_path(path: Path) -> None:
    try:
        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink(
            missing_ok=True
        )
    except OSError:
        pass


def _remove_tree(path: object) -> None:
    try:
        if type(path) is type(Path()) and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
    except OSError:
        pass


__all__ = [
    "DatabaseCounts",
    "MigrationValidationResult",
    "OpsMigrationError",
    "SqliteBackupResult",
    "WorkspaceCopyResult",
    "WorkspaceMaterializationResult",
    "backup_sqlite_database",
    "cleanup_disposable_workspace",
    "copy_workspace_members",
    "inspect_database",
    "materialize_workspace",
    "validate_disposable_database_copy",
]
