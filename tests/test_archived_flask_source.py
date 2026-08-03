from __future__ import annotations

import ast
import hashlib
import importlib.util
import sqlite3
import tomllib
from pathlib import Path

import yaml
import pytest

from career_agent_workbench import _archived_flask_source as archived
from career_agent_workbench.application_state import (
    ApplicationStateStore,
    ResumeVariantWrite,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths
from career_agent_workbench.webapp_archive_runtime import create_app

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "career_agent_workbench" / "_archived_flask_source.py"
EXPECTED_FRONTEND_SHA256 = {
    "ADD_APPLICATION_TEMPLATE": (
        "9e49e596446e4dee63ababcde1204da6c101ef78c62fc891f9a4367ee590a0ed"
    ),
    "COVER_LETTER_EDIT_TEMPLATE": (
        "13084856731c7a55b663f386e533b3c3b9242cb03dfc043cb913e65470d2ec5b"
    ),
    "DESCRIPTION_COMPARE_TEMPLATE": (
        "d1acfb6dc4f0dc1c8b700b1b1836b5321210687efec6112046a0baf814dd467d"
    ),
    "INDEX_TEMPLATE": (
        "52138815afa4242e16b99d3a0788a385ea479abe3f51a23f91480f01a774961d"
    ),
    "RESUME_EDIT_TEMPLATE": (
        "f2edaed4c4e1ace6830424bf9375ac3763adfefd9585a48664fcd8f73c9b735e"
    ),
    "RESUME_VARIANTS_TEMPLATE": (
        "a2800f7d570f7d6c437082ed8201c55129f05a80212b7c6154880f1b5bba3f5b"
    ),
    "_PLAYWRIGHT_CHROMIUM_SCRIPT": (
        "3d227487d19656936ff427d0375db9543a1cd6a49c9ba99e7cfa0b66161b2f15"
    ),
}


def _frontend_literals() -> dict[str, str]:
    tree = ast.parse(SOURCE.read_text("utf-8"))
    values: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in EXPECTED_FRONTEND_SHA256
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values[node.targets[0].id] = node.value.value
    return values


def _runtime(tmp_path: Path) -> RuntimeConfig:
    workspace = tmp_path / "workspace"
    output = workspace / "output"
    downloads = workspace / "downloads"
    temporary = workspace / "tmp"
    for path in (output, downloads, temporary):
        path.mkdir(parents=True, exist_ok=True)
    return RuntimeConfig(
        paths=WorkspacePaths(
            root=workspace,
            output_dir=output,
            database=output / "tracking" / "applications.sqlite3",
            download_dir=downloads,
            tmp_dir=temporary,
        ),
        settings=Settings(),
        env_file=None,
    )


def _demo_runtime(tmp_path: Path) -> RuntimeConfig:
    path = ROOT / "scripts" / "create_demo_workspace.py"
    spec = importlib.util.spec_from_file_location("p03_demo_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_demo_workspace(
        ROOT / "examples" / "demo-workspace",
        tmp_path / "workspace",
    )


def test_archived_frontend_literals_remain_exact() -> None:
    values = _frontend_literals()

    assert set(values) == set(EXPECTED_FRONTEND_SHA256)
    assert {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in values.items()
    } == EXPECTED_FRONTEND_SHA256


def test_archived_runtime_serves_the_bound_index(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    app = create_app(runtime, project_root=ROOT)

    response = app.test_client().get("/")
    routes = [
        rule.rule for rule in app.url_map.iter_rules() if rule.endpoint != "static"
    ]

    assert response.status_code == 200
    assert b"Actions" in response.data
    assert b"Add" in response.data
    assert len(routes) == 31
    assert len(set(routes)) == 28
    assert runtime.paths.database is not None
    assert runtime.paths.database.is_file()
    assert archived._RUNTIME_PROCESS_ENV["CAREER_AGENT_WORKBENCH_DATABASE"] == str(
        runtime.paths.database
    )


def test_archived_runtime_reads_do_not_rewrite_initialized_database(
    tmp_path: Path,
) -> None:
    runtime = _demo_runtime(tmp_path)
    assert runtime.paths.database is not None
    before = hashlib.sha256(runtime.paths.database.read_bytes()).hexdigest()

    app = create_app(runtime, project_root=ROOT)
    after_create = hashlib.sha256(runtime.paths.database.read_bytes()).hexdigest()
    response = app.test_client().get("/")
    after_get = hashlib.sha256(runtime.paths.database.read_bytes()).hexdigest()

    assert response.status_code == 200
    assert before == after_create == after_get


def test_archived_resume_edit_updates_canonical_variant_and_revision(
    tmp_path: Path,
) -> None:
    runtime = _demo_runtime(tmp_path)
    app = create_app(runtime, project_root=ROOT)
    assert app.test_client().get("/").status_code == 200
    assert runtime.paths.database is not None
    store = ApplicationStateStore(runtime.paths)
    before = store.get_workflow_snapshot("demo-platform-001")
    original_yaml = before.active_resume_yaml
    assert original_yaml is not None
    edited = yaml.safe_load(original_yaml)
    edited["basics"]["summary"] = "Synthetic compatibility edit."
    edited_yaml = yaml.safe_dump(edited, sort_keys=False, allow_unicode=False)

    archived.save_application_resume_edit(
        database_path=runtime.paths.database,
        job_id="demo-platform-001",
        application_resume_object=edited_yaml,
        template_path=(
            ROOT / "src/career_agent_workbench/templates/resume/master_resume.html.j2"
        ),
    )

    after = store.get_workflow_snapshot("demo-platform-001")
    selected = after.application.selected_variant
    assert after.active_resume_yaml == edited_yaml
    assert selected is not None
    assert selected.variant_key == "v1"
    assert selected.application_resume["basics"]["summary"] == (
        "Synthetic compatibility edit."
    )
    assert after.application.application_resume["basics"]["summary"] == (
        "Synthetic compatibility edit."
    )
    assert after.application.application_resume_backup_target == "v1"
    assert after.revision != before.revision

    archived.revert_application_resume_edit(
        database_path=runtime.paths.database,
        job_id="demo-platform-001",
        template_path=(
            ROOT / "src/career_agent_workbench/templates/resume/master_resume.html.j2"
        ),
    )

    reverted = store.get_workflow_snapshot("demo-platform-001")
    assert reverted.active_resume_yaml == original_yaml
    assert reverted.application.application_resume_backup_target == "v1"


def test_archived_resume_edit_rejects_revision_drift_during_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _demo_runtime(tmp_path)
    create_app(runtime, project_root=ROOT)
    assert runtime.paths.database is not None
    store = ApplicationStateStore(runtime.paths)
    before = store.get_workflow_snapshot("demo-platform-001")
    assert before.active_resume_yaml is not None
    attempted = yaml.safe_load(before.active_resume_yaml)
    attempted["basics"]["summary"] = "Synthetic losing edit."
    attempted_yaml = yaml.safe_dump(attempted, sort_keys=False, allow_unicode=False)
    concurrent = yaml.safe_load(before.active_resume_yaml)
    concurrent["basics"]["summary"] = "Synthetic concurrent winner."
    concurrent_yaml = yaml.safe_dump(
        concurrent,
        sort_keys=False,
        allow_unicode=False,
    )

    def concurrent_write(_resume_html: str) -> bytes:
        current = store.get_resume_variant("demo-platform-001", "v1")
        store.upsert_resume_variant(
            "demo-platform-001",
            ResumeVariantWrite(
                variant_key="v1",
                variant_label=current.variant_label,
                source=current.source,
                application_resume_yaml=concurrent_yaml,
                resume_html="<html><body>Synthetic concurrent winner.</body></html>",
                resume_pdf=b"synthetic-concurrent-pdf",
            ),
        )
        return b"synthetic-losing-pdf"

    monkeypatch.setattr(archived, "render_resume_pdf_from_html", concurrent_write)

    with pytest.raises(ValueError, match="workflow state could not be updated"):
        archived.save_application_resume_edit(
            database_path=runtime.paths.database,
            job_id="demo-platform-001",
            application_resume_object=attempted_yaml,
            template_path=(
                ROOT
                / "src/career_agent_workbench/templates/resume/master_resume.html.j2"
            ),
        )

    after = store.get_workflow_snapshot("demo-platform-001")
    assert after.active_resume_yaml == concurrent_yaml
    assert after.application.application_resume_backup is None


def test_archived_variant_selection_updates_canonical_projection(
    tmp_path: Path,
) -> None:
    runtime = _demo_runtime(tmp_path)
    create_app(runtime, project_root=ROOT)
    assert runtime.paths.database is not None
    store = ApplicationStateStore(runtime.paths)
    snapshot = store.get_workflow_snapshot("demo-platform-001")
    assert snapshot.active_resume_yaml is not None
    refined = yaml.safe_load(snapshot.active_resume_yaml)
    refined["basics"]["summary"] = "Synthetic refined selection."
    refined_yaml = yaml.safe_dump(refined, sort_keys=False, allow_unicode=False)
    store.upsert_resume_variant(
        "demo-platform-001",
        ResumeVariantWrite(
            variant_key="v2",
            variant_label="Synthetic v2",
            source="synthetic_test",
            parent_variant_key="v1",
            application_resume_yaml=refined_yaml,
            resume_html="<html><body>Synthetic refined selection.</body></html>",
            resume_pdf=b"synthetic-pdf",
        ),
    )
    store.select_resume_variant("demo-platform-001", "v1")

    selected = archived.select_application_resume_variant(
        database_path=runtime.paths.database,
        job_id="demo-platform-001",
        variant_key="v2",
    )

    after = store.get_workflow_snapshot("demo-platform-001")
    assert selected == {"variant_key": "v2", "variant_label": "Refined v2"}
    assert after.application.selected_resume_variant == "v2"
    assert after.application.resume_variant_selection_mode == "manual"
    assert after.active_resume_yaml == refined_yaml
    assert after.application.application_resume["basics"]["summary"] == (
        "Synthetic refined selection."
    )


def test_archived_add_and_delete_preserve_public_state_contract(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    create_app(runtime, project_root=ROOT)
    assert runtime.paths.database is not None
    archived.upsert_application_artifact(
        database_path=runtime.paths.database,
        job_id="fictional-job-001",
        company="Example Cooperative",
        job_title="Synthetic Platform Engineer",
        linkedin_url="https://jobs.example.com/fictional-job-001",
        resume_path=None,
        job_description="Synthetic Python and SQLite role.",
        prompt_job_description="Python and SQLite.",
        date_posted="2042-04-10",
        experience_level="Senior",
    )
    store = ApplicationStateStore(runtime.paths)

    record = store.get_application("fictional-job-001")
    assert record.company == "Example Cooperative"
    assert record.job_url == "https://jobs.example.com/fictional-job-001"
    assert record.source == "generic"
    assert record.prompt_job_description == "Python and SQLite."

    assert (
        archived.delete_applications(
            database_path=runtime.paths.database,
            job_ids=["fictional-job-001"],
        )
        == 1
    )
    with sqlite3.connect(runtime.paths.database) as connection:
        application_count = connection.execute(
            "SELECT COUNT(*) FROM applications"
        ).fetchone()[0]
        variant_count = connection.execute(
            "SELECT COUNT(*) FROM application_resume_variants"
        ).fetchone()[0]
    assert application_count == 0
    assert variant_count == 0


def test_archived_runtime_copies_only_to_configured_downloads(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    app = create_app(runtime, project_root=ROOT)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert runtime.paths.database is not None
    with sqlite3.connect(runtime.paths.database) as connection:
        connection.execute(
            """
            INSERT INTO applications (
                job_id, company, job_title, linkedin_url, resume_filename,
                resume_content, resume_mime_type, source_resume_path,
                applied_to, notes, imported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fixture-job",
                "Example Company",
                "Platform Engineer",
                "https://jobs.example.com/fixture-job",
                "fixture_resume.pdf",
                b"fixture-pdf",
                "application/pdf",
                "",
                "No",
                "",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()

    response = client.post(
        "/resumes/fixture-job/copy-to-downloads",
        data={"return_to": "/"},
    )

    assert response.status_code == 302
    assert runtime.paths.download_dir is not None
    assert (
        runtime.paths.download_dir / "fixture_resume.pdf"
    ).read_bytes() == b"fixture-pdf"


def test_archived_runtime_is_the_packaged_console_entrypoint() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    included = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    assert metadata["project"]["scripts"]["career-agent-workbench-webapp"] == (
        "career_agent_workbench.webapp_archive_runtime:main"
    )
    assert "/src/career_agent_workbench/_archived_flask_source.py" in included
    assert "/src/career_agent_workbench/webapp_archive_runtime.py" in included
    assert "/tests/test_archived_flask_source.py" in included
