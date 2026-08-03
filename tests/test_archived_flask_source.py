from __future__ import annotations

import ast
import hashlib
import sqlite3
import tomllib
from pathlib import Path

from career_agent_workbench import _archived_flask_source as archived
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
