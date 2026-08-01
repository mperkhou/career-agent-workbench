from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from career_agent_workbench.application_resume import (
    initialize_application_resume_object,
)
from career_agent_workbench.application_state import ApplicationStateStore
from career_agent_workbench.config import (
    RuntimeOverrides,
    load_runtime_config,
)
from career_agent_workbench.models import JobDetails
from career_agent_workbench.webapp import create_app

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "demo-workspace"
MANIFEST = {
    Path("README.md"),
    Path(".blacklist"),
    Path("profile/MASTER-RESUME.yml"),
    Path("profile/MP-MASTER-RESUME.txt"),
    Path("jobs/demo-platform-engineer.json"),
}


def _load_factory():
    path = ROOT / "scripts" / "create_demo_workspace.py"
    spec = importlib.util.spec_from_file_location("p13_demo_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_digests() -> dict[Path, str]:
    return {
        relative: hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest()
        for relative in MANIFEST
    }


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _runtime(workspace: Path):
    return load_runtime_config(
        overrides=RuntimeOverrides(workspace=workspace),
        environ={},
        cwd=workspace,
    )


def test_tracked_demo_source_is_small_coherent_and_fictional() -> None:
    actual = {path.relative_to(SOURCE) for path in SOURCE.rglob("*") if path.is_file()}
    assert actual == MANIFEST
    assert all(0 < (SOURCE / path).stat().st_size < 20_000 for path in MANIFEST)
    assert not any(
        path.suffix.casefold() in {".db", ".sqlite", ".sqlite3", ".pdf", ".docx"}
        for path in actual
    )

    resume_path = SOURCE / "profile/MASTER-RESUME.yml"
    resume = yaml.safe_load(resume_path.read_text("utf-8"))
    assert initialize_application_resume_object(resume_path)["basics"]["name"] == (
        "Avery Demo"
    )
    email_fields = [resume["basics"]["email"]]
    email_fields.extend(
        item for item in resume["header_top"]["contact_items"] if "@" in item
    )
    email_fields.extend(
        link["label"] for link in resume["header_top"]["links"] if "@" in link["label"]
    )
    assert email_fields
    assert all(email.endswith("@example.test") for email in email_fields)
    web_fields = [resume["basics"]["url"]]
    web_fields.extend(link["url"] for link in resume["header_top"]["links"])
    for value in web_fields:
        parsed = urlsplit(value)
        if parsed.scheme == "mailto":
            assert parsed.path.endswith("@example.test")
        else:
            assert parsed.scheme == "https"
            assert parsed.hostname is not None
            assert parsed.hostname.endswith(".example.test")
    source_text = (SOURCE / "profile/MP-MASTER-RESUME.txt").read_text("utf-8")
    assert "Avery Demo" in source_text
    assert "avery.demo@example.test" in source_text
    assert "Nimbus Quay Example Labs" in source_text
    assert "Cedar & Comet Example Cooperative" in source_text

    job = JobDetails.model_validate_json(
        (SOURCE / "jobs/demo-platform-engineer.json").read_text("utf-8")
    )
    assert job.job_id == "demo-platform-001"
    assert job.company == "Nimbus Quay Example Labs"
    assert job.title == "Demo Platform Engineer"
    assert job.location == "Example City, ZZ"
    assert str(job.job_url).startswith("https://jobs.example.test/")
    assert job.workplace_type == "remote"
    assert job.employment_type == "full_time"
    assert job.seniority_level == "mid_senior"
    assert "Python" in (job.description or "")
    blacklist = (SOURCE / ".blacklist").read_text("utf-8").strip()
    assert blacklist == "Obsidian Kite Demo Works"
    assert blacklist != job.company
    readme = (SOURCE / "README.md").read_text("utf-8")
    assert "fictional" in readme.casefold()
    assert "SQLite" in readme and "PDF" in readme


def test_factory_materializes_one_governed_demo_and_flask_row(tmp_path: Path) -> None:
    factory = _load_factory()
    before = _source_digests()
    workspace = tmp_path / "workspace"
    runtime = factory.create_demo_workspace(SOURCE, workspace)
    resolved = _runtime(workspace)
    assert runtime.paths == resolved.paths

    store = ApplicationStateStore(resolved.paths)
    rows = store.list_applications("all")
    assert len(rows) == 1
    record = rows[0]
    assert record.job_id == "demo-platform-001"
    assert record.company == "Nimbus Quay Example Labs"
    assert record.job_title == "Demo Platform Engineer"
    assert record.job_description and "Python services" in record.job_description
    assert record.prompt_job_description
    assert record.selected_resume_variant == "v1"
    assert record.resume_variant_selection_mode == "auto"
    assert record.application_resume is not None
    assert record.cover_letter is not None
    assert record.cover_letter["requires_human_review"] is True
    variants = store.list_resume_variants(record.job_id)
    assert len(variants) == 1
    assert variants[0].variant_key == "v1"
    assert variants[0].application_resume == record.application_resume

    examples = workspace / "output" / "demo-examples"
    generated = {path.name: path.read_text("utf-8") for path in examples.iterdir()}
    assert set(generated) == {
        "application-resume-v1.yml",
        "cover-letter.json",
        "resume-v1.html",
    }
    assert yaml.safe_load(generated["application-resume-v1.yml"]) == _plain_json(
        variants[0].application_resume
    )
    assert json.loads(generated["cover-letter.json"]) == _plain_json(
        record.cover_letter
    )
    assert generated["resume-v1.html"] == variants[0].resume_html
    assert "Avery Demo" in generated["resume-v1.html"]
    assert "Cedar &amp; Comet Example Cooperative" in generated["resume-v1.html"]

    app = create_app(resolved, command_executor=lambda _argv: 0, project_root=ROOT)
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"demo-platform-001" in response.data
    assert b"Nimbus Quay Example Labs" in response.data
    assert _source_digests() == before


def test_factory_rerun_is_semantically_idempotent_and_preserves_files(
    tmp_path: Path,
) -> None:
    factory = _load_factory()
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside sentinel", encoding="utf-8")
    factory.create_demo_workspace(SOURCE, workspace)
    unrelated = workspace / "keep-me.txt"
    unrelated.write_text("workspace sentinel", encoding="utf-8")
    examples = workspace / "output" / "demo-examples"
    before = {path.name: path.read_text("utf-8") for path in examples.iterdir()}

    factory.create_demo_workspace(SOURCE, workspace)
    after = {path.name: path.read_text("utf-8") for path in examples.iterdir()}
    store = ApplicationStateStore(_runtime(workspace).paths)
    rows = store.list_applications("all")
    assert len(rows) == 1
    assert len(store.list_resume_variants(rows[0].job_id)) == 1
    assert rows[0].selected_resume_variant == "v1"
    assert before == after
    assert unrelated.read_text("utf-8") == "workspace sentinel"
    assert outside.read_text("utf-8") == "outside sentinel"


def test_factory_ignores_dotenv_redirects_and_rejects_escaped_members(
    tmp_path: Path,
) -> None:
    factory = _load_factory()
    redirected_workspace = tmp_path / "redirected-workspace"
    redirected_workspace.mkdir()
    redirected_output = tmp_path / "redirected-output"
    redirected_output.mkdir()
    redirect_sentinel = redirected_output / "sentinel.txt"
    redirect_sentinel.write_text("outside sentinel", encoding="utf-8")
    (redirected_workspace / ".env").write_text(
        "\n".join(
            (
                f"CAREER_AGENT_WORKBENCH_PROFILE_DIR={redirected_output}",
                f"CAREER_AGENT_WORKBENCH_OUTPUT_DIR={redirected_output}",
                f"CAREER_AGENT_WORKBENCH_DATABASE={redirected_output / 'demo.sqlite3'}",
                f"CAREER_AGENT_WORKBENCH_BLACKLIST={redirected_output / 'blacklist'}",
            )
        ),
        encoding="utf-8",
    )

    runtime = factory.create_demo_workspace(SOURCE, redirected_workspace)

    assert runtime.env_file is None
    assert runtime.paths.profile_dir == redirected_workspace / "profile"
    assert runtime.paths.output_dir == redirected_workspace / "output"
    assert runtime.paths.database == (
        redirected_workspace / "output/tracking/applications.sqlite3"
    )
    assert runtime.paths.blacklist == redirected_workspace / ".blacklist"
    assert (redirected_workspace / "output/demo-examples/resume-v1.html").is_file()
    assert redirect_sentinel.read_text("utf-8") == "outside sentinel"
    assert set(redirected_output.iterdir()) == {redirect_sentinel}

    escaped_workspace = tmp_path / "escaped-workspace"
    escaped_workspace.mkdir()
    outside_profile = tmp_path / "outside-profile"
    outside_profile.mkdir()
    resume_sentinel = outside_profile / "MASTER-RESUME.yml"
    text_sentinel = outside_profile / "MP-MASTER-RESUME.txt"
    resume_sentinel.write_text("outside resume sentinel", encoding="utf-8")
    text_sentinel.write_text("outside text sentinel", encoding="utf-8")
    (escaped_workspace / "profile").symlink_to(
        outside_profile,
        target_is_directory=True,
    )

    with pytest.raises(factory.DemoWorkspaceError) as caught:
        factory.create_demo_workspace(SOURCE, escaped_workspace)

    assert str(caught.value) == "Demo workspace could not be created."
    assert resume_sentinel.read_text("utf-8") == "outside resume sentinel"
    assert text_sentinel.read_text("utf-8") == "outside text sentinel"
    assert not (escaped_workspace / ".blacklist").exists()
    assert not (escaped_workspace / "output").exists()
