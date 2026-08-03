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


def test_tracked_demo_source_is_bounded_coherent_and_fictional() -> None:
    actual = {path.relative_to(SOURCE) for path in SOURCE.rglob("*") if path.is_file()}
    assert actual == MANIFEST
    size_limits = {path: 20_000 for path in MANIFEST}
    size_limits[Path("profile/MP-MASTER-RESUME.txt")] = 40_000
    size_limits[Path("profile/MASTER-RESUME.yml")] = 100_000
    assert all(
        0 < (SOURCE / path).stat().st_size < size_limits[path] for path in MANIFEST
    )
    assert not any(
        path.suffix.casefold() in {".db", ".sqlite", ".sqlite3", ".pdf", ".docx"}
        for path in actual
    )

    resume_path = SOURCE / "profile/MASTER-RESUME.yml"
    resume = yaml.safe_load(resume_path.read_text("utf-8"))
    assert initialize_application_resume_object(resume_path)["basics"]["name"] == (
        "Tessa Rowan"
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
    assert source_text.startswith("Tessa Rowan\nSenior Platform & Reliability Engineer")
    assert "tessa.rowan@example.test" in source_text
    assert "https://tessa-rowan.example.test" in source_text
    assert (
        "The person, organizations, locations, dates, credentials, and project "
        "examples in this resume are fictionalized for a public demonstration"
        in source_text
    )
    assert "Northbridge Systems Cooperative" in source_text
    assert "Alder Creek Medical Group" in source_text
    section_headings = (
        "Professional Summary",
        "Core Technical Skills",
        "Professional Experience",
        "Education",
        "Certifications",
        "Portfolio",
    )
    section_positions = [
        source_text.index(f"\n{heading}\n") for heading in section_headings
    ]
    assert section_positions == sorted(section_positions)

    skills_text = source_text.split("\nCore Technical Skills\n", 1)[1].split(
        "\nProfessional Experience\n", 1
    )[0]
    assert sum(line.startswith("- ") for line in skills_text.splitlines()) == 14

    role_markers = (
        "Northbridge Systems Cooperative | Remote",
        "Alder Creek Medical Group | Fort Haven, CO",
        "Grayhaven Health Network | Pinehaven, CO",
        "Harborline Media Partners | Cedar Ridge, CO",
        "Lumina Imaging Systems | Silver Falls, CO",
        "Calder Ridge Research Institute | Fairmont, CO",
        "Redstone Valley University | Fairmont, CO",
        "Education",
    )
    role_bullets = (17, 7, 9, 7, 10, 7, 3)
    for start, end, expected in zip(role_markers, role_markers[1:], role_bullets):
        role_text = source_text.split(f"\n{start}\n", 1)[1].split(f"\n{end}\n", 1)[0]
        assert sum(line.startswith("- ") for line in role_text.splitlines()) == expected
    long_role = source_text.split(f"\n{role_markers[0]}\n", 1)[1].split(
        f"\n{role_markers[1]}\n", 1
    )[0]
    assert sum(line.startswith("Category: ") for line in long_role.splitlines()) == 13

    education_text = source_text.split("\nEducation\n", 1)[1].split(
        "\nCertifications\n", 1
    )[0]
    certifications_text = source_text.split("\nCertifications\n", 1)[1].split(
        "\nPortfolio\n", 1
    )[0]
    assert sum(line.startswith("- ") for line in education_text.splitlines()) == 2
    assert sum(line.startswith("- ") for line in certifications_text.splitlines()) == 3
    assert "https://code.example.test/" in source_text
    normal_entries = source_text.split("\nProfessional Summary\n", 1)[1].replace(
        "example.test", "reserved-domain"
    )
    placeholder_words = {"demo", "example", "zz", "invented", "fictional"}
    visible_words = {
        word.strip(".,:;()[]").casefold() for word in normal_entries.split()
    }
    assert placeholder_words.isdisjoint(visible_words)

    required_sections = (
        "header_top",
        "professional_summary",
        "core_technical_skills",
        "professional_experience",
        "education",
        "certifications",
        "portfolio",
    )
    assert resume["section_order"] == list(required_sections)
    assert all(resume[section]["render"] is True for section in required_sections)
    assert resume["basics"]["render"] is True
    assert resume["source"] == {
        "text_path": "profile/MP-MASTER-RESUME.txt",
        "pdf_path": None,
        "page_count": None,
        "extraction_method": "g01_approved_public_text",
    }
    assert (
        resume["basics"]["name"]
        == resume["header_top"]["line_1_name_header_text"]
        == "Tessa Rowan"
    )
    assert resume["basics"]["summary"] == resume["professional_summary"]["paragraph"]
    assert resume["professional_summary"]["summary_note"] in source_text

    source_skill_lines = [
        line.removeprefix("- ")
        for line in skills_text.splitlines()
        if line.startswith("- ")
    ]
    source_skills = {
        category: [item.strip() for item in values.split(",")]
        for category, values in (line.split(": ", 1) for line in source_skill_lines)
    }
    skill_buckets = resume["core_technical_skills"]["bullet_points"]
    assert len(skill_buckets) == 14
    assert [bucket["order"] for bucket in skill_buckets] == list(range(1, 15))
    assert [bucket["category"] for bucket in skill_buckets] == list(source_skills)
    skill_inventory = {}
    for bucket in skill_buckets:
        category = bucket["category"]
        items = bucket["items"]
        combined = items["primary"] + items["additional"]
        assert combined == source_skills[category]
        assert not set(items["primary"]) & set(items["additional"])
        assert set(items["match_terms"]).issubset(combined)
        assert all(items["match_terms"].values())
        assert bucket["jod_matched_items"] == []
        skill_inventory[category] = set(combined)

    jobs = resume["professional_experience"]["jobs"]
    assert [job["order"] for job in jobs] == list(range(1, 8))
    assert [len(job["bullet_points"]) for job in jobs] == list(role_bullets)
    assert all(job["render"] is True for job in jobs)
    assert sum(bool(job["line_2"]["position_intro_text"]) for job in jobs) == 1
    assigned_category_links = 0
    linked_category_groups = 0
    linked_skill_values = 0
    for job in jobs:
        line_1 = job["line_1"]
        assert line_1["company_name_text"] in source_text
        assert line_1["position_name_text"] in source_text
        assert line_1["position_dates_text"] in source_text
        intro = job["line_2"]["position_intro_text"]
        if intro:
            assert f"Role Overview: {intro}" in source_text
        assert [bullet["order"] for bullet in job["bullet_points"]] == list(
            range(1, len(job["bullet_points"]) + 1)
        )
        for bullet in job["bullet_points"]:
            assert bullet["render"] is True
            assert f"- {bullet['text']}" in source_text
            assert bullet["bullet_point_total_match_count"] == 0
            assert bullet["categories"]["matched"] == []
            assert set(bullet["categories"]["assigned"]).issubset(skill_inventory)
            assigned_category_links += len(bullet["categories"]["assigned"])
            for skill_group in bullet["skills"]:
                category = skill_group["category"]
                assert category in skill_inventory
                assert set(skill_group["matched"]).issubset(skill_inventory[category])
                assert skill_group["jod_match_count"] == 0
                linked_category_groups += 1
                linked_skill_values += len(skill_group["matched"])
    assert assigned_category_links == 59
    assert linked_category_groups == 146
    assert linked_skill_values == 228
    assert "job_opening_description" not in resume

    education = resume["education"]["entries"]
    assert len(education) == 1
    assert len(education[0]["bullet_points"]) == 2
    assert education[0]["line_1"]["institution_name_text"] in source_text
    assert education[0]["line_2"]["degree_name_text"] in source_text
    assert education[0]["line_2"]["degree_dates_text"] in source_text
    assert all(
        f"- {bullet['text']}" in source_text for bullet in education[0]["bullet_points"]
    )
    certification_bullets = resume["certifications"]["bullet_points"]
    assert len(certification_bullets) == 3
    assert all(f"- {bullet['text']}" in source_text for bullet in certification_bullets)
    projects = resume["portfolio"]["projects"]
    assert len(projects) == 1
    project = projects[0]
    assert (
        f"{project['title_text']} | {project['url']} | "
        f"{project['description_text']}" in source_text
    )

    job = JobDetails.model_validate_json(
        (SOURCE / "jobs/demo-platform-engineer.json").read_text("utf-8")
    )
    assert job.job_id == "demo-platform-001"
    assert job.company == "Rivermark Platform Services"
    assert job.title == "Senior Platform Automation Engineer"
    assert job.location == "Remote, United States"
    assert str(job.job_url).startswith("https://jobs.example.test/")
    assert job.workplace_type == "remote"
    assert job.employment_type == "full_time"
    assert job.seniority_level == "mid_senior"
    assert "Python" in (job.description or "")
    blacklist = (SOURCE / ".blacklist").read_text("utf-8").strip()
    assert blacklist == "Granite Harbor Consulting"
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
    assert record.company == "Rivermark Platform Services"
    assert record.job_title == "Senior Platform Automation Engineer"
    assert record.job_description and "Python automation services" in (
        record.job_description
    )
    assert record.prompt_job_description
    assert record.selected_resume_variant == "v1"
    assert record.resume_variant_selection_mode == "auto"
    assert record.application_resume is not None
    assert record.cover_letter is not None
    assert record.cover_letter["requires_human_review"] is True
    assert record.cover_letter["company"] == "Rivermark Platform Services"
    assert "Tessa Rowan" in record.cover_letter["paragraphs"][0]
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
    assert "Tessa Rowan" in generated["resume-v1.html"]
    assert "Alder Creek Medical Group" in generated["resume-v1.html"]

    app = create_app(resolved, project_root=ROOT)
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"demo-platform-001" in response.data
    assert b"Rivermark Platform Services" in response.data
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
