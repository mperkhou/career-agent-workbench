from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P13_SDIST_PATHS = {
    "/docs/behavior-coverage.md",
    "/examples/demo-workspace/README.md",
    "/examples/demo-workspace/.blacklist",
    "/examples/demo-workspace/profile/MASTER-RESUME.yml",
    "/examples/demo-workspace/profile/MP-MASTER-RESUME.txt",
    "/examples/demo-workspace/jobs/demo-platform-engineer.json",
    "/scripts/create_demo_workspace.py",
    "/tests/test_demo_workspace.py",
    "/tests/test_release_metadata.py",
}
P16_ASSETS = {
    "/docs/assets/aro-application-workflow.svg",
    "/docs/assets/background-progress-annotated.png",
    "/docs/assets/cover-letter-editor-annotated.png",
    "/docs/assets/job-description-diff-annotated.png",
    "/docs/assets/job-description-editor-annotated.png",
    "/docs/assets/master-resume-object-build.svg",
    "/docs/assets/resume-editor-annotated.png",
    "/docs/assets/resume-variant-review-annotated.png",
    "/docs/assets/tracker-actions-menu-annotated.png",
    "/docs/assets/tracker-add-seed-annotated.png",
    "/docs/assets/tracker-main-annotated.png",
}
CONSOLE_SCRIPTS = {
    "career-agent-workbench": "career_agent_workbench.__main__:main",
    "career-agent-workbench-audit-jods": (
        "career_agent_workbench.jod_cleaner_audit:main"
    ),
    "career-agent-workbench-refine-resume": (
        "career_agent_workbench.resume_refinement_cli:main"
    ),
    "career-agent-workbench-seed-jobs": (
        "career_agent_workbench.workflows.matching:main"
    ),
    "career-agent-workbench-webapp": (
        "career_agent_workbench.webapp_archive_runtime:main"
    ),
    "career-agent-workbench-mcp": "career_agent_workbench.server:main",
}


def _metadata() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))


def _top_changelog_version() -> str:
    changelog = (ROOT / "CHANGELOG.md").read_text("utf-8")
    match = re.search(r"^## \[([^]]+)]", changelog, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_project_changelog_and_release_note_align() -> None:
    version = _metadata()["project"]["version"]
    assert version == _top_changelog_version()
    assert (ROOT / "docs" / "release-notes" / f"{version}.md").is_file()


def test_p13_paths_are_explicit_sdist_members() -> None:
    metadata = _metadata()
    included = set(metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert P13_SDIST_PATHS <= included


def test_p16_readme_assets_and_local_links_are_aligned() -> None:
    metadata = _metadata()
    assert metadata["project"]["readme"] == "README.md"
    included = set(metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert {"/README.md", *P16_ASSETS} <= included

    readme = (ROOT / "README.md").read_text("utf-8")
    assert all(f"]({asset[1:]})" in readme for asset in P16_ASSETS)

    root = ROOT.resolve()
    local_targets = re.findall(r"!?\[[^]]*]\(([^)]+)\)", readme)
    assert local_targets
    for target in local_targets:
        assert not target.startswith(("http://", "https://", "mailto:"))
        relative = target.split("#", 1)[0]
        resolved = (root / relative).resolve()
        resolved.relative_to(root)
        assert resolved.exists()


def test_restored_modules_resources_tests_scripts_and_skills_are_in_sdist() -> None:
    metadata = _metadata()
    declared = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    included = set(declared)
    assert len(included) == len(declared)

    expected_paths = {
        "/scripts/check_public_safety.py",
        "/scripts/workbench_operator.py",
        "/src/career_agent_workbench/cover_letter_rendering.py",
        "/tests/test_cover_letter_rendering.py",
        "/tests/test_guidance.py",
        "/tests/test_makefile_workflows.py",
        "/tests/test_mcp_integration.py",
        "/tests/test_operator_helpers.py",
        "/tests/test_public_safety.py",
        "/tests/test_release_metadata.py",
    }
    for pattern in (
        "src/career_agent_workbench/webapp*.py",
        "src/career_agent_workbench/templates/webapp/*",
        "src/career_agent_workbench/static/webapp/*",
        "tests/test_webapp*.py",
        "skills/**/*",
    ):
        expected_paths.update(
            f"/{path.relative_to(ROOT).as_posix()}"
            for path in ROOT.glob(pattern)
            if path.is_file()
        )
    assert expected_paths <= included
    for relative in included:
        assert (ROOT / relative.removeprefix("/")).is_file()


def test_six_console_scripts_and_operator_guidance_are_current() -> None:
    metadata = _metadata()
    assert metadata["project"]["scripts"] == CONSOLE_SCRIPTS
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for marker in (
        "make install",
        "make start-website",
        "make skill-link",
        "career-agent-workbench-mcp",
    ):
        assert marker in readme
    assert "CAREER_AGENT_WORKBENCH_PRIVATE_ENV_FILE" in env_example
    assert "CAREER_AGENT_WORKBENCH_DOWNLOAD_DIR" in env_example
