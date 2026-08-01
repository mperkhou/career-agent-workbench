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
