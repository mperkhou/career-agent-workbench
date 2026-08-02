from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
EXACT_SKILLS = {
    "career-agent-workbench",
    "master-resume-yaml",
    "manual-resume-passthrough",
    "agentic-workflow-init",
    "agentic-workflow-controller",
}


def test_exact_five_public_skill_trees_are_product_current() -> None:
    assert {path.name for path in SKILLS.iterdir() if path.is_dir()} == EXACT_SKILLS
    for name in EXACT_SKILLS:
        assert (SKILLS / name / "SKILL.md").is_file()

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SKILLS.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".json"}
    )
    assert "linkedin-career-mcp" not in text.casefold()
    assert "/Users/" not in text
    assert "linkedin-career-" not in text


def test_controller_includes_only_its_required_helper_assets() -> None:
    controller = SKILLS / "agentic-workflow-controller"
    required = {
        "SKILL.md",
        "agents/openai.yaml",
        "assets/artifact-manifest.schema.json",
        "assets/evidence-route.prompt.md",
        "assets/workflow-plan.template.md",
        "assets/workflow-tracker.schema.json",
        "assets/workflow-tracker.template.json",
        "scripts/workflow_state.py",
    }
    assert {
        path.relative_to(controller).as_posix()
        for path in controller.rglob("*")
        if path.is_file()
    } == required

    completed = subprocess.run(
        [sys.executable, controller / "scripts" / "workflow_state.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "validate" in completed.stdout
