from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATEFUL_TARGETS = (
    "seed-jobs",
    "audit-jods",
    "generate-draft-resumes",
    "regenerate-draft-resumes",
    "regenerate-aro-objects",
    "sync-draft-to-aro",
    "refine-draft-resumes",
    "highlight-draft-resumes",
    "manual-pass-resumes",
    "launch-website",
)
STATE_FLAGS = (
    "--workspace",
    "--database",
    "--output-dir",
    "--profile-dir",
    "--master-resume",
    "--master-resume-text",
    "--blacklist-path",
    "--tmp-dir",
    "--template",
)
PRIVATE_LITERALS = ("profile/", "output/", ".blacklist", "tmp/")
BATCH_TARGETS = (
    "generate-draft-resumes",
    "regenerate-draft-resumes",
    "regenerate-aro-objects",
    "sync-draft-to-aro",
    "highlight-draft-resumes",
)


def _dry_run(target: str, *variables: str) -> str:
    completed = subprocess.run(
        ["make", "--no-print-directory", "-n", target, *variables],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_setup_targets_use_one_public_venv_and_explicit_chromium_install(
    tmp_path: Path,
) -> None:
    isolated_venv = tmp_path / "operator-venv"
    assert not isolated_venv.exists()

    venv = _dry_run("venv", f"VENV={isolated_venv}")
    browser = _dry_run("install-browser", f"VENV={isolated_venv}")
    assert f'python3 -m venv "{isolated_venv}"' in venv
    assert f'{isolated_venv}/bin/python -m pip install -e ".[dev,browser]"' in venv
    assert f"{isolated_venv}/bin/python -m playwright install chromium" in browser
    assert "curl" not in browser
    assert "ollama" not in browser.casefold()


def test_operator_targets_use_public_venv_and_exact_bounded_helpers() -> None:
    linked = _dry_run("skill-link", "CODEX_SKILLS_DIR=temporary-skills")
    assert (
        ".venv/bin/python scripts/workbench_operator.py skills link --destination "
        '"temporary-skills"'
    ) in linked

    started = _dry_run("start-website")
    assert ".venv/bin/python scripts/workbench_operator.py website start" in started
    assert "--open-browser" not in started
    assert "lsof" not in started
    assert "kill" not in started

    opted_in = _dry_run("start-website", "OPEN_BROWSER=true")
    assert opted_in.count("--open-browser") == 1
    stopped = _dry_run("stop-website")
    assert (
        stopped.strip() == ".venv/bin/python scripts/workbench_operator.py website stop"
    )


def test_public_commands_resolve_from_public_venv() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for command in (
        "career-agent-workbench-seed-jobs",
        "career-agent-workbench-audit-jods",
        "career-agent-workbench-refine-resume",
        "career-agent-workbench-webapp",
    ):
        assert f"$(VENV)/bin/{command}" in makefile
    assert "PYTHON ?= $(VENV_PYTHON)" in makefile


@pytest.mark.parametrize("target", STATEFUL_TARGETS)
def test_default_stateful_make_targets_emit_no_private_state(target: str) -> None:
    output = _dry_run(target)
    for flag in STATE_FLAGS:
        assert flag not in output
    for literal in PRIVATE_LITERALS:
        assert literal not in output


def test_seed_make_overrides_emit_one_exact_flag_each() -> None:
    output = _dry_run(
        "seed-jobs",
        "WORKSPACE=workspace-value",
        "DATABASE=database-value",
        "OUTPUT_DIR=output-value",
        "PROFILE_DIR=profile-value",
        "MASTER_RESUME=master-value",
        "BLACKLIST=blacklist-value",
    )
    expected = {
        "--workspace": "workspace-value",
        "--database": "database-value",
        "--output-dir": "output-value",
        "--profile-dir": "profile-value",
        "--master-resume": "master-value",
        "--blacklist-path": "blacklist-value",
    }
    for flag, value in expected.items():
        assert output.count(flag) == 1
        assert value in output


def test_web_make_overrides_emit_one_exact_flag_each() -> None:
    output = _dry_run(
        "launch-website",
        "WORKSPACE=workspace-value",
        "DATABASE=database-value",
        "OUTPUT_DIR=output-value",
        "PROFILE_DIR=profile-value",
        "MASTER_RESUME=master-value",
        "MASTER_RESUME_TEXT=source-value",
        "BLACKLIST=blacklist-value",
        "TMP_DIR=temporary-value",
        "HOST=127.0.0.2",
        "PORT=9876",
    )
    tokens = shlex.split(output)
    expected = {
        "--workspace": "workspace-value",
        "--database": "database-value",
        "--output-dir": "output-value",
        "--profile-dir": "profile-value",
        "--master-resume": "master-value",
        "--master-resume-text": "source-value",
        "--blacklist-path": "blacklist-value",
        "--tmp-dir": "temporary-value",
        "--host": "127.0.0.2",
        "--port": "9876",
    }
    for flag, value in expected.items():
        assert tokens.count(flag) == 1
        assert value in tokens


def test_model_and_remaining_path_overrides_are_presence_aware() -> None:
    output = _dry_run(
        "highlight-draft-resumes",
        "WORKSPACE=workspace-value",
        "DATABASE=database-value",
        "OUTPUT_DIR=output-value",
        "MASTER_RESUME=master-value",
        "MASTER_RESUME_TEXT=source-value",
        "TMP_DIR=temporary-value",
        "CODEX_MODEL=shared-model",
        "HIGHLIGHT_CODEX_MODEL=workflow-model",
        "CODEX_REASONING_EFFORT=shared-effort",
        "HIGHLIGHT_CODEX_REASONING_EFFORT=workflow-effort",
    )
    tokens = shlex.split(output)
    for flag in (
        "--workspace",
        "--database",
        "--output-dir",
        "--master-resume",
        "--master-resume-text",
        "--tmp-dir",
        "--codex-model",
        "--codex-reasoning-effort",
    ):
        assert tokens.count(flag) == 1
    assert "workflow-model" in output and "shared-model" not in output
    assert "workflow-effort" in output and "shared-effort" not in output


def test_resume_template_and_highlight_selectors_propagate_once() -> None:
    tokens = shlex.split(
        _dry_run(
            "highlight-draft-resumes",
            "RESUME_TEMPLATE=private-template.html",
            "HIGHLIGHT_MAX_STRONG_SPANS_PER_BULLET=2",
            "HIGHLIGHT_EXPERIENCE_COMPANY=Example Cooperative",
            "HIGHLIGHT_EXPERIENCE_JOB_ORDER=3",
        )
    )
    expected = {
        "--template": "private-template.html",
        "--max-strong-spans-per-bullet": "2",
        "--experience-company": "Example Cooperative",
        "--experience-job-order": "3",
    }
    for flag, value in expected.items():
        assert tokens.count(flag) == 1
        assert value in tokens


@pytest.mark.parametrize("value", ["1", "true"])
def test_regeneration_force_is_explicit_and_conditional(value: str) -> None:
    assert (
        shlex.split(
            _dry_run("regenerate-draft-resumes", f"FIRST_DRAFT_FORCE={value}")
        ).count("--force")
        == 1
    )


@pytest.mark.parametrize("value", [None, "", "0", "false", "unexpected"])
def test_regeneration_without_enabled_force_preserves_existing_drafts(
    value: str | None,
) -> None:
    variables = () if value is None else (f"FIRST_DRAFT_FORCE={value}",)
    assert "--force" not in shlex.split(
        _dry_run("regenerate-draft-resumes", *variables)
    )


@pytest.mark.parametrize("target", BATCH_TARGETS)
@pytest.mark.parametrize("selection", [None, "", "all"])
def test_batch_make_all_selectors_emit_no_job_id(
    target: str,
    selection: str | None,
) -> None:
    variables = () if selection is None else (f"JOB_IDS={selection}",)
    assert "--job-id" not in shlex.split(_dry_run(target, *variables))


@pytest.mark.parametrize("target", BATCH_TARGETS)
def test_batch_make_explicit_ids_emit_one_flag_per_token(target: str) -> None:
    tokens = shlex.split(_dry_run(target, "JOB_IDS=fictional-a fictional-b"))
    assert tokens.count("--job-id") == 2
    assert "fictional-a" in tokens and "fictional-b" in tokens


@pytest.mark.parametrize("selection", [None, "", "all"])
def test_refinement_make_all_selectors_emit_all_active_once(
    selection: str | None,
) -> None:
    variables = () if selection is None else (f"JOB_IDS={selection}",)
    tokens = shlex.split(_dry_run("refine-draft-resumes", *variables))
    assert tokens.count("--all-active") == 1
    assert "--job-id" not in tokens


def test_refinement_make_explicit_ids_exclude_all_active() -> None:
    tokens = shlex.split(
        _dry_run("refine-draft-resumes", "JOB_IDS=fictional-a fictional-b")
    )
    assert tokens.count("--job-id") == 2
    assert "--all-active" not in tokens


@pytest.mark.parametrize("selection", [None, "", "all"])
def test_manual_make_all_selectors_stop_before_python(
    selection: str | None,
) -> None:
    variables = () if selection is None else (f"JOB_IDS={selection}",)
    output = _dry_run("manual-pass-resumes", *variables)
    assert "application_resume_manual_pass.py" not in output
    assert "explicit job IDs" in output


@pytest.mark.parametrize("selection", ["", "all"])
def test_manual_make_guard_fails_before_command_execution(selection: str) -> None:
    completed = subprocess.run(
        [
            "make",
            "manual-pass-resumes",
            f"JOB_IDS={selection}",
            "PYTHON=python-boundary-must-not-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "explicit job IDs" in completed.stderr
    assert "python-boundary-must-not-run" not in completed.stderr


@pytest.mark.parametrize(
    ("variables", "expected_model", "expected_effort"),
    [
        (
            (
                "CODEX_MODEL=shared-model",
                "CODEX_REASONING_EFFORT=shared-effort",
                "MANUAL_PASS_CODEX_MODEL=workflow-model",
                "MANUAL_PASS_CODEX_REASONING_EFFORT=workflow-effort",
            ),
            "workflow-model",
            "workflow-effort",
        ),
        (
            ("CODEX_MODEL=shared-model", "CODEX_REASONING_EFFORT=shared-effort"),
            "shared-model",
            "shared-effort",
        ),
        ((), None, None),
    ],
)
def test_manual_model_and_effort_precedence(
    variables: tuple[str, ...],
    expected_model: str | None,
    expected_effort: str | None,
) -> None:
    tokens = shlex.split(
        _dry_run("manual-pass-resumes", "JOB_IDS=fictional-job", *variables)
    )
    assert tokens.count("--codex-model") == int(expected_model is not None)
    assert tokens.count("--codex-reasoning-effort") == int(expected_effort is not None)
    if expected_model is not None:
        assert expected_model in tokens
    if expected_effort is not None:
        assert expected_effort in tokens


@pytest.mark.parametrize(
    "target",
    ["regenerate-resumes", "regenerate-resume-variants"],
)
def test_regeneration_aliases_preserve_all_and_explicit_selection(target: str) -> None:
    default_tokens = shlex.split(_dry_run(target))
    assert default_tokens.count("--all-active") == 1
    assert "--job-id" not in default_tokens

    explicit_tokens = shlex.split(_dry_run(target, "JOB_IDS=fictional-a fictional-b"))
    assert explicit_tokens.count("--job-id") == 4
    assert "--all-active" not in explicit_tokens


def test_second_pass_alias_preserves_all_selection() -> None:
    tokens = shlex.split(_dry_run("second-pass-refinement"))
    assert tokens.count("--all-active") == 1
    assert "--job-id" not in tokens
