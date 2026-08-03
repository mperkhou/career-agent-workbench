from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_public_safety.py"
    spec = importlib.util.spec_from_file_location("p14_public_safety", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _bad_content(kind: str) -> tuple[str, str, str]:
    if kind == "email":
        marker = "mailbox" + "@" + "vendor" + ".example"
        return "public.txt", marker, "TEXT_EMAIL"
    if kind == "personal-number":
        phone = "212" + "-555-" + "0200"
        ssn = "321" + "-54-" + "9876"
        return "public.txt", f"{phone} {ssn}", "TEXT_PERSONAL_NUMBER"
    if kind == "ticket":
        marker = "ticket " + "OPS" + "-4242"
        return "public.txt", marker, "TEXT_INTERNAL_TICKET"
    if kind == "machine-path":
        marker = "/" + "Users" + "/" + "sample-user" + "/workspace"
        return "public.txt", marker, "TEXT_MACHINE_PATH"
    if kind == "credential":
        token = "gh" + "p_" + ("x" * 32)
        header = ("-" * 5) + "BEGIN " + "PRIVATE" + " KEY" + ("-" * 5)
        return "public.txt", f"{token}\n{header}", "TEXT_CREDENTIAL"
    if kind == "example-domain":
        marker = "https://" + "portfolio." + "vendor" + ".example/path"
        return "examples/demo.txt", marker, "TEXT_EXAMPLE_DOMAIN"
    raise AssertionError("unsupported synthetic case")


@pytest.mark.parametrize(
    ("kind", "relative", "expected_rule"),
    [
        ("private-root", "profile/source.txt", "PATH_PRIVATE_STATE"),
        ("dotenv", "docs/.env.local", "PATH_DOTENV"),
        ("database", "data/sample.sqlite3", "PATH_DATABASE"),
        ("generated", "generated-resumes/sample.txt", "PATH_GENERATED_ARTIFACT"),
        ("email", "", "TEXT_EMAIL"),
        ("personal-number", "", "TEXT_PERSONAL_NUMBER"),
        ("ticket", "", "TEXT_INTERNAL_TICKET"),
        ("machine-path", "", "TEXT_MACHINE_PATH"),
        ("credential", "", "TEXT_CREDENTIAL"),
        ("example-domain", "", "TEXT_EXAMPLE_DOMAIN"),
    ],
)
def test_representative_public_rule_classes(
    tmp_path: Path,
    kind: str,
    relative: str,
    expected_rule: str,
) -> None:
    if relative:
        content = "fictional public text"
    else:
        relative, content, expected_rule = _bad_content(kind)
    _write(tmp_path, relative, content)

    findings = CHECKER.scan_paths(tmp_path, [relative])

    assert expected_rule in {finding.rule for finding in findings}


def test_paths_are_normalized_and_nonregular_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("safe", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(target)

    findings = CHECKER.scan_paths(tmp_path, ["../escape", "linked.txt"])

    assert findings == (
        CHECKER.Finding("PATH_INVALID", "<invalid>"),
        CHECKER.Finding("PATH_NON_REGULAR", "linked.txt"),
    )


def test_findings_never_expose_the_matching_value(tmp_path: Path) -> None:
    relative, marker, _ = _bad_content("email")
    _write(tmp_path, relative, marker)

    findings = CHECKER.scan_paths(tmp_path, [relative])
    rendered = " ".join((*map(str, findings), *map(repr, findings)))

    assert marker not in rendered
    assert findings == (CHECKER.Finding("TEXT_EMAIL", relative),)


def test_public_identifiers_and_sanitized_cli_errors_remain_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content = (
        "CVE-2024-0001 SHA-256 ISO-27001 demo-platform-001 "
        "ticket CVE-2024 /browse/CVE-2024 "
        "https://github.com/example/repository/issues/123"
    )
    _write(tmp_path, "public.txt", content)
    assert CHECKER.scan_paths(tmp_path, ["public.txt"]) == ()

    marker = "unknown-" + "private-marker"
    with pytest.raises(SystemExit) as caught:
        CHECKER.main([marker])
    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert marker not in captured.out
    assert marker not in captured.err


def _wheel(path: Path, *, prohibited: bool = False) -> None:
    members = {
        "career_agent_workbench/_archived_flask_source.py": "",
        "career_agent_workbench/__init__.py": '__version__ = "1.0.0"\n',
        "career_agent_workbench/__main__.py": "def main(): return 0\n",
        "career_agent_workbench/cover_letter_rendering.py": "",
        "career_agent_workbench/static/webapp/app.js": "void 0;\n",
        "career_agent_workbench/templates/resume/master_resume.html.j2": (
            "{{ data.header_top.line_1_name_header_text }}"
        ),
        "career_agent_workbench/templates/webapp/add.html": "<main>Add</main>",
        "career_agent_workbench/templates/webapp/cover_letter_edit.html": (
            "<main>Cover letter</main>"
        ),
        "career_agent_workbench/templates/webapp/index.html": "<main>Demo</main>",
        "career_agent_workbench/templates/webapp/jod.html": "<main>JOD</main>",
        "career_agent_workbench/templates/webapp/resume_edit.html": (
            "<main>Resume</main>"
        ),
        "career_agent_workbench/templates/webapp/variant_review.html": (
            "<main>Variants</main>"
        ),
        "career_agent_workbench/webapp_actions.py": "",
        "career_agent_workbench/webapp_archive_runtime.py": "",
        "career_agent_workbench/webapp_artifacts.py": "",
        "career_agent_workbench/webapp_editors.py": "",
        "career_agent_workbench/webapp_ingestion.py": "",
        "career_agent_workbench/webapp_tracker.py": "",
        "career_agent_workbench-1.0.0.dist-info/METADATA": (
            "Name: career-agent-workbench\nVersion: 1.0.0\n"
        ),
    }
    if prohibited:
        members["career_agent_workbench/profile/private-state.txt"] = ""
        members["career_agent_workbench/tests/test_hidden.py"] = "assert True\n"
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _sdist(path: Path, *, public_text: str = "fictional public text") -> None:
    prefix = "career_agent_workbench-1.0.0"
    members = {
        "pyproject.toml": (
            '[tool.hatch.build.targets.sdist]\ninclude = ["/pyproject.toml", '
            '"/public.txt"]\n'
        ),
        "public.txt": public_text,
        "PKG-INFO": "Name: career-agent-workbench\nVersion: 1.0.0\n",
    }
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.sdist]\ninclude = ["/pyproject.toml", '
        '"/public.txt"]\n',
        encoding="utf-8",
    )
    return root


def test_synthetic_safe_wheel_and_sdist_use_the_artifact_scanner(
    tmp_path: Path,
) -> None:
    root = _artifact_root(tmp_path)
    wheel = tmp_path / "career_agent_workbench-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "career_agent_workbench-1.0.0.tar.gz"
    _wheel(wheel)
    _sdist(sdist)

    assert CHECKER.scan_artifacts(root, wheel, sdist) == ()


def test_wheel_rejects_prohibited_membership(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    wheel = tmp_path / "career_agent_workbench-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "career_agent_workbench-1.0.0.tar.gz"
    _wheel(wheel, prohibited=True)
    _sdist(sdist)

    findings = CHECKER.scan_artifacts(root, wheel, sdist)

    assert (
        CHECKER.Finding(
            "PATH_PRIVATE_STATE", "career_agent_workbench/profile/private-state.txt"
        )
        in findings
    )
    assert (
        CHECKER.Finding(
            "WHEEL_MEMBERSHIP", "career_agent_workbench/tests/test_hidden.py"
        )
        in findings
    )


def test_sdist_rejects_prohibited_text_content(tmp_path: Path) -> None:
    root = _artifact_root(tmp_path)
    wheel = tmp_path / "career_agent_workbench-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "career_agent_workbench-1.0.0.tar.gz"
    _wheel(wheel)
    _, marker, _ = _bad_content("email")
    _sdist(sdist, public_text=marker)

    findings = CHECKER.scan_artifacts(root, wheel, sdist)

    assert CHECKER.Finding("TEXT_EMAIL", "public.txt") in findings


def test_current_tracked_tree_passes() -> None:
    assert CHECKER.scan_tree(ROOT) == ()


def test_scan_tree_sanitizes_missing_root(tmp_path: Path) -> None:
    marker = "absent-" + "root-marker"
    missing = tmp_path / marker

    with pytest.raises(CHECKER.SafetyCheckError) as caught:
        CHECKER.scan_tree(missing)

    assert type(caught.value) is CHECKER.SafetyCheckError
    assert str(caught.value) == "Public-safety check failed."
    assert marker not in str(caught.value)
    assert str(missing) not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_ci_workflow_and_sdist_membership_are_proportional() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow_text = workflow_path.read_text("utf-8")
    workflow = yaml.safe_load(workflow_text)
    triggers = workflow.get("on", workflow.get(True))

    assert workflow["permissions"] == {"contents": "read"}
    assert set(triggers) == {"push", "pull_request"}
    assert triggers["push"]["branches"] == ["main"]
    assert set(workflow["jobs"]) == {"quality", "package-smoke"}
    assert workflow_text.count("actions/checkout@v6") == 2
    assert workflow_text.count("actions/setup-python@v6") == 2
    assert workflow_text.count("persist-credentials: false") == 2
    assert workflow_text.count("python -m build ") == 1
    assert workflow_text.count("shopt -s nullglob") == 2
    assert 'python-version: "3.12"' in workflow_text

    quality_runs = [
        step["run"] for step in workflow["jobs"]["quality"]["steps"] if "run" in step
    ]
    assert quality_runs.index("python scripts/check_public_safety.py tree --root .") < (
        quality_runs.index("python -m pip install '.[dev]'")
    )
    assert {"make lint", "make format-check", "make test"} <= set(quality_runs)

    package_runs = "\n".join(
        step["run"]
        for step in workflow["jobs"]["package-smoke"]["steps"]
        if "run" in step
    )
    package_run_steps = [
        step["run"]
        for step in workflow["jobs"]["package-smoke"]["steps"]
        if "run" in step
    ]
    assert package_run_steps.index(
        "python scripts/check_public_safety.py tree --root ."
    ) < package_run_steps.index("python -m pip install build")
    for required in (
        "scripts/check_public_safety.py artifacts",
        "python -m venv",
        'bin/career-agent-workbench" --version',
        'bin/career-agent-workbench" --help',
        "installed-smoke --expected-prefix",
    ):
        assert required in package_runs
    forbidden = (
        "pull_request_target",
        "upload-artifact",
        "secrets.",
        "playwright",
        "publish",
        "deploy",
        "provider",
        "browser",
    )
    assert all(value not in workflow_text.casefold() for value in forbidden)

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    included = set(metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert {
        "/.github/workflows/ci.yml",
        "/scripts/check_public_safety.py",
        "/tests/test_public_safety.py",
    } <= included
