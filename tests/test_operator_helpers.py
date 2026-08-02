from __future__ import annotations

import importlib.util
import json
import os
import signal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "career_agent_workbench_operator", ROOT / "scripts" / "workbench_operator.py"
)
assert SPEC is not None and SPEC.loader is not None
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return project


def test_skill_link_is_exact_idempotent_and_retargets_symlinks(tmp_path: Path) -> None:
    destination = tmp_path / "skills"
    assert operator.link_skills(destination) == 5
    assert tuple(sorted(path.name for path in destination.iterdir())) == tuple(
        sorted(operator.EXACT_SKILLS)
    )
    first_targets = {path.name: os.readlink(path) for path in destination.iterdir()}
    assert operator.link_skills(destination) == 5
    assert {
        path.name: os.readlink(path) for path in destination.iterdir()
    } == first_targets

    selected = destination / operator.EXACT_SKILLS[0]
    selected.unlink()
    selected.symlink_to(tmp_path / "old-skill", target_is_directory=True)
    assert operator.link_skills(destination) == 5
    assert os.readlink(selected) == str(
        (ROOT / "skills" / operator.EXACT_SKILLS[0]).resolve()
    )


def test_skill_link_refuses_any_non_symlink_before_mutation(tmp_path: Path) -> None:
    destination = tmp_path / "skills"
    destination.mkdir()
    blocked = destination / operator.EXACT_SKILLS[2]
    blocked.mkdir()

    with pytest.raises(operator.OperatorError, match="Operator command failed"):
        operator.link_skills(destination)

    assert tuple(destination.iterdir()) == (blocked,)


def test_website_lifecycle_records_and_stops_only_owned_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    opened: list[str] = []
    signals: list[tuple[int, signal.Signals]] = []

    class FakeProcess:
        pid = 4321

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def poll(self) -> None:
            return None

    monkeypatch.setattr(operator, "_project_root", lambda: project)
    monkeypatch.setattr(operator.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(operator.webbrowser, "open", opened.append)
    monkeypatch.setattr(operator.time, "sleep", lambda _seconds: None)

    assert (
        operator.start_website(host="127.0.0.1", port=8765, open_browser=False)
        == "Website started."
    )
    assert opened == []
    record_path = project / "tmp" / "website" / "website.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["pid"] == 4321
    assert len(record["token"]) == 32
    assert (record_path.stat().st_mode & 0o777) == 0o600

    alive = iter((True, False, False))
    monkeypatch.setattr(operator, "_pid_is_alive", lambda _pid: next(alive, False))
    monkeypatch.setattr(operator, "_owns_process", lambda pid, token: True)
    monkeypatch.setattr(
        operator.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )
    assert operator.stop_website() == "Website stopped."
    assert signals == [(4321, signal.SIGTERM)]
    assert not record_path.exists()

    assert (
        operator.start_website(host="localhost", port=8765, open_browser=True)
        == "Website started."
    )
    assert opened == ["http://localhost:8765/"]


def test_stale_or_mismatched_website_record_never_signals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    monkeypatch.setattr(operator, "_project_root", lambda: project)
    state = operator._state_directory(project)
    record = state / "website.json"
    operator._write_record(record, 9999, "a" * 32)
    monkeypatch.setattr(operator, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(operator, "_owns_process", lambda _pid, _token: False)
    monkeypatch.setattr(
        operator.os,
        "kill",
        lambda *_args: pytest.fail("mismatched process was signaled"),
    )

    assert operator.stop_website() == "Stale website state removed."
    assert not record.exists()
