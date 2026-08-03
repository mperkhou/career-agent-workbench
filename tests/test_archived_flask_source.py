from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import career_agent_workbench.webapp as active_webapp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "career_agent_workbench" / "_archived_flask_source.py"
EXPECTED_SHA256 = "c9e7ede302e97e6f9ec44ac9c861598a9c0225866b65738af74813a755c011c6"
EXPECTED_LINES = 8244


def test_archived_flask_source_is_exact_and_inactive() -> None:
    source_bytes = SOURCE.read_bytes()

    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED_SHA256
    assert len(source_bytes.splitlines()) == EXPECTED_LINES
    assert Path(active_webapp.__file__).name == "webapp.py"


def test_archived_flask_source_is_explicit_sdist_input() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    included = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    assert "/src/career_agent_workbench/_archived_flask_source.py" in included
    assert "/tests/test_archived_flask_source.py" in included
