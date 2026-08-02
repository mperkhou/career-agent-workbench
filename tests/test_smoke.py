"""Synthetic smoke tests for the public package scaffold."""

import pytest

from career_agent_workbench import __version__
from career_agent_workbench.__main__ import main


def test_package_version() -> None:
    assert __version__ == "1.2.0"


@pytest.mark.parametrize(
    ("flag", "expected_output"),
    [
        ("--help", "Career Agent Workbench"),
        ("--version", "career-agent-workbench 1.2.0"),
    ],
)
def test_cli_help_and_version(
    flag: str,
    expected_output: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([flag])

    assert exit_info.value.code == 0
    assert expected_output in capsys.readouterr().out
