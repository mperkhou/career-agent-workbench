"""Synthetic tests for deterministic runtime configuration."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

import career_agent_workbench.config as config_module
from career_agent_workbench.__main__ import main
from career_agent_workbench.config import (
    EnvironmentFileError,
    InvalidConfigurationError,
    RuntimeConfig,
    RuntimeOverrides,
    Settings,
    WorkspaceMember,
    WorkspacePathRequiredError,
    WorkspacePaths,
    load_runtime_config,
    load_settings,
)

CANONICAL = "CAREER_AGENT_WORKBENCH_"
COMPATIBILITY = "LINKEDIN_CAREER_MCP_"

SETTING_CASES = (
    ("user_agent", "USER_AGENT", "fictional-browser-agent", "fictional-browser-agent"),
    ("timeout_seconds", "TIMEOUT_SECONDS", "14.5", 14.5),
    ("max_results", "MAX_RESULTS", "31", 31),
    (
        "ollama_base_url",
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:12000",
        "http://127.0.0.1:12000",
    ),
    ("ollama_model", "OLLAMA_MODEL", "fictional-local-model", "fictional-local-model"),
    ("ollama_timeout_seconds", "OLLAMA_TIMEOUT_SECONDS", "181.5", 181.5),
    (
        "llm_api_base_url",
        "LLM_API_BASE_URL",
        "https://example.invalid/api",
        "https://example.invalid/api",
    ),
    (
        "llm_api_model",
        "LLM_API_MODEL",
        "fictional/chat-model",
        "fictional/chat-model",
    ),
    (
        "llm_planner_api_model",
        "LLM_PLANNER_API_MODEL",
        "fictional/planner-model",
        "fictional/planner-model",
    ),
    ("llm_api_key", "LLM_API_KEY", "fictional-token-marker", "fictional-token-marker"),
    ("llm_api_timeout_seconds", "LLM_API_TIMEOUT_SECONDS", "361.5", 361.5),
    ("llm_provider", "LLM_PROVIDER", "OlLaMa", "ollama"),
    ("jod_model", "JOD_MODEL", "fictional/jod-model", "fictional/jod-model"),
    (
        "core_skill_model",
        "CORE_SKILL_MODEL",
        "fictional/core-model",
        "fictional/core-model",
    ),
    (
        "second_pass_model",
        "SECOND_PASS_MODEL",
        "fictional/second-model",
        "fictional/second-model",
    ),
    ("codex_model", "CODEX_MODEL", "fictional-shared-model", "fictional-shared-model"),
    (
        "codex_reasoning_effort",
        "CODEX_REASONING_EFFORT",
        "fictional-shared-effort",
        "fictional-shared-effort",
    ),
)

WORKSPACE_CASES = (
    ("root", "WORKSPACE"),
    ("profile_dir", "PROFILE_DIR"),
    ("master_resume", "MASTER_RESUME"),
    ("master_resume_text", "MASTER_RESUME_TEXT"),
    ("output_dir", "OUTPUT_DIR"),
    ("database", "DATABASE"),
    ("blacklist", "BLACKLIST"),
    ("tmp_dir", "TMP_DIR"),
)


@pytest.fixture(autouse=True)
def disable_real_editable_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from consulting any checkout-local dotenv."""
    monkeypatch.setattr(config_module, "_editable_source_root", lambda: None)


def _load(
    tmp_path: Path,
    environ: dict[str, str] | None = None,
    overrides: RuntimeOverrides | None = None,
) -> RuntimeConfig:
    return load_runtime_config(
        environ={} if environ is None else environ,
        overrides=overrides,
        cwd=tmp_path,
    )


def _write_private_env(
    bootstrap_dir: Path,
    lines: list[str] | tuple[str, ...],
    *,
    workspace: Path | None = None,
) -> tuple[Path, Path]:
    selected_workspace = workspace or (bootstrap_dir / "fictional-ops")
    selected_workspace.mkdir(parents=True, exist_ok=True)
    private_env = selected_workspace / ".env"
    private_env.write_text("\n".join(lines), encoding="utf-8")
    private_env.chmod(0o600)
    bootstrap = bootstrap_dir / ".env"
    bootstrap.write_text(
        "\n".join(
            (
                f"{CANONICAL}WORKSPACE={selected_workspace}",
                f"{CANONICAL}PRIVATE_ENV_FILE={private_env}",
            )
        ),
        encoding="utf-8",
    )
    return bootstrap, private_env


def test_configuration_dataclasses_are_frozen_and_slotted(tmp_path: Path) -> None:
    values: tuple[Any, ...] = (
        RuntimeOverrides(),
        WorkspacePaths(),
        Settings(),
        _load(tmp_path),
    )
    for value in values:
        assert value.__dataclass_params__.frozen
        assert "__slots__" in type(value).__dict__
        assert "__dict__" not in dir(value)
        with pytest.raises(FrozenInstanceError):
            setattr(value, fields(value)[0].name, None)


def test_defaults_require_no_workspace_or_dotenv(tmp_path: Path) -> None:
    loaded = _load(tmp_path)

    assert loaded == RuntimeConfig(
        paths=WorkspacePaths(),
        settings=Settings(),
        env_file=None,
    )
    assert all(
        getattr(loaded.paths, member.value) is None for member in WorkspaceMember
    )
    assert load_settings(environ={}, cwd=tmp_path) == Settings()


def test_require_returns_resolved_member_and_rejects_safely(tmp_path: Path) -> None:
    root = tmp_path / "fictional-workspace"
    paths = _load(
        tmp_path,
        overrides=RuntimeOverrides(workspace=root),
    ).paths

    assert paths.require(WorkspaceMember.ROOT) == root
    assert paths.require("database") == root / "output/tracking/applications.sqlite3"
    with pytest.raises(WorkspacePathRequiredError, match="profile_dir"):
        WorkspacePaths().require("profile_dir")
    with pytest.raises(InvalidConfigurationError, match="Unknown workspace member"):
        paths.require("untrusted-member-name")


@pytest.mark.parametrize(("field", "suffix", "raw", "expected"), SETTING_CASES)
@pytest.mark.parametrize("prefix", (CANONICAL, COMPATIBILITY))
def test_every_setting_accepts_canonical_and_compatibility_names(
    tmp_path: Path,
    field: str,
    suffix: str,
    raw: str,
    expected: Any,
    prefix: str,
) -> None:
    loaded = _load(tmp_path, {f"{prefix}{suffix}": raw})
    assert getattr(loaded.settings, field) == expected


@pytest.mark.parametrize(("field", "suffix"), WORKSPACE_CASES)
@pytest.mark.parametrize("prefix", (CANONICAL, COMPATIBILITY))
def test_every_workspace_path_accepts_both_names(
    tmp_path: Path,
    field: str,
    suffix: str,
    prefix: str,
) -> None:
    selected = (tmp_path / f"fictional-{field}").resolve()
    loaded = _load(tmp_path, {f"{prefix}{suffix}": os.fspath(selected)})
    assert getattr(loaded.paths, field) == selected


@pytest.mark.parametrize(
    ("process_values", "override", "expected"),
    [
        ({}, None, "dotenv-compatibility"),
        ({}, None, "dotenv-canonical"),
        (
            {f"{COMPATIBILITY}USER_AGENT": "process-compatibility"},
            None,
            "process-compatibility",
        ),
        (
            {
                f"{CANONICAL}USER_AGENT": "process-canonical",
                f"{COMPATIBILITY}USER_AGENT": "process-compatibility",
            },
            None,
            "process-canonical",
        ),
        (
            {f"{CANONICAL}USER_AGENT": "process-canonical"},
            "explicit-value",
            "explicit-value",
        ),
    ],
)
def test_complete_precedence_order(
    tmp_path: Path,
    process_values: dict[str, str],
    override: str | None,
    expected: str,
) -> None:
    dotenv_canonical = "dotenv-canonical" if expected != "dotenv-compatibility" else ""
    lines = [f"{COMPATIBILITY}USER_AGENT=dotenv-compatibility"]
    if dotenv_canonical:
        lines.append(f"{CANONICAL}USER_AGENT={dotenv_canonical}")
    _write_private_env(tmp_path, lines)

    loaded = _load(
        tmp_path,
        process_values,
        RuntimeOverrides(user_agent=override),
    )
    assert loaded.settings.user_agent == expected


def test_workflow_defaults_preserve_general_provider_model(tmp_path: Path) -> None:
    settings = _load(tmp_path).settings

    assert settings.llm_api_model == "deepseek/deepseek-chat"
    assert settings.jod_model == "z-ai/glm-5.2"
    assert settings.core_skill_model == "z-ai/glm-5.2"
    assert settings.second_pass_model == "z-ai/glm-5.2"
    assert settings.manual_pass_codex_model == ""
    assert settings.manual_pass_codex_reasoning_effort == ""
    assert settings.highlight_codex_model == "gpt-5.6-sol"
    assert settings.highlight_codex_reasoning_effort == "high"


@pytest.mark.parametrize(
    ("field", "suffix", "shared_suffix", "default"),
    (
        (
            "manual_pass_codex_model",
            "MANUAL_PASS_CODEX_MODEL",
            "CODEX_MODEL",
            "",
        ),
        (
            "manual_pass_codex_reasoning_effort",
            "MANUAL_PASS_CODEX_REASONING_EFFORT",
            "CODEX_REASONING_EFFORT",
            "",
        ),
        (
            "highlight_codex_model",
            "HIGHLIGHT_CODEX_MODEL",
            "CODEX_MODEL",
            "gpt-5.6-sol",
        ),
        (
            "highlight_codex_reasoning_effort",
            "HIGHLIGHT_CODEX_REASONING_EFFORT",
            "CODEX_REASONING_EFFORT",
            "high",
        ),
    ),
)
def test_workflow_codex_precedence_and_shared_fallback(
    tmp_path: Path,
    field: str,
    suffix: str,
    shared_suffix: str,
    default: str,
) -> None:
    _write_private_env(
        tmp_path,
        (
            f"{COMPATIBILITY}{shared_suffix}=private-compatibility-shared",
            f"{COMPATIBILITY}{suffix}=private-compatibility-workflow",
            f"{CANONICAL}{shared_suffix}=private-canonical-shared",
            f"{CANONICAL}{suffix}=private-canonical-workflow",
        ),
    )
    process = {
        f"{COMPATIBILITY}{suffix}": "process-compatibility-workflow",
        f"{CANONICAL}{shared_suffix}": "process-canonical-shared",
        f"{CANONICAL}{suffix}": "process-canonical-workflow",
    }

    assert getattr(_load(tmp_path).settings, field) == "private-canonical-workflow"
    assert (
        getattr(
            _load(
                tmp_path,
                {f"{COMPATIBILITY}{shared_suffix}": "process-compatibility-shared"},
            ).settings,
            field,
        )
        == "process-compatibility-shared"
    )
    assert (
        getattr(_load(tmp_path, process).settings, field)
        == "process-canonical-workflow"
    )
    assert (
        getattr(
            _load(
                tmp_path,
                process,
                RuntimeOverrides(**{field: "explicit-workflow"}),
            ).settings,
            field,
        )
        == "explicit-workflow"
    )
    assert (
        getattr(
            _load(
                tmp_path,
                {f"{CANONICAL}{suffix}": ""},
            ).settings,
            field,
        )
        == default
    )


def test_blank_canonical_values_block_lower_layers_and_reset(tmp_path: Path) -> None:
    root = tmp_path / "fictional-workspace"
    process_values = {
        f"{CANONICAL}WORKSPACE": os.fspath(root),
        f"{CANONICAL}PROFILE_DIR": " ",
        f"{COMPATIBILITY}PROFILE_DIR": os.fspath(tmp_path / "blocked-member"),
        f"{CANONICAL}USER_AGENT": "\t",
        f"{COMPATIBILITY}USER_AGENT": "blocked-agent",
        f"{CANONICAL}LLM_API_KEY": "",
        f"{COMPATIBILITY}LLM_API_KEY": "blocked-token-marker",
    }
    loaded = _load(tmp_path, process_values)

    assert loaded.paths.profile_dir == root / "profile"
    assert loaded.settings.user_agent == Settings().user_agent
    assert loaded.settings.llm_api_key == ""

    root_reset = _load(
        tmp_path,
        {
            f"{CANONICAL}WORKSPACE": "",
            f"{COMPATIBILITY}WORKSPACE": os.fspath(root),
        },
    )
    assert root_reset.paths == WorkspacePaths()


def test_valueless_dotenv_key_is_a_blank_barrier(tmp_path: Path) -> None:
    _write_private_env(
        tmp_path,
        (
            f"{CANONICAL}USER_AGENT",
            f"{COMPATIBILITY}USER_AGENT=blocked-agent",
        ),
    )

    assert _load(tmp_path).settings.user_agent == Settings().user_agent


def test_valid_dotenv_parser_forms_remain_supported(tmp_path: Path) -> None:
    _write_private_env(
        tmp_path,
        (
            "# fictional comment",
            "",
            f'export {CANONICAL}MAX_RESULTS = "74"',
            f"{CANONICAL}USER_AGENT='fictional quoted agent'",
            f"{CANONICAL}LLM_API_KEY",
        ),
    )

    loaded = _load(tmp_path)
    assert loaded.settings.max_results == 74
    assert loaded.settings.user_agent == "fictional quoted agent"
    assert loaded.settings.llm_api_key == ""


def test_malformed_dotenv_fails_before_value_loading_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed_marker = "fictional-malformed-marker"
    later_marker = "fictional-later-value-marker"
    _bootstrap, selected = _write_private_env(tmp_path, ())
    selected.write_text(
        "\n".join(
            (
                f'{CANONICAL}USER_AGENT="{malformed_marker}',
                f"{CANONICAL}USER_AGENT={later_marker}",
                f"{CANONICAL}MAX_RESULTS=75",
            )
        ),
        encoding="utf-8",
    )
    value_loader_called = False

    real_value_loader = config_module.dotenv_values

    def track_value_loader(*args: Any, **kwargs: Any) -> dict[str, str]:
        nonlocal value_loader_called
        if Path(args[0]) == selected:
            value_loader_called = True
        return dict(real_value_loader(*args, **kwargs))

    monkeypatch.setattr(config_module, "dotenv_values", track_value_loader)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(EnvironmentFileError) as caught:
        _load(tmp_path)
    captured = capsys.readouterr()

    assert not value_loader_called
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert str(caught.value) == "Selected environment file is not usable."
    for marker in (malformed_marker, later_marker, selected.name):
        assert marker not in str(caught.value)
        assert marker not in repr(caught.value)
        assert marker not in caplog.text
        assert marker not in captured.out
        assert marker not in captured.err


@pytest.mark.parametrize("kind", ("missing", "mode", "directory", "symlink"))
def test_private_env_file_fails_closed_without_path_disclosure(
    tmp_path: Path,
    kind: str,
) -> None:
    workspace = tmp_path / "fictional-ops"
    workspace.mkdir()
    private_env = workspace / ".env"
    if kind == "mode":
        private_env.write_text(f"{CANONICAL}MAX_RESULTS=41\n", encoding="utf-8")
        private_env.chmod(0o644)
    elif kind == "directory":
        private_env.mkdir()
    elif kind == "symlink":
        target = workspace / "private-settings"
        target.write_text(f"{CANONICAL}MAX_RESULTS=41\n", encoding="utf-8")
        target.chmod(0o600)
        private_env.symlink_to(target)
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                f"{CANONICAL}WORKSPACE={workspace}",
                f"{CANONICAL}PRIVATE_ENV_FILE={private_env}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentFileError) as caught:
        _load(tmp_path)

    assert str(caught.value) == "Private environment file is not usable."
    assert os.fspath(private_env) not in str(caught.value)


def test_private_env_must_be_contained_by_workspace_and_outside_public_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_root = tmp_path / "fictional-public"
    workspace = tmp_path / "fictional-ops"
    public_root.mkdir()
    workspace.mkdir()
    private_env = public_root / "private-settings"
    private_env.write_text(f"{CANONICAL}MAX_RESULTS=41\n", encoding="utf-8")
    private_env.chmod(0o600)
    (public_root / ".env").write_text(
        "\n".join(
            (
                f"{CANONICAL}WORKSPACE={workspace}",
                f"{CANONICAL}PRIVATE_ENV_FILE={private_env}",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_editable_source_root", lambda: public_root)

    with pytest.raises(EnvironmentFileError, match="not usable"):
        _load(public_root)


def test_bootstrap_resolves_relative_private_file_after_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "fictional-ops"
    workspace.mkdir()
    private_env = workspace / ".env"
    private_env.write_text(f"{CANONICAL}MAX_RESULTS=43\n", encoding="utf-8")
    private_env.chmod(0o600)
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                f"{CANONICAL}WORKSPACE=fictional-ops",
                f"{CANONICAL}PRIVATE_ENV_FILE=fictional-ops/.env",
            )
        ),
        encoding="utf-8",
    )

    loaded = _load(tmp_path)

    assert loaded.paths.root == workspace
    assert loaded.private_env_file == private_env
    assert loaded.settings.max_results == 43


def test_explicit_workspace_override_keeps_lower_layer_private_settings(
    tmp_path: Path,
) -> None:
    _write_private_env(tmp_path, [f"{CANONICAL}MAX_RESULTS=43"])
    override = tmp_path / "fictional-explicit-workspace"

    loaded = _load(
        tmp_path,
        overrides=RuntimeOverrides(workspace=override),
    )

    assert loaded.paths.root == override
    assert loaded.settings.max_results == 43


def test_valid_scalar_parsing_accepts_natural_and_string_overrides(
    tmp_path: Path,
) -> None:
    loaded = _load(
        tmp_path,
        overrides=RuntimeOverrides(
            timeout_seconds=13,
            max_results="42",
            ollama_timeout_seconds="182.25",
            llm_api_timeout_seconds=362.5,
            llm_provider=" OLLAMA ",
        ),
    )

    assert loaded.settings.timeout_seconds == 13.0
    assert loaded.settings.max_results == 42
    assert loaded.settings.ollama_timeout_seconds == 182.25
    assert loaded.settings.llm_api_timeout_seconds == 362.5
    assert loaded.settings.llm_provider == "ollama"


@pytest.mark.parametrize(
    ("suffix", "rejected", "safe_field"),
    [
        ("TIMEOUT_SECONDS", "malformed-number-marker", "timeout_seconds"),
        ("TIMEOUT_SECONDS", "nan", "timeout_seconds"),
        ("TIMEOUT_SECONDS", "inf", "timeout_seconds"),
        ("TIMEOUT_SECONDS", "0", "timeout_seconds"),
        ("OLLAMA_TIMEOUT_SECONDS", "-1", "ollama_timeout_seconds"),
        ("LLM_API_TIMEOUT_SECONDS", "-2", "llm_api_timeout_seconds"),
        ("MAX_RESULTS", "1.5", "max_results"),
        ("MAX_RESULTS", "0", "max_results"),
        ("LLM_PROVIDER", "unsupported-provider-marker", "llm_provider"),
    ],
)
def test_invalid_scalar_values_fail_without_echo(
    tmp_path: Path,
    suffix: str,
    rejected: str,
    safe_field: str,
) -> None:
    with pytest.raises(InvalidConfigurationError) as caught:
        _load(tmp_path, {f"{CANONICAL}{suffix}": rejected})

    assert safe_field in str(caught.value)
    assert rejected not in str(caught.value)


def test_boolean_is_not_a_positive_integer_or_timeout(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigurationError, match="max_results"):
        _load(tmp_path, overrides=RuntimeOverrides(max_results=True))
    with pytest.raises(InvalidConfigurationError, match="timeout_seconds"):
        _load(tmp_path, overrides=RuntimeOverrides(timeout_seconds=True))


def test_unsupported_float_inputs_fail_without_conversion_or_echo(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "fictional-conversion-marker"

    class ConversionTrap:
        def __float__(self) -> float:
            raise RuntimeError(marker)

    class BrokenFloat(float):
        def __float__(self) -> float:
            raise RuntimeError(marker)

    for rejected in (b"12", ConversionTrap(), BrokenFloat(12)):
        with pytest.raises(InvalidConfigurationError) as caught:
            _load(
                tmp_path,
                overrides=RuntimeOverrides(timeout_seconds=rejected),  # type: ignore[arg-type]
            )
        assert caught.value.__cause__ is None
        assert marker not in str(caught.value)
        assert marker not in repr(caught.value)

    captured = capsys.readouterr()
    assert marker not in caplog.text
    assert marker not in captured.out
    assert marker not in captured.err


@pytest.mark.parametrize("target", ("workspace", "cwd"))
def test_broken_pathlike_inputs_fail_without_exception_leakage(
    tmp_path: Path,
    target: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "fictional-pathlike-marker"

    class BrokenPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError(marker)

    with pytest.raises(InvalidConfigurationError) as caught:
        if target == "workspace":
            _load(
                tmp_path,
                overrides=RuntimeOverrides(workspace=BrokenPath()),
            )
        else:
            load_runtime_config(environ={}, cwd=BrokenPath())

    captured = capsys.readouterr()
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert marker not in str(caught.value)
    assert marker not in repr(caught.value)
    assert marker not in caplog.text
    assert marker not in captured.out
    assert marker not in captured.err


def test_pathlike_interrupts_are_not_caught(tmp_path: Path) -> None:
    class InterruptingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _load(
            tmp_path,
            overrides=RuntimeOverrides(workspace=InterruptingPath()),
        )


def test_relative_explicit_env_file_resolves_from_invocation_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "fictional-ops"
    workspace.mkdir()
    private_env = workspace / ".env"
    private_env.write_text(f"{CANONICAL}MAX_RESULTS=44\n", encoding="utf-8")
    private_env.chmod(0o600)
    selected = tmp_path / "fictional-bootstrap.env"
    selected.write_text(
        "\n".join(
            (
                f"{CANONICAL}WORKSPACE={workspace}",
                f"{CANONICAL}PRIVATE_ENV_FILE={private_env}",
            )
        ),
        encoding="utf-8",
    )
    loaded = _load(
        tmp_path,
        {f"{CANONICAL}ENV_FILE": selected.name},
    )

    assert loaded.env_file == selected
    assert loaded.private_env_file == private_env
    assert loaded.settings.max_results == 44


def test_cwd_dotenv_wins_over_editable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "fictional-source"
    source_root.mkdir()
    _write_private_env(
        source_root,
        [f"{CANONICAL}USER_AGENT=editable-value"],
        workspace=tmp_path / "source-ops",
    )
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    _write_private_env(invocation, [f"{CANONICAL}USER_AGENT=cwd-value"])
    monkeypatch.setattr(config_module, "_editable_source_root", lambda: source_root)

    loaded = _load(invocation)
    assert loaded.env_file == invocation / ".env"
    assert loaded.settings.user_agent == "cwd-value"


def test_editable_root_fallback_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "fictional-source"
    source_root.mkdir()
    source_env, _private_env = _write_private_env(
        source_root,
        [f"{CANONICAL}MAX_RESULTS=45"],
        workspace=tmp_path / "source-ops",
    )
    invocation = tmp_path / "nested" / "invocation"
    invocation.mkdir(parents=True)
    monkeypatch.setattr(config_module, "_editable_source_root", lambda: source_root)

    loaded = _load(invocation)
    assert loaded.env_file == source_env
    assert loaded.settings.max_results == 45


def test_dotenv_discovery_can_be_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "fictional-source"
    source_root.mkdir()
    (source_root / ".env").write_text(
        f"{CANONICAL}MAX_RESULTS=91\n",
        encoding="utf-8",
    )
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    (invocation / ".env").write_text(
        f"{CANONICAL}MAX_RESULTS=92\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_editable_source_root", lambda: source_root)
    reads = 0

    def track_read(_env_file: Path) -> dict[str, str]:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(config_module, "_read_env_file", track_read)
    workspace = invocation / "workspace"
    loaded = load_runtime_config(
        overrides=RuntimeOverrides(workspace=workspace),
        environ={},
        cwd=invocation,
        discover_dotenv=False,
    )

    assert loaded.env_file is None
    assert loaded.settings.max_results == Settings().max_results
    assert loaded.paths.root == workspace
    assert reads == 0


def test_parent_and_home_style_dotenv_files_are_not_discovered(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"{CANONICAL}MAX_RESULTS=46\n",
        encoding="utf-8",
    )
    invocation = tmp_path / "child"
    invocation.mkdir()
    fake_home = tmp_path / "fictional-home"
    fake_home.mkdir()
    (fake_home / ".env").write_text(
        f"{CANONICAL}MAX_RESULTS=47\n",
        encoding="utf-8",
    )

    loaded = _load(invocation)
    assert loaded.env_file is None
    assert loaded.settings.max_results == Settings().max_results


@pytest.mark.parametrize("kind", ("missing", "directory", "invalid-utf8"))
def test_invalid_explicit_env_file_never_falls_back(
    tmp_path: Path,
    kind: str,
) -> None:
    fallback = tmp_path / ".env"
    fallback.write_text(f"{CANONICAL}MAX_RESULTS=48\n", encoding="utf-8")
    selected = tmp_path / f"fictional-{kind}.env"
    if kind == "directory":
        selected.mkdir()
    elif kind == "invalid-utf8":
        selected.write_bytes(b"\xff")

    with pytest.raises(EnvironmentFileError) as caught:
        _load(tmp_path, {f"{CANONICAL}ENV_FILE": selected.name})

    assert selected.name not in str(caught.value)


def test_blank_explicit_env_file_selector_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentFileError, match="selector"):
        _load(tmp_path, {f"{CANONICAL}ENV_FILE": " "})


def test_bootstrap_rejects_settings_and_chained_selector(tmp_path: Path) -> None:
    primary = tmp_path / ".env"
    primary.write_text(
        "\n".join(
            (
                f"{CANONICAL}WORKSPACE=fictional-ops",
                f"{CANONICAL}ENV_FILE=chained.env",
                f"{CANONICAL}MAX_RESULTS=50",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentFileError, match="not usable"):
        _load(tmp_path)


def test_dotenv_interpolation_is_disabled(tmp_path: Path) -> None:
    _write_private_env(
        tmp_path,
        (
            "FICTIONAL_BASE=expanded-value",
            f"{CANONICAL}USER_AGENT=${{FICTIONAL_BASE}}",
        ),
    )
    assert _load(tmp_path).settings.user_agent == "${FICTIONAL_BASE}"


def test_relative_explicit_paths_resolve_from_cwd(tmp_path: Path) -> None:
    loaded = _load(
        tmp_path,
        overrides=RuntimeOverrides(
            workspace="fictional-workspace",
            profile_dir="fictional-workspace/custom-member",
        ),
    )

    assert loaded.paths.root == tmp_path / "fictional-workspace"
    assert loaded.paths.profile_dir == (tmp_path / "fictional-workspace/custom-member")


@pytest.mark.parametrize("field", ("workspace", "profile_dir"))
def test_blank_explicit_path_is_invalid(tmp_path: Path, field: str) -> None:
    with pytest.raises(InvalidConfigurationError, match=field):
        _load(tmp_path, overrides=RuntimeOverrides(**{field: " "}))


@pytest.mark.parametrize("root_layer", ("process", "dotenv"))
def test_relative_environment_root_resolves_from_selected_file(
    tmp_path: Path,
    root_layer: str,
) -> None:
    env_dir = tmp_path / "fictional-env-dir"
    env_dir.mkdir()
    selected = env_dir / "settings.env"
    lines: list[str] = []
    process_values = {f"{CANONICAL}ENV_FILE": os.fspath(selected)}
    if root_layer == "process":
        process_values[f"{CANONICAL}WORKSPACE"] = "fictional-ops"
    else:
        lines.append(f"{CANONICAL}WORKSPACE=fictional-ops")
    selected.write_text("\n".join(lines), encoding="utf-8")

    assert _load(tmp_path, process_values).paths.root == env_dir / "fictional-ops"


def test_relative_process_root_without_env_file_fails_safely(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigurationError, match="workspace"):
        _load(tmp_path, {f"{CANONICAL}WORKSPACE": "fictional-relative-root"})


@pytest.mark.parametrize("member_layer", ("process", "dotenv"))
def test_relative_environment_members_resolve_beneath_root(
    tmp_path: Path,
    member_layer: str,
) -> None:
    root = tmp_path / "fictional-workspace"
    selected = tmp_path / ".env"
    private = root / ".env"
    root.mkdir()
    private_lines: list[str] = []
    lines = [
        f"{CANONICAL}WORKSPACE={root}",
        f"{CANONICAL}PRIVATE_ENV_FILE={private}",
    ]
    process_values: dict[str, str] = {}
    if member_layer == "process":
        process_values[f"{CANONICAL}PROFILE_DIR"] = "custom-member"
    else:
        private_lines.append(f"{CANONICAL}PROFILE_DIR=custom-member")
    private.write_text("\n".join(private_lines), encoding="utf-8")
    private.chmod(0o600)
    selected.write_text("\n".join(lines), encoding="utf-8")

    loaded = _load(tmp_path, process_values)
    assert loaded.paths.profile_dir == root / "custom-member"


def test_absolute_paths_are_not_rebased_and_partial_paths_are_valid(
    tmp_path: Path,
) -> None:
    absolute_root = (tmp_path / "absolute-root").resolve()
    absolute_member = (tmp_path / "standalone-member").resolve()
    loaded = _load(
        tmp_path / "different-cwd",
        {f"{CANONICAL}WORKSPACE": os.fspath(absolute_root)},
    )
    partial = _load(
        tmp_path,
        overrides=RuntimeOverrides(database=absolute_member),
    )

    assert loaded.paths.root == absolute_root
    assert partial.paths.root is None
    assert partial.paths.database == absolute_member


def test_conventional_members_are_exact_and_sibling_independent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fictional-workspace"
    custom_profile = root / "custom-profile"
    custom_output = root / "custom-output"
    loaded = _load(
        tmp_path,
        overrides=RuntimeOverrides(
            workspace=root,
            profile_dir=custom_profile,
            output_dir=custom_output,
        ),
    ).paths

    assert loaded == WorkspacePaths(
        root=root,
        profile_dir=custom_profile,
        master_resume=root / "profile/MASTER-RESUME.yml",
        master_resume_text=root / "profile/MP-MASTER-RESUME.txt",
        output_dir=custom_output,
        database=root / "output/tracking/applications.sqlite3",
        blacklist=root / ".blacklist",
        tmp_dir=root / "tmp",
    )


@pytest.mark.parametrize(
    ("symlink_name", "safe_field"),
    (("profile", "profile_dir"), ("output", "output_dir")),
)
def test_conventional_member_symlink_escapes_are_rejected(
    tmp_path: Path,
    symlink_name: str,
    safe_field: str,
) -> None:
    root = tmp_path / "fictional-workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / symlink_name).symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidConfigurationError, match=safe_field) as caught:
        _load(
            tmp_path,
            overrides=RuntimeOverrides(workspace=root),
        )
    assert os.fspath(outside) not in str(caught.value)


def test_blank_reset_conventional_member_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "fictional-workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "profile").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidConfigurationError, match="profile_dir"):
        _load(
            tmp_path,
            {f"{CANONICAL}PROFILE_DIR": ""},
            RuntimeOverrides(workspace=root),
        )


def test_conventional_internal_symlink_remains_valid(tmp_path: Path) -> None:
    root = tmp_path / "fictional-workspace"
    internal = root / "internal-profile"
    root.mkdir()
    internal.mkdir()
    (root / "profile").symlink_to(internal, target_is_directory=True)

    loaded = _load(
        tmp_path,
        overrides=RuntimeOverrides(workspace=root),
    ).paths
    assert loaded.profile_dir == internal
    assert loaded.master_resume == internal / "MASTER-RESUME.yml"
    assert loaded.master_resume_text == internal / "MP-MASTER-RESUME.txt"


def test_parent_traversal_and_existing_symlink_escape_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fictional-workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape-link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidConfigurationError, match="profile_dir"):
        _load(
            tmp_path,
            {
                f"{CANONICAL}WORKSPACE": os.fspath(root),
                f"{CANONICAL}PROFILE_DIR": "../outside",
            },
        )
    with pytest.raises(InvalidConfigurationError, match="profile_dir"):
        _load(
            tmp_path,
            {
                f"{CANONICAL}WORKSPACE": os.fspath(root),
                f"{CANONICAL}PROFILE_DIR": "escape-link/member",
            },
        )


def test_repeated_loads_are_equal_without_cross_call_state(tmp_path: Path) -> None:
    process_values = {f"{CANONICAL}MAX_RESULTS": "51"}
    first = _load(tmp_path, process_values)
    second = _load(tmp_path, process_values)
    changed = _load(tmp_path, {f"{CANONICAL}MAX_RESULTS": "52"})

    assert first == second
    assert first is not second
    assert changed.settings.max_results == 52
    assert first.settings.max_results == 51


def test_injected_and_global_environment_mappings_remain_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = {f"{CANONICAL}MAX_RESULTS": "53"}
    injected_before = dict(injected)
    _load(tmp_path, injected)
    _load(tmp_path, injected)
    assert injected == injected_before

    synthetic_global = {f"{CANONICAL}MAX_RESULTS": "54"}
    monkeypatch.setattr(config_module.os, "environ", synthetic_global)
    global_before = dict(synthetic_global)
    load_runtime_config(cwd=tmp_path)
    load_runtime_config(cwd=tmp_path)
    assert synthetic_global == global_before


def test_values_paths_errors_and_output_are_secret_safe(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token_marker = "fictional-private-token-marker"
    path_marker = "fictional-private-path-marker"
    rejected_marker = "fictional-rejected-provider-marker"
    selected_root = tmp_path / path_marker
    overrides = RuntimeOverrides(
        workspace=selected_root,
        llm_api_key=token_marker,
    )
    loaded = _load(tmp_path, overrides=overrides)

    rendered = "\n".join(
        (
            repr(overrides),
            repr(loaded.paths),
            repr(loaded.settings),
            repr(loaded),
        )
    )
    with pytest.raises(InvalidConfigurationError) as invalid:
        _load(
            tmp_path,
            {f"{CANONICAL}LLM_PROVIDER": rejected_marker},
        )
    with pytest.raises(EnvironmentFileError) as env_error:
        _load(
            tmp_path,
            {f"{CANONICAL}ENV_FILE": f"{path_marker}.env"},
        )
    logging.getLogger("fictional-config-test").info("safe-static-message")
    captured = capsys.readouterr()
    logs = caplog.text

    for marker in (token_marker, path_marker, rejected_marker):
        assert marker not in rendered
        assert marker not in str(invalid.value)
        assert marker not in str(env_error.value)
        assert marker not in captured.out
        assert marker not in captured.err
        assert marker not in logs


@pytest.mark.parametrize(
    ("flag", "expected"),
    (
        ("--help", "Career Agent Workbench"),
        ("--version", "career-agent-workbench 1.1.0"),
    ),
)
def test_help_and_version_need_no_configuration(
    flag: str,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main([flag])
    assert caught.value.code == 0
    assert expected in capsys.readouterr().out


def test_env_example_is_parseable_trackable_and_public_safe() -> None:
    repository = Path(__file__).resolve().parents[1]
    example = repository / ".env.example"
    parsed = dotenv_values(example, encoding="utf-8", interpolate=False)

    assert parsed == {
        f"{CANONICAL}WORKSPACE": "../career-agent-workbench-ops",
        f"{CANONICAL}PRIVATE_ENV_FILE": "../career-agent-workbench-ops/.env",
    }
    assert not (repository / ".env").exists()
    ignored_env = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", ".env"],
        cwd=repository,
        check=False,
    )
    ignored_example = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", ".env.example"],
        cwd=repository,
        check=False,
    )
    assert ignored_env.returncode == 0
    assert ignored_example.returncode == 1


def test_import_is_side_effect_free_in_fresh_interpreter(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; before = dict(os.environ); "
                "import career_agent_workbench.config; "
                "raise SystemExit(0 if before == dict(os.environ) else 1)"
            ),
        ],
        cwd=tmp_path,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.fspath(repository / "src"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert process.stdout == ""
    assert process.stderr == ""
