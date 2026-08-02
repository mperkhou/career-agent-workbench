"""Deterministic, secret-safe runtime configuration."""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from dotenv.parser import parse_stream

_CANONICAL_PREFIX = "CAREER_AGENT_WORKBENCH_"
_COMPATIBILITY_PREFIX = "LINKEDIN_CAREER_MCP_"
_ENV_FILE_KEY = f"{_CANONICAL_PREFIX}ENV_FILE"
_PRIVATE_ENV_FILE_KEY = f"{_CANONICAL_PREFIX}PRIVATE_ENV_FILE"
_MISSING = object()

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36 career-agent-workbench/1.1.0"
)


class ConfigurationError(Exception):
    """Base class for safe configuration failures."""

    __slots__ = ()


class EnvironmentFileError(ConfigurationError):
    """Raised when the selected dotenv file cannot be used."""

    __slots__ = ()


class InvalidConfigurationError(ConfigurationError):
    """Raised when a supported configuration field is invalid."""

    __slots__ = ()


class WorkspacePathRequiredError(ConfigurationError):
    """Raised when an operation requires an unresolved workspace path."""

    __slots__ = ()


class WorkspaceMember(StrEnum):
    """Names of independently configurable workspace paths."""

    ROOT = "root"
    PROFILE_DIR = "profile_dir"
    MASTER_RESUME = "master_resume"
    MASTER_RESUME_TEXT = "master_resume_text"
    OUTPUT_DIR = "output_dir"
    DATABASE = "database"
    BLACKLIST = "blacklist"
    TMP_DIR = "tmp_dir"


@dataclass(frozen=True, slots=True)
class RuntimeOverrides:
    """Explicit runtime values supplied by a future CLI or Make integration."""

    workspace: str | os.PathLike[str] | None = None
    profile_dir: str | os.PathLike[str] | None = None
    master_resume: str | os.PathLike[str] | None = None
    master_resume_text: str | os.PathLike[str] | None = None
    output_dir: str | os.PathLike[str] | None = None
    database: str | os.PathLike[str] | None = None
    blacklist: str | os.PathLike[str] | None = None
    tmp_dir: str | os.PathLike[str] | None = None
    user_agent: str | None = None
    timeout_seconds: float | str | None = None
    max_results: int | str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    ollama_timeout_seconds: float | str | None = None
    llm_api_base_url: str | None = None
    llm_api_model: str | None = None
    llm_planner_api_model: str | None = None
    llm_api_key: str | None = None
    llm_api_timeout_seconds: float | str | None = None
    llm_provider: str | None = None
    jod_model: str | None = None
    core_skill_model: str | None = None
    second_pass_model: str | None = None
    codex_model: str | None = None
    codex_reasoning_effort: str | None = None
    manual_pass_codex_model: str | None = None
    manual_pass_codex_reasoning_effort: str | None = None
    highlight_codex_model: str | None = None
    highlight_codex_reasoning_effort: str | None = None

    def __repr__(self) -> str:
        configured = (
            f"{field}=configured"
            for field in self.__dataclass_fields__
            if getattr(self, field) is not None
        )
        return f"RuntimeOverrides({', '.join(configured)})"


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Resolved, independently optional workspace paths."""

    root: Path | None = None
    profile_dir: Path | None = None
    master_resume: Path | None = None
    master_resume_text: Path | None = None
    output_dir: Path | None = None
    database: Path | None = None
    blacklist: Path | None = None
    tmp_dir: Path | None = None

    def __repr__(self) -> str:
        configured = (
            f"{field}=configured"
            for field in self.__dataclass_fields__
            if getattr(self, field) is not None
        )
        return f"WorkspacePaths({', '.join(configured)})"

    def require(self, member: WorkspaceMember | str) -> Path:
        """Return one resolved member or raise a sanitized typed error."""
        try:
            selected = (
                member
                if isinstance(member, WorkspaceMember)
                else WorkspaceMember(member)
            )
        except (TypeError, ValueError):
            raise InvalidConfigurationError("Unknown workspace member.") from None

        value = getattr(self, selected.value)
        if value is None:
            raise WorkspacePathRequiredError(
                f"Required workspace path is not configured: {selected.value}."
            )
        return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings."""

    user_agent: str = _DEFAULT_USER_AGENT
    timeout_seconds: float = 12.0
    max_results: int = 25
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_seconds: float = 180.0
    llm_api_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_model: str = "deepseek/deepseek-chat"
    llm_planner_api_model: str = "deepseek/deepseek-v4-flash"
    llm_api_key: str = ""
    llm_api_timeout_seconds: float = 360.0
    llm_provider: str = "api"
    jod_model: str = "z-ai/glm-5.2"
    core_skill_model: str = "z-ai/glm-5.2"
    second_pass_model: str = "z-ai/glm-5.2"
    codex_model: str = ""
    codex_reasoning_effort: str = ""
    manual_pass_codex_model: str = ""
    manual_pass_codex_reasoning_effort: str = ""
    highlight_codex_model: str = "gpt-5.6-sol"
    highlight_codex_reasoning_effort: str = "high"

    def __repr__(self) -> str:
        return "Settings(configured=True)"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """One complete configuration load."""

    paths: WorkspacePaths
    settings: Settings
    env_file: Path | None
    private_env_file: Path | None = None

    def __repr__(self) -> str:
        return (
            "RuntimeConfig("
            f"paths_configured={any(getattr(self.paths, field) is not None for field in self.paths.__dataclass_fields__)}, "
            "settings_configured=True, "
            f"env_file_configured={self.env_file is not None}, "
            f"private_env_file_configured={self.private_env_file is not None})"
        )


_WORKSPACE_SPECS = (
    ("profile_dir", "PROFILE_DIR", ("profile",)),
    ("master_resume", "MASTER_RESUME", ("profile", "MASTER-RESUME.yml")),
    (
        "master_resume_text",
        "MASTER_RESUME_TEXT",
        ("profile", "MP-MASTER-RESUME.txt"),
    ),
    ("output_dir", "OUTPUT_DIR", ("output",)),
    (
        "database",
        "DATABASE",
        ("output", "tracking", "applications.sqlite3"),
    ),
    ("blacklist", "BLACKLIST", (".blacklist",)),
    ("tmp_dir", "TMP_DIR", ("tmp",)),
)

_STRING_SETTING_SPECS = (
    ("user_agent", "USER_AGENT"),
    ("ollama_base_url", "OLLAMA_BASE_URL"),
    ("ollama_model", "OLLAMA_MODEL"),
    ("llm_api_base_url", "LLM_API_BASE_URL"),
    ("llm_api_model", "LLM_API_MODEL"),
    ("llm_planner_api_model", "LLM_PLANNER_API_MODEL"),
    ("llm_api_key", "LLM_API_KEY"),
    ("jod_model", "JOD_MODEL"),
    ("core_skill_model", "CORE_SKILL_MODEL"),
    ("second_pass_model", "SECOND_PASS_MODEL"),
    ("codex_model", "CODEX_MODEL"),
    ("codex_reasoning_effort", "CODEX_REASONING_EFFORT"),
)

_WORKFLOW_SETTING_SPECS = (
    (
        "manual_pass_codex_model",
        "MANUAL_PASS_CODEX_MODEL",
        "CODEX_MODEL",
    ),
    (
        "manual_pass_codex_reasoning_effort",
        "MANUAL_PASS_CODEX_REASONING_EFFORT",
        "CODEX_REASONING_EFFORT",
    ),
    (
        "highlight_codex_model",
        "HIGHLIGHT_CODEX_MODEL",
        "CODEX_MODEL",
    ),
    (
        "highlight_codex_reasoning_effort",
        "HIGHLIGHT_CODEX_REASONING_EFFORT",
        "CODEX_REASONING_EFFORT",
    ),
)

_FLOAT_SETTING_SPECS = (
    ("timeout_seconds", "TIMEOUT_SECONDS"),
    ("ollama_timeout_seconds", "OLLAMA_TIMEOUT_SECONDS"),
    ("llm_api_timeout_seconds", "LLM_API_TIMEOUT_SECONDS"),
)


def load_runtime_config(
    *,
    overrides: RuntimeOverrides | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    discover_dotenv: bool = True,
) -> RuntimeConfig:
    """Load a fresh runtime configuration without global environment mutation."""
    process_values = dict(os.environ) if environ is None else dict(environ)
    invocation_cwd = _snapshot_cwd(cwd)
    explicit = overrides if overrides is not None else RuntimeOverrides()

    env_file = (
        _select_env_file(process_values, invocation_cwd) if discover_dotenv else None
    )
    bootstrap_data = _read_bootstrap_env_file(env_file) if env_file is not None else {}

    bootstrap_root = _resolve_workspace_root(
        None,
        process_values,
        bootstrap_data,
        env_file,
        invocation_cwd,
    )
    root = _resolve_workspace_root(
        explicit.workspace,
        process_values,
        bootstrap_data,
        env_file,
        invocation_cwd,
    )
    private_env_file = _select_private_env_file(
        process_values,
        bootstrap_data,
        env_file,
        invocation_cwd,
        bootstrap_root if bootstrap_root is not None else root,
    )
    private_data = (
        _read_env_file(private_env_file) if private_env_file is not None else {}
    )
    paths = _resolve_workspace_paths(
        explicit,
        process_values,
        private_data,
        root,
        invocation_cwd,
    )
    settings = _resolve_settings(explicit, process_values, private_data)
    return RuntimeConfig(
        paths=paths,
        settings=settings,
        env_file=env_file,
        private_env_file=private_env_file,
    )


def load_settings(
    *,
    overrides: RuntimeOverrides | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> Settings:
    """Compatibility wrapper returning settings from the central loader."""
    return load_runtime_config(
        overrides=overrides,
        environ=environ,
        cwd=cwd,
    ).settings


def _snapshot_cwd(cwd: str | os.PathLike[str] | None) -> Path:
    try:
        selected = Path.cwd() if cwd is None else Path(cwd)
        return selected.resolve(strict=False)
    except Exception:  # noqa: BLE001 - sanitize ordinary PathLike failures
        raise InvalidConfigurationError("Invalid configuration for 'cwd'.") from None


def _editable_source_root() -> Path | None:
    module_path = Path(__file__).resolve(strict=False)
    package_dir = module_path.parent
    source_dir = package_dir.parent
    if package_dir.name != "career_agent_workbench" or source_dir.name != "src":
        return None
    root = source_dir.parent
    if (root / "pyproject.toml").is_file():
        return root
    return None


def _select_env_file(process_values: Mapping[str, Any], cwd: Path) -> Path | None:
    if _ENV_FILE_KEY in process_values:
        raw_selector = process_values[_ENV_FILE_KEY]
        if _is_blank(raw_selector):
            raise EnvironmentFileError("Environment file selector is invalid.")
        try:
            selector = _coerce_path(raw_selector, "env_file")
            selected = selector if selector.is_absolute() else cwd / selector
            selected = _normalize_path(selected, "env_file")
        except InvalidConfigurationError:
            raise EnvironmentFileError(
                "Environment file selector is invalid."
            ) from None
        if not selected.is_file():
            raise EnvironmentFileError("Selected environment file is not usable.")
        return selected

    cwd_candidate = cwd / ".env"
    if cwd_candidate.is_file():
        return _normalize_path(cwd_candidate, "env_file")

    source_root = _editable_source_root()
    if source_root is not None:
        source_candidate = source_root / ".env"
        if source_candidate.is_file():
            return _normalize_path(source_candidate, "env_file")
    return None


def _read_env_file(env_file: Path) -> dict[str, str | None]:
    try:
        content = env_file.read_text(encoding="utf-8")
        if any(binding.error for binding in parse_stream(StringIO(content))):
            raise EnvironmentFileError(
                "Selected environment file is not usable."
            ) from None
        values = dotenv_values(
            env_file,
            encoding="utf-8",
            interpolate=False,
        )
        return dict(values)
    except EnvironmentFileError:
        raise
    except Exception:  # noqa: BLE001 - sanitize library and file failures
        raise EnvironmentFileError("Selected environment file is not usable.") from None


def _read_bootstrap_env_file(env_file: Path) -> dict[str, str | None]:
    values = _read_env_file(env_file)
    allowed = {
        f"{_CANONICAL_PREFIX}WORKSPACE",
        f"{_COMPATIBILITY_PREFIX}WORKSPACE",
        _PRIVATE_ENV_FILE_KEY,
    }
    if any(key not in allowed for key in values):
        raise EnvironmentFileError("Selected environment file is not usable.")
    return values


def _select_private_env_file(
    process_values: Mapping[str, Any],
    bootstrap_data: Mapping[str, Any],
    bootstrap_file: Path | None,
    cwd: Path,
    root: Path | None,
) -> Path | None:
    raw_selector: Any = _MISSING
    for values in (process_values, bootstrap_data):
        if _PRIVATE_ENV_FILE_KEY in values:
            raw_selector = values[_PRIVATE_ENV_FILE_KEY]
            break
    if raw_selector is _MISSING:
        return None
    if _is_blank(raw_selector) or root is None:
        raise EnvironmentFileError("Private environment file selector is invalid.")
    try:
        selector = _coerce_path(raw_selector, "private_env_file")
        base = bootstrap_file.parent if bootstrap_file is not None else cwd
        candidate = selector if selector.is_absolute() else base / selector
    except InvalidConfigurationError:
        raise EnvironmentFileError(
            "Private environment file selector is invalid."
        ) from None
    if _has_symlink_component(candidate):
        raise EnvironmentFileError("Private environment file is not usable.")
    try:
        selected = _normalize_path(candidate, "private_env_file")
    except InvalidConfigurationError:
        raise EnvironmentFileError(
            "Private environment file selector is invalid."
        ) from None
    public_root = _editable_source_root()
    if (
        not _is_within(selected, root)
        or (public_root is not None and _is_within(selected, public_root))
        or not _is_secure_private_env_file(selected)
    ):
        raise EnvironmentFileError("Private environment file is not usable.")
    return selected


def _is_secure_private_env_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        return False
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
        return False
    return True


def _has_symlink_component(path: Path) -> bool:
    try:
        absolute = path if path.is_absolute() else Path.cwd() / path
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            try:
                if current.is_symlink():
                    return True
            except OSError:
                return True
        return False
    except (OSError, RuntimeError, ValueError):
        return True


def _resolve_workspace_root(
    explicit_value: str | os.PathLike[str] | None,
    process_values: Mapping[str, Any],
    dotenv_data: Mapping[str, Any],
    env_file: Path | None,
    cwd: Path,
) -> Path | None:
    value, layer = _select_value(
        explicit_value,
        "WORKSPACE",
        process_values,
        dotenv_data,
    )
    if value is _MISSING:
        return None
    if _is_blank(value):
        if layer == "explicit":
            raise InvalidConfigurationError("Invalid configuration for 'workspace'.")
        return None

    raw_path = _coerce_path(value, "workspace")
    if raw_path.is_absolute():
        return _normalize_path(raw_path, "workspace")
    if layer == "explicit":
        return _normalize_path(cwd / raw_path, "workspace")
    if env_file is None:
        raise InvalidConfigurationError("Invalid configuration for 'workspace'.")
    return _normalize_path(env_file.parent / raw_path, "workspace")


def _resolve_workspace_paths(
    explicit: RuntimeOverrides,
    process_values: Mapping[str, Any],
    dotenv_data: Mapping[str, Any],
    root: Path | None,
    cwd: Path,
) -> WorkspacePaths:
    resolved: dict[str, Path | None] = {"root": root}
    for field, suffix, conventional_parts in _WORKSPACE_SPECS:
        value, layer = _select_value(
            getattr(explicit, field),
            suffix,
            process_values,
            dotenv_data,
        )
        if value is _MISSING or (layer != "explicit" and _is_blank(value)):
            resolved[field] = (
                _resolve_conventional_member(root, conventional_parts, field)
                if root is not None
                else None
            )
            continue
        if _is_blank(value):
            raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")

        raw_path = _coerce_path(value, field)
        if raw_path.is_absolute():
            resolved[field] = _normalize_path(raw_path, field)
            continue
        if root is None or ".." in raw_path.parts:
            raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
        base = cwd if layer == "explicit" else root
        candidate = _normalize_path(base / raw_path, field)
        if not _is_within(candidate, root):
            raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
        resolved[field] = candidate
    return WorkspacePaths(**resolved)


def _resolve_settings(
    explicit: RuntimeOverrides,
    process_values: Mapping[str, Any],
    dotenv_data: Mapping[str, Any],
) -> Settings:
    defaults = Settings()
    resolved: dict[str, str | float | int] = {}

    for field, suffix in _STRING_SETTING_SPECS:
        value, _ = _select_value(
            getattr(explicit, field),
            suffix,
            process_values,
            dotenv_data,
        )
        if value is _MISSING or _is_blank(value):
            resolved[field] = getattr(defaults, field)
        elif isinstance(value, str):
            resolved[field] = value
        else:
            raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")

    for field, workflow_suffix, shared_suffix in _WORKFLOW_SETTING_SPECS:
        value, _ = _select_workflow_value(
            getattr(explicit, field),
            workflow_suffix,
            shared_suffix,
            process_values,
            dotenv_data,
        )
        if value is _MISSING or _is_blank(value):
            resolved[field] = getattr(defaults, field)
        elif isinstance(value, str):
            resolved[field] = value
        else:
            raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")

    for field, suffix in _FLOAT_SETTING_SPECS:
        value, _ = _select_value(
            getattr(explicit, field),
            suffix,
            process_values,
            dotenv_data,
        )
        resolved[field] = (
            getattr(defaults, field)
            if value is _MISSING or _is_blank(value)
            else _parse_positive_float(value, field)
        )

    max_results, _ = _select_value(
        explicit.max_results,
        "MAX_RESULTS",
        process_values,
        dotenv_data,
    )
    resolved["max_results"] = (
        defaults.max_results
        if max_results is _MISSING or _is_blank(max_results)
        else _parse_positive_int(max_results, "max_results")
    )

    provider, _ = _select_value(
        explicit.llm_provider,
        "LLM_PROVIDER",
        process_values,
        dotenv_data,
    )
    resolved["llm_provider"] = (
        defaults.llm_provider
        if provider is _MISSING or _is_blank(provider)
        else _parse_provider(provider)
    )
    return Settings(**resolved)


def _select_value(
    explicit_value: Any,
    suffix: str,
    process_values: Mapping[str, Any],
    dotenv_data: Mapping[str, Any],
) -> tuple[Any, str]:
    if explicit_value is not None:
        return explicit_value, "explicit"
    for values, prefix, layer in (
        (process_values, _CANONICAL_PREFIX, "process"),
        (process_values, _COMPATIBILITY_PREFIX, "process"),
        (dotenv_data, _CANONICAL_PREFIX, "dotenv"),
        (dotenv_data, _COMPATIBILITY_PREFIX, "dotenv"),
    ):
        key = f"{prefix}{suffix}"
        if key in values:
            return values[key], layer
    return _MISSING, "default"


def _select_workflow_value(
    explicit_value: Any,
    workflow_suffix: str,
    shared_suffix: str,
    process_values: Mapping[str, Any],
    dotenv_data: Mapping[str, Any],
) -> tuple[Any, str]:
    if explicit_value is not None:
        return explicit_value, "explicit"
    for values, prefix, layer in (
        (process_values, _CANONICAL_PREFIX, "process"),
        (process_values, _COMPATIBILITY_PREFIX, "process"),
        (dotenv_data, _CANONICAL_PREFIX, "dotenv"),
        (dotenv_data, _COMPATIBILITY_PREFIX, "dotenv"),
    ):
        for suffix in (workflow_suffix, shared_suffix):
            key = f"{prefix}{suffix}"
            if key in values:
                return values[key], layer
    return _MISSING, "default"


def _coerce_path(value: Any, field: str) -> Path:
    try:
        raw = os.fspath(value)
        if isinstance(raw, bytes):
            raise TypeError
        return Path(raw)
    except Exception:  # noqa: BLE001 - sanitize ordinary PathLike failures
        raise InvalidConfigurationError(
            f"Invalid configuration for '{field}'."
        ) from None


def _normalize_path(value: Path, field: str) -> Path:
    try:
        return value.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise InvalidConfigurationError(
            f"Invalid configuration for '{field}'."
        ) from None


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_conventional_member(
    root: Path,
    conventional_parts: tuple[str, ...],
    field: str,
) -> Path:
    candidate = _normalize_path(root.joinpath(*conventional_parts), field)
    if not _is_within(candidate, root):
        raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
    return candidate


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not str.strip(value))


def _parse_positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
    try:
        parsed = float(value)
    except Exception:  # noqa: BLE001 - sanitize supported conversion failures
        raise InvalidConfigurationError(
            f"Invalid configuration for '{field}'."
        ) from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
    return parsed


def _parse_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
    try:
        parsed = int(value)
    except Exception:  # noqa: BLE001 - sanitize supported conversion failures
        raise InvalidConfigurationError(
            f"Invalid configuration for '{field}'."
        ) from None
    if parsed <= 0:
        raise InvalidConfigurationError(f"Invalid configuration for '{field}'.")
    return parsed


def _parse_provider(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidConfigurationError("Invalid configuration for 'llm_provider'.")
    normalized = str.lower(str.strip(value))
    if normalized not in {"api", "ollama"}:
        raise InvalidConfigurationError("Invalid configuration for 'llm_provider'.")
    return normalized


__all__ = [
    "ConfigurationError",
    "EnvironmentFileError",
    "InvalidConfigurationError",
    "RuntimeConfig",
    "RuntimeOverrides",
    "Settings",
    "WorkspaceMember",
    "WorkspacePathRequiredError",
    "WorkspacePaths",
    "load_runtime_config",
    "load_settings",
]
