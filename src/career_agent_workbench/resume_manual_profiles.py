"""Immutable logical model policy for the governed manual review pass."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from career_agent_workbench.codex_cli import (
    CodexConfigurationError,
    CodexModelConfig,
    resolve_codex_model_config,
)


class ManualPassProfileKey(StrEnum):
    """Exact allowlisted manual-pass profile keys."""

    ECONOMY = "economy"
    REGULAR = "regular"
    PREMIUM = "premium"


@dataclass(frozen=True, slots=True, repr=False)
class ManualPassProfile:
    """One immutable public model policy."""

    key: ManualPassProfileKey
    model: str
    reasoning_effort: str

    def __post_init__(self) -> None:
        if type(self.key) is not ManualPassProfileKey:
            raise CodexConfigurationError("Invalid manual-pass profile configuration.")
        CodexModelConfig(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            workflow="manual_pass",
            profile=self.key.value,
        )

    def __repr__(self) -> str:
        return "ManualPassProfile(configured=True)"


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedManualPassConfig:
    """Presence-aware manual-pass configuration."""

    profile: ManualPassProfile
    model: str
    reasoning_effort: str

    def __post_init__(self) -> None:
        if type(self.profile) is not ManualPassProfile:
            raise CodexConfigurationError("Invalid manual-pass profile configuration.")
        CodexModelConfig(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            workflow="manual_pass",
            profile=self.profile.key.value,
        )

    def to_model_config(self) -> CodexModelConfig:
        """Return the manual workflow's independent logical runner config."""

        return CodexModelConfig(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            workflow="manual_pass",
            profile=self.profile.key.value,
        )

    def __repr__(self) -> str:
        return "ResolvedManualPassConfig(configured=True)"


DEFAULT_MANUAL_PASS_PROFILE = ManualPassProfileKey.REGULAR
MANUAL_PASS_PROFILES: Mapping[ManualPassProfileKey, ManualPassProfile] = (
    MappingProxyType(
        {
            ManualPassProfileKey.ECONOMY: ManualPassProfile(
                key=ManualPassProfileKey.ECONOMY,
                model="gpt-5.6-terra",
                reasoning_effort="high",
            ),
            ManualPassProfileKey.REGULAR: ManualPassProfile(
                key=ManualPassProfileKey.REGULAR,
                model="gpt-5.6-sol",
                reasoning_effort="high",
            ),
            ManualPassProfileKey.PREMIUM: ManualPassProfile(
                key=ManualPassProfileKey.PREMIUM,
                model="gpt-5.6-sol",
                reasoning_effort="xhigh",
            ),
        }
    )
)


def parse_manual_pass_profile(
    value: str | ManualPassProfileKey,
) -> ManualPassProfile:
    """Parse an exact profile key without aliases, coercion, or leakage."""

    if type(value) is ManualPassProfileKey:
        key = value
    elif type(value) is str:
        try:
            key = ManualPassProfileKey(value)
        except ValueError:
            raise CodexConfigurationError(
                "Invalid manual-pass profile configuration."
            ) from None
    else:
        raise CodexConfigurationError("Invalid manual-pass profile configuration.")
    return MANUAL_PASS_PROFILES[key]


def parse_manual_pass_profile_key(value: str) -> ManualPassProfileKey:
    """Return only the exact parsed key."""

    return parse_manual_pass_profile(value).key


def resolve_manual_pass_config(
    *,
    profile: str | ManualPassProfileKey = DEFAULT_MANUAL_PASS_PROFILE,
    workflow_model_override: str | None = None,
    workflow_reasoning_effort_override: str | None = None,
) -> ResolvedManualPassConfig:
    """Resolve manual profile defaults with independent explicit overrides."""

    selected = parse_manual_pass_profile(profile)
    logical = resolve_codex_model_config(
        default_model=selected.model,
        default_reasoning_effort=selected.reasoning_effort,
        workflow_model_override=workflow_model_override,
        workflow_reasoning_effort_override=workflow_reasoning_effort_override,
        workflow="manual_pass",
        profile=selected.key.value,
    )
    return ResolvedManualPassConfig(
        profile=selected,
        model=logical.model,
        reasoning_effort=logical.reasoning_effort,
    )


__all__ = [
    "DEFAULT_MANUAL_PASS_PROFILE",
    "MANUAL_PASS_PROFILES",
    "ManualPassProfile",
    "ManualPassProfileKey",
    "ResolvedManualPassConfig",
    "parse_manual_pass_profile",
    "parse_manual_pass_profile_key",
    "resolve_manual_pass_config",
]
