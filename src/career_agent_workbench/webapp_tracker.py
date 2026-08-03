"""Deterministic tracker views over immutable application snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from career_agent_workbench.application_state import (
    APPLICATION_STATUSES,
    ApplicationRecord,
    ApplicationStateStore,
)

TRACKER_SCOPES = ("active", "archived", "all")
TRACKER_SORTS = (
    "company",
    "title",
    "matched",
    "posted",
    "applied",
    "updated",
    "ats",
    "selected",
    "clo",
)
TRACKER_DIRECTIONS = ("asc", "desc")
TRACKER_STATUSES = ("all", *sorted(APPLICATION_STATUSES))
_MAX_SEARCH_CHARS = 256


class TrackerViewError(ValueError):
    """Raised when a tracker view contains a non-allowlisted value."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TrackerView:
    """One validated local tracker view."""

    q: str = ""
    status: str = "all"
    scope: str = "active"
    sort: str = "updated"
    direction: str = "desc"

    @classmethod
    def parse(cls, values: Mapping[str, Any], *, prefix: str = "") -> TrackerView:
        """Parse exact query or form fields without accepting redirect targets."""

        q = _value(values, f"{prefix}q", "").strip()
        status = _value(values, f"{prefix}status", "all")
        scope = _value(values, f"{prefix}scope", "active")
        sort = _value(values, f"{prefix}sort", "updated")
        direction = _value(values, f"{prefix}direction", "desc")
        if (
            len(q) > _MAX_SEARCH_CHARS
            or _has_control(q)
            or status not in TRACKER_STATUSES
            or scope not in TRACKER_SCOPES
            or sort not in TRACKER_SORTS
            or direction not in TRACKER_DIRECTIONS
        ):
            raise TrackerViewError("Tracker view is invalid.")
        return cls(
            q=q,
            status=status,
            scope=scope,
            sort=sort,
            direction=direction,
        )

    @property
    def index_url(self) -> str:
        """Rebuild a same-origin index URL from allowlisted fields only."""

        return f"/?{urlencode(self.query_items)}"

    @property
    def add_url(self) -> str:
        """Build the local Add-page URL without accepting a return target."""

        return f"/applications/add?{urlencode(self.query_items)}"

    @property
    def query_items(self) -> tuple[tuple[str, str], ...]:
        return (
            ("q", self.q),
            ("status", self.status),
            ("scope", self.scope),
            ("sort", self.sort),
            ("direction", self.direction),
        )

    @property
    def form_items(self) -> tuple[tuple[str, str], ...]:
        return tuple((f"view_{key}", value) for key, value in self.query_items)


@dataclass(frozen=True, slots=True)
class TrackerRow:
    """Dense presentation metadata for one immutable application row."""

    application: ApplicationRecord
    variant_keys: tuple[str, ...]
    status_key: str


def tracker_rows(
    store: ApplicationStateStore,
    applications: Sequence[ApplicationRecord],
) -> tuple[TrackerRow, ...]:
    """Attach bounded variant and status metadata without mutating state."""

    return tuple(
        TrackerRow(
            application=application,
            variant_keys=tuple(
                variant.variant_key
                for variant in store.list_resume_variants(application.job_id)
            ),
            status_key=_status_key(application.applied_to),
        )
        for application in applications
    )


def tracker_applications(
    store: ApplicationStateStore,
    view: TrackerView,
) -> tuple[ApplicationRecord, ...]:
    """Return a filtered and deterministically sorted tracker snapshot."""

    applications = store.list_applications(view.scope)
    query = view.q.casefold()
    filtered = tuple(
        application
        for application in applications
        if (view.status == "all" or application.applied_to == view.status)
        and (
            not query
            or any(
                query in value.casefold()
                for value in (
                    application.job_id,
                    application.company,
                    application.job_title,
                    application.source,
                    application.notes,
                )
            )
        )
    )
    ordered = sorted(filtered, key=lambda item: item.job_id)
    ordered.sort(
        key=lambda item: _sort_value(item, view.sort),
        reverse=view.direction == "desc",
    )
    return tuple(ordered)


def tracker_counts(applications: Sequence[ApplicationRecord]) -> dict[str, int]:
    """Return bounded presentation counts without hidden content."""

    return {
        "shown": len(applications),
        "applied": sum(item.applied_to == "Yes" for item in applications),
        "pending": sum(item.applied_to == "No" for item in applications),
        "archived": sum(item.archived_at is not None for item in applications),
    }


def _sort_value(application: ApplicationRecord, selected: str) -> object:
    if selected == "company":
        return (application.company.casefold(), application.job_title.casefold())
    if selected == "title":
        return (application.job_title.casefold(), application.company.casefold())
    if selected == "matched":
        return application.date_matched or ""
    if selected == "posted":
        return application.date_posted or ""
    if selected == "applied":
        return application.date_applied or ""
    if selected == "ats":
        return -1 if application.ats.score is None else application.ats.score
    if selected == "selected":
        return (
            application.resume_variant_selection_mode,
            application.selected_resume_variant or "",
        )
    if selected == "clo":
        return (
            application.cover_letter is not None
            or application.cover_letter_pdf is not None
        )
    return application.updated_at


def _value(values: Mapping[str, Any], key: str, default: str) -> str:
    value = values.get(key, default)
    if type(value) is not str:
        raise TrackerViewError("Tracker view is invalid.")
    return value


def _has_control(value: str) -> bool:
    return any(
        ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
    )


def _status_key(value: str) -> str:
    return {
        "No": "pending",
        "Yes": "applied",
        "N/A": "not-applicable",
        "Rejected": "rejected",
        "Accepted for interview": "interview",
    }[value]


__all__ = [
    "TRACKER_DIRECTIONS",
    "TRACKER_SCOPES",
    "TRACKER_SORTS",
    "TRACKER_STATUSES",
    "TrackerView",
    "TrackerViewError",
    "TrackerRow",
    "tracker_applications",
    "tracker_counts",
    "tracker_rows",
]
