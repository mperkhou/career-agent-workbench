"""Provider-neutral, caller-injected public job matching workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import yaml
from pydantic import HttpUrl

from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationSeedOutcome,
    ApplicationStateStore,
    MAX_QUERY_RESULTS,
    QueryOutcomeWrite,
)
from career_agent_workbench.cli_paths import (
    CliConfigurationError,
    add_runtime_path_arguments,
    load_command_config,
    seed_job_derived_paths,
)
from career_agent_workbench.config import RuntimeConfig, WorkspaceMember, WorkspacePaths
from career_agent_workbench.jod import job_description_context, usable_job_description
from career_agent_workbench.llm import build_llm_client
from career_agent_workbench.models import (
    DatePosted,
    ExperienceLevel,
    JobDetails,
    JobPosting,
    JobSearchQuery,
    JobSearchResult,
    JobType,
    WorkplaceType,
)
from career_agent_workbench.query_optimizer import (
    ScoredQuery,
    StoredQueryOutcome,
    historical_query_candidates,
    rank_search_queries,
)
from career_agent_workbench.providers.linkedin_public import LinkedInPublicJobsProvider
from career_agent_workbench.services.job_search import JobSearchService

_PATH_TYPE = type(Path())
_HTTP_URL_TYPE = type(HttpUrl("https://example.com"))
_MAX_MASTER_RESUME_BYTES = 1_000_000
_MAX_PROFILE_CONTEXT_CHARS = 120_000
_MAX_PROFILE_TREE_DEPTH = 20
_MAX_PROFILE_TREE_NODES = 20_000
_MAX_PROFILE_STRING_CHARS = 100_000
_MAX_BLACKLIST_BYTES = 128_000
_MAX_BLACKLIST_LINES = 2_000
_MAX_BLACKLIST_PATTERNS = 1_000
_MAX_BLACKLIST_LINE_CHARS = 512
_MAX_QUERY_INPUTS = 200
_MAX_RESULT_JOBS = 100
_MAX_JOB_IDS = 10_000
_MAX_PROGRESS_COUNT = 1_000_000

_MATCHING_ERROR = "Matching workflow failed."
_CONFIGURATION_ERROR = "Matching workflow configuration is invalid."
_BLACKLIST_ERROR = "Configured blacklist is invalid."
_PROFILE_ERROR = "Configured master resume is invalid."
_PLANNER_ERROR = "Matching planner returned an invalid result."
_STATE_ERROR = "Application state seed failed."

MatchingStage = Literal[
    "planned",
    "searched",
    "details",
    "filtered",
    "seeded",
]


class MatchingWorkflowError(Exception):
    """Stable content-free matching failure."""

    __slots__ = ()


class MatchingConfigurationError(MatchingWorkflowError):
    """Raised before a workflow touches caller-supplied capabilities."""

    __slots__ = ()


class MatchingBlacklistError(MatchingWorkflowError):
    """Raised when an explicitly configured blacklist is not safe to use."""

    __slots__ = ()


class MatchingProfileError(MatchingWorkflowError):
    """Raised when the exact configured master resume cannot be read."""

    __slots__ = ()


class MatchingPlannerError(MatchingWorkflowError):
    """Raised for invalid planner output."""

    __slots__ = ()


class MatchingStateError(MatchingWorkflowError):
    """Raised after one candidate's atomic state seed failed."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class MatchingProgress:
    """Content-free progress containing only a stage and bounded counts."""

    stage: MatchingStage
    completed: int
    total: int


@dataclass(frozen=True, slots=True)
class MatchingPlanRequest:
    """Bounded planner input whose representation hides profile content."""

    profile_context: str = field(repr=False)
    location: str
    date_posted: DatePosted
    workplace_types: tuple[WorkplaceType, ...]
    experience_levels: tuple[ExperienceLevel, ...]
    job_types: tuple[JobType, ...]
    limit_per_query: int
    max_queries: int

    def __repr__(self) -> str:
        return (
            "MatchingPlanRequest("
            f"workplace_type_count={len(self.workplace_types)}, "
            f"experience_level_count={len(self.experience_levels)}, "
            f"job_type_count={len(self.job_types)}, "
            f"limit_per_query={self.limit_per_query}, "
            f"max_queries={self.max_queries})"
        )


@runtime_checkable
class QueryPlanner(Protocol):
    """Caller-supplied bounded query planner."""

    def plan_queries(
        self,
        request: MatchingPlanRequest,
    ) -> Awaitable[Sequence[JobSearchQuery]]:
        """Return proposed public search queries."""


@runtime_checkable
class PublicJobSearch(Protocol):
    """Narrow injected public search/detail capability."""

    def search(self, query: JobSearchQuery) -> Awaitable[JobSearchResult]:
        """Search through one already constructed public service."""

    def get_details(self, job_id_or_url: str) -> Awaitable[JobDetails]:
        """Fetch normalized public details."""


@dataclass(frozen=True, slots=True)
class MatchingBounds:
    """Exact caller-supplied search and filter policy."""

    location: str
    date_posted: DatePosted
    workplace_types: tuple[WorkplaceType, ...]
    experience_levels: tuple[ExperienceLevel, ...]
    job_types: tuple[JobType, ...]
    limit_per_query: int = 10
    max_queries: int = 6
    max_jobs: int = 10

    def __post_init__(self) -> None:
        _validate_bounds(self)


@dataclass(frozen=True, slots=True)
class MatchingQueryOutcome:
    """Content-free outcome for one ranked query."""

    ordinal: int
    results_returned: int
    details_missed: int
    skipped_existing: int
    skipped_blacklisted: int
    skipped_workplace_type: int
    skipped_experience_level: int
    skipped_job_type: int
    skipped_description: int
    seeded: int
    provider_missed: bool


@dataclass(frozen=True, slots=True, repr=False)
class MatchingWorkflowResult:
    """Immutable in-memory result with job and query identities hidden."""

    query_outcomes: tuple[MatchingQueryOutcome, ...] = field(repr=False)
    newly_seeded_job_ids: tuple[str, ...] = field(repr=False)
    queries_planned: int
    queries_searched: int
    jobs_seen: int
    jobs_seeded: int

    def __repr__(self) -> str:
        return (
            "MatchingWorkflowResult("
            f"queries_planned={self.queries_planned}, "
            f"queries_searched={self.queries_searched}, "
            f"jobs_seen={self.jobs_seen}, "
            f"jobs_seeded={self.jobs_seeded})"
        )


class MatchingWorkflow:
    """Plan, search, filter, and atomically seed public job metadata."""

    __slots__ = ("_paths", "_planner", "_service", "_store")

    def __init__(
        self,
        *,
        service: PublicJobSearch,
        planner: QueryPlanner,
        store: ApplicationStateStore,
        paths: WorkspacePaths,
    ) -> None:
        if type(paths) is not WorkspacePaths:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        if service is None or planner is None:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        if type(store) is not ApplicationStateStore:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        self._service = service
        self._planner = planner
        self._store = store
        self._paths = paths

    async def run(
        self,
        *,
        bounds: MatchingBounds,
        history: tuple[StoredQueryOutcome, ...] | None = None,
        supplemental_queries: tuple[JobSearchQuery, ...] = (),
        existing_job_ids: tuple[str, ...] = (),
        progress_callback: Callable[[MatchingProgress], None] | None = None,
    ) -> MatchingWorkflowResult:
        """Run without model calls or generated-artifact construction."""

        if type(bounds) is not MatchingBounds:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        _validate_queries(supplemental_queries, limit=_MAX_QUERY_INPUTS)
        existing = _validate_job_ids(existing_job_ids)
        if progress_callback is not None and not callable(progress_callback):
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)

        # This assertion is deliberately the first capability/file boundary.
        try:
            self._store.assert_workspace_binding(self._paths)
        except Exception:  # noqa: BLE001 - sanitize an injected state boundary
            raise MatchingConfigurationError(_CONFIGURATION_ERROR) from None

        if history is None:
            try:
                history = self._store.load_query_outcomes()
            except Exception:  # noqa: BLE001 - sanitize an injected state boundary
                raise MatchingStateError(_STATE_ERROR) from None
        _validate_history(history)

        master_resume = _required_exact_path(
            self._paths,
            WorkspaceMember.MASTER_RESUME,
            error_type=MatchingProfileError,
            message=_PROFILE_ERROR,
        )
        profile_context = _read_master_resume_context(master_resume)
        blacklist = _load_blacklist(self._paths.blacklist)
        request = MatchingPlanRequest(
            profile_context=profile_context,
            location=bounds.location,
            date_posted=bounds.date_posted,
            workplace_types=bounds.workplace_types,
            experience_levels=bounds.experience_levels,
            job_types=bounds.job_types,
            limit_per_query=bounds.limit_per_query,
            max_queries=bounds.max_queries,
        )
        planned = await self._plan(request)
        try:
            ranked = _rank_queries(
                planned=planned,
                supplemental=supplemental_queries,
                history=history,
                profile_context=profile_context,
                bounds=bounds,
            )
        except MatchingWorkflowError:
            raise
        except Exception:  # noqa: BLE001 - sanitize protected ranking boundaries
            raise MatchingPlannerError(_PLANNER_ERROR) from None
        _progress(progress_callback, "planned", len(ranked), len(ranked))

        seeded_ids: list[str] = []
        seen_job_ids = set(existing)
        jobs_seen = 0
        outcomes: list[MatchingQueryOutcome] = []
        persisted_outcomes: list[QueryOutcomeWrite] = []
        for ordinal, scored in enumerate(ranked, start=1):
            if len(seeded_ids) >= bounds.max_jobs:
                break
            counters = _MutableOutcome(ordinal=ordinal)
            try:
                result = await self._service.search(scored.query)
                if (
                    type(result) is not JobSearchResult
                    or type(result.count) is not int
                    or type(result.jobs) is not list
                    or result.count != len(result.jobs)
                    or result.count > _MAX_RESULT_JOBS
                    or any(not _posting_is_inert(item) for item in result.jobs)
                ):
                    raise TypeError
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a provider is an injected boundary
                counters.provider_missed = True
                outcomes.append(counters.freeze())
                persisted_outcomes.append(_persisted_outcome(scored, counters))
                _progress(progress_callback, "searched", len(outcomes), len(ranked))
                continue
            counters.results_returned = min(result.count, _MAX_RESULT_JOBS)
            jobs_seen += counters.results_returned
            _progress(progress_callback, "searched", len(outcomes) + 1, len(ranked))

            for posting in result.jobs[:_MAX_RESULT_JOBS]:
                if len(seeded_ids) >= bounds.max_jobs:
                    break
                if type(posting) is not JobPosting:
                    counters.details_missed += 1
                    continue
                if posting.job_id in seen_job_ids:
                    counters.skipped_existing += 1
                    continue
                details = await self._details_or_none(posting)
                if details is None:
                    counters.details_missed += 1
                    continue
                if details.job_id in seen_job_ids:
                    counters.skipped_existing += 1
                    continue
                _progress(
                    progress_callback,
                    "details",
                    counters.results_returned,
                    counters.results_returned,
                )
                if blacklist.matches(details.company):
                    counters.skipped_blacklisted += 1
                    continue
                if not _allowed(details.workplace_type, bounds.workplace_types):
                    counters.skipped_workplace_type += 1
                    continue
                if not _allowed(details.seniority_level, bounds.experience_levels):
                    counters.skipped_experience_level += 1
                    continue
                if not _allowed(details.employment_type, bounds.job_types):
                    counters.skipped_job_type += 1
                    continue
                source_text = usable_job_description(details.description)
                if (
                    source_text is None
                    or details.company is None
                    or details.job_url is None
                ):
                    counters.skipped_description += 1
                    continue
                prompt_text = job_description_context(
                    details.model_copy(update={"description": source_text})
                )
                metadata = ApplicationMetadata(
                    job_id=details.job_id,
                    company=details.company,
                    job_title=details.title,
                    job_url=str(details.job_url),
                    source=details.source,
                    date_posted=_posted_date(details),
                    experience_level=details.seniority_level,
                )
                try:
                    seed_outcome = self._store.seed_application_with_outcome(
                        metadata,
                        source_text=source_text,
                        prompt_text=prompt_text,
                    )
                except Exception:  # noqa: BLE001 - sanitize an injected state boundary
                    raise MatchingStateError(_STATE_ERROR) from None
                seen_job_ids.add(details.job_id)
                created: object = None
                malformed_outcome = False
                if type(seed_outcome) is ApplicationSeedOutcome:
                    try:
                        created = object.__getattribute__(seed_outcome, "created")
                    except (AttributeError, TypeError):
                        malformed_outcome = True
                else:
                    malformed_outcome = True
                if malformed_outcome or type(created) is not bool:
                    raise MatchingStateError(_STATE_ERROR)
                if not created:
                    counters.skipped_existing += 1
                    continue
                seeded_ids.append(details.job_id)
                counters.seeded += 1
                _progress(progress_callback, "seeded", len(seeded_ids), bounds.max_jobs)
            _progress(
                progress_callback,
                "filtered",
                counters.results_returned,
                counters.results_returned,
            )
            outcomes.append(counters.freeze())
            persisted_outcomes.append(_persisted_outcome(scored, counters))

        if persisted_outcomes:
            try:
                self._store.record_query_outcomes(persisted_outcomes)
            except Exception:  # noqa: BLE001 - sanitize an injected state boundary
                raise MatchingStateError(_STATE_ERROR) from None

        return MatchingWorkflowResult(
            query_outcomes=tuple(outcomes),
            newly_seeded_job_ids=tuple(seeded_ids),
            queries_planned=len(ranked),
            queries_searched=len(outcomes),
            jobs_seen=jobs_seen,
            jobs_seeded=len(seeded_ids),
        )

    async def _plan(
        self,
        request: MatchingPlanRequest,
    ) -> tuple[JobSearchQuery, ...]:
        try:
            result = await self._planner.plan_queries(request)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a planner is an injected boundary
            raise MatchingPlannerError(_PLANNER_ERROR) from None
        if type(result) not in {list, tuple}:
            raise MatchingPlannerError(_PLANNER_ERROR)
        try:
            return _validate_queries(tuple(result), limit=request.max_queries)
        except MatchingConfigurationError:
            raise MatchingPlannerError(_PLANNER_ERROR) from None

    async def _details_or_none(self, posting: JobPosting) -> JobDetails | None:
        lookup = str(posting.job_url) if posting.job_url is not None else posting.job_id
        try:
            details = await self._service.get_details(lookup)
            if not _details_is_inert(details):
                return None
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a provider is an injected boundary
            return None
        return details.model_copy(
            update={
                "job_id": details.job_id or posting.job_id,
                "title": details.title or posting.title,
                "company": details.company or posting.company,
                "location": details.location or posting.location,
                "listed_at": details.listed_at or posting.listed_at,
                "posted_text": details.posted_text or posting.posted_text,
                "job_url": details.job_url or posting.job_url,
                "company_url": details.company_url or posting.company_url,
                "workplace_type": details.workplace_type or posting.workplace_type,
                "source": details.source or posting.source,
            }
        )


@dataclass(slots=True)
class _MutableOutcome:
    ordinal: int
    results_returned: int = 0
    details_missed: int = 0
    skipped_existing: int = 0
    skipped_blacklisted: int = 0
    skipped_workplace_type: int = 0
    skipped_experience_level: int = 0
    skipped_job_type: int = 0
    skipped_description: int = 0
    seeded: int = 0
    provider_missed: bool = False

    def freeze(self) -> MatchingQueryOutcome:
        return MatchingQueryOutcome(
            ordinal=self.ordinal,
            results_returned=self.results_returned,
            details_missed=self.details_missed,
            skipped_existing=self.skipped_existing,
            skipped_blacklisted=self.skipped_blacklisted,
            skipped_workplace_type=self.skipped_workplace_type,
            skipped_experience_level=self.skipped_experience_level,
            skipped_job_type=self.skipped_job_type,
            skipped_description=self.skipped_description,
            seeded=self.seeded,
            provider_missed=self.provider_missed,
        )


def _persisted_outcome(
    scored: ScoredQuery,
    counters: _MutableOutcome,
) -> QueryOutcomeWrite:
    query = scored.query
    return QueryOutcomeWrite(
        keywords=query.keywords,
        location=query.location,
        date_posted=query.date_posted,
        workplace_type=query.workplace_type,
        experience_level=query.experience_level,
        job_type=query.job_type,
        sort_by=query.sort_by,
        limit=query.limit,
        page=1,
        profile_match=scored.profile_match,
        query_score=scored.score,
        results_returned=counters.results_returned,
        fresh_jobs_accepted=counters.seeded,
        skipped_existing=counters.skipped_existing,
        skipped_blacklisted=counters.skipped_blacklisted,
        skipped_workplace_type=counters.skipped_workplace_type,
        skipped_experience_level=counters.skipped_experience_level,
    )


class _CompanyBlacklist:
    __slots__ = ("_patterns",)

    def __init__(self, patterns: tuple[str, ...]) -> None:
        self._patterns = patterns

    def matches(self, company: str | None) -> bool:
        if company is None or type(company) is not str:
            return False
        candidate = company.casefold()
        return any(
            _bounded_glob_matches(pattern.casefold(), candidate)
            for pattern in self._patterns
        )


def _bounded_glob_matches(pattern: str, candidate: str) -> bool:
    pattern_index = 0
    candidate_index = 0
    last_star = -1
    retry_index = 0
    while candidate_index < len(candidate):
        if pattern_index < len(pattern) and pattern[pattern_index] in {
            "?",
            candidate[candidate_index],
        }:
            pattern_index += 1
            candidate_index += 1
        elif pattern_index < len(pattern) and pattern[pattern_index] == "*":
            last_star = pattern_index
            retry_index = candidate_index
            pattern_index += 1
        elif last_star >= 0:
            retry_index += 1
            candidate_index = retry_index
            pattern_index = last_star + 1
        else:
            return False
    while pattern_index < len(pattern) and pattern[pattern_index] == "*":
        pattern_index += 1
    return pattern_index == len(pattern)


def _validate_bounds(bounds: MatchingBounds) -> None:
    if type(bounds.location) is not str:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    _bounded_text(bounds.location, 256, required=True)
    if type(bounds.date_posted) is not str:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    if bounds.date_posted not in {
        "any_time",
        "past_24_hours",
        "past_week",
        "past_month",
    }:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    _enum_tuple(
        bounds.workplace_types,
        {"on_site", "remote", "hybrid"},
        required=True,
    )
    _enum_tuple(
        bounds.experience_levels,
        {
            "internship",
            "entry_level",
            "associate",
            "mid_senior",
            "director",
            "executive",
        },
        required=False,
    )
    _enum_tuple(
        bounds.job_types,
        {
            "full_time",
            "part_time",
            "contract",
            "temporary",
            "volunteer",
            "internship",
            "other",
        },
        required=False,
    )
    for value, upper in (
        (bounds.limit_per_query, 100),
        (bounds.max_queries, 100),
        (bounds.max_jobs, 1_000),
    ):
        if type(value) is not int or not 1 <= value <= upper:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)


def _enum_tuple(
    value: object,
    allowed: set[str],
    *,
    required: bool,
) -> None:
    if type(value) is not tuple or required and not value or len(value) > len(allowed):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    if any(type(item) is not str or item not in allowed for item in value):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    if len(set(value)) != len(value):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)


def _validate_history(history: object) -> None:
    if type(history) is not tuple or len(history) > _MAX_QUERY_INPUTS:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    for item in history:
        if type(item) is not StoredQueryOutcome:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        _bounded_text(item.keywords, 256, required=True)
        _bounded_text(item.location, 256, required=True)
        if (
            type(item.date_posted) is not str
            or item.date_posted
            not in {"any_time", "past_24_hours", "past_week", "past_month"}
            or (
                item.workplace_type is not None
                and (
                    type(item.workplace_type) is not str
                    or item.workplace_type not in {"on_site", "remote", "hybrid"}
                )
            )
            or (
                item.experience_level is not None
                and (
                    type(item.experience_level) is not str
                    or item.experience_level
                    not in {
                        "internship",
                        "entry_level",
                        "associate",
                        "mid_senior",
                        "director",
                        "executive",
                    }
                )
            )
            or (
                item.job_type is not None
                and (
                    type(item.job_type) is not str
                    or item.job_type
                    not in {
                        "full_time",
                        "part_time",
                        "contract",
                        "temporary",
                        "volunteer",
                        "internship",
                        "other",
                    }
                )
            )
            or type(item.sort_by) is not str
            or item.sort_by not in {"relevance", "recent"}
            or type(item.limit) is not int
            or not 1 <= item.limit <= 100
        ):
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        for score in (item.profile_match, item.query_score):
            if not _bounded_number(score, minimum=0, maximum=1):
                raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        for count in (
            item.results_returned,
            item.fresh_jobs_accepted,
            item.skipped_existing,
            item.skipped_blacklisted,
            item.skipped_workplace_type,
            item.skipped_experience_level,
            item.resumes_generated,
        ):
            if type(count) is not int or not 0 <= count <= _MAX_PROGRESS_COUNT:
                raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        if item.average_ats_score is not None and not _bounded_number(
            item.average_ats_score,
            minimum=0,
            maximum=100,
        ):
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)


def _validate_queries(
    queries: object,
    *,
    limit: int,
) -> tuple[JobSearchQuery, ...]:
    if type(queries) is not tuple or len(queries) > min(limit, _MAX_QUERY_INPUTS):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    for query in queries:
        _validate_query(query)
    return queries


def _validate_query(query: object) -> None:
    if type(query) is not JobSearchQuery:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    _bounded_text(query.keywords, 256, required=True)
    _bounded_text(query.location, 256, required=True)
    if (
        type(query.date_posted) is not str
        or query.date_posted
        not in {"any_time", "past_24_hours", "past_week", "past_month"}
        or (
            query.job_type is not None
            and (
                type(query.job_type) is not str
                or query.job_type
                not in {
                    "full_time",
                    "part_time",
                    "contract",
                    "temporary",
                    "volunteer",
                    "internship",
                    "other",
                }
            )
        )
        or (
            query.workplace_type is not None
            and (
                type(query.workplace_type) is not str
                or query.workplace_type not in {"on_site", "remote", "hybrid"}
            )
        )
        or (
            query.experience_level is not None
            and (
                type(query.experience_level) is not str
                or query.experience_level
                not in {
                    "internship",
                    "entry_level",
                    "associate",
                    "mid_senior",
                    "director",
                    "executive",
                }
            )
        )
        or type(query.sort_by) is not str
        or query.sort_by not in {"relevance", "recent"}
        or (
            query.distance is not None
            and (type(query.distance) is not int or not 0 <= query.distance <= 100)
        )
        or type(query.limit) is not int
        or not 1 <= query.limit <= 100
        or type(query.page) is not int
        or not 0 <= query.page <= 10_000
        or type(query.exclude_job_ids) is not set
        or len(query.exclude_job_ids) > 500
    ):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    for job_id in query.exclude_job_ids:
        _bounded_text(job_id, 128, required=True)


def _validate_job_ids(values: object) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) > _MAX_JOB_IDS:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    checked: list[str] = []
    for value in values:
        if type(value) is not str:
            raise MatchingConfigurationError(_CONFIGURATION_ERROR)
        checked.append(_bounded_text(value, 128, required=True))
    if len(set(checked)) != len(checked):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    return tuple(checked)


def _rank_queries(
    *,
    planned: tuple[JobSearchQuery, ...],
    supplemental: tuple[JobSearchQuery, ...],
    history: tuple[StoredQueryOutcome, ...],
    profile_context: str,
    bounds: MatchingBounds,
) -> tuple[ScoredQuery, ...]:
    historical = tuple(
        query.model_copy(update={"workplace_type": None})
        for query in historical_query_candidates(
            history,
            location=bounds.location,
            date_posted=bounds.date_posted,
            limit_per_query=bounds.limit_per_query,
            max_candidates=bounds.max_queries,
        )
    )
    candidates: list[JobSearchQuery] = []
    for query in (*planned, *supplemental, *historical):
        if len(candidates) >= _MAX_QUERY_INPUTS:
            break
        workplace_values = (
            (query.workplace_type,)
            if query.workplace_type in bounds.workplace_types
            else bounds.workplace_types
        )
        for workplace in workplace_values:
            if (
                query.experience_level is not None
                and bounds.experience_levels
                and query.experience_level not in bounds.experience_levels
            ):
                continue
            if (
                query.job_type is not None
                and bounds.job_types
                and query.job_type not in bounds.job_types
            ):
                continue
            candidates.append(
                query.model_copy(
                    update={
                        "location": bounds.location,
                        "date_posted": bounds.date_posted,
                        "workplace_type": workplace,
                        "limit": min(query.limit, bounds.limit_per_query),
                        "page": 0,
                    }
                )
            )
    ranked = rank_search_queries(
        candidates,
        profile_context=profile_context,
        history=history,
        max_queries=bounds.max_queries,
    )
    return tuple(ranked)


def _load_blacklist(path: Path | None) -> _CompanyBlacklist:
    if path is None:
        return _CompanyBlacklist(())
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise MatchingBlacklistError(_BLACKLIST_ERROR)
    raw = _read_regular_file(
        path,
        max_bytes=_MAX_BLACKLIST_BYTES,
        error_type=MatchingBlacklistError,
        message=_BLACKLIST_ERROR,
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise MatchingBlacklistError(_BLACKLIST_ERROR) from None
    if any(
        character not in {"\r", "\n"}
        and (
            _has_control(character)
            or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        )
        for character in text
    ):
        raise MatchingBlacklistError(_BLACKLIST_ERROR)
    lines = text.splitlines()
    if len(lines) > _MAX_BLACKLIST_LINES:
        raise MatchingBlacklistError(_BLACKLIST_ERROR)
    patterns: list[str] = []
    for line in lines:
        if len(line) > _MAX_BLACKLIST_LINE_CHARS or _has_control(line):
            raise MatchingBlacklistError(_BLACKLIST_ERROR)
        selected = line.strip()
        if not selected or selected.startswith("#"):
            continue
        if "[" in selected or "]" in selected or "\\" in selected:
            raise MatchingBlacklistError(_BLACKLIST_ERROR)
        patterns.append(selected)
        if len(patterns) > _MAX_BLACKLIST_PATTERNS:
            raise MatchingBlacklistError(_BLACKLIST_ERROR)
    return _CompanyBlacklist(tuple(patterns))


def _read_master_resume_context(path: Path) -> str:
    raw = _read_regular_file(
        path,
        max_bytes=_MAX_MASTER_RESUME_BYTES,
        error_type=MatchingProfileError,
        message=_PROFILE_ERROR,
    )
    try:
        text = raw.decode("utf-8")
        parsed = yaml.safe_load(text)
    except (UnicodeDecodeError, yaml.YAMLError, RecursionError, TypeError, ValueError):
        raise MatchingProfileError(_PROFILE_ERROR) from None
    if type(parsed) is not dict:
        raise MatchingProfileError(_PROFILE_ERROR)
    try:
        _validate_inert_tree(parsed)
        context = json.dumps(
            parsed,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, MatchingProfileError):
        raise MatchingProfileError(_PROFILE_ERROR) from None
    if not context or len(context) > _MAX_PROFILE_CONTEXT_CHARS:
        raise MatchingProfileError(_PROFILE_ERROR)
    return context


def _validate_inert_tree(root: object) -> None:
    pending: list[tuple[object, int]] = [(root, 1)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_PROFILE_TREE_NODES or depth > _MAX_PROFILE_TREE_DEPTH:
            raise MatchingProfileError(_PROFILE_ERROR)
        if value is None or type(value) in {bool, int}:
            continue
        if type(value) is float:
            if not math.isfinite(value):
                raise MatchingProfileError(_PROFILE_ERROR)
            continue
        if type(value) is str:
            if len(value) > _MAX_PROFILE_STRING_CHARS or _has_control(value):
                raise MatchingProfileError(_PROFILE_ERROR)
            continue
        if type(value) is list:
            if len(value) > _MAX_PROFILE_TREE_NODES:
                raise MatchingProfileError(_PROFILE_ERROR)
            pending.extend((item, depth + 1) for item in value)
            continue
        if type(value) is dict:
            if len(value) > _MAX_PROFILE_TREE_NODES:
                raise MatchingProfileError(_PROFILE_ERROR)
            for key, item in value.items():
                if type(key) is not str or len(key) > 256 or _has_control(key):
                    raise MatchingProfileError(_PROFILE_ERROR)
                pending.append((item, depth + 1))
            continue
        raise MatchingProfileError(_PROFILE_ERROR)


def _required_exact_path(
    paths: WorkspacePaths,
    member: WorkspaceMember,
    *,
    error_type: type[MatchingWorkflowError],
    message: str,
) -> Path:
    try:
        selected = paths.require(member)
    except Exception:  # noqa: BLE001 - sanitize WorkspacePaths failures
        raise error_type(message) from None
    if type(selected) is not _PATH_TYPE or not selected.is_absolute():
        raise error_type(message)
    return selected


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int,
    error_type: type[MatchingWorkflowError],
    message: str,
) -> bytes:
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        visible = os.stat(path, follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > max_bytes
            or (visible.st_dev, visible.st_ino) != identity
            or not stat.S_ISREG(visible.st_mode)
        ):
            raise OSError
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise OSError
        visible = os.stat(path, follow_symlinks=False)
        if (visible.st_dev, visible.st_ino) != identity or not stat.S_ISREG(
            visible.st_mode
        ):
            raise OSError
        return raw
    except Exception:  # noqa: BLE001 - sanitize OS and path failures
        raise error_type(message) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _posted_date(job: JobDetails) -> str | None:
    value = job.listed_at or job.posted_text
    if value is None:
        return None
    text = value.strip()
    match = re.match(r"^\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match is not None else text


def _posting_is_inert(value: object) -> bool:
    if type(value) is not JobPosting:
        return False
    return _job_fields_are_inert(value)


def _details_is_inert(value: object) -> bool:
    if type(value) is not JobDetails:
        return False
    if not _job_fields_are_inert(value):
        return False
    return bool(
        (
            value.description is None
            or _inert_text(
                value.description,
                500_000,
                required=False,
                allow_layout=True,
            )
        )
        and all(
            item is None or _inert_text(item, 1_024, required=False)
            for item in (
                value.seniority_level,
                value.employment_type,
                value.job_function,
                value.industries,
            )
        )
    )


def _job_fields_are_inert(value: JobPosting | JobDetails) -> bool:
    if not (
        _inert_text(value.job_id, 128, required=True)
        and _inert_text(value.title, 1_024, required=False)
        and _inert_text(value.source, 1_024, required=False)
    ):
        return False
    if not all(
        item is None or _inert_text(item, 1_024, required=False)
        for item in (
            value.company,
            value.location,
            value.listed_at,
            value.posted_text,
            value.workplace_type,
        )
    ):
        return False
    return all(
        item is None or type(item) is _HTTP_URL_TYPE
        for item in (value.job_url, value.company_url)
    )


def _inert_text(
    value: object,
    limit: int,
    *,
    required: bool,
    allow_layout: bool = False,
) -> bool:
    return bool(
        type(value) is str
        and len(value) <= limit
        and (not required or value.strip())
        and not any(
            (
                ord(character) < 32
                and (not allow_layout or character not in {"\t", "\n", "\r"})
            )
            or 127 <= ord(character) <= 159
            for character in value
        )
    )


def _bounded_number(
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> bool:
    if type(value) is int:
        return minimum <= value <= maximum
    if type(value) is float:
        return math.isfinite(value) and minimum <= value <= maximum
    return False


def _allowed(value: str | None, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    if value is None or type(value) is not str:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    aliases = {
        "onsite": "on_site",
        "entry": "entry_level",
        "entrylevel": "entry_level",
        "mid_senior_level": "mid_senior",
        "fulltime": "full_time",
        "parttime": "part_time",
    }
    return aliases.get(normalized, normalized) in allowed


def _bounded_text(value: str, limit: int, *, required: bool) -> str:
    if type(value) is not str or len(value) > limit or _has_control(value):
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    selected = value.strip()
    if required and not selected:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    return selected


def _has_control(value: str) -> bool:
    return any(
        ord(character) < 32 or 127 <= ord(character) <= 159 for character in value
    )


def _progress(
    callback: Callable[[MatchingProgress], None] | None,
    stage: MatchingStage,
    completed: int,
    total: int,
) -> None:
    if callback is None:
        return
    safe_completed = min(max(completed, 0), _MAX_PROGRESS_COUNT)
    safe_total = min(max(total, 0), _MAX_PROGRESS_COUNT)
    try:
        callback(
            MatchingProgress(
                stage=stage,
                completed=safe_completed,
                total=safe_total,
            )
        )
    except Exception:  # noqa: BLE001 - callbacks are injected active boundaries
        raise MatchingWorkflowError(_MATCHING_ERROR) from None


async def run_matching_workflow(
    *,
    service: PublicJobSearch,
    planner: QueryPlanner,
    store: ApplicationStateStore,
    paths: WorkspacePaths,
    bounds: MatchingBounds,
    history: tuple[StoredQueryOutcome, ...] = (),
    supplemental_queries: tuple[JobSearchQuery, ...] = (),
    existing_job_ids: tuple[str, ...] = (),
    progress_callback: Callable[[MatchingProgress], None] | None = None,
) -> MatchingWorkflowResult:
    """Convenience wrapper retaining explicit caller composition."""

    workflow = MatchingWorkflow(
        service=service,
        planner=planner,
        store=store,
        paths=paths,
    )
    return await workflow.run(
        bounds=bounds,
        history=history,
        supplemental_queries=supplemental_queries,
        existing_job_ids=existing_job_ids,
        progress_callback=progress_callback,
    )


class _LlmQueryPlanner:
    """Adapt one configured JSON client to the bounded planner protocol."""

    __slots__ = ("_client",)

    def __init__(self, client: object) -> None:
        self._client = client

    async def plan_queries(
        self,
        request: MatchingPlanRequest,
    ) -> Sequence[JobSearchQuery]:
        prompt = json.dumps(
            {
                "task": "Return public job search queries as a queries array.",
                "profile": request.profile_context,
                "location": request.location,
                "date_posted": request.date_posted,
                "workplace_types": request.workplace_types,
                "experience_levels": request.experience_levels,
                "job_types": request.job_types,
                "limit_per_query": request.limit_per_query,
                "max_queries": request.max_queries,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        generate = getattr(self._client, "generate_json", None)
        if not callable(generate):
            raise MatchingPlannerError(_PLANNER_ERROR)
        payload = await generate(prompt)
        if not isinstance(payload, Mapping):
            raise MatchingPlannerError(_PLANNER_ERROR)
        values = payload.get("queries")
        if not isinstance(values, list):
            raise MatchingPlannerError(_PLANNER_ERROR)
        queries: list[JobSearchQuery] = []
        for value in values[: request.max_queries]:
            if not isinstance(value, Mapping):
                raise MatchingPlannerError(_PLANNER_ERROR)
            data = dict(value)
            data.setdefault("location", request.location)
            data.setdefault("date_posted", request.date_posted)
            data["limit"] = min(
                request.limit_per_query,
                int(data.get("limit") or request.limit_per_query),
            )
            try:
                queries.append(JobSearchQuery.model_validate(data))
            except Exception:
                raise MatchingPlannerError(_PLANNER_ERROR) from None
        return queries


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free public job-seeding parser."""

    parser = argparse.ArgumentParser(
        description="Search public jobs and seed governed application state."
    )
    add_runtime_path_arguments(
        parser,
        "workspace",
        "profile_dir",
        "blacklist",
        "output_dir",
        "database",
        "master_resume",
    )
    parser.add_argument("--master-resume-name", default="MASTER-RESUME.yml")
    parser.add_argument("--location", default="United States")
    parser.add_argument(
        "--date-posted",
        choices=("any_time", "past_24_hours", "past_week", "past_month"),
        default="past_week",
    )
    parser.add_argument("--limit-per-query", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=6)
    parser.add_argument("--max-jobs", type=int, default=10)
    return parser


async def run_from_cli(args: argparse.Namespace) -> MatchingWorkflowResult:
    """Resolve configuration once and run the public matching core."""

    database, master_resume = seed_job_derived_paths(args)
    config = load_command_config(
        args,
        required=(
            WorkspaceMember.MASTER_RESUME,
            WorkspaceMember.OUTPUT_DIR,
            WorkspaceMember.DATABASE,
        ),
        database=database,
        master_resume=master_resume,
    )
    return await run_matching_from_runtime(
        config,
        bounds=MatchingBounds(
            location=args.location,
            date_posted=args.date_posted,
            workplace_types=("remote", "hybrid"),
            experience_levels=("associate", "mid_senior", "director"),
            job_types=("full_time", "contract"),
            limit_per_query=args.limit_per_query,
            max_queries=args.max_queries,
            max_jobs=args.max_jobs,
        ),
    )


async def run_matching_from_runtime(
    runtime: RuntimeConfig,
    *,
    bounds: MatchingBounds,
    service: PublicJobSearch | None = None,
    planner: QueryPlanner | None = None,
) -> MatchingWorkflowResult:
    """Compose matching from one caller-resolved runtime configuration."""

    if type(runtime) is not RuntimeConfig or type(bounds) is not MatchingBounds:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR)
    try:
        runtime.paths.require(WorkspaceMember.MASTER_RESUME)
        runtime.paths.require(WorkspaceMember.OUTPUT_DIR)
        runtime.paths.require(WorkspaceMember.DATABASE)
    except Exception:
        raise MatchingConfigurationError(_CONFIGURATION_ERROR) from None

    store = ApplicationStateStore(runtime.paths)
    store.initialize()
    existing = tuple(
        item.job_id for item in store.list_applications("all", limit=MAX_QUERY_RESULTS)
    )

    owned_provider: LinkedInPublicJobsProvider | None = None
    owned_planner_client: object | None = None
    selected_service = service
    selected_planner = planner
    try:
        if selected_service is None:
            owned_provider = LinkedInPublicJobsProvider(
                user_agent=runtime.settings.user_agent,
                timeout_seconds=runtime.settings.timeout_seconds,
            )
            selected_service = JobSearchService(
                provider=owned_provider,
                max_results=runtime.settings.max_results,
            )
        if selected_planner is None:
            owned_planner_client = build_llm_client(
                runtime.settings,
                api_model=runtime.settings.llm_planner_api_model,
            )
            selected_planner = _LlmQueryPlanner(owned_planner_client)

        workflow = MatchingWorkflow(
            service=selected_service,
            planner=selected_planner,
            store=store,
            paths=runtime.paths,
        )
        return await workflow.run(bounds=bounds, existing_job_ids=existing)
    finally:
        try:
            if owned_provider is not None:
                await owned_provider.aclose()
        finally:
            if owned_planner_client is not None:
                await owned_planner_client.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with all state resolution after parsing."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_from_cli(args))
    except (CliConfigurationError, MatchingWorkflowError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "jobs_seeded": result.jobs_seeded,
                "jobs_seen": result.jobs_seen,
                "queries_planned": result.queries_planned,
                "queries_searched": result.queries_searched,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "MatchingBlacklistError",
    "MatchingBounds",
    "MatchingConfigurationError",
    "MatchingPlanRequest",
    "MatchingPlannerError",
    "MatchingProgress",
    "MatchingQueryOutcome",
    "MatchingStateError",
    "MatchingWorkflow",
    "MatchingWorkflowError",
    "MatchingWorkflowResult",
    "PublicJobSearch",
    "QueryPlanner",
    "build_arg_parser",
    "main",
    "run_from_cli",
    "run_matching_from_runtime",
    "run_matching_workflow",
]


if __name__ == "__main__":
    raise SystemExit(main())
