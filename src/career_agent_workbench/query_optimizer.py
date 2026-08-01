"""Pure, deterministic in-memory query ranking."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from career_agent_workbench.models import DatePosted, JobSearchQuery

DEFAULT_EXPLORATION_RATE = 0.20
DISALLOWED_REUSED_EXPERIENCE_LEVELS = frozenset({"internship", "entry_level"})
STOPWORDS = frozenset(
    {
        "and",
        "are",
        "for",
        "from",
        "jobs",
        "level",
        "role",
        "senior",
        "software",
        "the",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class StoredQueryOutcome:
    """Caller-supplied immutable historical outcome."""

    keywords: str
    location: str
    date_posted: str
    workplace_type: str | None
    experience_level: str | None
    job_type: str | None
    sort_by: str
    limit: int
    profile_match: float
    query_score: float
    results_returned: int
    fresh_jobs_accepted: int
    skipped_existing: int = 0
    skipped_blacklisted: int = 0
    skipped_workplace_type: int = 0
    skipped_experience_level: int = 0
    resumes_generated: int = 0
    average_ats_score: float | None = None

    @property
    def query(self) -> JobSearchQuery:
        return JobSearchQuery(
            keywords=self.keywords,
            location=self.location,
            date_posted=self.date_posted,  # type: ignore[arg-type]
            workplace_type=self.workplace_type,  # type: ignore[arg-type]
            experience_level=self.experience_level,  # type: ignore[arg-type]
            job_type=self.job_type,  # type: ignore[arg-type]
            sort_by=self.sort_by,  # type: ignore[arg-type]
            limit=min(max(self.limit, 1), 100),
        )


@dataclass(frozen=True, slots=True)
class QueryPerformanceEstimate:
    """Bounded immutable performance estimate."""

    history_count: int
    expected_fresh_results: float
    historical_acceptance_rate: float
    historical_ats_score: float
    duplicate_or_skip_rate: float


@dataclass(frozen=True, slots=True)
class ScoredQuery:
    """A query, score, and deterministic selection reason."""

    query: JobSearchQuery
    score: float
    profile_match: float
    performance: QueryPerformanceEstimate
    selection_reason: Literal["exploit", "explore"] = "exploit"


def historical_query_candidates(
    history: Sequence[StoredQueryOutcome],
    *,
    location: str,
    date_posted: DatePosted,
    limit_per_query: int,
    max_candidates: int = 8,
) -> list[JobSearchQuery]:
    """Reuse only productive, non-intern/entry historical keyword groups."""
    grouped: dict[str, dict[str, object]] = {}
    for outcome in history:
        key = _normalize_keywords(outcome.keywords)
        if not key:
            continue
        group = grouped.setdefault(
            key,
            {
                "keywords": outcome.keywords,
                "accepted": 0,
                "resumes": 0,
                "score": 0.0,
                "count": 0,
                "experience_level": outcome.experience_level,
                "job_type": outcome.job_type or "full_time",
            },
        )
        group["accepted"] = int(group["accepted"]) + max(
            0,
            outcome.fresh_jobs_accepted,
        )
        group["resumes"] = int(group["resumes"]) + max(0, outcome.resumes_generated)
        group["score"] = float(group["score"]) + _clamp_unit(outcome.query_score)
        group["count"] = int(group["count"]) + 1
        current = str(group["experience_level"] or "")
        replacement = str(outcome.experience_level or "")
        if (
            current in DISALLOWED_REUSED_EXPERIENCE_LEVELS or not current
        ) and replacement not in DISALLOWED_REUSED_EXPERIENCE_LEVELS:
            group["experience_level"] = replacement or None

    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            -int(item["accepted"]),
            -int(item["resumes"]),
            -(float(item["score"]) / max(int(item["count"]), 1)),
            str(item["keywords"]).casefold(),
        ),
    )
    candidates: list[JobSearchQuery] = []
    for item in ranked:
        if len(candidates) >= max(0, max_candidates):
            break
        if int(item["accepted"]) <= 0 and int(item["resumes"]) <= 0:
            continue
        experience = str(item["experience_level"] or "")
        if experience in DISALLOWED_REUSED_EXPERIENCE_LEVELS:
            experience = ""
        candidates.append(
            JobSearchQuery(
                keywords=str(item["keywords"]),
                location=location,
                date_posted=date_posted,
                job_type=str(item["job_type"] or "full_time"),  # type: ignore[arg-type]
                workplace_type="remote",
                experience_level=experience or None,  # type: ignore[arg-type]
                sort_by="recent",
                limit=min(max(limit_per_query, 1), 100),
            )
        )
    return candidates


def rank_search_queries(
    queries: Sequence[JobSearchQuery],
    *,
    profile_context: str,
    history: Sequence[StoredQueryOutcome],
    max_queries: int,
    exploration_rate: float = DEFAULT_EXPLORATION_RATE,
) -> list[ScoredQuery]:
    """Deduplicate, score, and deterministically mix exploit/explore queries."""
    if max_queries <= 0:
        return []
    scored = [
        _score_query(query=query, profile_context=profile_context, history=history)
        for query in _dedupe_queries(queries)
    ]
    scored.sort(
        key=lambda item: (
            -item.score,
            -item.profile_match,
            item.query.keywords.casefold(),
            item.query.workplace_type or "",
        )
    )
    if len(scored) <= max_queries:
        return scored

    safe_rate = exploration_rate if math.isfinite(exploration_rate) else 0.0
    explore_count = 0
    if max_queries > 1:
        explore_count = min(
            math.ceil(max_queries * _clamp_unit(safe_rate)),
            max_queries - 1,
        )
    exploit = scored[: max_queries - explore_count]
    selected_keys = {_query_key(item.query) for item in exploit}
    pool = [
        item
        for item in scored[max_queries - explore_count :]
        if _query_key(item.query) not in selected_keys
    ]
    pool.sort(
        key=lambda item: (
            item.performance.history_count,
            -item.profile_match,
            -item.score,
            item.query.keywords.casefold(),
        )
    )
    explore = [
        ScoredQuery(
            query=item.query,
            score=item.score,
            profile_match=item.profile_match,
            performance=item.performance,
            selection_reason="explore",
        )
        for item in pool[:explore_count]
    ]
    return [*exploit, *explore]


def query_profile_match(query: JobSearchQuery, profile_context: str) -> float:
    """Return token overlap between a query and caller-supplied context."""
    query_tokens = _tokens(query.keywords)
    if not query_tokens:
        return 0.0
    profile_tokens = _tokens(profile_context)
    if not profile_tokens:
        return 0.25
    return _clamp_unit(len(query_tokens & profile_tokens) / len(query_tokens))


def _score_query(
    *,
    query: JobSearchQuery,
    profile_context: str,
    history: Sequence[StoredQueryOutcome],
) -> ScoredQuery:
    profile_match = query_profile_match(query, profile_context)
    performance = _estimate_performance(query=query, history=history)
    score = _clamp_unit(
        0.35 * profile_match
        + 0.25 * performance.expected_fresh_results
        + 0.20 * performance.historical_acceptance_rate
        + 0.15 * performance.historical_ats_score
        - 0.05 * performance.duplicate_or_skip_rate
    )
    return ScoredQuery(
        query=query,
        score=score,
        profile_match=profile_match,
        performance=performance,
    )


def _estimate_performance(
    *,
    query: JobSearchQuery,
    history: Sequence[StoredQueryOutcome],
) -> QueryPerformanceEstimate:
    weighted = [
        (outcome, similarity)
        for outcome in history
        if (similarity := _query_similarity(query, outcome.query)) >= 0.20
    ]
    if not weighted:
        return QueryPerformanceEstimate(0, 0.50, 0.50, 0.50, 0.0)
    total_weight = sum(weight for _, weight in weighted)

    def average(values: Iterable[tuple[float, float]]) -> float:
        return sum(value * weight for value, weight in values) / max(
            total_weight,
            0.0001,
        )

    fresh = average(
        (
            (
                min(max(outcome.fresh_jobs_accepted, 0) / max(query.limit, 1), 1.0),
                weight,
            )
            for outcome, weight in weighted
        )
    )
    acceptance = average(
        (
            (
                max(outcome.fresh_jobs_accepted, 0) / max(outcome.results_returned, 1),
                weight,
            )
            for outcome, weight in weighted
        )
    )
    ats = average(
        (
            (
                outcome.average_ats_score / 100
                if outcome.average_ats_score is not None
                else 0.50,
                weight,
            )
            for outcome, weight in weighted
        )
    )
    skipped = average(
        (
            (
                (
                    max(outcome.skipped_existing, 0)
                    + max(outcome.skipped_blacklisted, 0)
                    + max(outcome.skipped_workplace_type, 0)
                    + max(outcome.skipped_experience_level, 0)
                )
                / max(outcome.results_returned, 1),
                weight,
            )
            for outcome, weight in weighted
        )
    )
    return QueryPerformanceEstimate(
        history_count=len(weighted),
        expected_fresh_results=_clamp_unit(fresh),
        historical_acceptance_rate=_clamp_unit(acceptance),
        historical_ats_score=_clamp_unit(ats),
        duplicate_or_skip_rate=_clamp_unit(skipped),
    )


def _query_similarity(
    query: JobSearchQuery,
    previous: JobSearchQuery,
) -> float:
    current = _tokens(query.keywords)
    old = _tokens(previous.keywords)
    token_similarity = (
        len(current & old) / len(current | old) if current and old else 0.0
    )
    score = token_similarity * 0.70
    score += 0.10 if query.workplace_type == previous.workplace_type else 0.0
    score += 0.08 if query.experience_level == previous.experience_level else 0.0
    score += 0.07 if query.date_posted == previous.date_posted else 0.0
    score += 0.05 if query.job_type == previous.job_type else 0.0
    return _clamp_unit(score)


def _dedupe_queries(queries: Sequence[JobSearchQuery]) -> list[JobSearchQuery]:
    deduped: list[JobSearchQuery] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for query in queries:
        key = _query_key(query)
        if key not in seen:
            seen.add(key)
            deduped.append(query)
    return deduped


def _query_key(query: JobSearchQuery) -> tuple[str, str, str, str, str]:
    return (
        _normalize_keywords(query.keywords),
        query.location.casefold().strip(),
        str(query.date_posted),
        query.workplace_type or "",
        query.experience_level or "",
    )


def _normalize_keywords(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9+#./-]*", value.casefold())
        if token not in STOPWORDS and len(token) >= 2
    }


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


__all__ = [
    "DEFAULT_EXPLORATION_RATE",
    "QueryPerformanceEstimate",
    "ScoredQuery",
    "StoredQueryOutcome",
    "historical_query_candidates",
    "query_profile_match",
    "rank_search_queries",
]
