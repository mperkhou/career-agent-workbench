"""Bounded public URL ingestion for the local Flask adapter."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from career_agent_workbench.api_client import _ResponseTooLarge, _bounded_http_request
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateStore,
)
from career_agent_workbench.generic_job_scraper import (
    MAX_HTML_CHARS,
    extract_generic_job_details_from_html,
    normalize_job_url,
)
from career_agent_workbench.jod import job_description_context, usable_job_description
from career_agent_workbench.models import JobDetails
from career_agent_workbench.providers.linkedin_public import (
    LINKEDIN_JOB_URL,
    LinkedInPublicJobsProvider,
    extract_job_id,
)

MAX_INGESTION_URLS = 20
MAX_URL_LIST_CHARS = 82_000

LinkedInDetailsFetcher = Callable[[str], JobDetails]
GenericHtmlFetcher = Callable[[str], str]


class WebIngestionError(ValueError):
    """Raised for a content-free ingestion boundary failure."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ParsedUrlBatch:
    """Normalized URL input with only a rejected-input count exposed."""

    urls: tuple[str, ...] = field(repr=False)
    rejected: int


@dataclass(frozen=True, slots=True, repr=False)
class IngestionResult:
    """Content-free URL ingestion result."""

    created: int
    refreshed: int
    failed: int
    job_ids: tuple[str, ...] = field(repr=False)

    @property
    def accepted(self) -> int:
        return self.created + self.refreshed

    def __repr__(self) -> str:
        return (
            "IngestionResult("
            f"created={self.created}, refreshed={self.refreshed}, "
            f"failed={self.failed}, identifiers_hidden=True)"
        )


def parse_job_url_batch(value: object, *, linkedin: bool) -> ParsedUrlBatch:
    """Normalize, deduplicate, and bound a submitted public URL list."""

    if type(value) is not str or not value.strip() or len(value) > MAX_URL_LIST_CHARS:
        raise WebIngestionError("Job URL input is invalid.")
    parts = tuple(part for part in re.split(r"[\s,]+", value.strip()) if part)
    if not parts or len(parts) > MAX_INGESTION_URLS:
        raise WebIngestionError("Job URL input is invalid.")
    urls: list[str] = []
    seen: set[str] = set()
    rejected = 0
    for part in parts:
        normalized = _normalize_submitted_url(part, linkedin=linkedin)
        if normalized is None:
            rejected += 1
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
    if not urls:
        raise WebIngestionError("Job URL input is invalid.")
    return ParsedUrlBatch(urls=tuple(urls), rejected=rejected)


def ingest_linkedin_urls(
    *,
    store: ApplicationStateStore,
    batch: ParsedUrlBatch,
    fetcher: LinkedInDetailsFetcher,
) -> IngestionResult:
    """Fetch guest-only details and atomically seed each accepted URL."""

    return _ingest_urls(
        store=store,
        batch=batch,
        details_fetcher=fetcher,
    )


def ingest_generic_urls(
    *,
    store: ApplicationStateStore,
    batch: ParsedUrlBatch,
    html_fetcher: GenericHtmlFetcher,
) -> IngestionResult:
    """Fetch bounded public HTML, parse it purely, and seed each URL."""

    def details_fetcher(url: str) -> JobDetails:
        html = html_fetcher(url)
        return extract_generic_job_details_from_html(html=html, url=url)

    return _ingest_urls(
        store=store,
        batch=batch,
        details_fetcher=details_fetcher,
    )


def fetch_linkedin_details(
    url: str,
    *,
    user_agent: str,
    timeout_seconds: float,
) -> JobDetails:
    """Use the existing guest-only provider for one normalized URL."""

    async def fetch() -> JobDetails:
        provider = LinkedInPublicJobsProvider(
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
        )
        try:
            return await provider.get_job_details(url)
        finally:
            await provider.aclose()

    try:
        return asyncio.run(fetch())
    except Exception:  # noqa: BLE001 - provider failures stay content-free.
        raise WebIngestionError("Public LinkedIn job fetch failed.") from None


def fetch_generic_html(
    url: str,
    *,
    user_agent: str,
    timeout_seconds: float,
) -> str:
    """Fetch one normalized public HTML page through a bounded HTTP client."""

    normalized = normalize_job_url(url)

    async def fetch() -> str:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
        ) as client:
            response, body = await _bounded_http_request(
                client,
                "GET",
                normalized,
                max_response_bytes=MAX_HTML_CHARS,
            )
            if not 200 <= response.status_code < 300:
                raise WebIngestionError("Public job fetch failed.")
            media_type = response.headers.get("content-type", "").split(";", 1)[0]
            if media_type.casefold().strip() not in {
                "text/html",
                "application/xhtml+xml",
            }:
                raise WebIngestionError("Public job fetch failed.")
            try:
                return body.decode(response.encoding or "utf-8")
            except (LookupError, UnicodeDecodeError):
                raise WebIngestionError("Public job fetch failed.") from None

    try:
        return asyncio.run(fetch())
    except (_ResponseTooLarge, httpx.HTTPError, WebIngestionError):
        raise WebIngestionError("Public job fetch failed.") from None


def _ingest_urls(
    *,
    store: ApplicationStateStore,
    batch: ParsedUrlBatch,
    details_fetcher: LinkedInDetailsFetcher,
) -> IngestionResult:
    created = 0
    refreshed = 0
    failed = batch.rejected
    job_ids: list[str] = []
    for url in batch.urls:
        try:
            details = details_fetcher(url)
            if type(details) is not JobDetails:
                raise TypeError
            source_text = usable_job_description(details.description)
            if source_text is None:
                raise ValueError
            prompt_text = job_description_context(
                details.model_copy(update={"description": source_text})
            )
            outcome = store.seed_application_with_outcome(
                ApplicationMetadata(
                    job_id=details.job_id,
                    company=details.company or "Unknown company",
                    job_title=details.title or "Unknown title",
                    job_url=str(details.job_url or url),
                    source=details.source,
                    date_posted=_posted_date(details),
                    experience_level=details.seniority_level,
                ),
                source_text=source_text,
                prompt_text=prompt_text,
            )
        except Exception:  # noqa: BLE001 - one URL failure is count-only.
            failed += 1
            continue
        if outcome.created:
            created += 1
        else:
            refreshed += 1
        job_ids.append(outcome.application.job_id)
    return IngestionResult(
        created=created,
        refreshed=refreshed,
        failed=failed,
        job_ids=tuple(job_ids),
    )


def _normalize_submitted_url(value: str, *, linkedin: bool) -> str | None:
    if linkedin:
        job_id = extract_job_id(value)
        return None if job_id is None else LINKEDIN_JOB_URL.format(job_id=job_id)
    try:
        return normalize_job_url(value)
    except Exception:  # noqa: BLE001 - invalid inputs are count-only.
        return None


def _posted_date(details: JobDetails) -> str | None:
    value = details.listed_at or details.posted_text
    if value is None:
        return None
    selected = value.strip()
    match = re.match(r"^\d{4}-\d{2}-\d{2}", selected)
    return match.group(0) if match is not None else selected


__all__ = [
    "GenericHtmlFetcher",
    "IngestionResult",
    "LinkedInDetailsFetcher",
    "MAX_INGESTION_URLS",
    "ParsedUrlBatch",
    "WebIngestionError",
    "fetch_generic_html",
    "fetch_linkedin_details",
    "ingest_generic_urls",
    "ingest_linkedin_urls",
    "parse_job_url_batch",
]
