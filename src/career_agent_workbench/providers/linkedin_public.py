"""Guest-only access to LinkedIn's public job endpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from career_agent_workbench.api_client import (
    _bounded_http_request,
    _ResponseTooLarge,
)
from career_agent_workbench.errors import JobNotFoundError, ProviderError
from career_agent_workbench.models import (
    JobDetails,
    JobPosting,
    JobRawPayload,
    JobSearchQuery,
)

SEARCH_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
DETAIL_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
LINKEDIN_JOB_URL = "https://www.linkedin.com/jobs/view/{job_id}"
MAX_RESPONSE_BYTES = 2_000_000

DATE_POSTED_FILTERS = {
    "past_24_hours": "r86400",
    "past_week": "r604800",
    "past_month": "r2592000",
}
JOB_TYPE_FILTERS = {
    "full_time": "F",
    "part_time": "P",
    "contract": "C",
    "temporary": "T",
    "volunteer": "V",
    "internship": "I",
    "other": "O",
}
WORKPLACE_FILTERS = {"on_site": "1", "remote": "2", "hybrid": "3"}
EXPERIENCE_FILTERS = {
    "internship": "1",
    "entry_level": "2",
    "associate": "3",
    "mid_senior": "4",
    "director": "5",
    "executive": "6",
}
SORT_FILTERS = {"relevance": "R", "recent": "DD"}
_FORBIDDEN_RESPONSE_MARKERS = (
    "authwall",
    "/uas/login",
    "/checkpoint/challenge",
    "linkedin login",
)
_TRACKING_KEYS = {"ref", "refid", "trk", "trackingid"}
_ASCII_JOB_ID = re.compile(r"[0-9]{1,128}\Z")


class LinkedInPublicJobsProvider:
    """Own a stateless, proxy-free guest-only HTTP client."""

    name = "linkedin_public"

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        client = None
        try:
            client = httpx.AsyncClient(
                timeout=timeout_seconds,
                transport=transport,
                trust_env=False,
                follow_redirects=False,
                event_hooks={
                    "request": [self._remove_request_cookies],
                    "response": [self._discard_response_cookies],
                },
                headers={
                    "User-Agent": str(user_agent),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
        except (TypeError, ValueError):
            pass
        if client is None:
            raise ProviderError("Public provider configuration is invalid.")
        self._client = client

    def __repr__(self) -> str:
        return "LinkedInPublicJobsProvider(name='linkedin_public')"

    async def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]:
        html, _, _ = await self._get_public_html(
            operation="search",
            query=query,
        )
        parser_failed = False
        try:
            jobs = _parse_search_results(html)
        except Exception:  # noqa: BLE001 - sanitize parser/model failures
            parser_failed = True
        if parser_failed:
            raise ProviderError("Public job search parsing failed.")
        return jobs

    async def get_job_details(self, job_id_or_url: str) -> JobDetails:
        return (await self.get_job_raw_payload(job_id_or_url)).parsed

    async def get_job_raw_payload(self, job_id_or_url: str) -> JobRawPayload:
        job_id = extract_job_id(job_id_or_url)
        if job_id is None:
            raise JobNotFoundError("Public job identifier is invalid.")
        detail_url = DETAIL_URL.format(job_id=job_id)
        html, status_code, content_type = await self._get_public_html(
            operation="detail",
            job_id=job_id,
        )
        parser_failed = False
        try:
            parsed = _parse_job_details(html, job_id)
            payload = JobRawPayload(
                job_id=job_id,
                detail_url=detail_url,
                status_code=status_code,
                content_type=content_type,
                payload_chars=len(html),
                payload=html,
                parsed=parsed,
            )
        except (JobNotFoundError, ProviderError):
            raise
        except Exception:  # noqa: BLE001 - sanitize parser/model failures
            parser_failed = True
        if parser_failed:
            raise ProviderError("Public job detail parsing failed.")
        return payload

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _remove_request_cookies(self, request: httpx.Request) -> None:
        request.headers.pop("cookie", None)

    async def _discard_response_cookies(self, response: httpx.Response) -> None:
        response.cookies.clear()
        self._client.cookies.clear()

    async def _get_public_html(
        self,
        url: str | None = None,
        *,
        params: Mapping[str, str | int] | None = None,
        operation: str,
        query: JobSearchQuery | None = None,
        job_id: str | None = None,
    ) -> tuple[str, int, str]:
        request = _approved_public_request(
            operation=operation,
            query=query,
            job_id=job_id,
            direct_url=url,
            direct_params=params,
        )
        if request is None:
            raise ProviderError("Public job endpoint is not allowed.")
        request_url, request_params = request
        request_error = None
        try:
            response, body = await _bounded_http_request(
                self._client,
                "GET",
                request_url,
                max_response_bytes=MAX_RESPONSE_BYTES,
                params=request_params,
            )
        except _ResponseTooLarge:
            request_error = ProviderError(
                f"Public job {operation} response exceeds the public size limit."
            )
        except httpx.HTTPError:
            request_error = ProviderError(f"Public job {operation} request failed.")
        finally:
            self._client.cookies.clear()
        if request_error is not None:
            raise request_error

        response.cookies.clear()
        if response.status_code == 404:
            raise JobNotFoundError("Public job was not found.")
        if 300 <= response.status_code < 400:
            raise ProviderError(f"Public job {operation} redirect was rejected.")
        if not 200 <= response.status_code < 300:
            raise ProviderError(
                f"Public job {operation} failed with HTTP status "
                f"{response.status_code}."
            )
        media_type = response.headers.get("content-type", "").split(";", 1)[0]
        media_type = media_type.casefold().strip()
        if media_type not in {"text/html", "application/xhtml+xml"}:
            raise ProviderError(
                f"Public job {operation} returned an unsupported media type."
            )
        decode_failed = False
        try:
            html = body.decode(response.encoding or "utf-8")
        except (LookupError, UnicodeDecodeError):
            decode_failed = True
        if decode_failed:
            raise ProviderError(f"Public job {operation} returned unreadable HTML.")
        lowered = html.casefold()
        if any(marker in lowered for marker in _FORBIDDEN_RESPONSE_MARKERS):
            raise ProviderError(
                f"Public job {operation} authentication boundary was rejected."
            )
        return html, response.status_code, media_type


def _approved_public_request(
    *,
    operation: str,
    query: JobSearchQuery | None,
    job_id: str | None,
    direct_url: str | None,
    direct_params: Mapping[str, str | int] | None,
) -> tuple[str, Mapping[str, str | int] | None] | None:
    if direct_url is not None or direct_params is not None:
        return None
    if operation == "search" and isinstance(query, JobSearchQuery) and job_id is None:
        return SEARCH_URL, _search_params(query)
    if operation == "detail" and query is None and _is_ascii_job_id(job_id):
        return DETAIL_URL.format(job_id=job_id), None
    return None


def _search_params(query: JobSearchQuery) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "keywords": query.keywords,
        "location": query.location,
        "start": query.page * query.limit,
        "sortBy": SORT_FILTERS[query.sort_by],
    }
    if query.date_posted != "any_time":
        params["f_TPR"] = DATE_POSTED_FILTERS[query.date_posted]
    if query.job_type:
        params["f_JT"] = JOB_TYPE_FILTERS[query.job_type]
    if query.workplace_type:
        params["f_WT"] = WORKPLACE_FILTERS[query.workplace_type]
    if query.experience_level:
        params["f_E"] = EXPERIENCE_FILTERS[query.experience_level]
    if query.distance is not None:
        params["distance"] = query.distance
    return params


def _parse_search_results(html: str) -> list[JobPosting]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("[data-entity-urn*='jobPosting']") or soup.select("li")
    jobs: list[JobPosting] = []
    for card in cards:
        job_id = _extract_card_job_id(card.attrs)
        title_node = card.select_one(".base-search-card__title")
        company_node = card.select_one(".base-search-card__subtitle")
        location_node = card.select_one(".job-search-card__location")
        link_node = card.select_one("a.base-card__full-link, a[href*='/jobs/view/']")
        date_node = card.select_one("time")
        metadata_node = card.select_one(".job-search-card__metadata")

        href = _clean_url(str(link_node.get("href") or "")) if link_node else None
        href_job_id = extract_job_id(href or "")
        job_id = job_id or href_job_id
        if not job_id:
            continue
        if href_job_id != job_id:
            href = None
        try:
            jobs.append(
                JobPosting(
                    job_id=job_id,
                    title=_text(title_node) or "Unknown title",
                    company=_text(company_node),
                    location=_text(location_node),
                    listed_at=(
                        str(date_node.get("datetime") or "") or None
                        if date_node
                        else None
                    ),
                    posted_text=_text(date_node),
                    job_url=href or LINKEDIN_JOB_URL.format(job_id=job_id),
                    company_url=(
                        _clean_url(str(company_node.get("href") or ""))
                        if company_node
                        else None
                    ),
                    workplace_type=_text(metadata_node),
                )
            )
        except Exception:  # noqa: BLE001,S112 - discard malformed public cards
            continue
    return jobs


def _parse_job_details(html: str, job_id: str) -> JobDetails:
    soup = BeautifulSoup(html, "html.parser")
    title = _text(soup.select_one(".top-card-layout__title, h2")) or "Unknown title"
    company_node = soup.select_one(".topcard__org-name-link, .topcard__flavor")
    location_node = soup.select_one(".topcard__flavor--bullet")
    date_node = soup.select_one("time")
    description_node = soup.select_one(".show-more-less-html__markup")
    criteria = _parse_criteria(soup)
    return JobDetails(
        job_id=job_id,
        title=title,
        company=_text(company_node),
        location=_text(location_node),
        listed_at=(str(date_node.get("datetime") or "") or None if date_node else None),
        posted_text=_text(date_node),
        job_url=LINKEDIN_JOB_URL.format(job_id=job_id),
        company_url=(
            _clean_url(str(company_node.get("href") or "")) if company_node else None
        ),
        description=_text(description_node),
        seniority_level=criteria.get("Seniority level"),
        employment_type=criteria.get("Employment type"),
        workplace_type=criteria.get("Workplace type") or criteria.get("Workplace"),
        job_function=criteria.get("Job function"),
        industries=criteria.get("Industries"),
    )


def _parse_criteria(soup: BeautifulSoup) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for item in soup.select(".description__job-criteria-item"):
        label = _text(item.select_one(".description__job-criteria-subheader"))
        value = _text(item.select_one(".description__job-criteria-text"))
        if label and value:
            criteria[label] = value
    return criteria


def extract_job_id(value: str) -> str | None:
    """Extract a numeric ID from a supported public LinkedIn identifier."""
    candidate = str(value or "").strip()
    if _is_ascii_job_id(candidate):
        return candidate
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.hostname != "www.linkedin.com"
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    query = parse_qs(parsed.query)
    if parsed.path.rstrip("/") == "/jobs/search":
        values = query.get("currentJobId", [])
        if values and _is_ascii_job_id(values[0]):
            return values[0]
    match = re.fullmatch(r"/jobs/view/(?:[^/?#]*-)?([^/?#]+)/?", parsed.path)
    if match and _is_ascii_job_id(match.group(1)):
        return match.group(1)
    return None


def _extract_card_job_id(attrs: Mapping[str, object]) -> str | None:
    for key in ("data-entity-urn", "data-id", "data-job-id"):
        value = attrs.get(key)
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        prefix = "urn:li:jobPosting:"
        if candidate.startswith(prefix):
            candidate = candidate.removeprefix(prefix)
        if _is_ascii_job_id(candidate):
            return candidate
    return None


def _is_ascii_job_id(value: object) -> bool:
    return isinstance(value, str) and _ASCII_JOB_ID.fullmatch(value) is not None


def _text(node: object | None) -> str | None:
    if node is None or not hasattr(node, "get_text"):
        return None
    text = node.get_text(" ", strip=True)
    return " ".join(str(text).split()) or None


def _clean_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlsplit(url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        and not 0 < port <= 65_535
    ):
        return None
    query = [
        (key, value)
        for key, values in parse_qs(parsed.query, keep_blank_values=False).items()
        for value in values
        if key.casefold() not in _TRACKING_KEYS
        and not key.casefold().startswith("utm_")
    ]
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path,
            urlencode(query, doseq=True),
            "",
        )
    )
