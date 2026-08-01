"""Strict, bounded models for public job research."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

DatePosted = Literal["any_time", "past_24_hours", "past_week", "past_month"]
JobType = Literal[
    "full_time",
    "part_time",
    "contract",
    "temporary",
    "volunteer",
    "internship",
    "other",
]
WorkplaceType = Literal["on_site", "remote", "hybrid"]
ExperienceLevel = Literal[
    "internship",
    "entry_level",
    "associate",
    "mid_senior",
    "director",
    "executive",
]
SortBy = Literal["relevance", "recent"]

SearchText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
JobId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
DisplayText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=1_024),
]
DescriptionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=500_000),
]

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    hide_input_in_errors=True,
)
_URL_MAX_CHARS = 4_096


def _validate_public_url(value: object) -> object:
    if value is None:
        return value
    if not isinstance(value, (str, HttpUrl)):
        raise ValueError("URL is not public-safe.")  # noqa: TRY004
    rendered = str(value)
    if len(rendered) > _URL_MAX_CHARS:
        raise ValueError("URL is not public-safe.")
    try:
        parts = urlsplit(rendered)
        hostname = parts.hostname
        port = parts.port
    except (TypeError, ValueError):
        raise ValueError("URL is not public-safe.") from None
    if (
        parts.scheme.casefold() not in {"http", "https"}
        or not hostname
        or parts.username is not None
        or parts.password is not None
        or port is not None
        and not 0 < port <= 65_535
    ):
        raise ValueError("URL is not public-safe.")
    return rendered


class JobSearchQuery(BaseModel):
    """One bounded public job-search request."""

    model_config = _MODEL_CONFIG

    keywords: SearchText
    location: SearchText
    date_posted: DatePosted = "any_time"
    job_type: JobType | None = None
    workplace_type: WorkplaceType | None = None
    experience_level: ExperienceLevel | None = None
    sort_by: SortBy = "recent"
    distance: int | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=10, ge=1, le=100)
    page: int = Field(default=0, ge=0, le=10_000)
    exclude_job_ids: set[JobId] = Field(default_factory=set, max_length=500)


class JobPosting(BaseModel):
    """Normalized public job-card data."""

    model_config = _MODEL_CONFIG

    job_id: JobId
    title: DisplayText
    company: DisplayText | None = None
    location: DisplayText | None = None
    listed_at: DisplayText | None = None
    posted_text: DisplayText | None = None
    job_url: HttpUrl | None = None
    company_url: HttpUrl | None = None
    workplace_type: DisplayText | None = None
    source: DisplayText = "linkedin_public"

    _safe_job_url = field_validator("job_url", mode="before")(_validate_public_url)
    _safe_company_url = field_validator("company_url", mode="before")(
        _validate_public_url
    )


class JobSearchResult(BaseModel):
    """Normalized results and the effective query used."""

    model_config = _MODEL_CONFIG

    query: JobSearchQuery
    count: int = Field(ge=0, le=100)
    jobs: list[JobPosting] = Field(max_length=100)
    provider: DisplayText = "linkedin_public"

    @model_validator(mode="after")
    def _count_matches_jobs(self) -> JobSearchResult:
        if self.count != len(self.jobs):
            raise ValueError("Result count does not match returned jobs.")
        return self


class JobDetails(JobPosting):
    """Normalized public job-detail data."""

    description: DescriptionText | None = None
    seniority_level: DisplayText | None = None
    employment_type: DisplayText | None = None
    job_function: DisplayText | None = None
    industries: DisplayText | None = None


class JobRawPayload(BaseModel):
    """One bounded in-memory public detail payload."""

    model_config = _MODEL_CONFIG

    job_id: JobId
    detail_url: HttpUrl
    status_code: int = Field(ge=100, le=599)
    content_type: DisplayText | None = None
    payload_type: Literal["html"] = "html"
    payload_chars: int = Field(ge=0, le=2_000_000)
    payload: str = Field(max_length=2_000_000, repr=False)
    parsed: JobDetails

    _safe_detail_url = field_validator("detail_url", mode="before")(
        _validate_public_url
    )

    @model_validator(mode="after")
    def _payload_length_matches(self) -> JobRawPayload:
        if self.payload_chars != len(self.payload):
            raise ValueError("Raw payload length metadata does not match.")
        return self
