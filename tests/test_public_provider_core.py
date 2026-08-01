from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx
import pytest

from career_agent_workbench import providers as providers_package
from career_agent_workbench.errors import JobNotFoundError, ProviderError
from career_agent_workbench.models import JobSearchQuery
from career_agent_workbench.providers import linkedin_public
from career_agent_workbench.providers.linkedin_public import (
    DETAIL_URL,
    SEARCH_URL,
    LinkedInPublicJobsProvider,
    extract_job_id,
)

CARD_HTML = """
<ul>
  <li data-entity-urn="urn:li:jobPosting:910000001">
    <h3 class="base-search-card__title">Synthetic Reliability Engineer</h3>
    <a class="base-search-card__subtitle"
       href="https://companies.example.invalid/harbor">Example Harbor Systems</a>
    <span class="job-search-card__location">Example City</span>
    <a class="base-card__full-link"
       href="https://www.linkedin.com/jobs/view/synthetic-role-910000001/?trk=guest&amp;keep=yes">
       View
    </a>
    <time datetime="2026-07-01">Recently</time>
    <span class="job-search-card__metadata">Remote</span>
  </li>
</ul>
"""

DETAIL_HTML = """
<main>
  <h2 class="top-card-layout__title">Synthetic Reliability Engineer</h2>
  <a class="topcard__org-name-link"
     href="https://companies.example.invalid/harbor?utm_source=test">
     Example Harbor Systems
  </a>
  <span class="topcard__flavor--bullet">Example City</span>
  <time datetime="2026-07-01">Recently</time>
  <div class="show-more-less-html__markup">
    Design reliable fictional services and public test automation.
  </div>
  <div class="description__job-criteria-item">
    <h3 class="description__job-criteria-subheader">Employment type</h3>
    <span class="description__job-criteria-text">Full-time</span>
  </div>
</main>
"""


class _TrackedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _BlockingAsyncStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await self.release.wait()
        yield b"<html></html>"

    async def aclose(self) -> None:
        self.closed = True


def _assert_sanitized_exception(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


def _query() -> JobSearchQuery:
    return JobSearchQuery(
        keywords="synthetic reliability",
        location="Example City",
        date_posted="past_week",
        job_type="full_time",
        workplace_type="remote",
        experience_level="associate",
        sort_by="recent",
        distance=25,
        limit=20,
        page=2,
    )


def test_guest_requests_filters_cookie_discard_parsing_and_raw() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.host == "www.linkedin.com"
        lowered_headers = {name.casefold() for name in request.headers}
        assert "authorization" not in lowered_headers
        assert "cookie" not in lowered_headers
        assert not any("csrf" in name or "session" in name for name in lowered_headers)
        if request.url.path.endswith("/search"):
            params = request.url.params
            assert params["keywords"] == "synthetic reliability"
            assert params["location"] == "Example City"
            assert params["start"] == "40"
            assert params["sortBy"] == "DD"
            assert params["f_TPR"] == "r604800"
            assert params["f_JT"] == "F"
            assert params["f_WT"] == "2"
            assert params["f_E"] == "3"
            assert params["distance"] == "25"
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/html; charset=utf-8",
                    "set-cookie": "synthetic_session=discarded",
                },
                text=CARD_HTML,
            )
        assert request.url.path == "/jobs-guest/jobs/api/jobPosting/910000001"
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=DETAIL_HTML,
        )

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        jobs = await provider.search_jobs(_query())
        assert len(jobs) == 1
        assert jobs[0].job_id == "910000001"
        assert "trk=" not in str(jobs[0].job_url)
        assert "keep=yes" in str(jobs[0].job_url)
        details = await provider.get_job_details("910000001")
        assert details.company == "Example Harbor Systems"
        assert details.employment_type == "Full-time"
        raw = await provider.get_job_raw_payload(
            "https://www.linkedin.com/jobs/view/synthetic-role-910000001/"
        )
        assert raw.payload == DETAIL_HTML
        assert raw.payload_chars == len(DETAIL_HTML)
        assert raw.parsed.job_id == "910000001"
        await provider.aclose()

    asyncio.run(scenario())
    assert len(requests) == 3
    assert requests[0].url.path == httpx.URL(SEARCH_URL).path
    assert DETAIL_URL.format(job_id="910000001").endswith(requests[1].url.path)


def test_overlapping_guest_requests_remain_cookie_free_and_cancellation_safe() -> None:
    first_stream = _BlockingAsyncStream()
    cancelled_stream = _BlockingAsyncStream()
    detail_streams = [
        _TrackedAsyncStream([DETAIL_HTML.encode()]),
        _TrackedAsyncStream([DETAIL_HTML.encode()]),
    ]
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "cookie" not in request.headers
        call = len(requests)
        if call == 1:
            response = httpx.Response(
                200,
                headers={
                    "content-type": "text/html",
                    "set-cookie": "synthetic_cookie=synthetic_value; Path=/",
                },
                stream=first_stream,
            )
        elif call == 2:
            response = httpx.Response(
                200,
                headers={"content-type": "text/html"},
                stream=detail_streams[0],
            )
        elif call == 3:
            response = httpx.Response(
                200,
                headers={
                    "content-type": "text/html",
                    "set-cookie": "synthetic_cookie=synthetic_value; Path=/",
                },
                stream=cancelled_stream,
            )
        else:
            response = httpx.Response(
                200,
                headers={"content-type": "text/html"},
                stream=detail_streams[1],
            )
        responses.append(response)
        return response

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )
    provider._client.cookies.set(
        "synthetic_seed",
        "synthetic_value",
        domain=".linkedin.com",
        path="/",
    )

    async def scenario() -> None:
        first = asyncio.create_task(provider.search_jobs(_query()))
        await first_stream.started.wait()
        assert len(responses[0].cookies) == 0
        assert len(provider._client.cookies) == 0

        second = asyncio.create_task(provider.get_job_details("910000011"))
        details = await second
        assert details.job_id == "910000011"
        assert len(requests) == 2
        assert not first.done()
        first_stream.release.set()
        assert await first == []

        cancelled = asyncio.create_task(provider.search_jobs(_query()))
        await cancelled_stream.started.wait()
        assert len(responses[2].cookies) == 0
        assert len(provider._client.cookies) == 0
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        follow_up = await provider.get_job_details("910000012")
        assert follow_up.job_id == "910000012"
        assert len(provider._client.cookies) == 0
        assert all(len(response.cookies) == 0 for response in responses)
        assert first_stream.closed
        assert cancelled_stream.closed
        assert all(stream.closed for stream in detail_streams)
        await provider.aclose()

    asyncio.run(scenario())
    assert len(requests) == 4
    assert all("cookie" not in request.headers for request in requests)


def test_public_job_id_extraction_is_bounded_to_supported_forms() -> None:
    assert extract_job_id("910000002") == "910000002"
    assert (
        extract_job_id("https://www.linkedin.com/jobs/view/synthetic-role-910000003/")
        == "910000003"
    )
    assert (
        extract_job_id("https://www.linkedin.com/jobs/search/?currentJobId=910000004")
        == "910000004"
    )
    for value in (
        "synthetic-text-id",
        "https://jobs.example.invalid/jobs/view/synthetic-role-910000003/",
        "https://user:marker@www.linkedin.com/jobs/view/910000003/",
        "https://www.linkedin.com/in/example-profile/",
    ):
        assert extract_job_id(value) is None


@pytest.mark.parametrize("job_id", ["１２３", "١٢٣", "²"])
def test_non_ascii_job_ids_are_rejected_at_every_provider_boundary(
    job_id: str,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            text=DETAIL_HTML,
            headers={"content-type": "text/html"},
        )

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )
    assert extract_job_id(job_id) is None
    assert (
        extract_job_id("https://www.linkedin.com/jobs/search/?currentJobId=" + job_id)
        is None
    )
    assert (
        extract_job_id(
            "https://www.linkedin.com/jobs/view/synthetic-role-" + job_id + "/"
        )
        is None
    )

    for attribute in ("data-entity-urn", "data-id", "data-job-id"):
        value = (
            f"urn:li:jobPosting:{job_id}" if attribute == "data-entity-urn" else job_id
        )
        html = (
            f'<li {attribute}="{value}">'
            '<h3 class="base-search-card__title">Synthetic role</h3>'
            "</li>"
        )
        assert linkedin_public._parse_search_results(html) == []

    async def scenario() -> None:
        with pytest.raises(JobNotFoundError) as public_error:
            await provider.get_job_details(job_id)
        _assert_sanitized_exception(public_error.value)
        with pytest.raises(ProviderError, match="endpoint is not allowed") as internal:
            await provider._get_public_html(
                operation="detail",
                job_id=job_id,
            )
        _assert_sanitized_exception(internal.value)
        await provider.aclose()

    asyncio.run(scenario())
    assert calls == 0


def test_redirect_is_rejected_before_second_request_and_cookies_are_not_reused() -> (
    None
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            302,
            headers={
                "location": "https://login.example.invalid/synthetic",
                "set-cookie": "synthetic_session=discarded",
                "content-type": "text/html",
            },
        )

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(ProviderError, match="redirect was rejected") as captured:
            await provider.search_jobs(_query())
        assert captured.value.__cause__ is None
        await provider.aclose()

    asyncio.run(scenario())
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("response", "error_type", "message"),
    [
        (
            httpx.Response(404, headers={"content-type": "text/html"}),
            JobNotFoundError,
            "not found",
        ),
        (
            httpx.Response(
                503,
                headers={"content-type": "text/html"},
                text="UPSTREAM-SENSITIVE-MARKER",
            ),
            ProviderError,
            "HTTP status 503",
        ),
        (
            httpx.Response(
                200,
                headers={"content-type": "application/json"},
                text="UPSTREAM-SENSITIVE-MARKER",
            ),
            ProviderError,
            "unsupported media type",
        ),
        (
            httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"x" * 2_000_001,
            ),
            ProviderError,
            "size limit",
        ),
        (
            httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<title>LinkedIn Login</title>UPSTREAM-SENSITIVE-MARKER",
            ),
            ProviderError,
            "authentication boundary",
        ),
    ],
)
def test_provider_bounds_status_and_sanitized_failures(
    response: httpx.Response,
    error_type: type[Exception],
    message: str,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return response

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(error_type, match=message) as captured:
            await provider.get_job_details("910000005")
        rendered = f"{captured.value!r} {captured.value}"
        assert "UPSTREAM-SENSITIVE-MARKER" not in rendered
        assert captured.value.__cause__ is None
        await provider.aclose()

    asyncio.run(scenario())


def test_provider_sanitizes_transport_failure() -> None:
    marker = "TRANSPORT-SENSITIVE-MARKER"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(marker, request=request)

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        with pytest.raises(ProviderError) as captured:
            await provider.search_jobs(_query())
        assert marker not in str(captured.value)
        _assert_sanitized_exception(captured.value)
        await provider.aclose()

    asyncio.run(scenario())


def test_provider_concurrent_success_logging_is_safe_and_restores_logger(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    markers = (
        "KEYWORDS-SYNTHETIC-MARKER",
        "LOCATION-SYNTHETIC-MARKER",
        "HTML-SYNTHETIC-MARKER",
        "USER-AGENT-SYNTHETIC-MARKER",
    )
    arrivals = 0
    both_arrived = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            both_arrived.set()
        await both_arrived.wait()
        body = (
            f"<html>{markers[2]}</html>"
            if request.url.path.endswith("/search")
            else DETAIL_HTML.replace("</main>", f"{markers[2]}</main>")
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/html"},
        )

    caplog.set_level(logging.INFO, logger="httpx")
    logger = logging.getLogger("httpx")
    state_before = (
        logger.level,
        logger.propagate,
        logger.disabled,
        tuple(logger.handlers),
        tuple(logger.filters),
    )
    provider = LinkedInPublicJobsProvider(
        user_agent=markers[3],
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )
    query = JobSearchQuery(
        keywords=markers[0],
        location=markers[1],
    )

    async def scenario() -> None:
        search, details = await asyncio.gather(
            provider.search_jobs(query),
            provider.get_job_details("910000009"),
        )
        assert search == []
        assert details.job_id == "910000009"
        await provider.aclose()

    asyncio.run(scenario())
    state_after = (
        logger.level,
        logger.propagate,
        logger.disabled,
        tuple(logger.handlers),
        tuple(logger.filters),
    )
    assert state_after == state_before
    streams = capsys.readouterr()
    rendered = f"{caplog.text}\n{streams.out}\n{streams.err}"
    assert all(marker not in rendered for marker in markers)


def test_provider_rejects_direct_or_off_contract_helpers_before_transport() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            text="<html></html>",
            headers={"content-type": "text/html"},
        )

    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        attempts = (
            lambda: provider._get_public_html(
                "https://off-contract.example.invalid/path",
                operation="search",
            ),
            lambda: provider._get_public_html(
                SEARCH_URL,
                params={"keywords": "direct-misuse"},
                operation="search",
            ),
            lambda: provider._get_public_html(operation="unsupported"),
            lambda: provider._get_public_html(
                operation="detail",
                job_id="not-a-public-job-id",
            ),
        )
        for attempt in attempts:
            with pytest.raises(ProviderError, match="endpoint is not allowed") as error:
                await attempt()
            _assert_sanitized_exception(error.value)
        await provider.aclose()

    asyncio.run(scenario())
    assert calls == 0


def test_provider_rejects_late_authwall_marker() -> None:
    html = "<html>" + "x" * 1_990_000 + "authwall</html>"
    assert len(html.encode()) < 2_000_000
    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                text=html,
                headers={"content-type": "text/html"},
            )
        ),
    )

    async def scenario() -> None:
        with pytest.raises(ProviderError, match="authentication boundary") as error:
            await provider.search_jobs(_query())
        _assert_sanitized_exception(error.value)
        await provider.aclose()

    asyncio.run(scenario())


def test_provider_stream_bounds_status_decode_and_closure() -> None:
    async def run_response(
        response: httpx.Response,
        stream: _TrackedAsyncStream,
        *,
        message: str,
    ) -> None:
        provider = LinkedInPublicJobsProvider(
            user_agent="Synthetic-Test-Agent/1.0",
            timeout_seconds=3,
            transport=httpx.MockTransport(lambda _: response),
        )
        with pytest.raises(ProviderError, match=message) as error:
            await provider.search_jobs(_query())
        _assert_sanitized_exception(error.value)
        assert stream.closed
        await provider.aclose()

    async def scenario() -> None:
        declared = _TrackedAsyncStream([b"<html></html>"])
        await run_response(
            httpx.Response(
                200,
                headers={
                    "content-type": "text/html",
                    "content-length": "2000001",
                },
                stream=declared,
            ),
            declared,
            message="size limit",
        )
        assert declared.consumed == 0

        crossing = _TrackedAsyncStream(
            [
                b"x" * 1_000_000,
                b"x" * 1_000_000,
                b"x",
                b"must-not-be-consumed",
            ]
        )
        await run_response(
            httpx.Response(
                200,
                headers={"content-type": "text/html", "content-length": "1"},
                stream=crossing,
            ),
            crossing,
            message="size limit",
        )
        assert crossing.consumed == 3

        redirect = _TrackedAsyncStream([b"redirect body"])
        await run_response(
            httpx.Response(
                302,
                headers={"content-type": "text/html"},
                stream=redirect,
            ),
            redirect,
            message="redirect was rejected",
        )
        assert redirect.consumed == 0

        undecodable = _TrackedAsyncStream([b"\xff"])
        await run_response(
            httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                stream=undecodable,
            ),
            undecodable,
            message="unreadable HTML",
        )
        assert undecodable.consumed == 1

    asyncio.run(scenario())


def test_provider_stream_closes_on_cancellation() -> None:
    stream = _BlockingAsyncStream()
    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                stream=stream,
            )
        ),
    )

    async def scenario() -> None:
        task = asyncio.create_task(provider.search_jobs(_query()))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        await provider.aclose()

    asyncio.run(scenario())


def test_provider_parser_and_url_failures_remove_exception_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PROVIDER-PARSER-SYNTHETIC-MARKER"
    provider = LinkedInPublicJobsProvider(
        user_agent="Synthetic-Test-Agent/1.0",
        timeout_seconds=3,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                text="<html></html>",
                headers={"content-type": "text/html"},
            )
        ),
    )

    def parser_failure(_: str) -> list[object]:
        raise ValueError(marker)

    monkeypatch.setattr(linkedin_public, "_parse_search_results", parser_failure)

    async def scenario() -> None:
        with pytest.raises(ProviderError, match="parsing failed") as parser_error:
            await provider.search_jobs(_query())
        _assert_sanitized_exception(parser_error.value)
        with pytest.raises(JobNotFoundError) as url_error:
            await provider.get_job_details("https://[")
        _assert_sanitized_exception(url_error.value)

        monkeypatch.undo()

        def model_failure(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise ValueError(marker)

        monkeypatch.setattr(linkedin_public, "JobRawPayload", model_failure)
        with pytest.raises(ProviderError, match="detail parsing failed") as model_error:
            await provider.get_job_raw_payload("910000010")
        _assert_sanitized_exception(model_error.value)
        await provider.aclose()

    caplog.set_level(logging.INFO)
    asyncio.run(scenario())
    streams = capsys.readouterr()
    assert marker not in f"{caplog.text}\n{streams.out}\n{streams.err}"


def test_provider_has_no_private_or_external_action_surface() -> None:
    assert providers_package.LinkedInPublicJobsProvider is LinkedInPublicJobsProvider
    for name in (
        "authenticate",
        "login",
        "profile",
        "connections",
        "message",
        "submit",
        "apply",
        "contact",
    ):
        assert not hasattr(LinkedInPublicJobsProvider, name)
