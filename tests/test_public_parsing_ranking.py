from __future__ import annotations

import json
import socket
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest
from pydantic import ValidationError

from career_agent_workbench import generic_job_scraper
from career_agent_workbench.errors import ProviderError
from career_agent_workbench.generic_job_scraper import (
    extract_generic_job_details_from_html,
    generic_job_id,
    normalize_job_url,
)
from career_agent_workbench.jod import (
    JOD_INPUT_MAX_CHARS,
    NO_PUBLIC_JOB_DESCRIPTION,
    clean_job_description_for_prompt,
    is_placeholder_job_description,
    job_description_context,
    limit_context,
    usable_job_description,
)
from career_agent_workbench.models import JobDetails, JobSearchQuery
from career_agent_workbench.query_optimizer import (
    ScoredQuery,
    StoredQueryOutcome,
    historical_query_candidates,
    query_profile_match,
    rank_search_queries,
)


def test_generic_url_normalization_id_and_rejection() -> None:
    source = (
        "https://jobs.example.invalid/synthetic-role/?utm_source=test"
        "&keep=yes#private-fragment"
    )
    normalized = normalize_job_url(source)
    assert normalized == "https://jobs.example.invalid/synthetic-role?keep=yes"
    assert generic_job_id(source) == generic_job_id(normalized)
    assert generic_job_id(source).startswith("url-")

    for value in (
        "ftp://jobs.example.invalid/posting",
        "https:///missing-host",
        "https://user:marker@jobs.example.invalid/posting",
        "http://127.0.0.1/posting",
        "http://10.0.0.1/posting",
        "http://169.254.1.1/posting",
        "http://224.0.0.1/posting",
        "http://0.0.0.0/posting",
        "http://127.1/synthetic-posting",
        "http://2130706433/synthetic-posting",
        "http://0x7f000001/synthetic-posting",
        "http://0177.0.0.1/synthetic-posting",
        "http://127.0.0.1./synthetic-posting",
        "http://127.1./synthetic-posting",
        "http://2130706433./synthetic-posting",
        "http://0x7f000001./synthetic-posting",
        "http://0177.0.0.1./synthetic-posting",
        "http://0x7f.0.0.1./synthetic-posting",
        "http://0x7f.0.0.1../synthetic-posting",
        "http://4294967296/synthetic-posting",
        "http://4294967296./synthetic-posting",
        "http://127.0.0.08/synthetic-posting",
        "http://127.0.0.08./synthetic-posting",
        "http://0xnotnumeric/synthetic-posting",
        "http://0xnotnumeric./synthetic-posting",
        "http://0129/synthetic-posting",
        "http://0129./synthetic-posting",
        "http://0x/synthetic-posting",
        "http://0x./synthetic-posting",
        "http://8.8.8.8../synthetic-posting",
        "http://0x08080808../synthetic-posting",
        "http://01002004010../synthetic-posting",
        "http://jobs.example.invalid../synthetic-posting",
        "http://1.2.3.4.5/synthetic-posting",
        "http://1.2.3.4.5./synthetic-posting",
        "http://0x1.0x2.0x3.0x4.0x5/synthetic-posting",
        "http://0x1.0x2.0x3.0x4.0x5./synthetic-posting",
        "http://1.0x2.03.4.5/synthetic-posting",
        "http://1.0x2.03.4.5./synthetic-posting",
        "http://.8.8.8.8/synthetic-posting",
        "http://.8.8.8.8./synthetic-posting",
        "http://8..8.8/synthetic-posting",
        "http://8..8.8./synthetic-posting",
        "http://.0x8.0x080808/synthetic-posting",
        "http://.0x8.0x080808./synthetic-posting",
        "http://0x8..0x080808/synthetic-posting",
        "http://0x8..0x080808./synthetic-posting",
        "http://.jobs.example.invalid/synthetic-posting",
        "http://.jobs.example.invalid./synthetic-posting",
        "http://jobs..example.invalid/synthetic-posting",
        "http://jobs..example.invalid./synthetic-posting",
    ):
        with pytest.raises(ProviderError) as captured:
            normalize_job_url(value)
        assert "marker" not in str(captured.value)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None

    assert (
        normalize_job_url("https://8.8.8.8/synthetic-posting/")
        == "https://8.8.8.8/synthetic-posting"
    )
    assert (
        normalize_job_url("https://[2606:4700:4700::1111]/synthetic-posting/")
        == "https://[2606:4700:4700::1111]/synthetic-posting"
    )
    assert (
        normalize_job_url("https://jobs.example.invalid/synthetic-posting/")
        == "https://jobs.example.invalid/synthetic-posting"
    )
    assert (
        normalize_job_url("https://jobs.example.invalid./synthetic-posting/")
        == "https://jobs.example.invalid./synthetic-posting"
    )
    assert (
        normalize_job_url("https://8.8.8.8./synthetic-posting/")
        == "https://8.8.8.8./synthetic-posting"
    )
    for value in (
        "https://134744072/synthetic-posting/",
        "https://134744072./synthetic-posting/",
        "https://0x08080808/synthetic-posting/",
        "https://0x08080808./synthetic-posting/",
        "https://01002004010/synthetic-posting/",
        "https://01002004010./synthetic-posting/",
        "https://8.8.2056/synthetic-posting/",
        "https://8.8.2056./synthetic-posting/",
        "https://0x8.0x080808/synthetic-posting/",
        "https://0x8.0x080808./synthetic-posting/",
        "https://127.example.invalid./synthetic-posting/",
        "https://1.2.example.invalid./synthetic-posting/",
        "https://xn--bcher-kva.example.invalid./synthetic-posting/",
        "https://bücher.example.invalid./synthetic-posting/",
        "https://0129.example.invalid/synthetic-posting/",
        "https://0129.example.invalid./synthetic-posting/",
        "https://0xjobs.example.invalid/synthetic-posting/",
        "https://0xjobs.example.invalid./synthetic-posting/",
        "https://0xnotnumeric.example.invalid/synthetic-posting/",
        "https://0xnotnumeric.example.invalid./synthetic-posting/",
        "https://١٢٣.example.invalid/synthetic-posting/",
        "https://١٢٣.example.invalid./synthetic-posting/",
    ):
        assert normalize_job_url(value) == value.rstrip("/")


def test_generic_url_idna_separator_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Generic URL normalization must remain pure.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)
    separators = ("\u3002", "\uff0e", "\uff61")
    for separator in separators:
        rejected_hosts = (
            separator.join(("127", "0", "0", "1")),
            separator.join(("127", "1")),
            separator.join(("0177", "0", "0", "1")),
            separator.join(("0x7f", "0", "0", "1")),
            f"2130706433{separator}",
            separator.join(("8", "8", "8", "8")) + separator * 2,
            f"{separator}8{separator}8{separator}8{separator}8",
            separator.join(("8", "", "8", "8")),
            separator.join(("1", "2", "3", "4", "5")),
            separator.join(("0x1", "0x2", "0x3", "0x4", "0x5")),
            separator.join(("１２７", "０", "０", "１")),
        )
        for host in rejected_hosts:
            with pytest.raises(ProviderError):
                normalize_job_url(f"https://{host}/synthetic-posting")

        assert (
            normalize_job_url(
                f"https://8{separator}8{separator}8{separator}8"
                f"{separator}/synthetic-posting/"
            )
            == "https://8.8.8.8./synthetic-posting"
        )
        assert (
            normalize_job_url(
                f"https://0x8{separator}0x080808{separator}/synthetic-posting/"
            )
            == "https://0x8.0x080808./synthetic-posting"
        )
        assert (
            normalize_job_url(
                f"https://bücher{separator}example{separator}invalid/synthetic-posting/"
            )
            == "https://bücher.example.invalid/synthetic-posting"
        )
        assert (
            normalize_job_url(
                f"https://jobs{separator}example{separator}invalid:8443"
                "/synthetic-posting/"
            )
            == "https://jobs.example.invalid:8443/synthetic-posting"
        )

    for host in (
        "127.0\u30020\uff0e1",
        "0177\u30020.0\uff0e1",
        "0x7f\uff610\u30020.1",
        "8\u30028\uff0e8\uff618\u3002\uff0e",
        "1\u30022.3\uff0e4\uff615",
        "jobs\u3002\uff0eexample.invalid",
    ):
        with pytest.raises(ProviderError):
            normalize_job_url(f"https://{host}/synthetic-posting")
    assert (
        normalize_job_url("https://jobs\u2024example.invalid/synthetic-posting/")
        == "https://jobs\u2024example.invalid/synthetic-posting"
    )
    assert network_calls == 0


def test_generic_idna_separator_rejection_precedes_model_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Generic parsing must not resolve or contact a host.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)
    description = (
        "Build synthetic public systems with bounded parsing and reliable "
        "automation for a fictional employer using only in-memory evidence."
    )
    html = (
        "<html><main><h1>Synthetic Platform Role</h1>"
        f"<p>{description}</p></main></html>"
    )
    with pytest.raises(ProviderError):
        extract_generic_job_details_from_html(
            html=html,
            url="https://127\u30020\u30020\u30021/synthetic-posting",
        )
    assert network_calls == 0


def test_generic_percent_authority_and_ports_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    network_calls = 0
    model_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Generic URL checks must not use DNS or networking.")

    def forbidden_model(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("Rejected URLs must not reach model construction.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)
    monkeypatch.setattr(generic_job_scraper, "JobDetails", forbidden_model)

    percent_hosts = [
        "127%2e0%2e0%2e1",
        "%31%32%37%2e%30%2e%30%2e%31",
        "%32%31%33%30%37%30%36%34%33%33",
        "%30%31%37%37%2e%30%2e%30%2e%31",
        "%30%78%37%66%30%30%30%30%30%31",
        "0x7f%2e0.0%2e1",
        quote("０ｘ７ｆ０００００１", safe=""),
        f"０ｘ{quote('７ｆ０００００１', safe='')}",
        "127.0%2e0.1",
        "jobs%2eexample.invalid",
        "jobs%example.invalid",
    ]
    for separator in ("\u3002", "\uff0e", "\uff61"):
        encoded = quote(separator, safe="")
        percent_hosts.extend(
            (
                f"127{encoded}0{encoded}0{encoded}1",
                f"127{encoded.lower()}0{encoded.lower()}0{encoded.lower()}1",
            )
        )

    malformed_authorities = [
        *(f"https://{host}/synthetic-posting" for host in percent_hosts),
        "https://jobs.example.invalid:65536/synthetic-posting",
        "https://jobs.example.invalid:alphabetic/synthetic-posting",
        "https://jobs.example.invalid:-1/synthetic-posting",
        "https://jobs.example.invalid:+443/synthetic-posting",
        "https://jobs.example.invalid:443:444/synthetic-posting",
        "https://jobs.example.invalid: 443/synthetic-posting",
        "https://jobs.example.invalid:443 /synthetic-posting",
        "https://jobs.example.invalid:\t443/synthetic-posting",
        "https://jobs.example.invalid:44\n3/synthetic-posting",
        "https://jobs.example.invalid:443 ",
        "https:\n//jobs.example.invalid:44\n3/synthetic-posting",
        "https://jobs.example.invalid:0/synthetic-posting",
        "https://[2606:4700:4700::1111]:0/synthetic-posting",
        "https://jobs.example.invalid:/synthetic-posting",
        "https://[2606:4700:4700::1111]:/synthetic-posting",
    ]
    description = (
        "Build synthetic public systems with bounded parsing and reliable "
        "automation for a fictional employer using only in-memory evidence."
    )
    html = (
        "<html><main><h1>Synthetic Platform Role</h1>"
        f"<p>{description}</p></main></html>"
    )

    for value in malformed_authorities:
        with pytest.raises(ProviderError) as normalized_error:
            normalize_job_url(value)
        with pytest.raises(ProviderError) as identifier_error:
            generic_job_id(value)
        with pytest.raises(ProviderError) as parser_error:
            extract_generic_job_details_from_html(html=html, url=value)
        for error in (normalized_error, identifier_error, parser_error):
            assert str(error.value) == "Public job URL is invalid."
            assert error.value.__cause__ is None
            assert error.value.__context__ is None

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert model_calls == 0
    assert network_calls == 0


def test_generic_raw_authority_controls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    network_calls = 0
    http_url_calls = 0
    model_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Generic URL checks must not use DNS or networking.")

    def forbidden_http_url(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal http_url_calls
        http_url_calls += 1
        raise AssertionError("Authority controls must reject before canonicalization.")

    def forbidden_model(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal model_calls
        model_calls += 1
        raise AssertionError(
            "Authority controls must reject before model construction."
        )

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)
    monkeypatch.setattr(generic_job_scraper, "HttpUrl", forbidden_http_url)
    monkeypatch.setattr(generic_job_scraper, "JobDetails", forbidden_model)
    caplog.clear()

    controls = [
        *(chr(code_point) for code_point in range(0x20)),
        chr(0x7F),
        *(chr(code_point) for code_point in range(0x80, 0xA0)),
    ]
    assert len(controls) == 65
    rejected_urls = [
        *(
            f"https://jobs{control}.example.invalid/synthetic-posting"
            for control in controls
        ),
    ]
    for control in (chr(0x01), chr(0x7F), chr(0x80)):
        rejected_urls.extend(
            (
                f"https://jobs.example.invalid:{control}443/synthetic-posting",
                f"https://jobs.example.invalid:4{control}43/synthetic-posting",
                f"https://jobs.example.invalid:443{control}/synthetic-posting",
                f"https://[2606:4700:4700::1111]{control}:8443/synthetic-posting",
                f"https://[2606:4700:4700::1111]:84{control}43/synthetic-posting",
                f"https://[2606:4700:4700::1111]:8443{control}/synthetic-posting",
                f"https://jobs.example.invalid{control}/synthetic-posting",
                f"https://jobs.example.invalid{control}?keep=yes",
                f"https://jobs.example.invalid{control}#fragment",
                f"https://user{control}@jobs.example.invalid/synthetic-posting",
                f"https://user@{control}jobs.example.invalid/synthetic-posting",
            )
        )

    description = (
        "Build synthetic public systems with bounded parsing and reliable "
        "automation for a fictional employer using only in-memory evidence."
    )
    html = (
        "<html><main><h1>Synthetic Platform Role</h1>"
        f"<p>{description}</p></main></html>"
    )
    operations = (
        normalize_job_url,
        generic_job_id,
        lambda value: extract_generic_job_details_from_html(html=html, url=value),
    )

    for value in rejected_urls:
        for operation in operations:
            with pytest.raises(ProviderError) as captured:
                operation(value)
            assert str(captured.value) == "Public job URL is invalid."
            assert captured.value.__cause__ is None
            assert captured.value.__context__ is None

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err == ""
    assert caplog.records == []
    assert http_url_calls == 0
    assert model_calls == 0
    assert network_calls == 0


def test_generic_raw_controls_outside_authority_preserve_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Generic URL normalization must remain pure.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)

    base_url = "https://jobs.example.invalid/synthetic-posting"
    for control in (chr(0x01), chr(0x7F), chr(0x80)):
        path_url = f"{base_url}/{control}path"
        assert normalize_job_url(path_url) == path_url
        assert generic_job_id(path_url) == generic_job_id(normalize_job_url(path_url))

        query_value = f"synthetic{control}value"
        query_url = f"{base_url}?keep={query_value}"
        normalized_query = f"{base_url}?{urlencode({'keep': query_value})}"
        assert normalize_job_url(query_url) == normalized_query
        assert generic_job_id(query_url) == generic_job_id(normalized_query)

        fragment_url = f"{base_url}#synthetic{control}fragment"
        assert normalize_job_url(fragment_url) == base_url
        assert generic_job_id(fragment_url) == generic_job_id(base_url)

    downstream_declined_url = "https://jobs\u2024example.invalid/synthetic-posting"
    assert normalize_job_url(downstream_declined_url) == downstream_declined_url
    assert generic_job_id(downstream_declined_url).startswith("url-")
    with pytest.raises(ValidationError):
        JobDetails(
            job_id="synthetic-downstream-declined",
            title="Synthetic Downstream Declined",
            job_url=downstream_declined_url,
        )
    assert network_calls == 0


def test_generic_downstream_canonical_host_rejects_compatibility_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0
    model_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Canonical host checks must remain pure.")

    def forbidden_model(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("Unsafe canonical hosts must reject before models.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)
    monkeypatch.setattr(generic_job_scraper, "JobDetails", forbidden_model)

    compatibility_hosts = [
        "０ｘ７ｆ０００００１",
        "⓪ⓧ⑦ⓕ⓪⓪⓪⓪⓪①",
        "⁰x7f000001",
        "0ˣ7f000001",
        "𝟎𝐱𝟕𝐟𝟎𝟎𝟎𝟎𝟎𝟏",
        "０ｘ0a000001",
        "０ｘa9fe0001",
        "０ｘe0000001",
        "０ｘf0000001",
        "０ｘ00000000",
        "０ｘnotnumeric",
        "０ｘ",
        *(
            separator.join(("０ｘ７ｆ", "０", "０", "１"))
            for separator in ("\u3002", "\uff0e", "\uff61")
        ),
    ]
    description = (
        "Build synthetic public systems with bounded parsing and reliable "
        "automation for a fictional employer using only in-memory evidence."
    )
    html = (
        "<html><main><h1>Synthetic Platform Role</h1>"
        f"<p>{description}</p></main></html>"
    )

    for host in compatibility_hosts:
        value = f"https://{host}/synthetic-posting"
        with pytest.raises(ProviderError):
            normalize_job_url(value)
        with pytest.raises(ProviderError):
            generic_job_id(value)
        with pytest.raises(ProviderError):
            extract_generic_job_details_from_html(html=html, url=value)

    assert model_calls == 0
    assert network_calls == 0


def test_generic_canonical_safety_preserves_safe_spelling_and_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("Safe URL controls must remain pure.")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_network)
    monkeypatch.setattr(socket, "socket", forbidden_network)

    path_query_url = (
        "https://jobs.example.invalid/synthetic%20role?keep=synthetic%2Fvalue"
    )
    assert normalize_job_url(path_query_url) == path_query_url
    assert generic_job_id(path_query_url).startswith("url-")

    for port in (1, 443, 8443, 65_535):
        value = f"https://jobs.example.invalid:{port}/synthetic-posting/"
        assert normalize_job_url(value) == value.rstrip("/")
    for value in (
        "https://[2606:4700:4700::1111]/synthetic-posting/",
        "https://[2606:4700:4700::1111]:8443/synthetic-posting/",
    ):
        assert normalize_job_url(value) == value.rstrip("/")

    public_compatibility_url = "https://０ｘ０８０８０８０８/synthetic-posting"
    normalized_public = normalize_job_url(public_compatibility_url)
    assert normalized_public == public_compatibility_url
    public_details = JobDetails(
        job_id="synthetic-public-control",
        title="Synthetic Public Control",
        job_url=normalized_public,
    )
    assert public_details.job_url is not None
    assert public_details.job_url.host == "8.8.8.8"

    for label in ("０ｘｊｏｂｓ", "0ｘnotnumeric"):
        value = f"https://{label}.example.invalid/synthetic-posting"
        assert normalize_job_url(value) == value
        details = JobDetails(
            job_id="synthetic-dns-control",
            title="Synthetic DNS Control",
            job_url=value,
        )
        assert details.job_url is not None
        assert details.job_url.host is not None
        assert details.job_url.host.endswith(".example.invalid")

    description = (
        "Build synthetic public systems with bounded parsing and reliable "
        "automation for a fictional employer using only in-memory evidence."
    )
    html = (
        "<html><main><h1>Synthetic Platform Role</h1>"
        f"<p>{description}</p></main></html>"
    )
    parsed_public = extract_generic_job_details_from_html(
        html=html,
        url=public_compatibility_url,
    )
    parsed_dns = extract_generic_job_details_from_html(
        html=html,
        url="https://０ｘｊｏｂｓ.example.invalid/synthetic-posting",
    )
    assert parsed_public.job_url is not None
    assert parsed_public.job_url.host == "8.8.8.8"
    assert parsed_dns.job_url is not None
    assert parsed_dns.job_url.host is not None
    assert parsed_dns.job_url.host.endswith(".example.invalid")
    assert network_calls == 0


def test_generic_jsonld_embedded_and_visible_fallback_parsing() -> None:
    description = (
        "Design fictional platform services, build reliable APIs, create public "
        "test automation, and improve observability for Example Harbor Systems."
    )
    jsonld = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "Synthetic Platform Engineer",
            "description": description,
            "hiringOrganization": {
                "@type": "Organization",
                "name": "Example Harbor Systems",
            },
            "jobLocation": {
                "@type": "Place",
                "address": {
                    "@type": "PostalAddress",
                    "addressLocality": "Example City",
                },
            },
        }
    )
    details = extract_generic_job_details_from_html(
        html=f"<html><script type='application/ld+json'>{jsonld}</script></html>",
        url="https://jobs.example.invalid/platform?utm_source=synthetic",
    )
    assert details.title == "Synthetic Platform Engineer"
    assert details.company == "Example Harbor Systems"
    assert details.description == description
    assert "utm_" not in str(details.job_url)

    embedded = json.dumps(
        {
            "props": {
                "job": {
                    "title": "Synthetic Automation Engineer",
                    "description": description,
                    "location": "Example City",
                },
                "company": {"name": "Example Beacon Works"},
            }
        }
    )
    embedded_details = extract_generic_job_details_from_html(
        html=f"<main data-page='{embedded}'></main>",
        url="https://jobs.example.invalid/automation",
    )
    assert embedded_details.company == "Example Beacon Works"

    visible = f"""
    <html><head><title>Synthetic Systems Role at Example Cedar Labs</title></head>
    <body><main><h1>Synthetic Systems Role</h1><p>{description}</p></main></body>
    </html>
    """
    visible_details = extract_generic_job_details_from_html(
        html=visible,
        url="https://jobs.example.invalid/systems",
    )
    assert visible_details.title == "Synthetic Systems Role"
    assert "fictional platform services" in (visible_details.description or "")


def test_generic_next_data_public_job_shape() -> None:
    description = (
        "Build reliable fictional Python services, maintain SQLite workflow "
        "state, add focused tests, and create useful observability for an "
        "offline demonstration platform with documented human review."
    )
    next_data = json.dumps(
        {
            "props": {
                "pageProps": {
                    "company": {
                        "candidateCorrespondenceClientName": (
                            "Nimbus Quay Example Labs"
                        )
                    },
                    "posting": {
                        "jobTitle": "Demo Platform Engineer",
                        "postingLocations": [{"formattedAddress": "Example City, ZZ"}],
                        "jobPostingContent": {"jobDescription": description},
                    },
                }
            }
        }
    )
    details = extract_generic_job_details_from_html(
        html=(
            "<html><script id='__NEXT_DATA__' type='application/json'>"
            f"{next_data}</script></html>"
        ),
        url="https://jobs.example.test/demo-next-platform",
    )
    assert details.title == "Demo Platform Engineer"
    assert details.company == "Nimbus Quay Example Labs"
    assert details.location == "Example City, ZZ"
    assert details.description == description


def test_generic_bounds_errors_and_parser_only_boundary() -> None:
    with pytest.raises(ProviderError, match="No usable"):
        extract_generic_job_details_from_html(
            html="<html><main>Short.</main></html>",
            url="https://jobs.example.invalid/short",
        )
    with pytest.raises(ProviderError, match="HTML exceeds"):
        extract_generic_job_details_from_html(
            html="x" * 2_000_001,
            url="https://jobs.example.invalid/large",
        )
    oversized = json.dumps(
        {
            "@type": "JobPosting",
            "title": "Synthetic Oversized Role",
            "description": "x" * 500_001,
        }
    )
    with pytest.raises(
        ProviderError,
        match=r"^Parsed job description exceeds the public size limit\.$",
    ) as captured:
        extract_generic_job_details_from_html(
            html=f"<script type='application/ld+json'>{oversized}</script>",
            url="https://jobs.example.invalid/oversized-description",
        )
    assert captured.value.__cause__ is None
    assert not hasattr(generic_job_scraper, "fetch_generic_job_details")


def test_generic_url_and_parser_failures_remove_exception_context(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "GENERIC-PARSER-SYNTHETIC-MARKER"
    with pytest.raises(ProviderError) as url_error:
        normalize_job_url("https://[")
    assert url_error.value.__cause__ is None
    assert url_error.value.__context__ is None

    def parser_failure(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise ValueError(marker)

    monkeypatch.setattr(generic_job_scraper, "get_base_url", parser_failure)
    with pytest.raises(ProviderError, match="HTML parsing failed") as parser_error:
        extract_generic_job_details_from_html(
            html="<html><main>Synthetic content</main></html>",
            url="https://jobs.example.invalid/synthetic-posting",
        )
    assert parser_error.value.__cause__ is None
    assert parser_error.value.__context__ is None

    monkeypatch.undo()
    monkeypatch.setattr(generic_job_scraper, "JobDetails", parser_failure)
    description = (
        "Design synthetic public systems with bounded parsing and reliable "
        "automation for an unmistakably fictional employer and test workflow."
    )
    with pytest.raises(ProviderError, match="HTML parsing failed") as model_error:
        extract_generic_job_details_from_html(
            html=f"<html><main><h1>Synthetic Role</h1><p>{description}</p></main></html>",
            url="https://jobs.example.invalid/synthetic-posting",
        )
    assert model_error.value.__cause__ is None
    assert model_error.value.__context__ is None
    streams = capsys.readouterr()
    assert marker not in f"{streams.out}\n{streams.err}"
    assert not hasattr(generic_job_scraper, "fetch_generic_job_details")


def test_jod_placeholder_trimming_limit_and_oversized_preprocessing() -> None:
    assert usable_job_description(None) is None
    assert usable_job_description(NO_PUBLIC_JOB_DESCRIPTION) is None
    assert is_placeholder_job_description(" No public job description was available. ")

    role = (
        "Our Mission\nA long fictional mission statement for Example Harbor Systems.\n"
        "Responsibilities\nDesign reliable APIs and automate public test systems. "
        "Build observability and secure platform infrastructure for engineering teams.\n"
        "Benefits\nThe base salary range and medical benefits are described here."
    )
    cleaned = clean_job_description_for_prompt(role)
    assert cleaned.startswith("Responsibilities")
    assert "Design reliable APIs" in cleaned
    assert "base salary" not in cleaned
    assert "automated tools" in clean_job_description_for_prompt(
        "Responsibilities\nBuild automated tools for public workflows."
    )
    assert limit_context("alpha\nbeta\ngamma", max_chars=10) == "alpha"

    details = JobDetails(
        job_id="920000001",
        title="Synthetic Role",
        description=role,
    )
    assert job_description_context(details) == cleaned

    oversized = "x" * (JOD_INPUT_MAX_CHARS + 1)
    direct_helpers = (
        lambda: usable_job_description(oversized),
        lambda: is_placeholder_job_description(oversized),
        lambda: clean_job_description_for_prompt(oversized),
        lambda: limit_context(oversized, max_chars=10),
    )
    for helper in direct_helpers:
        with pytest.raises(
            ValueError,
            match=r"^Job description exceeds the 500000-character input limit\.$",
        ) as captured:
            helper()
        assert captured.value.__cause__ is None
    assert clean_job_description_for_prompt(role) == cleaned


def _outcome(
    keywords: str,
    *,
    accepted: int,
    experience: str | None = "associate",
    history_score: float = 0.8,
) -> StoredQueryOutcome:
    return StoredQueryOutcome(
        keywords=keywords,
        location="Example City",
        date_posted="past_week",
        workplace_type="remote",
        experience_level=experience,
        job_type="full_time",
        sort_by="recent",
        limit=10,
        profile_match=0.8,
        query_score=history_score,
        results_returned=10,
        fresh_jobs_accepted=accepted,
        resumes_generated=1 if accepted else 0,
        average_ats_score=80,
    )


def test_query_ranking_dedup_history_exploration_and_no_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    duplicate = JobSearchQuery(
        keywords=" Platform Engineer ",
        location="Example City",
        workplace_type="remote",
    )
    queries = [
        duplicate,
        JobSearchQuery(
            keywords="platform engineer",
            location="Example City",
            workplace_type="remote",
        ),
        JobSearchQuery(
            keywords="automation engineer",
            location="Example City",
            workplace_type="remote",
        ),
        JobSearchQuery(
            keywords="observability engineer",
            location="Example City",
            workplace_type="remote",
        ),
    ]
    history = [_outcome("platform engineer", accepted=8)]
    ranked = rank_search_queries(
        queries,
        profile_context="Python platform automation observability",
        history=history,
        max_queries=2,
        exploration_rate=0.5,
    )
    assert len(ranked) == 2
    assert all(isinstance(item, ScoredQuery) for item in ranked)
    assert [item.selection_reason for item in ranked] == ["exploit", "explore"]
    assert len({item.query.keywords.casefold() for item in ranked}) == 2
    assert query_profile_match(duplicate, "Python platform engineering") > 0
    assert set(tmp_path.iterdir()) == before


def test_historical_candidates_suppress_invalid_levels_and_unproductive_rows() -> None:
    history = [
        _outcome("platform engineer", accepted=4, experience="entry_level"),
        _outcome("platform engineer", accepted=2, experience="mid_senior"),
        _outcome("unused role", accepted=0, experience="internship"),
    ]
    candidates = historical_query_candidates(
        history,
        location="Example City",
        date_posted="past_week",
        limit_per_query=200,
    )
    assert len(candidates) == 1
    assert candidates[0].keywords == "platform engineer"
    assert candidates[0].experience_level == "mid_senior"
    assert candidates[0].limit == 100
