"""Pure, bounded HTML parsing for public job-posting pages."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from html import unescape
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import extruct
import trafilatura
from bs4 import BeautifulSoup
from pydantic import HttpUrl
from w3lib.html import get_base_url

from career_agent_workbench.errors import ProviderError
from career_agent_workbench.models import JobDetails

MAX_HTML_CHARS = 2_000_000
MAX_DESCRIPTION_CHARS = 500_000
MIN_DESCRIPTION_CHARS = 120
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "source",
    "trk",
}
_HOSTNAME_SEPARATOR_TRANSLATION = str.maketrans(
    {
        "\u3002": ".",
        "\uff0e": ".",
        "\uff61": ".",
    }
)


def normalize_job_url(url: str) -> str:
    """Normalize a public HTTP(S) URL without resolving or fetching it."""
    parsed = None
    try:
        raw_value = str(url or "")
        if any(character in "\t\r\n" for character in raw_value):
            raise ValueError
        if _raw_authority_has_forbidden_syntax(raw_value):
            raise ValueError
        value = raw_value.strip()
        candidate_parts = urlsplit(value)
        if "%" in candidate_parts.netloc:
            raise ValueError
        candidate_parts = candidate_parts._replace(
            netloc=candidate_parts.netloc.translate(_HOSTNAME_SEPARATOR_TRANSLATION)
        )
        if _has_invalid_port_syntax(candidate_parts.netloc):
            raise ValueError
        candidate_hostname = candidate_parts.hostname
        candidate_port = candidate_parts.port
        parsed = (
            candidate_parts,
            candidate_hostname,
            candidate_port,
            candidate_parts.username,
            candidate_parts.password,
        )
    except (TypeError, ValueError):
        pass
    if parsed is None:
        raise ProviderError("Public job URL is invalid.")
    parts, hostname, port, username, password = parsed
    if (
        parts.scheme.casefold() not in {"http", "https"}
        or not hostname
        or username is not None
        or password is not None
        or port is not None
        and not 0 < port <= 65_535
    ):
        raise ProviderError("Public job URL is invalid.")
    _validate_public_hostname(hostname)

    query_items = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking_query_key(key)
    ]
    path = parts.path.rstrip("/") or "/"
    normalized_url = urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            path,
            urlencode(query_items, doseq=True),
            "",
        )
    )
    canonical_hostname = _downstream_canonical_hostname(normalized_url)
    if canonical_hostname is not None:
        _validate_public_hostname(canonical_hostname)
    return normalized_url


def _raw_authority_has_forbidden_syntax(url: str) -> bool:
    """Reject percent, whitespace, or controls before URL parsing."""
    scheme_end = url.find("://")
    if scheme_end < 0:
        return False
    authority_start = scheme_end + 3
    authority_end = len(url)
    for delimiter in "/?#":
        position = url.find(delimiter, authority_start)
        if position >= 0:
            authority_end = min(authority_end, position)
    authority = url[authority_start:authority_end]
    return "%" in authority or any(
        character.isspace() or _is_url_control(character) for character in authority
    )


def _is_url_control(character: str) -> bool:
    code_point = ord(character)
    return code_point <= 0x1F or 0x7F <= code_point <= 0x9F


def _has_invalid_port_syntax(netloc: str) -> bool:
    """Return whether an authority contains a malformed explicit port."""
    authority = netloc.rsplit("@", 1)[-1]
    if authority.startswith("["):
        bracket_end = authority.find("]")
        if bracket_end < 0:
            return True
        suffix = authority[bracket_end + 1 :]
        if not suffix:
            return False
        if not suffix.startswith(":") or ":" in suffix[1:]:
            return True
        port_text = suffix[1:]
    else:
        separator_count = authority.count(":")
        if separator_count == 0:
            return False
        if separator_count != 1:
            return True
        port_text = authority.rsplit(":", 1)[1]
    return not port_text or not port_text.isascii() or not port_text.isdigit()


def _validate_public_hostname(hostname: str) -> None:
    """Reject malformed or unsafe literal destinations without resolution."""
    if hostname.endswith(".."):
        raise ProviderError("Public job URL is invalid.")
    numeric_candidate = hostname.removesuffix(".")
    if (
        not numeric_candidate
        or ":" not in numeric_candidate
        and any(not component for component in numeric_candidate.split("."))
    ):
        raise ProviderError("Public job URL is invalid.")
    numeric_host = False
    try:
        address = ipaddress.ip_address(numeric_candidate)
    except ValueError:
        numeric_host, address = _legacy_numeric_ipv4(numeric_candidate)
    if numeric_host and address is None:
        raise ProviderError("Public job URL destination is not allowed.")
    if address is not None and (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise ProviderError("Public job URL destination is not allowed.")


def _downstream_canonical_hostname(url: str) -> str | None:
    """Return Pydantic's accepted canonical host without changing output."""
    canonical_hostname = None
    try:
        canonical_hostname = HttpUrl(url).host
    except (TypeError, ValueError):
        pass
    return canonical_hostname


def _legacy_numeric_ipv4(
    hostname: str,
) -> tuple[bool, ipaddress.IPv4Address | None]:
    """Parse legacy inet-style IPv4 spellings without DNS or socket access."""
    components = hostname.casefold().split(".")
    numeric_like = all(
        component.isdigit() or component.startswith("0x") for component in components
    )
    if not numeric_like:
        return False, None
    if not 1 <= len(components) <= 4:
        return True, None
    values: list[int] = []
    for component in components:
        base = 10
        digits = component
        if component.startswith("0x"):
            base = 16
            digits = component[2:]
        elif len(component) > 1 and component.startswith("0"):
            base = 8
            digits = component[1:]
        allowed = {
            8: "01234567",
            10: "0123456789",
            16: "0123456789abcdef",
        }[base]
        if not digits or any(character not in allowed for character in digits):
            return True, None
        try:
            values.append(int(digits, base))
        except (ValueError, OverflowError):
            return True, None

    limits = {
        1: (0xFFFFFFFF,),
        2: (0xFF, 0xFFFFFF),
        3: (0xFF, 0xFF, 0xFFFF),
        4: (0xFF, 0xFF, 0xFF, 0xFF),
    }[len(values)]
    if any(value > limit for value, limit in zip(values, limits, strict=True)):
        return True, None
    if len(values) == 1:
        integer = values[0]
    elif len(values) == 2:
        integer = values[0] << 24 | values[1]
    elif len(values) == 3:
        integer = values[0] << 24 | values[1] << 16 | values[2]
    else:
        integer = values[0] << 24 | values[1] << 16 | values[2] << 8 | values[3]
    return True, ipaddress.IPv4Address(integer)


def generic_job_id(url: str) -> str:
    """Return a deterministic synthetic identifier for a normalized URL."""
    digest = hashlib.sha256(normalize_job_url(url).encode()).hexdigest()
    return f"url-{digest[:16]}"


def extract_generic_job_details_from_html(*, html: str, url: str) -> JobDetails:
    """Extract one normalized job without performing network activity."""
    if not isinstance(html, str) or len(html) > MAX_HTML_CHARS:
        raise ProviderError("Public job HTML exceeds the public size limit.")
    normalized_url = normalize_job_url(url)
    parser_failed = False
    try:
        result = _extract_generic_job_details_unchecked(
            html=html,
            normalized_url=normalized_url,
        )
    except ProviderError:
        raise
    except Exception:  # noqa: BLE001 - sanitize parser/model failures
        parser_failed = True
    if parser_failed:
        raise ProviderError("Public job HTML parsing failed.")
    return result


def _extract_generic_job_details_unchecked(
    *,
    html: str,
    normalized_url: str,
) -> JobDetails:
    parser_failed = False
    try:
        base_url = get_base_url(html, normalized_url)
        structured = _find_structured_job_posting(html=html, base_url=base_url)
        embedded = _embedded_job_fields(html)
    except Exception:  # noqa: BLE001 - sanitize third-party parser failures
        parser_failed = True
    if parser_failed:
        raise ProviderError("Public job HTML parsing failed.")

    fallback: dict[str, str] | None = None

    def fallback_value(key: str) -> str:
        nonlocal fallback
        if fallback is None:
            fallback = _fallback_job_fields(html=html, url=normalized_url)
        return fallback.get(key, "")

    title = _clean_text(
        _first_value(structured, "title", "name")
        or embedded.get("title")
        or fallback_value("title")
        or "Job Posting"
    )
    description = _clean_description(
        _first_value(
            structured,
            "description",
            "responsibilities",
            "qualifications",
        )
        or embedded.get("description")
        or fallback_value("description")
    )
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ProviderError("Parsed job description exceeds the public size limit.")
    if len(description) < MIN_DESCRIPTION_CHARS:
        raise ProviderError("No usable public job description was found.")

    organization = _first_mapping(structured.get("hiringOrganization"))
    company = _clean_text(
        _first_value(organization, "name", "legalName")
        or embedded.get("company")
        or fallback_value("company")
    )
    return JobDetails(
        job_id=generic_job_id(normalized_url),
        title=title or "Job Posting",
        company=company or None,
        location=_job_location(structured)
        or _clean_text(embedded.get("location"))
        or None,
        listed_at=_clean_text(
            _first_value(structured, "datePosted") or embedded.get("listed_at")
        )
        or None,
        job_url=normalized_url,
        company_url=_clean_text(
            _first_value(organization, "sameAs", "url") or embedded.get("company_url")
        )
        or None,
        workplace_type=_clean_text(_first_value(structured, "jobLocationType")) or None,
        source="generic_url",
        description=description,
        seniority_level=_clean_text(
            _first_value(structured, "experienceRequirements")
            or embedded.get("seniority_level")
        )
        or None,
        employment_type=_join_values(structured.get("employmentType"))
        or _join_values(embedded.get("employment_type")),
        job_function=_clean_text(
            _first_value(structured, "occupationalCategory")
            or embedded.get("job_function")
        )
        or None,
        industries=_clean_text(
            _first_value(structured, "industry") or embedded.get("industries")
        )
        or None,
    )


def _is_tracking_query_key(key: str) -> bool:
    lowered = key.casefold()
    return lowered in TRACKING_QUERY_KEYS or lowered.startswith(TRACKING_QUERY_PREFIXES)


def _find_structured_job_posting(
    *,
    html: str,
    base_url: str,
) -> dict[str, object]:
    data = extruct.extract(
        html,
        base_url=base_url,
        syntaxes=["json-ld", "microdata", "rdfa"],
        uniform=True,
    )
    for item in _iter_mappings(data):
        raw_types = item.get("@type") or item.get("type")
        types = raw_types if isinstance(raw_types, list) else [raw_types]
        if any(str(value or "").casefold().endswith("jobposting") for value in types):
            return item
    return {}


def _embedded_job_fields(html: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("[data-page]")
    if root is not None:
        data = _loads_mapping(str(root.get("data-page") or ""))
        props = _first_mapping(data.get("props"))
        job = _first_mapping(props.get("job"))
        company = _first_mapping(props.get("company"))
        if _first_value(job, "descriptionHtml", "description"):
            return {
                "title": _first_value(job, "title"),
                "company": _first_value(company, "name"),
                "company_url": _first_value(company, "url"),
                "location": _first_value(job, "location"),
                "description": _first_value(
                    job,
                    "descriptionHtml",
                    "description",
                ),
                "seniority_level": _first_value(job, "minExperience"),
                "employment_type": _first_value(job, "jobType"),
                "industries": _first_value(company, "industry"),
            }

    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        return {}
    data = _loads_mapping(str(script.string or script.get_text() or ""))
    for item in _iter_mappings(data):
        content = _first_mapping(item.get("jobPostingContent"))
        description = _first_value(content, "jobDescription", "description")
        if description:
            return {
                "title": _first_value(item, "jobTitle", "title"),
                "company": _company_name(data),
                "location": _posting_locations_text(item.get("postingLocations")),
                "description": description,
                "listed_at": _first_value(
                    item,
                    "datePosted",
                    "postingStartTimestampUTC",
                ).split("T", 1)[0],
            }
    return {}


def _loads_mapping(value: str) -> dict[str, object]:
    for candidate in (value.strip(), unescape(value.strip())):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _iter_mappings(value: object) -> list[dict[str, object]]:
    mappings: list[dict[str, object]] = []
    stack = [value]
    while stack and len(mappings) < 20_000:
        node = stack.pop()
        if isinstance(node, dict):
            mappings.append(node)
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))
    return mappings


def _company_name(data: object) -> str:
    for item in _iter_mappings(data):
        name = _first_value(
            item,
            "candidateCorrespondenceClientName",
            "clientName",
            "companyName",
            "legalEntityName",
        )
        if name:
            return name
    return ""


def _posting_locations_text(value: object) -> str:
    values = value if isinstance(value, list) else [value]
    locations: list[str] = []
    for item in values:
        mapping = _first_mapping(item)
        if not mapping:
            continue
        location = _first_value(mapping, "formattedAddress", "name") or (
            _join_values(
                [
                    _first_value(mapping, "cityName"),
                    _first_value(mapping, "stateCode"),
                    _first_value(mapping, "isoCountryCode"),
                ]
            )
            or ""
        )
        if location:
            locations.append(location)
    return ", ".join(dict.fromkeys(locations))


def _fallback_job_fields(*, html: str, url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    page_title = _text(soup.select_one("title"))
    title = _clean_text(
        _meta_content(soup, "og:title")
        or _meta_content(soup, "twitter:title")
        or _text(soup.select_one("h1"))
        or page_title
    )
    company = _clean_text(
        _meta_content(soup, "og:site_name")
        or _meta_content(soup, "application-name")
        or _company_from_page_title(page_title)
        or (urlsplit(url).hostname or "")
    )
    description = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        favor_recall=True,
    )
    if not description:
        description = (
            _meta_content(soup, "description")
            or _meta_content(soup, "og:description")
            or _main_text_fallback(soup)
        )
    return {
        "title": title,
        "company": company,
        "description": _clean_description(description),
    }


def _company_from_page_title(title: str) -> str:
    match = re.search(r"\bat\s+(.+?)\s*$", title.strip(), flags=re.IGNORECASE)
    return _clean_text(match.group(1).strip(" -|")) if match else ""


def _meta_content(soup: BeautifulSoup, name: str) -> str:
    node = soup.find("meta", attrs={"property": name}) or soup.find(
        "meta",
        attrs={"name": name},
    )
    return str(node.get("content") or "") if node else ""


def _main_text_fallback(soup: BeautifulSoup) -> str:
    node = soup.select_one("main, article, [role='main'], body")
    return node.get_text("\n", strip=True) if node else ""


def _job_location(job: dict[str, object]) -> str | None:
    raw = job.get("jobLocation")
    values = raw if isinstance(raw, list) else [raw]
    locations: list[str] = []
    for item in values:
        mapping = _first_mapping(item)
        address = _first_mapping(mapping.get("address"))
        text = _join_values(
            [
                _first_value(address, "streetAddress"),
                _first_value(address, "addressLocality"),
                _first_value(address, "addressRegion"),
                _first_value(address, "postalCode"),
                _first_value(address, "addressCountry"),
            ]
        ) or _first_value(mapping, "name")
        if text:
            locations.append(text)
    return ", ".join(dict.fromkeys(locations)) or None


def _first_value(mapping: object, *keys: str) -> str:
    if not isinstance(mapping, dict):
        return ""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            selected = _first_value(value, "@value", "name", "text")
        elif isinstance(value, list):
            selected = _join_values(value) or ""
        else:
            selected = _clean_text(value)
        if selected:
            return selected
    return ""


def _first_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict)), {})
    return {}


def _join_values(value: object) -> str | None:
    values = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for item in values:
        part = (
            _first_value(item, "@value", "name", "text")
            if isinstance(item, dict)
            else _clean_text(item)
        )
        if part:
            parts.append(part)
    return ", ".join(dict.fromkeys(parts)) or None


def _clean_description(value: object) -> str:
    text = str(value or "").strip()
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text("\n", strip=True)
    return _clean_text(unescape(text))


def _clean_text(value: object) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", str(value or ""))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _text(node: object | None) -> str:
    if node is None or not hasattr(node, "get_text"):
        return ""
    return str(node.get_text(" ", strip=True))


__all__ = [
    "extract_generic_job_details_from_html",
    "generic_job_id",
    "normalize_job_url",
]
