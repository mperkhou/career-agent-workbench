"""Deterministic job-description cleanup for bounded prompt context."""

from __future__ import annotations

import re

from career_agent_workbench.models import JobDetails

NO_PUBLIC_JOB_DESCRIPTION = "No public job description was available."
JOB_DESCRIPTION_PROMPT_MAX_CHARS = 12_000
JOD_INPUT_MAX_CHARS = 500_000

ROLE_HEADINGS = (
    "Job Summary",
    "Position Summary",
    "Role Summary",
    "The Role",
    "About the Role",
    "What You'll Do",
    "What You Will Do",
    "Key Responsibilities",
    "Responsibilities",
    "What You'll Bring",
    "What You Bring",
    "Required Qualifications",
    "Minimum Qualifications",
    "Basic Qualifications",
    "Qualifications",
    "Requirements",
    "Skills and Experience",
    "Who You Are",
)
LOW_SIGNAL_HEADINGS = (
    "Our Mission",
    "Our Values",
    "Our Culture",
    "Life at",
    "Why Join Us",
)
TRAILING_HEADINGS = (
    "Benefits",
    "Benefits & Perks",
    "Pay & Benefits",
    "Compensation",
    "Compensation Range",
    "Salary Range",
    "Equal Opportunity",
    "Accommodations",
    "Privacy Notice",
    "Applicant Privacy Notice",
    "Application Process",
)
ROLE_MARKERS = (
    "responsibil",
    "you will",
    "you'll",
    "build",
    "design",
    "develop",
    "operate",
    "automate",
    "analyze",
    "platform",
    "infrastructure",
    "api",
    "security",
    "reliability",
    "python",
)
BOILERPLATE_MARKERS = (
    "base salary",
    "salary range",
    "compensation",
    "medical, dental",
    "retirement savings",
    "parental leave",
    "equal opportunity",
    "reasonable accommodation",
    "applicant privacy",
    "personal information",
    "application process",
    "interview process",
    "we may use automated tools",
)

# Public, fictional examples anchor the vendor-neutral chunk heuristic.
JOD_CHUNK_TRAINING_EXAMPLES = (
    (
        "keep",
        (
            "Responsibilities: design reliable APIs and automate cloud operations "
            "for Example Harbor Systems."
        ),
    ),
    (
        "keep",
        (
            "Qualifications: experience with Python, observability, testing, and "
            "secure service delivery."
        ),
    ),
    (
        "drop",
        "Benefits include medical coverage, retirement savings, and paid leave.",
    ),
    (
        "drop",
        (
            "Example employer is an equal opportunity employer and provides "
            "reasonable accommodation."
        ),
    ),
)


def _guard_description(value: str | None) -> str:
    text = str(value or "")
    if len(text) > JOD_INPUT_MAX_CHARS:
        raise ValueError(
            "Job description exceeds the 500000-character input limit."
        ) from None
    return text


def usable_job_description(value: str | None) -> str | None:
    """Return non-placeholder description text."""
    text = _guard_description(value).strip()
    if not text or is_placeholder_job_description(text):
        return None
    return text


def is_placeholder_job_description(value: str) -> bool:
    """Recognize the stable public placeholder."""
    normalized = re.sub(r"\s+", " ", _guard_description(value))
    return normalized.strip().casefold().rstrip(
        "."
    ) == NO_PUBLIC_JOB_DESCRIPTION.casefold().rstrip(".")


def job_description_context(job: JobDetails) -> str:
    """Return deterministic prompt context from an already bounded model."""
    description = clean_job_description_for_prompt(
        job.description or NO_PUBLIC_JOB_DESCRIPTION
    )
    return limit_context(description, max_chars=JOB_DESCRIPTION_PROMPT_MAX_CHARS)


def clean_job_description_for_prompt(description: str) -> str:
    """Remove predictable low-signal and trailing public boilerplate."""
    original = _normalize_job_description_text(_guard_description(description))
    if not original:
        return NO_PUBLIC_JOB_DESCRIPTION
    selected = _trim_low_signal_preamble(original)
    selected = _trim_trailing_boilerplate(selected)
    selected = _select_relevant_chunks(selected)
    return selected.strip() or original


def limit_context(text: str, *, max_chars: int) -> str:
    """Limit context at a deterministic line boundary."""
    bounded = _guard_description(text)
    if max_chars < 0:
        raise ValueError("Context limit must be nonnegative.") from None
    if len(bounded) <= max_chars:
        return bounded
    truncated = bounded[:max_chars].rsplit("\n", 1)[0].strip()
    return truncated or bounded[:max_chars].strip()


def _trim_low_signal_preamble(description: str) -> str:
    role = _first_heading(description, ROLE_HEADINGS)
    if role is None or role.start() == 0:
        return description
    prefix = description[: role.start()]
    if _first_heading(prefix, LOW_SIGNAL_HEADINGS) is not None or (
        len(prefix) > 3_000 and _role_score(prefix) < 3
    ):
        return description[role.start() :].lstrip(" :-\n")
    return description


def _trim_trailing_boilerplate(description: str) -> str:
    role_positions = [
        match.start() for match in _heading_matches(description, ROLE_HEADINGS)
    ]
    last_role = max(role_positions, default=-1)
    for match in _heading_matches(description, TRAILING_HEADINGS):
        if match.start() >= max(300, last_role):
            return description[: match.start()].rstrip(" :-\n")
    return description


def _select_relevant_chunks(description: str) -> str:
    if not any(marker in description.casefold() for marker in BOILERPLATE_MARKERS):
        return description
    chunks = _description_chunks(description)
    if len(chunks) < 2:
        return description
    selected = [chunk for chunk in chunks if _keep_chunk(chunk)]
    joined = "\n".join(selected).strip()
    if not joined or len(joined) < min(600, int(len(description) * 0.30)):
        return description
    return joined


def _description_chunks(description: str) -> list[str]:
    boundaries = {0, len(description)}
    for match in _heading_matches(
        description,
        (*ROLE_HEADINGS, *LOW_SIGNAL_HEADINGS, *TRAILING_HEADINGS),
    ):
        boundaries.add(match.start())
    chunks: list[str] = []
    ordered = sorted(boundaries)
    for index, start in enumerate(ordered[:-1]):
        segment = description[start : ordered[index + 1]].strip(" :-\n")
        if not segment:
            continue
        chunks.extend(
            part.strip(" :-\n")
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", segment)
            if part.strip(" :-\n")
        )
    return chunks


def _keep_chunk(chunk: str) -> bool:
    normalized = " ".join(chunk.split())
    if len(normalized) < 24:
        return False
    lowered = normalized.casefold()
    role_score = _role_score(normalized)
    boilerplate = sum(marker in lowered for marker in BOILERPLATE_MARKERS)
    if boilerplate and role_score < 2:
        return False
    # The weighted decision mirrors the fictional public examples above.
    return role_score - (2 * boilerplate) >= 0


def _role_score(text: str) -> int:
    lowered = text.casefold()
    return sum(marker in lowered for marker in ROLE_MARKERS)


def _first_heading(
    text: str,
    headings: tuple[str, ...],
) -> re.Match[str] | None:
    matches = _heading_matches(text, headings)
    return matches[0] if matches else None


def _heading_matches(
    text: str,
    headings: tuple[str, ...],
) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for heading in headings:
        pattern = r"\s+".join(re.escape(part) for part in heading.split())
        compiled = re.compile(
            rf"(?<![\w/]){pattern}(?=\s|[:?!.,;()\-/–—]|$)",
            re.IGNORECASE,
        )
        for match in compiled.finditer(text):
            first_alpha = next(
                (character for character in match.group() if character.isalpha()),
                "",
            )
            if first_alpha and first_alpha.isupper():
                matches.append(match)
    return sorted(matches, key=lambda match: match.start())


def _normalize_job_description_text(description: str) -> str:
    lines = [" ".join(line.split()) for line in description.splitlines()]
    normalized = [line for line in lines if line]
    return (
        "\n".join(normalized) if len(normalized) > 1 else " ".join(description.split())
    )


__all__ = [
    "JOB_DESCRIPTION_PROMPT_MAX_CHARS",
    "JOD_INPUT_MAX_CHARS",
    "NO_PUBLIC_JOB_DESCRIPTION",
    "clean_job_description_for_prompt",
    "is_placeholder_job_description",
    "job_description_context",
    "limit_context",
    "usable_job_description",
]
