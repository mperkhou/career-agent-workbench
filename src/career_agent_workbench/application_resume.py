"""Pure helpers for building and tailoring an Application Resume Object."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from career_agent_workbench.errors import WorkflowError

CORE_SKILLS_PROMPT_JOD_MAX_CHARS = 12_000
JOD_TARGET_PROMPT_JOD_MAX_CHARS = 12_000
JOB_OPENING_DESCRIPTION_SCHEMA_VERSION = "job_opening_description.v1"
MAX_RESUME_YAML_BYTES = 2_000_000

_LOAD_ERROR_MESSAGE = "Unable to load application resume YAML."
_ROOT_ERROR_MESSAGE = "Application resume YAML must contain a mapping."
_ORDER_ERROR_MESSAGE = "Experience job order is invalid."
_MAX_ORDER_DECIMAL_CHARS = 4_096


class ApplicationResumeError(WorkflowError):
    """Raised when application-resume input cannot be processed safely."""

    __slots__ = ()


def initialize_application_resume_object(master_resume_path: Path) -> dict[str, Any]:
    """Load an exact caller-supplied YAML path and reset job-specific fields."""

    if not isinstance(master_resume_path, Path):
        raise ApplicationResumeError(_LOAD_ERROR_MESSAGE)

    value, load_failed = _load_yaml(master_resume_path)
    if load_failed:
        raise ApplicationResumeError(_LOAD_ERROR_MESSAGE)
    if not isinstance(value, Mapping):
        raise ApplicationResumeError(_ROOT_ERROR_MESSAGE)
    return reset_application_resume_jod_state(value)


def reset_application_resume_jod_state(
    application_resume: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a defensive ARO copy with all job-specific match state reset."""

    aro = copy.deepcopy(dict(application_resume))
    aro.pop("job_opening_description", None)

    for bucket in _core_skill_buckets(aro):
        bucket["jod_matched_items"] = []

    for bullet in _professional_experience_bullets(aro):
        bullet["bullet_point_total_match_count"] = 0
        categories = bullet.get("categories")
        if isinstance(categories, Mapping):
            bullet["categories"] = copy.deepcopy(dict(categories))
            bullet["categories"]["matched"] = []
        for skill_entry in _skill_entries(bullet):
            skill_entry["jod_match_count"] = 0

    return aro


def build_core_skills_jod_match_prompt(
    *,
    application_resume: Mapping[str, Any],
    trimmed_job_description: str,
    max_jod_chars: int = CORE_SKILLS_PROMPT_JOD_MAX_CHARS,
) -> str:
    """Build a bounded prompt using only supplied skill and job evidence."""

    core_skills = _core_skill_prompt_payload(application_resume)
    core_skills_json = json.dumps(core_skills, ensure_ascii=True, indent=2)
    jod = _limit_text(
        trimmed_job_description,
        max_chars=_prompt_char_limit(
            max_jod_chars,
            hard_limit=CORE_SKILLS_PROMPT_JOD_MAX_CHARS,
        ),
    )
    return f"""
You select which factual resume skills match one job opening description.
Return only valid JSON. Do not return markdown fences, commentary, advice, or a resume.

Context:
- ARO means Application Resume Object.
- You only update Core Technical Skills by selecting jod_matched_items.
- The caller will apply your response before generating experience bullets.

Rules:
- Preserve the category names exactly.
- Only select display skills already present in that category's primary or additional lists.
- Use match_terms as non-display aliases or evidence for those display skills. Do not
  return match_terms directly.
- Include primary skills when the job description asks for them.
- Include additional skills when the job description asks for them.
- Select the strongest overlaps only; prefer no more than 8 additional display skills
  per category.
- Do not invent skills, employers, tools, responsibilities, outcomes, or credentials.
- If a category has no clear overlap, return an empty jod_matched_items list.

Return this exact JSON shape:
{{
  "core_technical_skills": [
    {{
      "category": "",
      "jod_matched_items": []
    }}
  ]
}}

Core Technical Skills inventory:
{core_skills_json}

Trimmed job opening description:
{jod}
""".strip()


def apply_core_skill_jod_matches(
    *,
    application_resume: Mapping[str, Any],
    core_skill_response: Any,
) -> dict[str, Any]:
    """Apply inventoried skill matches to a defensive ARO copy."""

    aro = copy.deepcopy(dict(application_resume))
    response_by_category = _extract_core_skill_match_response(core_skill_response)
    inventory_by_category = _core_skill_inventory_by_category(aro)

    for bucket in _core_skill_buckets(aro):
        category = str(bucket.get("category") or "").strip()
        normalized_category = _normalize(category)
        inventory = inventory_by_category.get(normalized_category, [])
        requested = response_by_category.get(normalized_category, set())
        bucket["jod_matched_items"] = [
            skill for skill, identity in inventory if identity in requested
        ]

    return aro


def build_jod_requirements_target_prompt(
    *,
    trimmed_job_description: str,
    max_jod_chars: int = JOD_TARGET_PROMPT_JOD_MAX_CHARS,
) -> str:
    """Build a bounded prompt for extracting resume-targetable requirements."""

    jod = _limit_text(
        trimmed_job_description,
        max_chars=_prompt_char_limit(
            max_jod_chars,
            hard_limit=JOD_TARGET_PROMPT_JOD_MAX_CHARS,
        ),
    )
    return f"""
You convert a job opening description into small, resume-targetable requirements.
Return only valid JSON. Do not return markdown fences, commentary, or advice.

Rules:
- Extract concrete responsibilities, qualifications, technologies, domains, and outcomes.
- Keep each target short enough to drive one resume-bullet rewrite.
- Do not invent requirements absent from the supplied job opening description.
- Merge duplicates and near-duplicates.
- Drop compensation, benefits, equal-opportunity, privacy, and application-process text.
- Prefer 6 to 14 targets unless the description has fewer meaningful requirements.

Return this exact JSON shape:
{{
  "job_opening_description": {{
    "requirements_targets": []
  }}
}}

Job opening description:
{jod}
""".strip()


def create_job_opening_description_object(
    *,
    trimmed_job_description: str,
    requirements_response: Any,
    model: str = "",
) -> dict[str, Any]:
    """Create a deterministic compact object from supplied requirement targets."""

    targets = _extract_jod_target_texts(requirements_response)
    if not targets:
        raise ApplicationResumeError(
            "Job requirements response did not contain targets."
        )

    llm: dict[str, str] = {}
    if isinstance(model, str) and model.strip():
        llm["model"] = model.strip()

    return {
        "schema_version": JOB_OPENING_DESCRIPTION_SCHEMA_VERSION,
        "source": {
            "type": "trimmed_job_description",
            "character_count": len(str(trimmed_job_description or "").strip()),
        },
        "llm": llm,
        "requirements_targets": [
            {"order": index, "text": target}
            for index, target in enumerate(targets, start=1)
        ],
    }


def attach_job_opening_description_object(
    *,
    application_resume: Mapping[str, Any],
    job_opening_description: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach supplied job-description data to a defensive ARO copy."""

    aro = copy.deepcopy(dict(application_resume))
    aro["job_opening_description"] = copy.deepcopy(dict(job_opening_description))
    return aro


def job_opening_description_target_texts(
    job_opening_description: Mapping[str, Any],
) -> list[str]:
    """Return normalized, deduplicated targets in source order."""

    return _extract_jod_target_texts(job_opening_description)


def experience_jobs_for_jod_bullet_rewrite(
    application_resume: Mapping[str, Any],
    *,
    include_orders: Sequence[Any] | None = None,
    exclude_orders: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Return rendered jobs selected by explicit generic order filters."""

    include = (
        None if include_orders is None else _normalize_order_sequence(include_orders)
    )
    exclude = _normalize_order_sequence(exclude_orders)

    jobs: list[dict[str, Any]] = []
    for job in _professional_experience_jobs(application_resume):
        if not _render_enabled(job.get("render")):
            continue
        order = _stored_order(job.get("order"))
        if include is not None and order not in include:
            continue
        if order in exclude:
            continue
        jobs.append(copy.deepcopy(job))
    return jobs


def experience_job_for_jod_bullet_rewrite(
    application_resume: Mapping[str, Any],
    *,
    job_order: Any,
) -> dict[str, Any]:
    """Return one exact caller-selected rendered job as a defensive copy."""

    expected_order = _normalize_positive_order(job_order)
    for job in _professional_experience_jobs(application_resume):
        if _stored_order(job.get("order")) != expected_order:
            continue
        if not _render_enabled(job.get("render")):
            raise ApplicationResumeError("Experience job is not enabled for rendering.")
        return copy.deepcopy(job)
    raise ApplicationResumeError("Experience job was not found.")


def build_experience_job_bullet_rewrite_prompt(
    *,
    job_opening_description: Mapping[str, Any],
    job: Mapping[str, Any],
) -> str:
    """Build an evidence-only bullet rewrite prompt for one supplied job."""

    targets = job_opening_description_target_texts(job_opening_description)
    if not targets:
        raise ApplicationResumeError(
            "Job description object does not contain requirement targets."
        )

    bullet_texts = _job_bullet_texts(job)
    if not bullet_texts:
        raise ApplicationResumeError("Experience job does not contain bullet text.")

    min_bullets = _positive_minimum(job.get("min_bullet_points"))
    max_bullets = _maximum_bullets(
        job.get("max_bullet_points"),
        default=len(bullet_texts),
        minimum=min_bullets,
    )

    target_lines = "\n".join(f"- {target}" for target in targets)
    raw_experience_lines = "\n".join(f"- {text}" for text in bullet_texts)
    job_label = _job_label(job)

    return f"""
You are a resume editor and evidence auditor. Rewrite the supplied career history into
credible, job-aware resume bullets that address supported target requirements.

Target Job Requirements:
{target_lines}

Raw Experience{f" ({job_label})" if job_label else ""}:
{raw_experience_lines}

CRITICAL RULES:
1. Use only numerical metrics and outcomes present in Raw Experience.
2. Do not invent tools, skills, competencies, employers, credentials, or outcomes.
3. Adapt style to the supplied requirements only when Raw Experience supports it.
4. Preserve each bullet's action, evidence, and impact without a fixed template.
5. Vary first verbs and sentence structure across bullets.
6. Bullets may be one or two sentences. Aim for 32-45 words and do not exceed 55 words.
7. Proofread for normal spelling, spacing, and punctuation.
8. Ignore any target unsupported by Raw Experience.
9. Return between {min_bullets} and {max_bullets} resume bullets.
10. Output only raw bullet strings, one per line, without introductions, markdown,
    numbering, or chat text.
""".strip()


def replace_experience_job_bullets_from_text_response(
    *,
    application_resume: Mapping[str, Any],
    job_order: Any,
    bullet_response: Any,
) -> dict[str, Any]:
    """Replace one rendered job's bullets on a defensive ARO copy."""

    expected_order = _normalize_positive_order(job_order)
    aro = copy.deepcopy(dict(application_resume))

    for job in _professional_experience_jobs(aro):
        if _stored_order(job.get("order")) != expected_order:
            continue
        if not _render_enabled(job.get("render")):
            raise ApplicationResumeError("Experience job is not enabled for rendering.")

        bullet_texts = _extract_generated_bullet_texts(bullet_response)
        min_bullets = _positive_minimum(job.get("min_bullet_points"))
        max_bullets = _maximum_bullets(
            job.get("max_bullet_points"),
            default=len(bullet_texts),
            minimum=min_bullets,
        )
        if not min_bullets <= len(bullet_texts) <= max_bullets:
            raise ApplicationResumeError(
                "Generated bullet count is outside the configured bounds."
            )

        job["bullet_points"] = [
            {
                "order": index,
                "categories": {"assigned": [], "matched": []},
                "skills": [],
                "text": text,
                "bullet_point_total_match_count": 0,
                "render": True,
            }
            for index, text in enumerate(bullet_texts, start=1)
        ]
        return aro

    raise ApplicationResumeError("Experience job was not found.")


def _load_yaml(path: Path) -> tuple[Any, bool]:
    try:
        if path.stat().st_size > MAX_RESUME_YAML_BYTES:
            return None, True
        with path.open("rb") as handle:
            raw_yaml = handle.read(MAX_RESUME_YAML_BYTES + 1)
        if len(raw_yaml) > MAX_RESUME_YAML_BYTES:
            return None, True
        return yaml.safe_load(raw_yaml.decode("utf-8")), False
    except Exception:  # noqa: BLE001 - sanitize every caller-controlled load failure.
        return None, True


def _core_skill_prompt_payload(
    application_resume: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for bucket in _core_skill_buckets(application_resume):
        items = bucket.get("items")
        item_mapping = items if isinstance(items, Mapping) else {}
        match_terms = _core_skill_match_terms_by_skill(item_mapping)
        payload.append(
            {
                "category": str(bucket.get("category") or "").strip(),
                "primary": _string_list(item_mapping.get("primary")),
                "additional": _string_list(item_mapping.get("additional")),
                "match_terms": [
                    {"skill": skill, "terms": terms}
                    for skill, terms in match_terms.items()
                ],
            }
        )
    return payload


def _core_skill_inventory_by_category(
    application_resume: Mapping[str, Any],
) -> dict[str, list[tuple[str, str]]]:
    inventory: dict[str, list[tuple[str, str]]] = {}
    for bucket in _core_skill_buckets(application_resume):
        category = str(bucket.get("category") or "").strip()
        items = bucket.get("items")
        item_mapping = items if isinstance(items, Mapping) else {}
        display_items = _dedupe_skills_preserve_order(
            [
                *_string_list(item_mapping.get("primary")),
                *_string_list(item_mapping.get("additional")),
            ]
        )
        inventory[_normalize(category)] = [
            (display_item, _skill_identity(display_item))
            for display_item in display_items
        ]
    return inventory


def _core_skill_match_terms_by_skill(
    item_mapping: Mapping[str, Any],
) -> dict[str, list[str]]:
    display_items = _dedupe_skills_preserve_order(
        [
            *_string_list(item_mapping.get("primary")),
            *_string_list(item_mapping.get("additional")),
        ]
    )
    display_by_key = {_skill_identity(item): item for item in display_items}
    raw_match_terms = item_mapping.get("match_terms")
    if not isinstance(raw_match_terms, Mapping):
        return {}

    terms_by_skill: dict[str, list[str]] = {}
    for raw_skill, raw_terms in raw_match_terms.items():
        skill = display_by_key.get(_skill_identity(str(raw_skill)))
        if skill is None:
            continue
        terms = _string_list(raw_terms)
        if terms:
            terms_by_skill[skill] = terms
    return terms_by_skill


def _extract_core_skill_match_response(response: Any) -> dict[str, set[str]]:
    raw_items: Any = response
    if isinstance(response, str):
        try:
            raw_items = json.loads(response)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw_items, Mapping):
        raw_items = raw_items.get("core_technical_skills", raw_items)
        if isinstance(raw_items, Mapping):
            raw_items = raw_items.get("bullet_points", raw_items)
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        return {}

    by_category: dict[str, set[str]] = {}
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("category") or item.get("name") or "").strip()
        if not category:
            continue
        matches = (
            item.get("jod_matched_items")
            or item.get("matched")
            or item.get("skills")
            or []
        )
        by_category[_normalize(category)] = {
            _skill_identity(skill) for skill in _string_list(matches)
        }
    return by_category


def _extract_jod_target_texts(response: Any) -> list[str]:
    raw_items: Any = response
    if isinstance(response, str):
        try:
            raw_items = json.loads(response)
        except json.JSONDecodeError:
            raw_items = _split_text_lines(response)
    if isinstance(raw_items, Mapping):
        raw_items = (
            raw_items.get("requirements_targets")
            or raw_items.get("targets")
            or raw_items.get("jod_targets")
            or raw_items.get("requirements")
            or raw_items.get("bullet_points")
            or raw_items.get("queries")
            or raw_items.get("job_opening_description")
            or raw_items.get("jod")
            or raw_items
        )
        if isinstance(raw_items, Mapping):
            raw_items = (
                raw_items.get("requirements_targets")
                or raw_items.get("targets")
                or raw_items.get("jod_targets")
                or raw_items.get("requirements")
                or raw_items.get("bullet_points")
                or raw_items.get("queries")
                or []
            )
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        return []

    targets: list[str] = []
    for item in raw_items:
        text = ""
        if isinstance(item, Mapping):
            text = str(
                item.get("text")
                or item.get("target")
                or item.get("requirement")
                or item.get("description")
                or ""
            )
        elif isinstance(item, str):
            text = item
        text = _clean_generated_line(text)
        if text:
            targets.append(text)
    return _dedupe_preserve_order(targets)


def _extract_generated_bullet_texts(response: Any) -> list[str]:
    raw_items: Any = response
    if isinstance(response, str):
        try:
            raw_items = json.loads(response)
        except json.JSONDecodeError:
            raw_items = _split_text_lines(response)
    if isinstance(raw_items, Mapping):
        raw_items = (
            raw_items.get("bullet_points") or raw_items.get("bullets") or raw_items
        )
        if isinstance(raw_items, Mapping):
            raw_items = raw_items.get("items") or raw_items.get("generated") or []
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        return []

    bullets: list[str] = []
    for item in raw_items:
        text = ""
        if isinstance(item, Mapping):
            text = str(
                item.get("text") or item.get("bullet") or item.get("content") or ""
            )
        elif isinstance(item, str):
            text = item
        text = _clean_generated_line(text)
        if text:
            bullets.append(text)
    return _dedupe_preserve_order(bullets)


def _split_text_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(stripped)
    return lines


def _clean_generated_line(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^\s*(?:[-*]+|\d+[.)])\s*", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    return cleaned.replace("`", "")


def _job_bullet_texts(job: Mapping[str, Any]) -> list[str]:
    raw_bullets = job.get("bullet_points")
    if not isinstance(raw_bullets, list):
        return []
    return [text for bullet in raw_bullets if (text := _bullet_text(bullet))]


def _bullet_text(bullet: Any) -> str:
    if isinstance(bullet, str):
        return bullet.strip()
    if not isinstance(bullet, Mapping):
        return ""
    return str(bullet.get("text") or "").strip()


def _job_label(job: Mapping[str, Any]) -> str:
    line_1 = job.get("line_1")
    line_mapping = line_1 if isinstance(line_1, Mapping) else {}
    parts = [
        str(line_mapping.get("company_name_text") or "").strip(),
        str(line_mapping.get("position_name_text") or "").strip(),
        str(line_mapping.get("position_dates_text") or "").strip(),
    ]
    return " | ".join(part for part in parts if part)


def _core_skill_buckets(
    application_resume: Mapping[str, Any],
) -> list[dict[str, Any]]:
    core_skills = application_resume.get("core_technical_skills")
    if not isinstance(core_skills, Mapping):
        return []
    buckets = core_skills.get("bullet_points")
    if not isinstance(buckets, list):
        return []
    return [bucket for bucket in buckets if isinstance(bucket, dict)]


def _professional_experience_bullets(
    application_resume: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bullets: list[dict[str, Any]] = []
    for job in _professional_experience_jobs(application_resume):
        bullets.extend(_job_bullets(job))
    return bullets


def _professional_experience_jobs(
    application_resume: Mapping[str, Any],
) -> list[dict[str, Any]]:
    experience = application_resume.get("professional_experience")
    if not isinstance(experience, Mapping):
        return []
    jobs = experience.get("jobs")
    if not isinstance(jobs, list):
        return []
    return [job for job in jobs if isinstance(job, dict)]


def _job_bullets(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_bullets = job.get("bullet_points")
    if not isinstance(raw_bullets, list):
        return []
    return [bullet for bullet in raw_bullets if isinstance(bullet, dict)]


def _skill_entries(bullet: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = bullet.get("skills")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize(item)
        if normalized not in seen:
            result.append(item)
            seen.add(normalized)
    return result


def _dedupe_skills_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        identity = _skill_identity(item)
        if identity not in seen:
            result.append(item)
            seen.add(identity)
    return result


def _positive_minimum(value: Any) -> int:
    normalized = _nonnegative_int(value, default=1)
    return max(normalized, 1)


def _maximum_bullets(value: Any, *, default: int, minimum: int) -> int:
    normalized = _nonnegative_int(value, default=default)
    if normalized <= 0:
        normalized = default
    return max(normalized, minimum)


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        try:
            return max(int(value), 0)
        except ValueError:
            return default
    return default


def _render_enabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() not in {"false", "no", "0", "off"}
    return bool(value)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _skill_identity(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalize_positive_order(value: Any) -> int:
    if isinstance(value, bool):
        raise ApplicationResumeError(_ORDER_ERROR_MESSAGE)
    if isinstance(value, int):
        if value > 0:
            return value
        raise ApplicationResumeError(_ORDER_ERROR_MESSAGE)
    if (
        isinstance(value, str)
        and value
        and len(value) <= _MAX_ORDER_DECIMAL_CHARS
        and value.isascii()
        and value.isdecimal()
    ):
        conversion_failed = False
        try:
            normalized = int(value)
        except ValueError:
            conversion_failed = True
            normalized = 0
        if not conversion_failed and normalized > 0:
            return normalized
    raise ApplicationResumeError(_ORDER_ERROR_MESSAGE)


def _normalize_order_sequence(values: Sequence[Any]) -> set[int]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ApplicationResumeError(_ORDER_ERROR_MESSAGE)
    return {_normalize_positive_order(value) for value in values}


def _stored_order(value: Any) -> int:
    return _normalize_positive_order(value)


def _prompt_char_limit(value: Any, *, hard_limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return hard_limit
    return min(max(value, 0), hard_limit)


def _limit_text(value: str, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n[truncated]"
