"""Lossless structured-field adapter for the local resume editor."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

from career_agent_workbench.application_state import MAX_ARO_YAML_BYTES

_MAX_ITEMS = 100
_MAX_TEXT_CHARS = 100_000
_VERSION = 1
_CAPABILITIES = {
    "version": _VERSION,
    "lossless_unknown_values": True,
    "advanced_yaml": True,
    "revision_bound": True,
    "sections": {
        "resume_layout": ("supporting_sections_start_on_page_2",),
        "header_top": (
            "render",
            "name",
            "headline",
            "applicant_info",
            "contact_items",
            "links",
        ),
        "professional_summary": ("render", "heading", "paragraph", "note"),
        "core_technical_skills": (
            "render",
            "heading",
            "categories",
            "primary",
            "additional",
            "jod_matches",
        ),
        "professional_experience": (
            "render",
            "heading",
            "employers",
            "roles",
            "dates",
            "introductions",
            "bullets",
        ),
        "education": ("render", "heading", "entries", "dates", "bullets"),
        "certifications": ("render", "heading", "bullets"),
        "portfolio": ("render", "heading", "projects", "links", "descriptions"),
    },
    "repeated_operations": ("add", "remove", "reorder", "validate"),
}


class ResumeFieldsError(ValueError):
    """Content-free structured resume input failure."""

    __slots__ = ()


def resume_field_capabilities() -> dict[str, Any]:
    """Return a content-free description of the structured editor surface."""

    return copy.deepcopy(_CAPABILITIES)


def resume_field_model(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return editable fields plus opaque source references for one resume."""

    source = _materialize(value)
    if type(source) is not dict:
        raise ResumeFieldsError("Structured resume data is invalid.")
    education_key = (
        "education"
        if "education" in source
        else "education_and_certifications"
        if "education_and_certifications" in source
        else "education"
    )
    education = _section(source, education_key)
    education_items_key = (
        "entries"
        if "entries" in education
        else "items"
        if "items" in education
        else "entries"
    )
    return {
        "version": _VERSION,
        "resume_layout": _layout_model(source),
        "header_top": _header_model(source),
        "professional_summary": _text_section_model(
            source,
            "professional_summary",
            ("header_text", "paragraph", "summary_note"),
        ),
        "core_technical_skills": _skills_model(source),
        "professional_experience": _experience_model(source),
        "education": {
            **_text_section_model(
                source,
                education_key,
                ("header_text",),
            ),
            "root_key": education_key,
            "items_key": education_items_key,
            "entries": _list_model(
                education.get(education_items_key), _education_item_model
            ),
        },
        "certifications": {
            **_text_section_model(
                source,
                "certifications",
                ("header_text",),
            ),
            "bullet_points": _list_model(
                _section(source, "certifications").get("bullet_points"),
                _bullet_model,
            ),
        },
        "portfolio": {
            **_text_section_model(source, "portfolio", ("header_text",)),
            "projects": _list_model(
                _section(source, "portfolio").get("projects"),
                _project_model,
            ),
        },
    }


def resume_field_payload_text(value: Mapping[str, Any]) -> str:
    """Serialize a structured field model for the browser form."""

    try:
        return json.dumps(
            resume_field_model(value),
            ensure_ascii=True,
            separators=(",", ":"),
        )
    except ResumeFieldsError:
        raise
    except Exception:  # noqa: BLE001 - mapping details remain private.
        raise ResumeFieldsError("Structured resume data is invalid.") from None


def apply_resume_field_payload(
    value: Mapping[str, Any], payload_text: object
) -> dict[str, Any]:
    """Patch supported fields while preserving all unsubmitted source values."""

    source = _materialize(value)
    try:
        if type(source) is not dict or type(payload_text) is not str:
            raise ValueError
        if len(payload_text.encode("utf-8")) > MAX_ARO_YAML_BYTES:
            raise ValueError
        payload = json.loads(payload_text)
        if type(payload) is not dict or payload.get("version") != _VERSION:
            raise ValueError
        result = copy.deepcopy(source)
        _apply_layout(result, _mapping(payload, "resume_layout"))
        _apply_header(result, _mapping(payload, "header_top"))
        _apply_text_section(
            result,
            "professional_summary",
            _mapping(payload, "professional_summary"),
            ("header_text", "paragraph", "summary_note"),
        )
        _apply_skills(result, _mapping(payload, "core_technical_skills"))
        _apply_experience(result, _mapping(payload, "professional_experience"))
        _apply_education(result, _mapping(payload, "education"))
        _apply_certifications(result, _mapping(payload, "certifications"))
        _apply_portfolio(result, _mapping(payload, "portfolio"))
        return result
    except ResumeFieldsError:
        raise
    except Exception:  # noqa: BLE001 - structured content remains private.
        raise ResumeFieldsError("Structured resume data is invalid.") from None


def _layout_model(source: dict[str, Any]) -> dict[str, Any]:
    section = _section(source, "resume_layout")
    return {
        "present": type(source.get("resume_layout")) is dict,
        "editable": "resume_layout" not in source
        or type(source.get("resume_layout")) is dict,
        "supporting_sections_start_on_page_2": _boolean(
            section.get("supporting_sections_start_on_page_2"), False
        ),
    }


def _header_model(source: dict[str, Any]) -> dict[str, Any]:
    section = _section(source, "header_top")
    return {
        **_base_section_model(source, "header_top"),
        **{
            key: _string(section.get(key))
            for key in (
                "line_1_name_header_text",
                "line_2_header_text",
                "line_2_applicant_info_text",
                "line_3_applicant_info_text",
            )
        },
        "contact_items": _list_model(section.get("contact_items"), _text_item_model),
        "links": _list_model(section.get("links"), _link_model),
    }


def _text_section_model(
    source: dict[str, Any], key: str, fields: Sequence[str]
) -> dict[str, Any]:
    section = _section(source, key)
    return {
        **_base_section_model(source, key),
        **{field: _string(section.get(field)) for field in fields},
    }


def _skills_model(source: dict[str, Any]) -> dict[str, Any]:
    section = _section(source, "core_technical_skills")
    return {
        **_text_section_model(source, "core_technical_skills", ("header_text",)),
        "bullet_points": _list_model(section.get("bullet_points"), _skill_item_model),
    }


def _experience_model(source: dict[str, Any]) -> dict[str, Any]:
    section = _section(source, "professional_experience")
    return {
        **_text_section_model(source, "professional_experience", ("header_text",)),
        "jobs": _list_model(section.get("jobs"), _job_model),
    }


def _base_section_model(source: dict[str, Any], key: str) -> dict[str, Any]:
    section = source.get(key)
    editable = key not in source or type(section) is dict
    values = section if type(section) is dict else {}
    return {
        "present": type(section) is dict,
        "editable": editable,
        "render": _boolean(values.get("render"), True),
    }


def _list_model(value: object, builder: Any) -> list[dict[str, Any]]:
    if type(value) not in {list, tuple}:
        return []
    return [builder(item, index) for index, item in enumerate(value[:_MAX_ITEMS])]


def _source_model(value: object, index: int) -> dict[str, Any]:
    return {"source": index, "kind": _kind(value)}


def _text_item_model(value: object, index: int) -> dict[str, Any]:
    return {**_source_model(value, index), "text": _string(value)}


def _link_model(value: object, index: int) -> dict[str, Any]:
    mapping = value if type(value) is dict else {}
    return {
        **_source_model(value, index),
        "label": _string(mapping.get("label")),
        "url": _string(mapping.get("url")),
    }


def _bullet_model(value: object, index: int) -> dict[str, Any]:
    mapping = value if type(value) is dict else {}
    return {
        **_source_model(value, index),
        "render": _boolean(mapping.get("render"), True),
        "text": _string(value if type(value) is str else mapping.get("text")),
        "bold_label": _string(mapping.get("bold_label")),
        "category": _string(mapping.get("category")),
    }


def _skill_item_model(value: object, index: int) -> dict[str, Any]:
    mapping = value if type(value) is dict else {}
    items = mapping.get("items") if type(mapping.get("items")) is dict else {}
    return {
        **_bullet_model(value, index),
        "primary": _string_items(items.get("primary")),
        "additional": _string_items(items.get("additional")),
        "jod_matched_items": _string_items(mapping.get("jod_matched_items")),
    }


def _job_model(value: object, index: int) -> dict[str, Any]:
    mapping = value if type(value) is dict else {}
    line_1 = mapping.get("line_1") if type(mapping.get("line_1")) is dict else {}
    line_2 = mapping.get("line_2") if type(mapping.get("line_2")) is dict else {}
    return {
        **_source_model(value, index),
        "render": _boolean(mapping.get("render"), True),
        "order": mapping.get("order")
        if type(mapping.get("order")) in {int, str}
        else "",
        "min_bullet_points": mapping.get("min_bullet_points")
        if type(mapping.get("min_bullet_points")) is int
        else "",
        "max_bullet_points": mapping.get("max_bullet_points")
        if type(mapping.get("max_bullet_points")) is int
        else "",
        "company": _string(line_1.get("company_name_text")),
        "role": _string(line_1.get("position_name_text")),
        "dates": _string(line_1.get("position_dates_text")),
        "intro": _string(line_2.get("position_intro_text")),
        "bullet_points": _list_model(mapping.get("bullet_points"), _bullet_model),
    }


def _education_item_model(value: object, index: int) -> dict[str, Any]:
    mapping = value if type(value) is dict else {}
    line_1 = mapping.get("line_1") if type(mapping.get("line_1")) is dict else {}
    line_2 = mapping.get("line_2") if type(mapping.get("line_2")) is dict else {}
    return {
        **_source_model(value, index),
        "render": _boolean(mapping.get("render"), True),
        "text": _string(value if type(value) is str else ""),
        "title": _string(mapping.get("title")),
        "institution": _string(line_1.get("institution_name_text")),
        "degree": _string(line_2.get("degree_name_text")),
        "dates": _string(line_2.get("degree_dates_text")),
        "bullet_points": _list_model(mapping.get("bullet_points"), _bullet_model),
    }


def _project_model(value: object, index: int) -> dict[str, Any]:
    mapping = value if type(value) is dict else {}
    return {
        **_source_model(value, index),
        "render": _boolean(mapping.get("render"), True),
        "text": _string(value if type(value) is str else ""),
        "title": _string(mapping.get("title_text", mapping.get("title"))),
        "url": _string(mapping.get("url")),
        "description": _string(
            mapping.get("description_text", mapping.get("description"))
        ),
    }


def _apply_layout(result: dict[str, Any], payload: dict[str, Any]) -> None:
    if not _require_editable(result, "resume_layout", payload):
        return
    _apply_section_presence(
        result,
        "resume_layout",
        payload,
        lambda section: _set_bool(
            section,
            "supporting_sections_start_on_page_2",
            _payload_bool(payload, "supporting_sections_start_on_page_2"),
            False,
        ),
    )


def _apply_header(result: dict[str, Any], payload: dict[str, Any]) -> None:
    if not _require_editable(result, "header_top", payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        for key in (
            "line_1_name_header_text",
            "line_2_header_text",
            "line_2_applicant_info_text",
            "line_3_applicant_info_text",
        ):
            _set_text(section, key, _payload_text(payload, key))
        _set_list(
            section,
            "contact_items",
            _apply_item_list(
                section.get("contact_items"),
                _payload_list(payload, "contact_items"),
                _patch_text_item,
            ),
        )
        _set_list(
            section,
            "links",
            _apply_item_list(
                section.get("links"),
                _payload_list(payload, "links"),
                _patch_link,
            ),
        )

    _apply_section_presence(result, "header_top", payload, patch)


def _apply_text_section(
    result: dict[str, Any],
    key: str,
    payload: dict[str, Any],
    fields: Sequence[str],
) -> None:
    if not _require_editable(result, key, payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        for field in fields:
            _set_text(section, field, _payload_text(payload, field))

    _apply_section_presence(result, key, payload, patch)


def _apply_skills(result: dict[str, Any], payload: dict[str, Any]) -> None:
    key = "core_technical_skills"
    if not _require_editable(result, key, payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        _set_text(section, "header_text", _payload_text(payload, "header_text"))
        _set_list(
            section,
            "bullet_points",
            _apply_item_list(
                section.get("bullet_points"),
                _payload_list(payload, "bullet_points"),
                _patch_skill_item,
            ),
        )

    _apply_section_presence(result, key, payload, patch)


def _apply_experience(result: dict[str, Any], payload: dict[str, Any]) -> None:
    key = "professional_experience"
    if not _require_editable(result, key, payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        _set_text(section, "header_text", _payload_text(payload, "header_text"))
        _set_list(
            section,
            "jobs",
            _apply_item_list(
                section.get("jobs"), _payload_list(payload, "jobs"), _patch_job
            ),
        )

    _apply_section_presence(result, key, payload, patch)


def _apply_education(result: dict[str, Any], payload: dict[str, Any]) -> None:
    root_key = _payload_choice(
        payload, "root_key", ("education", "education_and_certifications")
    )
    original_key = (
        "education"
        if "education" in result
        else "education_and_certifications"
        if "education_and_certifications" in result
        else "education"
    )
    if root_key != original_key:
        raise ResumeFieldsError("Structured resume data is invalid.")
    items_key = _payload_choice(payload, "items_key", ("entries", "items"))
    original = _section(result, root_key)
    original_items_key = (
        "entries"
        if "entries" in original
        else "items"
        if "items" in original
        else "entries"
    )
    if items_key != original_items_key:
        raise ResumeFieldsError("Structured resume data is invalid.")
    if not _require_editable(result, root_key, payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        _set_text(section, "header_text", _payload_text(payload, "header_text"))
        _set_list(
            section,
            items_key,
            _apply_item_list(
                section.get(items_key),
                _payload_list(payload, "entries"),
                _patch_education_item,
            ),
        )

    _apply_section_presence(result, root_key, payload, patch)


def _apply_certifications(result: dict[str, Any], payload: dict[str, Any]) -> None:
    key = "certifications"
    if not _require_editable(result, key, payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        _set_text(section, "header_text", _payload_text(payload, "header_text"))
        _set_list(
            section,
            "bullet_points",
            _apply_item_list(
                section.get("bullet_points"),
                _payload_list(payload, "bullet_points"),
                _patch_bullet,
            ),
        )

    _apply_section_presence(result, key, payload, patch)


def _apply_portfolio(result: dict[str, Any], payload: dict[str, Any]) -> None:
    key = "portfolio"
    if not _require_editable(result, key, payload):
        return

    def patch(section: dict[str, Any]) -> None:
        _set_bool(section, "render", _payload_bool(payload, "render"), True)
        _set_text(section, "header_text", _payload_text(payload, "header_text"))
        _set_list(
            section,
            "projects",
            _apply_item_list(
                section.get("projects"),
                _payload_list(payload, "projects"),
                _patch_project,
            ),
        )

    _apply_section_presence(result, key, payload, patch)


def _apply_section_presence(
    result: dict[str, Any], key: str, payload: dict[str, Any], patch: Any
) -> None:
    present = _payload_bool(payload, "present")
    if not present:
        result.pop(key, None)
        return
    existing = result.get(key)
    section = copy.deepcopy(existing) if type(existing) is dict else {}
    patch(section)
    result[key] = section


def _apply_item_list(original: object, payload: list[Any], patcher: Any) -> list[Any]:
    values = list(original) if type(original) in {list, tuple} else []
    if len(payload) > _MAX_ITEMS:
        raise ResumeFieldsError("Structured resume data is invalid.")
    seen: set[int] = set()
    result: list[Any] = []
    for item_payload in payload:
        if type(item_payload) is not dict:
            raise ResumeFieldsError("Structured resume data is invalid.")
        source_index = item_payload.get("source")
        if source_index is None:
            original_item = None
        elif (
            type(source_index) is not int
            or source_index < 0
            or source_index >= len(values)
            or source_index in seen
        ):
            raise ResumeFieldsError("Structured resume data is invalid.")
        else:
            seen.add(source_index)
            original_item = values[source_index]
        result.append(patcher(original_item, item_payload))
    return result


def _patch_text_item(original: object, payload: dict[str, Any]) -> str:
    _require_kind(original, payload, ("text",))
    value = _payload_text(payload, "text")
    if original is None and not value:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return value


def _patch_link(original: object, payload: dict[str, Any]) -> dict[str, Any]:
    _require_kind(original, payload, ("mapping",))
    result = copy.deepcopy(original) if type(original) is dict else {}
    label = _payload_text(payload, "label")
    url = _payload_text(payload, "url")
    if original is None and (not label or not url):
        raise ResumeFieldsError("Structured resume data is invalid.")
    _set_text(result, "label", label)
    _set_text(result, "url", url)
    return result


def _patch_bullet(original: object, payload: dict[str, Any]) -> Any:
    kind = _require_kind(original, payload, ("text", "mapping"))
    if kind == "opaque":
        return copy.deepcopy(original)
    if kind == "text":
        value = _payload_text(payload, "text")
        if original is None and not value:
            raise ResumeFieldsError("Structured resume data is invalid.")
        return value
    result = copy.deepcopy(original) if type(original) is dict else {}
    _set_bool(result, "render", _payload_bool(payload, "render"), True)
    _set_text(result, "text", _payload_text(payload, "text"))
    _set_text(result, "bold_label", _payload_text(payload, "bold_label"))
    _set_text(result, "category", _payload_text(payload, "category"))
    if original is None and not any(
        _payload_text(payload, key) for key in ("text", "bold_label", "category")
    ):
        raise ResumeFieldsError("Structured resume data is invalid.")
    return result


def _patch_skill_item(original: object, payload: dict[str, Any]) -> Any:
    kind = _require_kind(original, payload, ("text", "mapping"))
    if kind != "mapping":
        return _patch_bullet(original, payload)
    result = copy.deepcopy(original) if type(original) is dict else {}
    _set_bool(result, "render", _payload_bool(payload, "render"), True)
    _set_text(result, "text", _payload_text(payload, "text"))
    _set_text(result, "category", _payload_text(payload, "category"))
    items_value = result.get("items")
    items = copy.deepcopy(items_value) if type(items_value) is dict else {}
    _set_list(items, "primary", _payload_string_list(payload, "primary"))
    _set_list(items, "additional", _payload_string_list(payload, "additional"))
    if type(items_value) is dict or items:
        result["items"] = items
    _set_list(
        result,
        "jod_matched_items",
        _payload_string_list(payload, "jod_matched_items"),
    )
    if original is None and not (
        _payload_text(payload, "text") or _payload_text(payload, "category") or items
    ):
        raise ResumeFieldsError("Structured resume data is invalid.")
    return result


def _patch_job(original: object, payload: dict[str, Any]) -> Any:
    kind = _require_kind(original, payload, ("mapping",))
    if kind == "opaque":
        return copy.deepcopy(original)
    result = copy.deepcopy(original) if type(original) is dict else {}
    _set_bool(result, "render", _payload_bool(payload, "render"), True)
    _set_number_or_text(result, "order", payload.get("order", ""))
    _set_optional_integer(
        result, "min_bullet_points", payload.get("min_bullet_points", "")
    )
    _set_optional_integer(
        result, "max_bullet_points", payload.get("max_bullet_points", "")
    )
    _patch_nested_texts(
        result,
        "line_1",
        payload,
        {
            "company_name_text": "company",
            "position_name_text": "role",
            "position_dates_text": "dates",
        },
    )
    _patch_nested_texts(
        result,
        "line_2",
        payload,
        {"position_intro_text": "intro"},
    )
    _set_list(
        result,
        "bullet_points",
        _apply_item_list(
            result.get("bullet_points"),
            _payload_list(payload, "bullet_points"),
            _patch_bullet,
        ),
    )
    if original is None and not any(
        _payload_text(payload, key) for key in ("company", "role", "dates", "intro")
    ):
        raise ResumeFieldsError("Structured resume data is invalid.")
    return result


def _patch_education_item(original: object, payload: dict[str, Any]) -> Any:
    kind = _require_kind(original, payload, ("text", "mapping"))
    if kind == "opaque":
        return copy.deepcopy(original)
    if kind == "text":
        return _patch_text_item(original, payload)
    result = copy.deepcopy(original) if type(original) is dict else {}
    _set_bool(result, "render", _payload_bool(payload, "render"), True)
    _set_text(result, "title", _payload_text(payload, "title"))
    _patch_nested_texts(
        result,
        "line_1",
        payload,
        {"institution_name_text": "institution"},
    )
    _patch_nested_texts(
        result,
        "line_2",
        payload,
        {"degree_name_text": "degree", "degree_dates_text": "dates"},
    )
    _set_list(
        result,
        "bullet_points",
        _apply_item_list(
            result.get("bullet_points"),
            _payload_list(payload, "bullet_points"),
            _patch_bullet,
        ),
    )
    if original is None and not any(
        _payload_text(payload, key)
        for key in ("title", "institution", "degree", "dates")
    ):
        raise ResumeFieldsError("Structured resume data is invalid.")
    return result


def _patch_project(original: object, payload: dict[str, Any]) -> Any:
    kind = _require_kind(original, payload, ("text", "mapping"))
    if kind == "opaque":
        return copy.deepcopy(original)
    if kind == "text":
        return _patch_text_item(original, payload)
    result = copy.deepcopy(original) if type(original) is dict else {}
    _set_bool(result, "render", _payload_bool(payload, "render"), True)
    title_key = "title_text" if "title_text" in result else "title"
    description_key = (
        "description_text" if "description_text" in result else "description"
    )
    _set_text(result, title_key, _payload_text(payload, "title"))
    _set_text(result, "url", _payload_text(payload, "url"))
    _set_text(result, description_key, _payload_text(payload, "description"))
    if original is None and not any(
        _payload_text(payload, key) for key in ("title", "url", "description")
    ):
        raise ResumeFieldsError("Structured resume data is invalid.")
    return result


def _patch_nested_texts(
    result: dict[str, Any],
    key: str,
    payload: dict[str, Any],
    fields: Mapping[str, str],
) -> None:
    original = result.get(key)
    nested = copy.deepcopy(original) if type(original) is dict else {}
    for target, source in fields.items():
        _set_text(nested, target, _payload_text(payload, source))
    if type(original) is dict or nested:
        result[key] = nested


def _require_editable(
    result: dict[str, Any], key: str, payload: dict[str, Any]
) -> bool:
    expected = key not in result or type(result.get(key)) is dict
    if _payload_bool(payload, "editable") is not expected:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return expected


def _require_kind(
    original: object, payload: dict[str, Any], allowed_new: Sequence[str]
) -> str:
    kind = payload.get("kind")
    expected = _kind(original)
    if original is None:
        if kind not in allowed_new:
            raise ResumeFieldsError("Structured resume data is invalid.")
        return str(kind)
    if kind != expected:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return expected


def _set_text(target: dict[str, Any], key: str, value: str) -> None:
    if key in target or value:
        target[key] = value


def _set_bool(target: dict[str, Any], key: str, value: bool, default: bool) -> None:
    if key in target or value is not default:
        target[key] = value


def _set_list(target: dict[str, Any], key: str, value: list[Any]) -> None:
    if key in target or value:
        target[key] = value


def _set_number_or_text(target: dict[str, Any], key: str, value: object) -> None:
    if type(value) is int:
        selected: int | str = value
    elif type(value) is str:
        selected = _checked_text(value)
    else:
        raise ResumeFieldsError("Structured resume data is invalid.")
    if key in target or selected != "":
        target[key] = selected


def _set_optional_integer(target: dict[str, Any], key: str, value: object) -> None:
    if value == "":
        if key in target:
            target[key] = ""
        return
    if type(value) is not int or value < 0 or value > _MAX_ITEMS:
        raise ResumeFieldsError("Structured resume data is invalid.")
    target[key] = value


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    selected = value.get(key)
    if type(selected) is not dict:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return selected


def _payload_text(value: Mapping[str, Any], key: str) -> str:
    selected = value.get(key)
    if type(selected) is not str:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return _checked_text(selected)


def _payload_bool(value: Mapping[str, Any], key: str) -> bool:
    selected = value.get(key)
    if type(selected) is not bool:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return selected


def _payload_list(value: Mapping[str, Any], key: str) -> list[Any]:
    selected = value.get(key)
    if type(selected) is not list or len(selected) > _MAX_ITEMS:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return selected


def _payload_string_list(value: Mapping[str, Any], key: str) -> list[str]:
    return [_checked_text(item) for item in _payload_list(value, key)]


def _payload_choice(value: Mapping[str, Any], key: str, choices: Sequence[str]) -> str:
    selected = value.get(key)
    if selected not in choices:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return str(selected)


def _section(source: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if type(value) is dict else {}


def _kind(value: object) -> str:
    if type(value) is str:
        return "text"
    if type(value) is dict:
        return "mapping"
    return "opaque"


def _string(value: object) -> str:
    return value if type(value) is str else ""


def _boolean(value: object, default: bool) -> bool:
    return value if type(value) is bool else default


def _string_items(value: object) -> list[str]:
    if type(value) not in {list, tuple}:
        return []
    return [_string(item) for item in value if type(item) is str][:_MAX_ITEMS]


def _checked_text(value: object) -> str:
    if type(value) is not str or len(value) > _MAX_TEXT_CHARS:
        raise ResumeFieldsError("Structured resume data is invalid.")
    return value


def _materialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_materialize(item) for item in value]
    return value


__all__ = [
    "ResumeFieldsError",
    "apply_resume_field_payload",
    "resume_field_capabilities",
    "resume_field_model",
    "resume_field_payload_text",
]
