"""Synthetic tests for pure application-resume transformations."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

import career_agent_workbench.application_resume as application_resume_module
from career_agent_workbench.application_resume import (
    CORE_SKILLS_PROMPT_JOD_MAX_CHARS,
    JOB_OPENING_DESCRIPTION_SCHEMA_VERSION,
    JOD_TARGET_PROMPT_JOD_MAX_CHARS,
    MAX_RESUME_YAML_BYTES,
    ApplicationResumeError,
    apply_core_skill_jod_matches,
    attach_job_opening_description_object,
    build_core_skills_jod_match_prompt,
    build_experience_job_bullet_rewrite_prompt,
    build_jod_requirements_target_prompt,
    create_job_opening_description_object,
    experience_job_for_jod_bullet_rewrite,
    experience_jobs_for_jod_bullet_rewrite,
    initialize_application_resume_object,
    job_opening_description_target_texts,
    replace_experience_job_bullets_from_text_response,
    reset_application_resume_jod_state,
)


def _resume() -> dict[str, Any]:
    return {
        "basics": {
            "name": "Jules Example",
            "email": "jules@example.test",
            "url": "https://portfolio.example",
        },
        "core_technical_skills": {
            "bullet_points": [
                {
                    "category": "Languages & Frameworks",
                    "items": {
                        "primary": ["Python", "FastAPI"],
                        "additional": ["PostgreSQL", "Python"],
                        "match_terms": {
                            "Python": ["CPython", "automation"],
                            "PostgreSQL": ["relational database"],
                            "Not Inventoried": ["ignored alias"],
                        },
                    },
                    "jod_matched_items": ["stale skill"],
                },
                {
                    "category": "Operations",
                    "items": {
                        "primary": ["Observability"],
                        "additional": ["Incident Response"],
                    },
                    "jod_matched_items": ["stale operation"],
                },
            ]
        },
        "professional_experience": {
            "jobs": [
                _job(
                    order=1,
                    company="Aster Forge Labs",
                    bullet="Built Python automation for fictional test systems.",
                ),
                _job(
                    order="2",
                    company="Cedar Vale Systems",
                    bullet="Improved a synthetic service using supplied evidence.",
                ),
                _job(
                    order=3,
                    company="Lantern Example Group",
                    bullet="Maintained a fictional operational dashboard.",
                    render=False,
                ),
                _job(
                    order="04",
                    company="Northwind Example Studio",
                    bullet="Documented a synthetic platform workflow.",
                ),
            ]
        },
        "job_opening_description": {
            "requirements_targets": [{"order": 1, "text": "stale target"}]
        },
    }


def _job(
    *,
    order: int | str,
    company: str,
    bullet: str,
    render: bool = True,
) -> dict[str, Any]:
    return {
        "order": order,
        "render": render,
        "min_bullet_points": 1,
        "max_bullet_points": 3,
        "line_1": {
            "company_name_text": company,
            "position_name_text": "Synthetic Systems Builder",
            "position_dates_text": "2024",
        },
        "bullet_points": [
            {
                "order": 1,
                "text": bullet,
                "render": True,
                "bullet_point_total_match_count": 7,
                "categories": {
                    "assigned": ["fictional-source"],
                    "matched": ["stale-target"],
                },
                "skills": [
                    {
                        "name": "Python",
                        "jod_match_count": 4,
                    }
                ],
            }
        ],
    }


def _assert_sanitized(error: ApplicationResumeError, *secrets: str) -> None:
    state = " ".join((str(error), repr(error), repr(error.args)))
    for secret in secrets:
        assert secret not in state
    assert error.__cause__ is None
    assert error.__context__ is None


def test_initialize_exact_path_returns_defensive_reset_copy(tmp_path: Path) -> None:
    source = _resume()
    yaml_path = tmp_path / "fictional-resume.yml"
    yaml_path.write_text(json.dumps(source), encoding="utf-8")

    initialized = initialize_application_resume_object(yaml_path)

    assert "job_opening_description" not in initialized
    assert [
        bucket["jod_matched_items"]
        for bucket in initialized["core_technical_skills"]["bullet_points"]
    ] == [[], []]
    for job in initialized["professional_experience"]["jobs"]:
        for bullet in job["bullet_points"]:
            assert bullet["bullet_point_total_match_count"] == 0
            assert bullet["categories"] == {
                "assigned": ["fictional-source"],
                "matched": [],
            }
            assert [skill["jod_match_count"] for skill in bullet["skills"]] == [0]

    initialized["basics"]["name"] = "Changed"
    initialized["professional_experience"]["jobs"][0]["bullet_points"][0]["text"] = (
        "Changed"
    )
    assert source["basics"]["name"] == "Jules Example"
    assert (
        source["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
        == "Built Python automation for fictional test systems."
    )


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("missing-secret.yml", None),
        ("malformed-secret.yml", "private-marker: [unterminated"),
        ("scalar-secret.yml", "private-marker"),
        (
            "unsafe-secret.yml",
            "!!python/object/apply:builtins.str [private-marker]",
        ),
    ],
)
def test_yaml_failures_are_sanitized(
    tmp_path: Path,
    filename: str,
    content: str | None,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level("DEBUG")
    yaml_path = tmp_path / filename
    if content is not None:
        yaml_path.write_text(content, encoding="utf-8")

    with pytest.raises(ApplicationResumeError) as error_info:
        initialize_application_resume_object(yaml_path)

    _assert_sanitized(
        error_info.value,
        str(yaml_path),
        filename,
        "private-marker",
        "unterminated",
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert caplog.records == []


def test_oversized_yaml_fails_before_parsing_with_sanitized_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml_path = tmp_path / "oversized-private-marker.yml"
    yaml_path.write_bytes(b"x" * (MAX_RESUME_YAML_BYTES + 1))

    safe_load_calls: list[object] = []

    def record_parse(value: object) -> object:
        safe_load_calls.append(value)
        return {}

    monkeypatch.setattr(application_resume_module.yaml, "safe_load", record_parse)

    with pytest.raises(ApplicationResumeError) as error_info:
        initialize_application_resume_object(yaml_path)

    _assert_sanitized(
        error_info.value,
        str(yaml_path),
        "oversized-private-marker",
    )
    assert safe_load_calls == []


def test_fresh_import_does_not_discover_filesystem_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "career_agent_workbench.application_resume"
    package = sys.modules["career_agent_workbench"]
    original_module = sys.modules.pop(module_name)
    original_package_attribute = package.application_resume

    def unexpected_access(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    try:
        with monkeypatch.context() as import_guard:
            import_guard.setattr(Path, "open", unexpected_access)
            import_guard.setattr(Path, "stat", unexpected_access)
            import_guard.setattr(Path, "exists", unexpected_access)
            import_guard.setattr(Path, "read_text", unexpected_access)
            import_guard.setattr(Path, "read_bytes", unexpected_access)
            import_guard.setattr(Path, "home", unexpected_access)
            import_guard.setattr(Path, "cwd", unexpected_access)
            import_guard.setattr(os, "getenv", unexpected_access)
            fresh_module = importlib.import_module(module_name)
        assert (
            fresh_module.JOB_OPENING_DESCRIPTION_SCHEMA_VERSION
            == JOB_OPENING_DESCRIPTION_SCHEMA_VERSION
        )
    finally:
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = original_module
        package.application_resume = original_package_attribute


def test_initializer_requires_a_path_without_coercing_caller_input() -> None:
    marker = "caller-secret-path"

    with pytest.raises(ApplicationResumeError) as error_info:
        initialize_application_resume_object(marker)  # type: ignore[arg-type]

    _assert_sanitized(error_info.value, marker)


def test_pure_mapping_operations_do_not_access_paths_or_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _resume()

    def unexpected_access(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(Path, "open", unexpected_access)
    monkeypatch.setattr(os, "getenv", unexpected_access)

    reset = reset_application_resume_jod_state(source)
    matched = apply_core_skill_jod_matches(
        application_resume=reset,
        core_skill_response={
            "core_technical_skills": [
                {
                    "category": "Languages & Frameworks",
                    "jod_matched_items": ["Python"],
                }
            ]
        },
    )
    jod = create_job_opening_description_object(
        trimmed_job_description="Build a fictional Python service.",
        requirements_response=["Build a Python service."],
    )
    attached = attach_job_opening_description_object(
        application_resume=matched,
        job_opening_description=jod,
    )

    assert experience_job_for_jod_bullet_rewrite(attached, job_order=1)["order"] == 1


def test_skill_matches_are_inventory_bound_ordered_and_deduplicated() -> None:
    source = _resume()
    items = source["core_technical_skills"]["bullet_points"][0]["items"]
    items["primary"] = ["C", "C++", "C#", "Python"]
    items["additional"] = ["PostgreSQL"]
    items["match_terms"] = {
        "C": ["c language"],
        "C++": ["c plus plus"],
        "C#": ["c sharp"],
        "Python": ["CPython"],
        "PostgreSQL": ["relational database"],
    }
    response = {
        "core_technical_skills": [
            {
                "category": "Languages & Frameworks",
                "jod_matched_items": [
                    "C#",
                    "C",
                    "relational database",
                    "PostgreSQL",
                    "C++",
                    "c plus plus",
                    "PYTHON",
                    "Python",
                    "invented skill",
                ],
            },
            {
                "category": "Operations",
                "jod_matched_items": ["incident response", "unlisted"],
            },
        ]
    }

    result = apply_core_skill_jod_matches(
        application_resume=source,
        core_skill_response=json.dumps(response),
    )

    buckets = result["core_technical_skills"]["bullet_points"]
    assert buckets[0]["jod_matched_items"] == [
        "C",
        "C++",
        "C#",
        "Python",
        "PostgreSQL",
    ]
    assert buckets[1]["jod_matched_items"] == ["Incident Response"]
    assert source["core_technical_skills"]["bullet_points"][0]["jod_matched_items"] == [
        "stale skill"
    ]

    alias_only = apply_core_skill_jod_matches(
        application_resume=source,
        core_skill_response={
            "core_technical_skills": [
                {
                    "category": "Languages & Frameworks",
                    "jod_matched_items": [
                        "c plus plus",
                        "c sharp",
                        "relational database",
                    ],
                }
            ]
        },
    )
    assert (
        alias_only["core_technical_skills"]["bullet_points"][0]["jod_matched_items"]
        == []
    )

    malformed = apply_core_skill_jod_matches(
        application_resume=source,
        core_skill_response="not valid JSON",
    )
    assert (
        malformed["core_technical_skills"]["bullet_points"][0]["jod_matched_items"]
        == []
    )


def test_prompts_apply_exact_existing_job_text_bounds() -> None:
    skill_job_text = "S" * (CORE_SKILLS_PROMPT_JOD_MAX_CHARS + 17)
    target_job_text = "T" * (JOD_TARGET_PROMPT_JOD_MAX_CHARS + 17)

    skill_prompt = build_core_skills_jod_match_prompt(
        application_resume=_resume(),
        trimmed_job_description=skill_job_text,
    )
    target_prompt = build_jod_requirements_target_prompt(
        trimmed_job_description=target_job_text
    )

    assert "S" * CORE_SKILLS_PROMPT_JOD_MAX_CHARS in skill_prompt
    assert "S" * (CORE_SKILLS_PROMPT_JOD_MAX_CHARS + 1) not in skill_prompt
    assert "T" * JOD_TARGET_PROMPT_JOD_MAX_CHARS in target_prompt
    assert "T" * (JOD_TARGET_PROMPT_JOD_MAX_CHARS + 1) not in target_prompt
    assert skill_prompt.endswith("[truncated]")
    assert target_prompt.endswith("[truncated]")

    caller_requested_unbounded = build_core_skills_jod_match_prompt(
        application_resume=_resume(),
        trimmed_job_description=skill_job_text,
        max_jod_chars=CORE_SKILLS_PROMPT_JOD_MAX_CHARS * 10,
    )
    assert "S" * CORE_SKILLS_PROMPT_JOD_MAX_CHARS in caller_requested_unbounded
    assert (
        "S" * (CORE_SKILLS_PROMPT_JOD_MAX_CHARS + 1) not in caller_requested_unbounded
    )


def test_jod_object_schema_targets_and_attachment_are_deterministic() -> None:
    response = {
        "job_opening_description": {
            "requirements_targets": [
                {"text": "Build fictional automation."},
                {"target": "Support synthetic services."},
                {"text": "build fictional automation"},
                {"text": ""},
            ]
        }
    }

    jod = create_job_opening_description_object(
        trimmed_job_description="  Fictional role text.  ",
        requirements_response=response,
    )

    assert jod == {
        "schema_version": JOB_OPENING_DESCRIPTION_SCHEMA_VERSION,
        "source": {
            "type": "trimmed_job_description",
            "character_count": 20,
        },
        "llm": {},
        "requirements_targets": [
            {"order": 1, "text": "Build fictional automation."},
            {"order": 2, "text": "Support synthetic services."},
        ],
    }
    assert job_opening_description_target_texts(jod) == [
        "Build fictional automation.",
        "Support synthetic services.",
    ]

    source = _resume()
    attached = attach_job_opening_description_object(
        application_resume=source,
        job_opening_description=jod,
    )
    jod["requirements_targets"][0]["text"] = "Changed"
    attached["basics"]["name"] = "Changed"
    assert (
        attached["job_opening_description"]["requirements_targets"][0]["text"]
        == "Build fictional automation."
    )
    assert source["basics"]["name"] == "Jules Example"


def test_jod_target_plain_text_parsing_and_explicit_model_metadata() -> None:
    jod = create_job_opening_description_object(
        trimmed_job_description="Synthetic description",
        requirements_response="""
        - Build deterministic automation.
        2. Document test behavior.
        - build deterministic automation
        """,
        model=" caller-supplied-model ",
    )

    assert jod["llm"] == {"model": "caller-supplied-model"}
    assert job_opening_description_target_texts(jod) == [
        "Build deterministic automation.",
        "Document test behavior.",
    ]


def test_generic_job_selection_has_no_implicit_special_order() -> None:
    source = _resume()

    assert [job["order"] for job in experience_jobs_for_jod_bullet_rewrite(source)] == [
        1,
        "2",
        "04",
    ]
    assert [
        job["order"]
        for job in experience_jobs_for_jod_bullet_rewrite(
            source,
            include_orders=["04", 1, "2"],
            exclude_orders=("2", 1),
        )
    ] == ["04"]
    assert experience_jobs_for_jod_bullet_rewrite(source, include_orders=[]) == []
    assert (
        experience_job_for_jod_bullet_rewrite(source, job_order="01")["line_1"][
            "company_name_text"
        ]
        == "Aster Forge Labs"
    )

    selected = experience_job_for_jod_bullet_rewrite(source, job_order=1)
    selected["line_1"]["company_name_text"] = "Changed"
    assert (
        source["professional_experience"]["jobs"][0]["line_1"]["company_name_text"]
        == "Aster Forge Labs"
    )


@pytest.mark.parametrize(
    "invalid_order",
    [
        True,
        False,
        0,
        -1,
        "0",
        "-1",
        "+1",
        " 1",
        "1 ",
        "1.0",
        "\u0661",
        1.0,
        None,
        object(),
    ],
)
def test_invalid_caller_orders_fail_with_content_free_error(
    invalid_order: object,
) -> None:
    with pytest.raises(ApplicationResumeError) as error_info:
        experience_job_for_jod_bullet_rewrite(
            _resume(),
            job_order=invalid_order,
        )

    assert str(error_info.value) == "Experience job order is invalid."
    assert error_info.value.__cause__ is None
    assert error_info.value.__context__ is None


@pytest.mark.parametrize(
    "invalid_orders",
    [
        "1",
        b"1",
        [True],
        [0],
        [" 1"],
        ["\u0661"],
        [object()],
    ],
)
def test_invalid_order_sequences_fail_closed(invalid_orders: object) -> None:
    with pytest.raises(ApplicationResumeError, match="order is invalid"):
        experience_jobs_for_jod_bullet_rewrite(
            _resume(),
            include_orders=invalid_orders,  # type: ignore[arg-type]
        )


def test_malformed_rendered_resume_order_fails_closed() -> None:
    source = _resume()
    source["professional_experience"]["jobs"][0]["order"] = True

    with pytest.raises(ApplicationResumeError, match="order is invalid"):
        experience_jobs_for_jod_bullet_rewrite(source)


def test_excessive_decimal_orders_are_sanitized_at_every_boundary() -> None:
    excessive_order = "9" * 5_000

    calls = (
        lambda: experience_job_for_jod_bullet_rewrite(
            _resume(),
            job_order=excessive_order,
        ),
        lambda: experience_jobs_for_jod_bullet_rewrite(
            _resume(),
            include_orders=[excessive_order],
        ),
        lambda: experience_jobs_for_jod_bullet_rewrite(
            _resume(),
            exclude_orders=[excessive_order],
        ),
        lambda: replace_experience_job_bullets_from_text_response(
            application_resume=_resume(),
            job_order=excessive_order,
            bullet_response="- Fictional bullet.",
        ),
    )
    for call in calls:
        with pytest.raises(ApplicationResumeError) as error_info:
            call()
        assert str(error_info.value) == "Experience job order is invalid."
        assert error_info.value.__cause__ is None
        assert error_info.value.__context__ is None

    source = _resume()
    source["professional_experience"]["jobs"][0]["order"] = excessive_order
    with pytest.raises(ApplicationResumeError) as stored_error:
        experience_jobs_for_jod_bullet_rewrite(source)
    assert str(stored_error.value) == "Experience job order is invalid."
    assert stored_error.value.__cause__ is None
    assert stored_error.value.__context__ is None


def test_single_job_selection_rejects_non_rendered_and_missing_jobs() -> None:
    with pytest.raises(ApplicationResumeError) as disabled:
        experience_job_for_jod_bullet_rewrite(_resume(), job_order=3)
    with pytest.raises(ApplicationResumeError) as missing:
        experience_job_for_jod_bullet_rewrite(_resume(), job_order=99)

    assert str(disabled.value) == "Experience job is not enabled for rendering."
    assert str(missing.value) == "Experience job was not found."


def test_bullet_prompt_contains_only_supplied_fictional_evidence() -> None:
    source = _resume()
    job = experience_job_for_jod_bullet_rewrite(source, job_order=1)
    jod = create_job_opening_description_object(
        trimmed_job_description="Build test automation.",
        requirements_response=["Build test automation."],
    )

    prompt = build_experience_job_bullet_rewrite_prompt(
        job_opening_description=jod,
        job=job,
    )

    assert "Aster Forge Labs" in prompt
    assert "Built Python automation for fictional test systems." in prompt
    assert "Build test automation." in prompt
    assert "Do not invent" in prompt


def test_bullet_replacement_enforces_bounds_and_defensive_copy() -> None:
    source = _resume()
    source["professional_experience"]["jobs"][1]["min_bullet_points"] = 2
    source["professional_experience"]["jobs"][1]["max_bullet_points"] = 2
    original = copy.deepcopy(source)

    result = replace_experience_job_bullets_from_text_response(
        application_resume=source,
        job_order="02",
        bullet_response="""
        - Built a fictional service from supplied evidence.
        2. Documented synthetic outcomes without adding claims.
        """,
    )

    bullets = result["professional_experience"]["jobs"][1]["bullet_points"]
    assert [bullet["text"] for bullet in bullets] == [
        "Built a fictional service from supplied evidence.",
        "Documented synthetic outcomes without adding claims.",
    ]
    assert [bullet["order"] for bullet in bullets] == [1, 2]
    assert all(
        bullet["categories"] == {"assigned": [], "matched": []}
        and bullet["skills"] == []
        and bullet["bullet_point_total_match_count"] == 0
        and bullet["render"] is True
        for bullet in bullets
    )
    assert source == original
    bullets[0]["text"] = "Changed"
    assert source == original

    for response in (
        "- Only one fictional bullet.",
        "- One.\n- Two.\n- Three.",
    ):
        with pytest.raises(ApplicationResumeError) as error_info:
            replace_experience_job_bullets_from_text_response(
                application_resume=source,
                job_order=2,
                bullet_response=response,
            )
        assert (
            str(error_info.value)
            == "Generated bullet count is outside the configured bounds."
        )


def test_bullet_replacement_reuses_strict_order_and_render_rules() -> None:
    for order, expected in (
        (True, "Experience job order is invalid."),
        (3, "Experience job is not enabled for rendering."),
        (99, "Experience job was not found."),
    ):
        with pytest.raises(ApplicationResumeError) as error_info:
            replace_experience_job_bullets_from_text_response(
                application_resume=_resume(),
                job_order=order,
                bullet_response="- Fictional bullet.",
            )
        assert str(error_info.value) == expected


def test_public_surface_has_no_layout_or_model_default() -> None:
    public_source = inspect.getsource(application_resume_module).casefold()
    prohibited_fragments = (
        "default_master",
        "default_jod_llm",
        "repository-relative",
        "hidden special",
    )

    assert all(fragment not in public_source for fragment in prohibited_fragments)
