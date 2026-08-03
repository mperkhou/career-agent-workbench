from __future__ import annotations

import copy
import json

import pytest

from career_agent_workbench.webapp_resume_fields import (
    ResumeFieldsError,
    apply_resume_field_payload,
    resume_field_capabilities,
    resume_field_model,
    resume_field_payload_text,
)


def _complete_resume() -> dict[str, object]:
    return {
        "resume_layout": {
            "supporting_sections_start_on_page_2": True,
            "unknown_layout": "preserve-layout",
        },
        "header_top": {
            "render": True,
            "line_1_name_header_text": "Jules Example",
            "line_2_header_text": "Synthetic systems engineer",
            "line_2_applicant_info_text": "Earlier contact fallback",
            "line_3_applicant_info_text": "Example City",
            "contact_items": ["jules@example.test", "Portfolio"],
            "links": [
                {
                    "label": "Portfolio",
                    "url": "https://portfolio.example.test",
                    "tracking_disabled": True,
                }
            ],
            "unknown_header": {"preserve": True},
        },
        "professional_summary": {
            "render": True,
            "header_text": "Summary",
            "paragraph": "Builds <strong>fictional systems</strong>.",
            "summary_note": "Synthetic evidence only.",
            "unknown_summary": ["preserve"],
        },
        "core_technical_skills": {
            "render": True,
            "header_text": "Skills",
            "bullet_points": [
                {
                    "category": "Languages",
                    "items": {
                        "primary": ["Python", "SQL"],
                        "additional": ["Rust"],
                        "match_terms": {"Rust": ["rustlang"]},
                    },
                    "jod_matched_items": ["Rust"],
                    "unknown_skill": "preserve-skill",
                },
                "Reliable delivery",
            ],
        },
        "professional_experience": {
            "render": True,
            "header_text": "Experience",
            "jobs": [
                {
                    "order": "01",
                    "render": True,
                    "min_bullet_points": 1,
                    "max_bullet_points": 4,
                    "line_1": {
                        "company_name_text": "Example Systems",
                        "position_name_text": "Engineer",
                        "position_dates_text": "2024–Present",
                        "unknown_line": "preserve-line",
                    },
                    "line_2": {"position_intro_text": "Fictional work."},
                    "bullet_points": [
                        {
                            "render": True,
                            "bold_label": "Impact",
                            "text": "Improved a synthetic workflow.",
                            "evidence_ids": ["example-1"],
                        },
                        "Documented a bounded process.",
                    ],
                    "unknown_job": {"preserve": True},
                }
            ],
        },
        "education_and_certifications": {
            "render": True,
            "header_text": "Education",
            "items": [
                {
                    "line_1": {
                        "institution_name_text": "Example Institute",
                    },
                    "line_2": {
                        "degree_name_text": "B.S. Example Studies",
                        "degree_dates_text": "2020",
                    },
                    "bullet_points": ["Synthetic honors"],
                    "unknown_education": "preserve-education",
                }
            ],
        },
        "certifications": {
            "render": False,
            "header_text": "Certifications",
            "bullet_points": [{"text": "Example Certificate", "issuer": "Example Org"}],
        },
        "portfolio": {
            "header_text": "Portfolio",
            "projects": [
                {
                    "title_text": "Example Project",
                    "url": "https://project.example.test",
                    "description_text": "A fictional demonstration.",
                    "unknown_project": 7,
                }
            ],
        },
        "optional_unknown_section": {
            "nested": [1, {"private_free": "preserve exactly"}]
        },
    }


def _payload(resume: dict[str, object]) -> dict[str, object]:
    return json.loads(resume_field_payload_text(resume))


def test_structured_noop_round_trip_is_complete_and_lossless() -> None:
    resume = _complete_resume()
    before = copy.deepcopy(resume)

    model = resume_field_model(resume)
    result = apply_resume_field_payload(resume, json.dumps(model))

    assert result == before
    assert resume == before
    assert model["education"]["root_key"] == "education_and_certifications"
    assert model["education"]["items_key"] == "items"


def test_structured_capabilities_are_content_free_and_complete() -> None:
    capabilities = resume_field_capabilities()

    assert set(capabilities["sections"]) == {
        "resume_layout",
        "header_top",
        "professional_summary",
        "core_technical_skills",
        "professional_experience",
        "education",
        "certifications",
        "portfolio",
    }
    assert capabilities["repeated_operations"] == (
        "add",
        "remove",
        "reorder",
        "validate",
    )
    assert "content" not in json.dumps(capabilities).casefold()


def test_structured_repeated_lists_add_remove_reorder_and_preserve_unknowns() -> None:
    resume = _complete_resume()
    payload = _payload(resume)
    header = payload["header_top"]
    header["contact_items"] = [
        header["contact_items"][1],
        {"source": None, "kind": "text", "text": "example.test/profile"},
    ]
    links = header["links"]
    links.append(
        {
            "source": None,
            "kind": "mapping",
            "label": "Profile",
            "url": "https://example.test/profile",
        }
    )
    skills = payload["core_technical_skills"]["bullet_points"]
    skills[0]["primary"] = ["SQL", "Python"]
    skills[:] = [skills[1], skills[0]]
    jobs = payload["professional_experience"]["jobs"]
    jobs[0]["role"] = "Senior Engineer"
    bullets = jobs[0]["bullet_points"]
    bullets[:] = [
        bullets[1],
        bullets[0],
        {
            "source": None,
            "kind": "mapping",
            "render": True,
            "text": "Added bounded evidence.",
            "bold_label": "Delivery",
            "category": "",
        },
    ]

    result = apply_resume_field_payload(resume, json.dumps(payload))

    assert result["header_top"]["contact_items"] == [
        "Portfolio",
        "example.test/profile",
    ]
    assert result["header_top"]["links"][0]["tracking_disabled"] is True
    assert (
        result["core_technical_skills"]["bullet_points"][1]["unknown_skill"]
        == "preserve-skill"
    )
    assert result["core_technical_skills"]["bullet_points"][1]["items"][
        "match_terms"
    ] == {"Rust": ["rustlang"]}
    job = result["professional_experience"]["jobs"][0]
    assert job["line_1"]["position_name_text"] == "Senior Engineer"
    assert job["line_1"]["unknown_line"] == "preserve-line"
    assert job["unknown_job"] == {"preserve": True}
    assert job["bullet_points"][1]["evidence_ids"] == ["example-1"]
    assert job["bullet_points"][2]["bold_label"] == "Delivery"
    assert result["optional_unknown_section"] == resume["optional_unknown_section"]


def test_structured_optional_sections_and_aliases_are_explicit() -> None:
    resume = _complete_resume()
    payload = _payload(resume)
    payload["certifications"]["present"] = False
    payload["portfolio"]["projects"] = []
    payload["professional_summary"]["paragraph"] = "Updated rich <b>text</b>."

    result = apply_resume_field_payload(resume, json.dumps(payload))

    assert "certifications" not in result
    assert result["portfolio"]["projects"] == []
    assert result["professional_summary"]["paragraph"] == "Updated rich <b>text</b>."
    assert "education" not in result
    assert "education_and_certifications" in result


def test_structured_save_preserves_opaque_optional_section_values() -> None:
    resume = _complete_resume()
    resume["portfolio"] = "opaque optional representation"
    payload = _payload(resume)
    payload["professional_summary"]["paragraph"] = "Edited elsewhere."

    result = apply_resume_field_payload(resume, json.dumps(payload))

    assert result["portfolio"] == "opaque optional representation"
    assert result["professional_summary"]["paragraph"] == "Edited elsewhere."


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(version=2),
        lambda payload: payload["header_top"]["contact_items"].append(
            {"source": 0, "kind": "text", "text": "duplicate"}
        ),
        lambda payload: payload["professional_experience"]["jobs"][0].update(source=99),
        lambda payload: payload["education"].update(root_key="education"),
        lambda payload: payload["portfolio"]["projects"].append(
            {
                "source": None,
                "kind": "mapping",
                "render": True,
                "text": "",
                "title": "",
                "url": "",
                "description": "",
            }
        ),
    ],
)
def test_structured_invalid_inputs_fail_without_mutating_source(mutate) -> None:
    resume = _complete_resume()
    before = copy.deepcopy(resume)
    payload = _payload(resume)
    mutate(payload)

    with pytest.raises(ResumeFieldsError, match="Structured resume data is invalid"):
        apply_resume_field_payload(resume, json.dumps(payload))

    assert resume == before
