"""Synthetic regressions for the evidence-grounded v1-to-v2 workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

import career_agent_workbench.resume_refinement as refinement
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateConfigurationError,
    ApplicationStateNotFoundError,
    ApplicationStateStore,
    ApplicationWorkflowSnapshot,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.ats import (
    AtsComponentScores,
    AtsDiagnostics,
    AtsProxyScore,
)
from career_agent_workbench.codex_cli import (
    CodexModelConfig,
    ModelRequest,
    ModelResult,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.resume_refinement import (
    RESUME_PATCH_SCHEMA_VERSION,
    ResumeEvidenceError,
    ResumeEvidenceItem,
    ResumeEvidenceSnapshot,
    ResumePatch,
    ResumePatchError,
    ResumePatchResponse,
    ResumePatchTarget,
    ResumeRefinementError,
    ResumeWorkflowConflictError,
    build_resume_patch_prompt,
    collect_resume_patch_targets,
    parse_resume_patch_response,
    read_resume_evidence_snapshot,
    refine_resume_for_job,
    validate_and_apply_resume_patches,
)

JOB_ID = "synthetic-job-1"
COMPANY = "Example Research Cooperative"
ROLE = "Synthetic Systems Analyst"
OTHER_COMPANY = "Fictional Harbor Laboratory"
OTHER_ROLE = "Example Platform Builder"
CURRENT_BULLET = "Built Python automation for synthetic services."
PROPOSED_BULLET = "Built reliable Python automation for synthetic services."
FRONTED_BULLET = "For synthetic services, built reliable Python automation."
CURRENT_SUMMARY = "Builds reliable tools for fictional teams."
PROPOSED_SUMMARY = "Reliable Python automation for synthetic services."


class _FakeRunner:
    def __init__(self, response: str, callback: Any = None) -> None:
        self.response = response
        self.callback = callback
        self.requests: list[ModelRequest] = []

    def run(self, request: ModelRequest, /) -> ModelResult:
        self.requests.append(request)
        if self.callback is not None:
            self.callback()
        return ModelResult(
            response=self.response,
            model_metadata={
                "workflow": request.config.workflow,
                "model": request.config.model,
                "reasoning_effort": request.config.reasoning_effort,
                "attempt": 1,
                "timestamp": "2042-04-05T06:07:08+00:00",
                "version": 1,
            },
        )


def _resume(*, bullet: str = CURRENT_BULLET) -> dict[str, Any]:
    return {
        "professional_summary": {"paragraph": CURRENT_SUMMARY},
        "professional_experience": {
            "jobs": [
                {
                    "order": "1",
                    "line_1": {
                        "company_name_text": COMPANY,
                        "position_name_text": ROLE,
                        "position_dates_text": "2040–2042",
                    },
                    "bullet_points": [
                        {
                            "order": "1",
                            "text": bullet,
                            "render": True,
                        }
                    ],
                    "render": True,
                }
            ]
        },
    }


def _master_resume() -> dict[str, Any]:
    value = _resume()
    value["professional_experience"]["jobs"][0]["bullet_points"][0]["text"] = (
        PROPOSED_BULLET
    )
    value["professional_experience"]["jobs"].append(
        {
            "order": "2",
            "line_1": {
                "company_name_text": OTHER_COMPANY,
                "position_name_text": OTHER_ROLE,
            },
            "bullet_points": [
                {
                    "order": "1",
                    "text": "Led Kubernetes work that reduced latency by 40%.",
                }
            ],
        }
    )
    return value


def _paths(tmp_path: Path, *, name: str = "one") -> WorkspacePaths:
    root = tmp_path / name
    root.mkdir()
    output = root / "output"
    temporary = root / "tmp"
    output.mkdir()
    temporary.mkdir()
    master = root / "MASTER-RESUME.yml"
    source = root / "MASTER-RESUME.txt"
    master.write_text(
        yaml.safe_dump(_master_resume(), sort_keys=False),
        encoding="utf-8",
    )
    source.write_text(
        "Corroborating synthetic resume source for the exact fictional roles.",
        encoding="utf-8",
    )
    return WorkspacePaths(
        database=(root / "applications.sqlite3").absolute(),
        master_resume=master.absolute(),
        master_resume_text=source.absolute(),
        output_dir=output.absolute(),
        tmp_dir=temporary.absolute(),
    )


def _store_with_v1(paths: WorkspacePaths) -> ApplicationStateStore:
    store = ApplicationStateStore(paths)
    store.initialize()
    store.seed_application(
        ApplicationMetadata(
            job_id=JOB_ID,
            company=COMPANY,
            job_title="Example Public Role",
            job_url=f"https://jobs.example.com/{JOB_ID}",
            source="synthetic",
        ),
        source_text="Full fictional job description asking for reliable automation.",
        prompt_text="Reliable Python automation.",
    )
    store.upsert_resume_variant(
        JOB_ID,
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Synthetic first draft",
            source="synthetic_test",
            application_resume_yaml=yaml.safe_dump(_resume(), sort_keys=False),
            resume_html="<main>Synthetic draft</main>",
            resume_pdf=b"synthetic-pdf-v1",
            ats=AtsFields(
                score=70,
                parsing_score=80,
                keyword_score=70,
                semantic_score=60,
                formatting_risk="low",
            ),
            evidence_packet={"seed": True},
            validation={"valid": True},
            model_metadata={"workflow": "synthetic"},
        ),
    )
    return store


def _diagnostics(score: int = 87) -> AtsDiagnostics:
    proxy = AtsProxyScore(
        overall_score=score,
        parsing_score=92,
        keyword_match_score=84,
        semantic_match_score=85,
        formatting_risk="low",
        missing_high_value_terms=("fictional-term",),
    )
    components = AtsComponentScores(
        overall_score=score,
        parsing_score=92,
        keyword_match_score=84,
        semantic_match_score=85,
        formatting_score=90,
        formatting_risk="low",
    )
    return AtsDiagnostics(
        score=proxy,
        component_scores=components,
        matched_terms=(),
        unmatched_weighted_terms=(),
        repeated_phrase_terms=(),
        likely_noisy_phrase_matches=(),
    )


def _patch_response(
    *,
    proposed: str = PROPOSED_BULLET,
    evidence_ref: str = "mro:job:1:bullet:1",
    change_id: str = "change-1",
) -> str:
    return json.dumps(
        {
            "schema_version": RESUME_PATCH_SCHEMA_VERSION,
            "changes": [
                {
                    "change_id": change_id,
                    "operation": "rewrite_bullet",
                    "target": {
                        "section": "professional_experience",
                        "field": "text",
                        "job_order": "1",
                        "bullet_order": "1",
                    },
                    "current_text": CURRENT_BULLET,
                    "proposed_text": proposed,
                    "rationale": "Uses canonical same-role evidence.",
                    "evidence_refs": [evidence_ref],
                }
            ],
        },
        separators=(",", ":"),
    )


def _summary_and_bullet_patch_response() -> str:
    evidence_refs = ["mro:job:1:bullet:1"]
    return json.dumps(
        {
            "schema_version": RESUME_PATCH_SCHEMA_VERSION,
            "changes": [
                {
                    "change_id": "change-summary",
                    "operation": "rewrite_summary",
                    "target": {
                        "section": "professional_summary",
                        "field": "paragraph",
                        "job_order": None,
                        "bullet_order": None,
                    },
                    "current_text": CURRENT_SUMMARY,
                    "proposed_text": PROPOSED_SUMMARY,
                    "rationale": "Uses canonical same-role evidence.",
                    "evidence_refs": evidence_refs,
                },
                {
                    "change_id": "change-bullet",
                    "operation": "rewrite_bullet",
                    "target": {
                        "section": "professional_experience",
                        "field": "text",
                        "job_order": "1",
                        "bullet_order": "1",
                    },
                    "current_text": CURRENT_BULLET,
                    "proposed_text": PROPOSED_BULLET,
                    "rationale": "Uses canonical same-role evidence.",
                    "evidence_refs": evidence_refs,
                },
            ],
        },
        separators=(",", ":"),
    )


def _fronted_patch_response(
    *,
    summary: bool,
    proposed: str = FRONTED_BULLET,
) -> str:
    return json.dumps(
        {
            "schema_version": RESUME_PATCH_SCHEMA_VERSION,
            "changes": [
                {
                    "change_id": (
                        "change-fronted-summary" if summary else "change-fronted-bullet"
                    ),
                    "operation": "rewrite_summary" if summary else "rewrite_bullet",
                    "target": {
                        "section": (
                            "professional_summary"
                            if summary
                            else "professional_experience"
                        ),
                        "field": "paragraph" if summary else "text",
                        "job_order": None if summary else "1",
                        "bullet_order": None if summary else "1",
                    },
                    "current_text": CURRENT_SUMMARY if summary else CURRENT_BULLET,
                    "proposed_text": proposed,
                    "rationale": "Moves one intact canonical adjunct.",
                    "evidence_refs": ["mro:job:1:bullet:1"],
                }
            ],
        },
        separators=(",", ":"),
    )


def _config(workflow: str = "refinement") -> CodexModelConfig:
    return CodexModelConfig(
        model="synthetic-model",
        reasoning_effort="high",
        workflow=workflow,
    )


@pytest.fixture
def fake_rendering(monkeypatch: pytest.MonkeyPatch) -> AtsDiagnostics:
    diagnostics = _diagnostics()
    monkeypatch.setattr(
        refinement,
        "render_resume_html_from_mapping",
        lambda *, resume: "<main>bounded synthetic output</main>",
    )
    monkeypatch.setattr(
        refinement,
        "render_resume_pdf_from_html",
        lambda html: b"synthetic-rendered-pdf",
    )
    monkeypatch.setattr(
        refinement,
        "calculate_ats_diagnostics",
        lambda *, resume_pdf, job_description: diagnostics,
    )
    return diagnostics


def test_refinement_commit_derives_v2_from_v1_with_structured_audit(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)
    runner = _FakeRunner(_patch_response())

    result = refine_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=runner,
        model_config=_config(),
        dry_run=False,
    )

    v1 = store.get_resume_variant(JOB_ID, "v1")
    v2 = store.get_resume_variant(JOB_ID, "v2")
    assert result.stored_variant == "v2"
    assert result.requires_human_review is True
    assert (
        v1.application_resume["professional_experience"]["jobs"][0]["bullet_points"][0][
            "text"
        ]
        == CURRENT_BULLET
    )
    assert (
        v2.application_resume["professional_experience"]["jobs"][0]["bullet_points"][0][
            "text"
        ]
        == PROPOSED_BULLET
    )
    assert v2.parent_variant_key == "v1"
    assert v2.critique_prompt is None
    assert v2.critique_response is None
    assert v2.validation["requires_human_review"] is True
    assert v2.model_metadata["review_state"] == "awaiting_user_review"
    assert len(runner.requests) == 1
    assert "Kubernetes work" in runner.requests[0].prompt
    assert "corroborating synthetic" in runner.requests[0].prompt.casefold()
    assert '"role_id":"mro:job:1"' in runner.requests[0].prompt


def test_supported_summary_and_bullet_paraphrases_persist_complete_provenance(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)

    result = refine_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_summary_and_bullet_patch_response()),
        model_config=_config(),
    )

    v1 = store.get_resume_variant(JOB_ID, "v1")
    v2 = store.get_resume_variant(JOB_ID, "v2")
    assert result.changed_count == 2
    assert result.requires_human_review is True
    assert v1.application_resume["professional_summary"]["paragraph"] == CURRENT_SUMMARY
    assert (
        v2.application_resume["professional_summary"]["paragraph"] == PROPOSED_SUMMARY
    )
    assert (
        v2.application_resume["professional_experience"]["jobs"][0]["bullet_points"][0][
            "text"
        ]
        == PROPOSED_BULLET
    )
    assert v2.resume_html == "<main>bounded synthetic output</main>"
    assert v2.resume_pdf == b"synthetic-rendered-pdf"
    assert v2.ats.score == fake_rendering.score.overall_score
    assert v2.ats_diagnostics is not None
    assert v2.evidence_packet is not None
    assert v2.evidence_packet["schema_version"] == "governed_resume_evidence.v1"
    assert v2.evidence_packet["mro_sha256"]
    assert v2.evidence_packet["source_text_sha256"]
    assert "mro:job:1:bullet:1" in v2.evidence_packet["canonical_evidence_ids"]
    assert v2.critique["schema_version"] == RESUME_PATCH_SCHEMA_VERSION
    assert v2.critique["accepted_change_ids"] == (
        "change-summary",
        "change-bullet",
    )
    assert v2.validation["requires_human_review"] is True
    assert v2.validation["change_count"] == 2
    assert [item["target_id"] for item in v2.validation["changes"]] == [
        "summary:paragraph",
        "experience:1:bullet:1",
    ]
    assert all(
        item["evidence_refs"] == ("mro:job:1:bullet:1",)
        for item in v2.validation["changes"]
    )
    assert v2.model_metadata["requires_human_review"] is True
    assert v2.model_metadata["review_state"] == "awaiting_user_review"
    assert v2.critique_prompt is None
    assert v2.critique_response is None


@pytest.mark.parametrize("summary", [False, True])
def test_exact_for_adjunct_fronting_is_supported_for_bullet_and_summary(
    tmp_path: Path,
    summary: bool,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    original = _resume()
    before = copy.deepcopy(original)

    candidate, audit = validate_and_apply_resume_patches(
        application_resume=original,
        response=parse_resume_patch_response(_fronted_patch_response(summary=summary)),
        evidence=evidence,
    )

    if summary:
        assert candidate["professional_summary"]["paragraph"] == FRONTED_BULLET
    else:
        assert (
            candidate["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
            == FRONTED_BULLET
        )
    assert FRONTED_BULLET.casefold().split() != PROPOSED_BULLET.casefold().split()
    assert audit["is_valid"] is True
    assert original == before


@pytest.mark.parametrize("summary", [False, True], ids=["bullet", "summary"])
@pytest.mark.parametrize(
    ("proposed", "accepted"),
    [
        (FRONTED_BULLET, True),
        ("For synthetic services. Built reliable Python automation.", False),
        ("For synthetic services? Built reliable Python automation.", False),
        ("For synthetic services! Built reliable Python automation.", False),
        ("For synthetic services; built reliable Python automation.", False),
        ("For synthetic services: built reliable Python automation.", False),
        ("For synthetic services,\nbuilt reliable Python automation.", False),
        ("For synthetic services,\rbuilt reliable Python automation.", False),
        ("For synthetic services,\u2028built reliable Python automation.", False),
        ("For synthetic services,\u2029built reliable Python automation.", False),
        ("For synthetic services,, built reliable Python automation.", False),
        ("For synthetic services,; built reliable Python automation.", False),
        ("For synthetic services,  built reliable Python automation.", False),
        ("For synthetic services,\tbuilt reliable Python automation.", False),
        ("For synthetic services,\u00a0built reliable Python automation.", False),
        ("For synthetic services , built reliable Python automation.", False),
        ("For synthetic services, Built reliable Python automation.", False),
        ("For synthetic services, built reliable Python automation", False),
        ("For synthetic services, built reliable Python automation..", False),
        ("For synthetic services, built reliable Python automation!", False),
        (
            (
                "For synthetic services, built reliable Python automation. "
                "Led fictional teams."
            ),
            False,
        ),
    ],
    ids=[
        "comma-space",
        "period-boundary",
        "question-boundary",
        "exclamation-boundary",
        "semicolon-boundary",
        "colon-boundary",
        "line-feed",
        "carriage-return",
        "unicode-line-separator",
        "unicode-paragraph-separator",
        "repeated-comma",
        "comma-plus-separator",
        "double-space",
        "tab-space",
        "nonbreaking-space",
        "space-before-comma",
        "action-capitalization",
        "missing-terminal",
        "repeated-terminal",
        "changed-terminal",
        "extra-clause",
    ],
)
def test_for_adjunct_fronting_requires_one_exact_comma_space_boundary(
    tmp_path: Path,
    summary: bool,
    proposed: str,
    accepted: bool,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    original = _resume()
    before = copy.deepcopy(original)
    parsed = parse_resume_patch_response(_fronted_patch_response(summary=summary))
    response = replace(
        parsed,
        changes=(replace(parsed.changes[0], proposed_text=proposed),),
    )

    if accepted:
        candidate, audit = validate_and_apply_resume_patches(
            application_resume=original,
            response=response,
            evidence=evidence,
        )
        target_text = (
            candidate["professional_summary"]["paragraph"]
            if summary
            else candidate["professional_experience"]["jobs"][0]["bullet_points"][0][
                "text"
            ]
        )
        assert target_text == FRONTED_BULLET
        assert audit["is_valid"] is True
    else:
        with pytest.raises(ResumePatchError) as captured:
            validate_and_apply_resume_patches(
                application_resume=original,
                response=response,
                evidence=evidence,
            )
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
    assert original == before


@pytest.mark.parametrize("summary", [False, True], ids=["bullet", "summary"])
@pytest.mark.parametrize(
    ("canonical", "proposed"),
    [
        (
            "Built reliable Python automation for synthetic services .",
            "For synthetic services , built reliable Python automation.",
        ),
        (
            "Built reliable Python automation  for synthetic services.",
            "For synthetic services, built reliable Python automation .",
        ),
        (
            "Built reliable Python automation\u00a0 for synthetic services.",
            "For synthetic services, built reliable Python automation\u00a0.",
        ),
        (
            "Built reliable services.Built Python automation for synthetic teams.",
            "For synthetic teams, built reliable services.Built Python automation.",
        ),
    ],
    ids=[
        "space-before-terminal",
        "double-space-before-for",
        "nonbreaking-space-before-for",
        "internal-period-clause",
    ],
)
def test_for_adjunct_fronting_rejects_malformed_canonical_text_shape(
    tmp_path: Path,
    summary: bool,
    canonical: str,
    proposed: str,
) -> None:
    paths = _paths(tmp_path)
    master = _master_resume()
    master["professional_experience"]["jobs"][0]["bullet_points"][0]["text"] = canonical
    paths.master_resume.write_text(
        yaml.safe_dump(master, sort_keys=False),
        encoding="utf-8",
    )
    original = _resume()
    before = copy.deepcopy(original)

    with pytest.raises(ResumePatchError) as captured:
        validate_and_apply_resume_patches(
            application_resume=original,
            response=parse_resume_patch_response(
                _fronted_patch_response(summary=summary, proposed=proposed)
            ),
            evidence=read_resume_evidence_snapshot(paths),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert original == before


@pytest.mark.parametrize("summary", [False, True], ids=["bullet", "summary"])
@pytest.mark.parametrize(
    ("proposed", "accepted"),
    [
        (PROPOSED_BULLET, True),
        ("Built reliable Python automation. For synthetic services.", False),
        ("Built reliable Python automation? For synthetic services.", False),
        ("Built reliable Python automation! For synthetic services.", False),
        ("Built reliable Python automation; for synthetic services.", False),
        ("Built reliable Python automation: for synthetic services.", False),
        ("Built reliable Python automation, for synthetic services.", False),
        ("Built reliable Python automation.\nFor synthetic services.", False),
        ("Built reliable Python automation.\rFor synthetic services.", False),
        ("Built reliable Python automation.\tFor synthetic services.", False),
        ("Built reliable Python automation.\u00a0For synthetic services.", False),
        ("Built reliable Python automation.\u2028For synthetic services.", False),
        ("Built reliable Python automation.\u2029For synthetic services.", False),
        ("Built reliable Python  automation for synthetic services.", False),
        ("Built reliable Python automation  for synthetic services.", False),
        ("Built reliable Python automation for synthetic services", False),
        ("Built reliable Python automation for synthetic services..", False),
        ("Built reliable Python automation for synthetic services?", False),
        ("built reliable Python automation for synthetic services.", False),
        ("Built reliable Python automation for Synthetic services.", False),
        (
            "Exact clause: Built reliable Python automation for synthetic services.",
            False,
        ),
        ("Built reliable Python tested automation for synthetic services.", False),
        (
            "Built reliable Python automation for synthetic services. Extra clause.",
            False,
        ),
    ],
)
def test_exact_canonical_policy_requires_complete_text_equality(
    tmp_path: Path,
    summary: bool,
    proposed: str,
    accepted: bool,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    original = _resume()
    before = copy.deepcopy(original)
    response = parse_resume_patch_response(
        _fronted_patch_response(summary=summary, proposed=PROPOSED_BULLET)
    )
    response = replace(
        response,
        changes=(replace(response.changes[0], proposed_text=proposed),),
    )

    if accepted:
        candidate, audit = validate_and_apply_resume_patches(
            application_resume=original,
            response=response,
            evidence=evidence,
        )
        target_text = (
            candidate["professional_summary"]["paragraph"]
            if summary
            else candidate["professional_experience"]["jobs"][0]["bullet_points"][0][
                "text"
            ]
        )
        assert target_text == PROPOSED_BULLET
        assert audit["is_valid"] is True
    else:
        with pytest.raises(
            ResumePatchError,
            match=r"\AResume patch validation failed\.\Z",
        ) as captured:
            validate_and_apply_resume_patches(
                application_resume=original,
                response=response,
                evidence=evidence,
            )
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
    assert original == before


@pytest.mark.parametrize(
    ("proposed", "accepted"),
    [
        (PROPOSED_SUMMARY, True),
        ("Reliable Python automation. For synthetic services.", False),
        ("Reliable Python automation? For synthetic services.", False),
        ("Reliable Python automation! For synthetic services.", False),
        ("Reliable Python automation; for synthetic services.", False),
        ("Reliable Python automation: for synthetic services.", False),
        ("Reliable Python automation, for synthetic services.", False),
        ("Reliable Python automation.\nFor synthetic services.", False),
        ("Reliable Python automation.\rFor synthetic services.", False),
        ("Reliable Python automation.\tFor synthetic services.", False),
        ("Reliable Python automation.\u00a0For synthetic services.", False),
        ("Reliable Python automation.\u2028For synthetic services.", False),
        ("Reliable Python automation.\u2029For synthetic services.", False),
        ("Reliable  Python automation for synthetic services.", False),
        ("Reliable Python automation  for synthetic services.", False),
        ("Reliable Python automation for synthetic services", False),
        ("Reliable Python automation for synthetic services..", False),
        ("Reliable Python automation for synthetic services?", False),
        ("reliable Python automation for synthetic services.", False),
        ("Reliable python automation for synthetic services.", False),
        ("Bounded clause: Reliable Python automation for synthetic services.", False),
        ("Reliable Python tested automation for synthetic services.", False),
        ("Reliable Python automation for synthetic services. Extra clause.", False),
    ],
)
def test_summary_nominal_extraction_preserves_complete_remaining_text(
    tmp_path: Path,
    proposed: str,
    accepted: bool,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    original = _resume()
    before = copy.deepcopy(original)
    response = parse_resume_patch_response(
        _fronted_patch_response(summary=True, proposed=PROPOSED_SUMMARY)
    )
    response = replace(
        response,
        changes=(replace(response.changes[0], proposed_text=proposed),),
    )

    if accepted:
        candidate, audit = validate_and_apply_resume_patches(
            application_resume=original,
            response=response,
            evidence=evidence,
        )
        assert candidate["professional_summary"]["paragraph"] == PROPOSED_SUMMARY
        assert audit["is_valid"] is True
    else:
        with pytest.raises(
            ResumePatchError,
            match=r"\AResume patch validation failed\.\Z",
        ) as captured:
            validate_and_apply_resume_patches(
                application_resume=original,
                response=response,
                evidence=evidence,
            )
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
    assert original == before


@pytest.mark.parametrize(
    ("canonical", "proposed", "accepted"),
    [
        ("Built reliable synthetic platform.", "Reliable synthetic platform.", True),
        ("built reliable synthetic platform.", "Reliable synthetic platform.", True),
        ("Built élan synthetic platform.", "élan synthetic platform.", True),
        ("Built élan synthetic platform.", "Élan synthetic platform.", False),
        ("Built ßeta synthetic platform.", "ßeta synthetic platform.", True),
        ("Built ßeta synthetic platform.", "SSeta synthetic platform.", False),
        ("Built Élan synthetic platform.", "Élan synthetic platform.", True),
        ("BUILT reliable synthetic platform.", "Reliable synthetic platform.", False),
        ("Built  reliable synthetic platform.", "Reliable synthetic platform.", False),
        ("Built\treliable synthetic platform.", "Reliable synthetic platform.", False),
        (
            "Built\u00a0reliable synthetic platform.",
            "Reliable synthetic platform.",
            False,
        ),
    ],
)
def test_summary_nominal_extraction_changes_only_first_ascii_lowercase(
    canonical: str,
    proposed: str,
    accepted: bool,
) -> None:
    assert (
        refinement._is_safe_summary_nominal_extraction(proposed, canonical) is accepted
    )


def test_exact_and_nominal_text_grammars_reject_unicode_normalization_changes() -> None:
    canonical = "Built café-safe automation for synthetic teams."
    decomposed_canonical = "Built cafe\u0301-safe automation for synthetic teams."
    nominal = "Café-safe automation for synthetic teams."
    decomposed_nominal = "Cafe\u0301-safe automation for synthetic teams."
    item = ResumeEvidenceItem(
        evidence_id="mro:job:1:bullet:1",
        role_id="mro:job:1",
        employer=COMPANY,
        role=ROLE,
        text=canonical,
    )

    assert refinement._new_claims_are_supported(
        current_text=CURRENT_BULLET,
        proposed_text=canonical,
        evidence_items=(item,),
        summary_target=False,
    )
    assert not refinement._new_claims_are_supported(
        current_text=CURRENT_BULLET,
        proposed_text=decomposed_canonical,
        evidence_items=(item,),
        summary_target=False,
    )
    assert refinement._is_safe_summary_nominal_extraction(nominal, canonical)
    assert not refinement._is_safe_summary_nominal_extraction(
        decomposed_nominal,
        canonical,
    )


def test_nominal_summary_shape_is_not_available_to_bullet_targets(
    tmp_path: Path,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    original = _resume()
    response = parse_resume_patch_response(
        _fronted_patch_response(summary=False, proposed=PROPOSED_SUMMARY)
    )

    with pytest.raises(
        ResumePatchError,
        match=r"\AResume patch validation failed\.\Z",
    ) as captured:
        validate_and_apply_resume_patches(
            application_resume=original,
            response=response,
            evidence=evidence,
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_evidence_snapshot_items_are_exactly_rederived_during_validation(
    tmp_path: Path,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    response = parse_resume_patch_response(_patch_response())
    original = _resume()
    before = copy.deepcopy(original)

    assert refinement._validate_evidence_snapshot_object(evidence) is evidence
    candidate, _audit = validate_and_apply_resume_patches(
        application_resume=original,
        response=response,
        evidence=evidence,
    )
    assert (
        candidate["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
        == PROPOSED_BULLET
    )
    assert original == before

    bullet_index = next(
        index
        for index, item in enumerate(evidence.items)
        if item.evidence_id == "mro:job:1:bullet:1"
    )
    changed_values = {
        "evidence_id": "mro:job:1:bullet:99",
        "role_id": "mro:job:2",
        "employer": OTHER_COMPANY,
        "role": OTHER_ROLE,
        "text": "Built reliable Python automation for fictional platforms.",
    }
    invalid_snapshots: list[ResumeEvidenceSnapshot] = [
        replace(
            evidence,
            master_resume=MappingProxyType({}),
            items=evidence.items,
        ),
        replace(evidence, items=tuple(reversed(evidence.items))),
        replace(evidence, items=(evidence.items[0], *evidence.items)),
    ]
    for field_name, value in changed_values.items():
        items = list(evidence.items)
        items[bullet_index] = replace(items[bullet_index], **{field_name: value})
        invalid_snapshots.append(replace(evidence, items=tuple(items)))

    for snapshot in invalid_snapshots:
        with pytest.raises(ResumePatchError) as captured:
            validate_and_apply_resume_patches(
                application_resume=original,
                response=response,
                evidence=snapshot,
            )
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert original == before


def test_evidence_reader_uses_exact_byte_digests_without_exposing_content(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    yaml_only_sentinel = "yaml-only-synthetic-comment"
    mro_bytes = paths.master_resume.read_bytes() + (
        f"\n# {yaml_only_sentinel}\r\n".encode()
    )
    source_bytes = "Synthetic corroboration with café.\r\n".encode()
    paths.master_resume.write_bytes(mro_bytes)
    paths.master_resume_text.write_bytes(source_bytes)
    evidence = read_resume_evidence_snapshot(paths)

    assert evidence.mro_sha256 == hashlib.sha256(mro_bytes).hexdigest()
    assert evidence.source_text_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert evidence.source_text == source_bytes.decode()

    store = _store_with_v1(paths)
    snapshot = store.get_workflow_snapshot(JOB_ID)
    base_variant = store.get_resume_variant(JOB_ID, "v1")
    prompt = build_resume_patch_prompt(
        workflow="refinement",
        snapshot=snapshot,
        base_variant=base_variant,
        base_resume=_resume(),
        targets=collect_resume_patch_targets(_resume()),
        evidence=evidence,
        ats_diagnostics=_diagnostics(),
        external_critique=None,
        additional_context=None,
    )
    candidate, audit = validate_and_apply_resume_patches(
        application_resume=_resume(),
        response=parse_resume_patch_response(_patch_response()),
        evidence=evidence,
    )

    exposed = " ".join((repr(evidence), prompt, repr(audit)))
    assert yaml_only_sentinel not in exposed
    assert "current_text of every change must byte-for-byte equal" in prompt
    assert "return an empty changes list" in prompt
    assert (
        candidate["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
        == PROPOSED_BULLET
    )


@pytest.mark.parametrize(
    "violation",
    ["missing", "malformed", "oversized", "symlink"],
)
def test_evidence_reader_rejects_file_boundary_violations(
    tmp_path: Path,
    violation: str,
) -> None:
    paths = _paths(tmp_path)
    private_sentinel = "private-synthetic-evidence-boundary"
    if violation == "missing":
        paths.master_resume.unlink()
    elif violation == "malformed":
        paths.master_resume.write_text("[", encoding="utf-8")
    elif violation == "oversized":
        paths.master_resume.write_bytes(b"x" * (refinement.MAX_EVIDENCE_YAML_BYTES + 1))
    else:
        target = paths.master_resume.with_name("synthetic-master-target.yml")
        target.write_text(private_sentinel, encoding="utf-8")
        paths.master_resume.unlink()
        paths.master_resume.symlink_to(target)

    with pytest.raises(
        ResumeEvidenceError,
        match=r"\AResume evidence could not be loaded\.\Z",
    ) as captured:
        read_resume_evidence_snapshot(paths)

    assert private_sentinel not in str(captured.value)
    assert private_sentinel not in repr(captured.value)


def test_evidence_snapshot_enforces_source_byte_bound(tmp_path: Path) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    response = parse_resume_patch_response(_patch_response())
    original = _resume()
    before = copy.deepcopy(original)
    oversized_source = "é" * (refinement.MAX_EVIDENCE_SOURCE_BYTES // 2 + 1)
    oversized = replace(
        evidence,
        source_text=oversized_source,
        source_text_sha256=hashlib.sha256(oversized_source.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(ResumePatchError) as captured:
        validate_and_apply_resume_patches(
            application_resume=original,
            response=response,
            evidence=oversized,
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert captured.value.args == ("Resume patch validation failed.",)
    assert original == before


def test_duplicate_label_roles_remain_bound_to_exact_canonical_role_id(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    master = _master_resume()
    second = master["professional_experience"]["jobs"][1]
    second["line_1"] = {
        "company_name_text": COMPANY,
        "position_name_text": ROLE,
    }
    second["bullet_points"][0]["text"] = (
        "Designed reliable Python tooling for fictional platforms."
    )
    paths.master_resume.write_text(
        yaml.safe_dump(master, sort_keys=False),
        encoding="utf-8",
    )
    evidence = read_resume_evidence_snapshot(paths)
    original = _resume()
    before = copy.deepcopy(original)
    transferred = parse_resume_patch_response(
        _patch_response(
            proposed="Designed reliable Python tooling for fictional platforms.",
            evidence_ref="mro:job:2:bullet:1",
        )
    )

    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=original,
            response=transferred,
            evidence=evidence,
        )
    assert original == before


def test_refinement_dry_run_returns_hidden_candidate_without_write(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)
    before = store.get_workflow_snapshot(JOB_ID)

    result = refine_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_patch_response()),
        model_config=_config(),
        dry_run=True,
    )

    after = store.get_workflow_snapshot(JOB_ID)
    assert result.dry_run is True
    assert result.stored_variant is None
    assert (
        result.candidate["professional_experience"]["jobs"][0]["bullet_points"][0][
            "text"
        ]
        == PROPOSED_BULLET
    )
    assert "synthetic-job" not in repr(result)
    assert before.revision == after.revision
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "v2")


def test_refinement_preserves_concurrent_pin_status_note_date_and_archive(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)

    def mutate_human_state() -> None:
        store.select_resume_variant(JOB_ID, "v1")
        store.update_application_status(
            JOB_ID,
            applied_to="Yes",
            date_applied="2042-04-05",
            notes="Synthetic human checkpoint.",
        )
        store.archive([JOB_ID])

    result = refine_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_patch_response(), mutate_human_state),
        model_config=_config(),
    )
    application = store.get_application(JOB_ID)
    assert result.stored_variant == "v2"
    assert application.selected_resume_variant == "v1"
    assert application.resume_variant_selection_mode == "manual"
    assert application.applied_to == "Yes"
    assert application.date_applied == "2042-04-05"
    assert application.notes == "Synthetic human checkpoint."
    assert application.archived_at is not None


def test_zero_change_refinement_writes_audited_v2(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)
    response = json.dumps(
        {
            "schema_version": RESUME_PATCH_SCHEMA_VERSION,
            "changes": [],
        },
        separators=(",", ":"),
    )
    result = refine_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(response),
        model_config=_config(),
    )
    v1 = store.get_resume_variant(JOB_ID, "v1")
    v2 = store.get_resume_variant(JOB_ID, "v2")
    assert result.changed_count == 0
    assert v2.application_resume == v1.application_resume
    assert v2.parent_variant_key == "v1"
    assert v2.validation["change_count"] == 0
    assert v2.validation["requires_human_review"] is True


def test_refinement_detects_source_digest_drift_before_cas(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)

    def mutate_source() -> None:
        paths.master_resume_text.write_text(
            "Changed synthetic source version.",
            encoding="utf-8",
        )

    with pytest.raises(ResumeWorkflowConflictError) as captured:
        refine_resume_for_job(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=_FakeRunner(_patch_response(), mutate_source),
            model_config=_config(),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "v2")


def test_refinement_maps_removed_evidence_to_stable_conflict(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)

    def remove_source() -> None:
        paths.master_resume_text.unlink()

    with pytest.raises(ResumeWorkflowConflictError) as captured:
        refine_resume_for_job(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=_FakeRunner(_patch_response(), remove_source),
            model_config=_config(),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "v2")


def test_refinement_maps_malformed_current_mro_to_stable_conflict(
    tmp_path: Path,
    fake_rendering: AtsDiagnostics,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1(paths)
    before_revision = store.get_workflow_snapshot(JOB_ID).revision
    before_database = paths.database.read_bytes()

    def corrupt_mro() -> None:
        paths.master_resume.write_bytes(b"professional_experience: [\n")

    with pytest.raises(ResumeWorkflowConflictError) as captured:
        refine_resume_for_job(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=_FakeRunner(_patch_response(), corrupt_mro),
            model_config=_config(),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert store.get_workflow_snapshot(JOB_ID).revision == before_revision
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "v2")
    assert paths.database.read_bytes() == before_database
    assert list(paths.output_dir.iterdir()) == []
    assert list(paths.tmp_dir.iterdir()) == []


def test_crossed_store_binding_rejects_before_file_or_model_access(
    tmp_path: Path,
) -> None:
    bound_paths = _paths(tmp_path, name="bound")
    crossed_paths = _paths(tmp_path, name="crossed")
    store = _store_with_v1(bound_paths)
    runner = _FakeRunner(_patch_response())

    with pytest.raises(ApplicationStateConfigurationError):
        refine_resume_for_job(
            store=store,
            paths=crossed_paths,
            job_id=JOB_ID,
            runner=runner,
            model_config=_config(),
        )
    assert runner.requests == []


def test_same_role_evidence_applies_and_cross_role_transfer_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    evidence = read_resume_evidence_snapshot(paths)
    resume = _resume()

    valid = parse_resume_patch_response(_patch_response())
    candidate, audit = validate_and_apply_resume_patches(
        application_resume=resume,
        response=valid,
        evidence=evidence,
    )
    assert (
        candidate["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
        == PROPOSED_BULLET
    )
    assert audit["is_valid"] is True

    transferred = parse_resume_patch_response(
        _patch_response(
            proposed="Led Kubernetes work that reduced latency by 40%.",
            evidence_ref="mro:job:2:bullet:1",
        )
    )
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=resume,
            response=transferred,
            evidence=evidence,
        )


@pytest.mark.parametrize(
    "response",
    [
        "",
        "[]",
        "null",
        "```json\n{}\n```",
        'prose {"schema_version":"governed_resume_patch.v1","changes":[]}',
        '{"schema_version":"governed_resume_patch.v1","changes":[]} trailing',
        '{"schema_version":"governed_resume_patch.v1","changes":NaN}',
        (
            '{"schema_version":"governed_resume_patch.v1",'
            '"schema_version":"governed_resume_patch.v1","changes":[]}'
        ),
        '{"schema_version":"wrong","changes":[]}',
        '{"schema_version":"governed_resume_patch.v1","changes":[],"extra":1}',
        '{"schema_version":"governed_resume_patch.v1","changes":{}}',
        '{"schema_version":"governed_resume_patch.v1","changes":[true]}',
    ],
)
def test_strict_patch_parser_rejects_noncontract_json(response: str) -> None:
    with pytest.raises(ResumeRefinementError):
        parse_resume_patch_response(response)


def test_patch_parser_rejects_duplicate_change_and_target_ids() -> None:
    first = json.loads(_patch_response())
    duplicate_id = copy.deepcopy(first)
    duplicate_id["changes"].append(copy.deepcopy(first["changes"][0]))
    with pytest.raises(ResumePatchError):
        parse_resume_patch_response(json.dumps(duplicate_id))

    duplicate_target = copy.deepcopy(first)
    second = copy.deepcopy(first["changes"][0])
    second["change_id"] = "change-2"
    duplicate_target["changes"].append(second)
    with pytest.raises(ResumePatchError):
        parse_resume_patch_response(json.dumps(duplicate_target))


def test_stale_text_and_unsupported_metric_reject_all_changes(
    tmp_path: Path,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    stale_payload = json.loads(_patch_response())
    stale_payload["changes"][0]["current_text"] = "Stale synthetic text."
    stale = parse_resume_patch_response(json.dumps(stale_payload))
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=stale,
            evidence=evidence,
        )

    metric = parse_resume_patch_response(
        _patch_response(proposed="Built Python automation that improved output by 99%.")
    )
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=metric,
            evidence=evidence,
        )


@pytest.mark.parametrize(
    "proposed",
    [
        "Built Python automation for Nova Dynamics services.",
        (
            "Built Python automation for synthetic services while mentoring "
            "engineers and directing payroll operations."
        ),
        "Built python automation for synthetic services at nova dynamics.",
        "Built python automation for synthetic services with pulumi.",
        "Built Python and Pulumi automation for synthetic services.",
        "Earned a fictional cloud credential while building Python automation.",
        "Led Python automation for synthetic services.",
        "Built compliant Python automation for a regulated service.",
        "Reduced synthetic service latency with Python automation.",
        "Administered Python automation for synthetic services.",
        "Owned Python automation for synthetic services.",
        "Built Kubernetes automation for synthetic services.",
    ],
)
def test_fabricated_claim_classes_are_rejected(
    tmp_path: Path,
    proposed: str,
) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    parsed = parse_resume_patch_response(_patch_response(proposed=proposed))
    original = _resume()
    before = copy.deepcopy(original)
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=original,
            response=parsed,
            evidence=evidence,
        )
    assert original == before


def test_source_jod_ats_and_model_context_cannot_authorize_claim(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.master_resume_text.write_text(
        "Terraform appears only in this corroborating source text.",
        encoding="utf-8",
    )
    evidence = read_resume_evidence_snapshot(paths)
    parsed = parse_resume_patch_response(
        _patch_response(
            proposed="Built Terraform automation for synthetic services.",
        )
    )
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=parsed,
            evidence=evidence,
        )


@pytest.mark.parametrize(
    ("canonical", "proposed"),
    [
        (
            "Did not lead migration using Python automation.",
            "Did lead migration using Python automation.",
        ),
        (
            "Mentored engineers while payroll operations failed.",
            "Mentored payroll operations while engineers failed.",
        ),
        (
            "Supported leaders who owned Python automation.",
            "Owned Python automation.",
        ),
        (
            "Helped build Python automation.",
            "Build Python automation.",
        ),
        (
            "Contributed to a team that reduced latency by 40%.",
            "Reduced latency by 40%.",
        ),
    ],
)
def test_grounding_preserves_negation_and_claim_order(
    tmp_path: Path,
    canonical: str,
    proposed: str,
) -> None:
    paths = _paths(tmp_path)
    master = _master_resume()
    master["professional_experience"]["jobs"][0]["bullet_points"][0]["text"] = canonical
    paths.master_resume.write_text(
        yaml.safe_dump(master, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=parse_resume_patch_response(_patch_response(proposed=proposed)),
            evidence=read_resume_evidence_snapshot(paths),
        )


@pytest.mark.parametrize(
    ("canonical", "proposed"),
    [
        (
            "Built reliable Python automation without synthetic services.",
            "For synthetic services, built reliable Python automation.",
        ),
        (
            "Built reliable Python automation for synthetic services.",
            "For Python automation, built reliable synthetic services.",
        ),
        (
            (
                "Built reliable Python automation for synthetic services while "
                "maintained fictional tools."
            ),
            (
                "Maintained fictional tools while built reliable Python automation "
                "for synthetic services."
            ),
        ),
        (
            "Contributed to a team that built Python automation.",
            "Owned Python automation.",
        ),
        (
            "Built reliable Python automation for synthetic services.",
            "For synthetic services, built reliable Python automation by 40%.",
        ),
        (
            "Built reliable Python automation for synthetic services.",
            "For synthetic services, built reliable Kubernetes automation.",
        ),
        (
            (
                "Built reliable Python automation for synthetic services for "
                "fictional teams."
            ),
            (
                "For synthetic services for fictional teams, built reliable "
                "Python automation."
            ),
        ),
    ],
)
def test_bounded_fronting_rejects_semantic_near_neighbors(
    tmp_path: Path,
    canonical: str,
    proposed: str,
) -> None:
    paths = _paths(tmp_path)
    master = _master_resume()
    master["professional_experience"]["jobs"][0]["bullet_points"][0]["text"] = canonical
    paths.master_resume.write_text(
        yaml.safe_dump(master, sort_keys=False),
        encoding="utf-8",
    )
    original = _resume()
    before = copy.deepcopy(original)

    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=original,
            response=parse_resume_patch_response(_patch_response(proposed=proposed)),
            evidence=read_resume_evidence_snapshot(paths),
        )
    assert original == before


def test_parser_rejects_controls_depth_width_oversize_and_active_subclass() -> None:
    control = json.loads(_patch_response())
    control["changes"][0]["rationale"] = "unsafe\u0000control"
    with pytest.raises(ResumeRefinementError):
        parse_resume_patch_response(json.dumps(control))

    deep: object = "leaf"
    for _ in range(24):
        deep = [deep]
    with pytest.raises(ResumeRefinementError):
        parse_resume_patch_response(
            json.dumps(
                {
                    "schema_version": RESUME_PATCH_SCHEMA_VERSION,
                    "changes": deep,
                }
            )
        )
    with pytest.raises(ResumeRefinementError):
        parse_resume_patch_response(
            json.dumps(
                {
                    "schema_version": RESUME_PATCH_SCHEMA_VERSION,
                    "changes": [[] for _ in range(513)],
                }
            )
        )
    with pytest.raises(ResumeRefinementError):
        parse_resume_patch_response("{" + ("x" * 200_001) + "}")

    class ActiveString(str):
        def strip(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("active method must not run")

    with pytest.raises(ResumeRefinementError):
        parse_resume_patch_response(ActiveString("{}"))

    class ActiveWorkflow(str):
        activated = False

        def __hash__(self) -> int:
            type(self).activated = True
            raise AssertionError("active hash must not run")

    with pytest.raises(ResumeRefinementError):
        build_resume_patch_prompt(
            workflow=ActiveWorkflow("refinement"),  # type: ignore[arg-type]
            snapshot=None,  # type: ignore[arg-type]
            base_variant=None,  # type: ignore[arg-type]
            base_resume={},  # type: ignore[arg-type]
            targets=(),
            evidence=None,  # type: ignore[arg-type]
            ats_diagnostics=None,  # type: ignore[arg-type]
            external_critique=None,
            additional_context=None,
        )
    assert ActiveWorkflow.activated is False


def test_patch_prompt_rejects_active_target_and_job_description_before_use(
    tmp_path: Path,
) -> None:
    class ActiveString(str):
        activated = False

        def __bool__(self) -> bool:
            type(self).activated = True
            raise AssertionError("active truthiness must not run")

        def __eq__(self, _other: object) -> bool:
            type(self).activated = True
            raise AssertionError("active equality must not run")

        def __hash__(self) -> int:
            type(self).activated = True
            raise AssertionError("active hashing must not run")

    paths = _paths(tmp_path)
    store = _store_with_v1(paths)
    snapshot = store.get_workflow_snapshot(JOB_ID)
    base_variant = store.get_resume_variant(JOB_ID, "v1")
    targets = collect_resume_patch_targets(_resume())
    evidence = read_resume_evidence_snapshot(paths)
    active_target = ResumePatchTarget(
        section=ActiveString("professional_summary"),  # type: ignore[arg-type]
        field="paragraph",
        job_order=None,
        bullet_order=None,
    )
    forged_targets = (replace(targets[0], target=active_target), *targets[1:])
    with pytest.raises(ResumeRefinementError):
        build_resume_patch_prompt(
            workflow="refinement",
            snapshot=snapshot,
            base_variant=base_variant,
            base_resume=_resume(),
            targets=forged_targets,
            evidence=evidence,
            ats_diagnostics=_diagnostics(),
            external_critique=None,
            additional_context=None,
        )

    forged_application = replace(
        snapshot.application,
        prompt_job_description=ActiveString("synthetic prompt"),
    )
    forged_snapshot = ApplicationWorkflowSnapshot(
        application=forged_application,
        variants=snapshot.variants,
        revision=snapshot.revision,
    )
    with pytest.raises(ResumeRefinementError):
        build_resume_patch_prompt(
            workflow="refinement",
            snapshot=forged_snapshot,
            base_variant=base_variant,
            base_resume=_resume(),
            targets=targets,
            evidence=evidence,
            ats_diagnostics=_diagnostics(),
            external_critique=None,
            additional_context=None,
        )

    with pytest.raises(ResumeRefinementError):
        build_resume_patch_prompt(
            workflow="refinement",
            snapshot=snapshot,
            base_variant=base_variant,
            base_resume=_resume(),
            targets=(targets[0], targets[0]),
            evidence=evidence,
            ats_diagnostics=_diagnostics(),
            external_critique=None,
            additional_context=None,
        )
    assert ActiveString.activated is False


def test_patch_repr_and_errors_hide_model_content(tmp_path: Path) -> None:
    secret = "private-synthetic-model-content"
    payload = json.loads(_patch_response())
    payload["changes"][0]["proposed_text"] = f"{secret} Kubernetes"
    parsed = parse_resume_patch_response(json.dumps(payload))
    assert secret not in repr(parsed)
    assert secret not in repr(parsed.changes[0])
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    with pytest.raises(ResumePatchError) as captured:
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=parsed,
            evidence=evidence,
        )
    state = " ".join(
        (str(captured.value), repr(captured.value), repr(captured.value.args))
    )
    assert secret not in state
    assert captured.value.__cause__ is None


def test_direct_patch_object_rejects_active_fields_before_use(
    tmp_path: Path,
) -> None:
    class ActiveString(str):
        activated = False

        def __hash__(self) -> int:
            type(self).activated = True
            raise AssertionError("active hash must not run")

        def __eq__(self, _other: object) -> bool:
            type(self).activated = True
            raise AssertionError("active equality must not run")

    parsed = parse_resume_patch_response(_patch_response())
    original = parsed.changes[0]
    active_change = ResumePatch(
        change_id=ActiveString(original.change_id),
        operation=original.operation,
        target=original.target,
        current_text=original.current_text,
        proposed_text=original.proposed_text,
        rationale=original.rationale,
        evidence_refs=original.evidence_refs,
    )
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=ResumePatchResponse(
                schema_version=RESUME_PATCH_SCHEMA_VERSION,
                changes=(active_change,),
            ),
            evidence=evidence,
        )

    assert ActiveString.activated is False


def test_direct_patch_object_rechecks_duplicate_parser_invariants(
    tmp_path: Path,
) -> None:
    parsed = parse_resume_patch_response(_patch_response())
    direct_duplicate = ResumePatchResponse(
        schema_version=RESUME_PATCH_SCHEMA_VERSION,
        changes=(parsed.changes[0], parsed.changes[0]),
    )
    with pytest.raises(ResumePatchError):
        validate_and_apply_resume_patches(
            application_resume=_resume(),
            response=direct_duplicate,
            evidence=read_resume_evidence_snapshot(_paths(tmp_path)),
        )


def test_collect_targets_rejects_missing_and_duplicate_exact_identities() -> None:
    missing = _resume()
    del missing["professional_experience"]["jobs"][0]["order"]
    with pytest.raises(ResumePatchError):
        collect_resume_patch_targets(missing)

    duplicate = _resume()
    duplicate["professional_experience"]["jobs"][0]["bullet_points"].append(
        copy.deepcopy(
            duplicate["professional_experience"]["jobs"][0]["bullet_points"][0]
        )
    )
    with pytest.raises(ResumePatchError):
        collect_resume_patch_targets(duplicate)


def test_result_and_evidence_reprs_hide_content(tmp_path: Path) -> None:
    evidence = read_resume_evidence_snapshot(_paths(tmp_path))
    state = repr(evidence)
    assert COMPANY not in state
    assert CURRENT_BULLET not in state
    assert evidence.mro_sha256 not in state
    assert asdict(_diagnostics())["score"]["overall_score"] == 87


def test_evidence_reader_rejects_deep_yaml_and_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    deep_paths = _paths(tmp_path, name="deep")
    deep_paths.master_resume.write_text(
        ("[" * 2_000) + "0" + ("]" * 2_000),
        encoding="utf-8",
    )
    with pytest.raises(ResumeEvidenceError) as deep_error:
        read_resume_evidence_snapshot(deep_paths)
    assert deep_error.value.__cause__ is None

    fifo_paths = _paths(tmp_path, name="fifo")
    fifo_paths.master_resume.unlink()
    os.mkfifo(fifo_paths.master_resume)
    with pytest.raises(ResumeEvidenceError):
        read_resume_evidence_snapshot(fifo_paths)
