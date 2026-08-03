"""Synthetic manual-pass, highlighting, and fake Codex-runner regressions."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

import career_agent_workbench.resume_refinement as refinement
from career_agent_workbench import codex_cli
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateNotFoundError,
    ApplicationStateStore,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.ats import (
    AtsComponentScores,
    AtsDiagnostics,
    AtsProxyScore,
)
from career_agent_workbench.codex_cli import (
    MAX_RESPONSE_BYTES,
    CodexCancellationError,
    CodexCleanupError,
    CodexConfigurationError,
    CodexExecutionError,
    CodexModelConfig,
    CodexOutputError,
    CodexProcessConfig,
    CodexProcessRunner,
    CodexTimeoutError,
    ModelRequest,
    ModelResult,
    ProcessResult,
    resolve_codex_model_config,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.resume_highlighting import (
    HIGHLIGHT_RESPONSE_SCHEMA_VERSION,
    HighlightResponse,
    HighlightUpdate,
    ResumeHighlightError,
    apply_highlight_response,
    build_resume_highlight_prompt,
    collect_highlight_bullets,
    highlight_resume_for_job,
    parse_highlight_response,
    resolve_highlight_target,
    validate_highlighted_text,
)
from career_agent_workbench.resume_manual_pass import run_manual_resume_pass
from career_agent_workbench.resume_manual_profiles import (
    MANUAL_PASS_PROFILES,
    ManualPassProfileKey,
    resolve_manual_pass_config,
)
from career_agent_workbench.resume_refinement import (
    RESUME_PATCH_SCHEMA_VERSION,
    ResumeEvidenceError,
    ResumePatchError,
    ResumeWorkflowConflictError,
)

JOB_ID = "synthetic-job-2"
COMPANY = "Example Automation Cooperative"
ROLE = "Fictional Reliability Engineer"
V1_BULLET = "Built Python automation for synthetic services."
V2_BULLET = "Built reliable Python automation for synthetic services."
MANUAL_BULLET = "Built resilient Python automation for synthetic services."
MANUAL_FRONTED_BULLET = "For synthetic services, built resilient Python automation."
SAFEGUARDED_SKILLS = (
    "DevOps",
    "Scalability",
    "CI/CD pipelines",
    "cloud environments",
    "GitHub Actions",
)


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
                "profile": request.config.profile or "regular",
                "model": request.config.model,
                "reasoning_effort": request.config.reasoning_effort,
                "attempt": 1,
                "timestamp": "2043-05-06T07:08:09+00:00",
                "version": 1,
            },
        )


class _ActiveReturnCode(int):
    activated = False

    def __bool__(self) -> bool:
        type(self).activated = True
        raise AssertionError("active truthiness must not run")

    def __eq__(self, _other: object) -> bool:
        type(self).activated = True
        raise AssertionError("active equality must not run")

    def __index__(self) -> int:
        type(self).activated = True
        raise AssertionError("active indexing must not run")


class _FakeExecutor:
    def __init__(
        self,
        actions: list[str],
        *,
        response: str = '{"synthetic":true}',
        external: Path | None = None,
        callback: Any = None,
    ) -> None:
        self.actions = list(actions)
        self.response = response
        self.external = external
        self.callback = callback
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        command: tuple[str, ...],
        *,
        input: str,
        cwd: Path,
        env: Any,
        timeout: float,
        stdout: int,
        stderr: int,
        check: bool,
        shell: bool,
        encoding: str,
        errors: str,
    ) -> ProcessResult:
        self.calls.append(
            {
                "command": command,
                "input": input,
                "cwd": cwd,
                "env": dict(env),
                "timeout": timeout,
                "stdout": stdout,
                "stderr": stderr,
                "check": check,
                "shell": shell,
                "encoding": encoding,
                "errors": errors,
            }
        )
        action = self.actions.pop(0)
        output = Path(command[command.index("--output-last-message") + 1])
        if action == "timeout_callback":
            assert self.callback is not None
            self.callback()
            raise subprocess.TimeoutExpired(command, timeout)
        if action == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if action.startswith("timeout_replace_"):
            target = action.removeprefix("timeout_replace_")
            if target == "executable":
                executable = Path(command[0])
                displaced = executable.with_name(f"{executable.name}.displaced")
                executable.rename(displaced)
                executable.write_text("#!/bin/sh\nexit 98\n", encoding="utf-8")
                executable.chmod(0o700)
            elif target == "working":
                displaced = cwd.with_name(f"{cwd.name}.displaced")
                cwd.rename(displaced)
                cwd.mkdir()
            elif target == "tmp":
                temporary = output.parent.parent
                displaced = temporary.with_name(f"{temporary.name}.displaced")
                temporary.rename(displaced)
                temporary.mkdir()
            else:
                raise AssertionError("unknown replacement target")
            raise subprocess.TimeoutExpired(command, timeout)
        if action == "cancel":
            raise asyncio.CancelledError
        if action == "error":
            raise OSError("synthetic private executor detail")
        if action == "nonzero":
            return ProcessResult(returncode=17)
        if action == "missing":
            return ProcessResult(returncode=0)
        if action == "oversized":
            output.write_bytes(b"x" * (MAX_RESPONSE_BYTES + 1))
            return ProcessResult(returncode=0)
        if action == "extra_output":
            output.write_text(self.response, encoding="utf-8")
            (output.parent / "unexpected.txt").write_text(
                "bounded",
                encoding="utf-8",
            )
            return ProcessResult(returncode=0)
        if action == "control_output":
            output.write_text("unsafe\u0000response", encoding="utf-8")
            return ProcessResult(returncode=0)
        if action == "fifo_output":
            os.mkfifo(output)
            return ProcessResult(returncode=0)
        if action == "invalid_result":
            return {"returncode": 0}  # type: ignore[return-value]
        if action == "missing_result_field":
            return object.__new__(ProcessResult)
        if action == "active_result_field":
            forged = object.__new__(ProcessResult)
            object.__setattr__(forged, "returncode", _ActiveReturnCode(0))
            return forged
        if action == "output_symlink":
            assert self.external is not None
            output.symlink_to(self.external)
            return ProcessResult(returncode=0)
        if action == "output_hardlink":
            assert self.external is not None
            os.link(self.external, output)
            return ProcessResult(returncode=0)
        if action == "replacement":
            child = output.parent
            displaced = child.with_name(f"{child.name}.displaced")
            child.rename(displaced)
            child.mkdir()
            (child / "do-not-remove.txt").write_text("replacement", encoding="utf-8")
            return ProcessResult(returncode=0)
        if action == "cleanup_failure":
            for index in range(65):
                (output.parent / f"entry-{index:02d}").write_text(
                    "bounded",
                    encoding="utf-8",
                )
            return ProcessResult(returncode=0)
        if action != "success":
            raise AssertionError("unknown synthetic action")
        output.write_text(self.response, encoding="utf-8")
        return ProcessResult(returncode=0)


def _resume(bullet: str) -> dict[str, Any]:
    return {
        "professional_summary": {
            "paragraph": "Builds reliable tools for fictional teams."
        },
        "core_technical_skills": {
            "bullet_points": [
                {
                    "category": "Platform Delivery",
                    "items": {
                        "primary": list(SAFEGUARDED_SKILLS),
                        "additional": ["Python"],
                        "match_terms": {"DevOps": ["automation"]},
                    },
                },
                {
                    "category": "Synthetic Data Systems",
                    "items": {
                        "primary": ["SQL"],
                        "additional": ["Example Warehouse"],
                        "match_terms": {"Example Warehouse": ["warehouse alias"]},
                    },
                },
            ]
        },
        "professional_experience": {
            "jobs": [
                {
                    "order": "1",
                    "line_1": {
                        "company_name_text": COMPANY,
                        "position_name_text": ROLE,
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


def _paths(tmp_path: Path) -> WorkspacePaths:
    output = tmp_path / "output"
    temporary = tmp_path / "tmp"
    output.mkdir()
    temporary.mkdir()
    master = tmp_path / "MASTER-RESUME.yml"
    source = tmp_path / "MASTER-RESUME.txt"
    master.write_text(
        yaml.safe_dump(
            _resume(MANUAL_BULLET),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source.write_text(
        "Synthetic corroboration for the exact example role.",
        encoding="utf-8",
    )
    return WorkspacePaths(
        database=(tmp_path / "applications.sqlite3").absolute(),
        master_resume=master.absolute(),
        master_resume_text=source.absolute(),
        output_dir=output.absolute(),
        tmp_dir=temporary.absolute(),
    )


def _write_variant(key: str, bullet: str, *, parent: str | None) -> ResumeVariantWrite:
    return ResumeVariantWrite(
        variant_key=key,
        variant_label=f"Synthetic {key} draft",
        source="synthetic_test",
        parent_variant_key=parent,
        application_resume_yaml=yaml.safe_dump(_resume(bullet), sort_keys=False),
        resume_html=f"<main>{key} synthetic</main>",
        resume_pdf=f"synthetic-pdf-{key}".encode(),
        source_resume_html_path=f"provenance/{key}.html",
        source_resume_path=f"provenance/{key}.pdf",
        ats=AtsFields(
            score=80,
            parsing_score=90,
            keyword_score=80,
            semantic_score=70,
            formatting_risk="low",
            diagnostics={"variant": key},
        ),
        ats_diagnostics={"variant": key, "score": 80},
        evidence_packet={"variant": key, "evidence": ["synthetic"]},
        external_critique={"present": False},
        critique_prompt=f"preserved-prompt-{key}",
        critique_response=f"preserved-response-{key}",
        critique={"accepted": [key]},
        validation={"valid": True, "variant": key},
        model_metadata={"generation": {"variant": key}},
    )


def _store_with_v1_v2(paths: WorkspacePaths) -> ApplicationStateStore:
    store = ApplicationStateStore(paths)
    store.initialize()
    store.seed_application(
        ApplicationMetadata(
            job_id=JOB_ID,
            company=COMPANY,
            job_title="Example Public Reliability Role",
            job_url=f"https://jobs.example.com/{JOB_ID}",
            source="synthetic",
        ),
        source_text="Full fictional job description for reliable Python systems.",
        prompt_text="Reliable Python systems.",
    )
    store.upsert_resume_variant(
        JOB_ID,
        _write_variant("v1", V1_BULLET, parent=None),
    )
    store.upsert_resume_variant(
        JOB_ID,
        _write_variant("v2", V2_BULLET, parent="v1"),
    )
    return store


def _diagnostics(score: int = 91) -> AtsDiagnostics:
    proxy = AtsProxyScore(
        overall_score=score,
        parsing_score=94,
        keyword_match_score=89,
        semantic_match_score=88,
        formatting_risk="low",
        missing_high_value_terms=("example-term",),
    )
    components = AtsComponentScores(
        overall_score=score,
        parsing_score=94,
        keyword_match_score=89,
        semantic_match_score=88,
        formatting_score=93,
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


@pytest.fixture
def fake_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        refinement,
        "render_resume_html_from_mapping",
        lambda *, resume: "<main>synthetic rendered resume</main>",
    )
    monkeypatch.setattr(
        refinement,
        "render_resume_pdf_from_html",
        lambda html: b"synthetic-rendered-pdf",
    )
    monkeypatch.setattr(
        refinement,
        "calculate_ats_diagnostics",
        lambda *, resume_pdf, job_description: _diagnostics(),
    )


def _patch_response(*, current: str, proposed: str) -> str:
    return json.dumps(
        {
            "schema_version": RESUME_PATCH_SCHEMA_VERSION,
            "changes": [
                {
                    "change_id": "manual-change-1",
                    "operation": "rewrite_bullet",
                    "target": {
                        "section": "professional_experience",
                        "field": "text",
                        "job_order": "1",
                        "bullet_order": "1",
                    },
                    "current_text": current,
                    "proposed_text": proposed,
                    "rationale": "Canonical same-role evidence supports the wording.",
                    "evidence_refs": ["mro:job:1:bullet:1"],
                }
            ],
        },
        separators=(",", ":"),
    )


def _skill_patch_response(*, proposed_matches: list[str]) -> str:
    current = json.dumps([], separators=(",", ":"))
    proposed = json.dumps(proposed_matches, separators=(",", ":"))
    return json.dumps(
        {
            "schema_version": RESUME_PATCH_SCHEMA_VERSION,
            "changes": [
                {
                    "change_id": "manual-skill-change-1",
                    "operation": "replace_skill_matches",
                    "target": {
                        "section": "core_technical_skills",
                        "field": "jod_matched_items",
                        "job_order": "1",
                        "bullet_order": None,
                    },
                    "current_text": current,
                    "proposed_text": proposed,
                    "rationale": "Select an inventoried same-category match.",
                    "evidence_refs": ["mro:job:1:bullet:1"],
                }
            ],
        },
        separators=(",", ":"),
    )


def _highlight_response(current: str, highlighted: str) -> str:
    return json.dumps(
        {
            "schema_version": HIGHLIGHT_RESPONSE_SCHEMA_VERSION,
            "updates": [
                {
                    "target_id": "experience:1:bullet:1",
                    "current_text": current,
                    "highlighted_text": highlighted,
                }
            ],
        },
        separators=(",", ":"),
    )


def _config(workflow: str) -> CodexModelConfig:
    return CodexModelConfig(
        model="synthetic-model",
        reasoning_effort="high",
        workflow=workflow,
        profile="regular",
    )


def test_manual_pass_derives_from_v2_and_stores_review_candidate(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    before_v1 = store.get_resume_variant(JOB_ID, "v1")
    before_v2 = store.get_resume_variant(JOB_ID, "v2")

    result = run_manual_resume_pass(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_patch_response(current=V2_BULLET, proposed=MANUAL_BULLET)),
        model_config=_config("manual_pass"),
    )

    manual = store.get_resume_variant(JOB_ID, "manual")
    assert result.stored_variant == "manual"
    assert result.review_state == "awaiting_user_review"
    assert manual.parent_variant_key == "v2"
    assert (
        manual.application_resume["professional_experience"]["jobs"][0][
            "bullet_points"
        ][0]["text"]
        == MANUAL_BULLET
    )
    assert manual.model_metadata["requires_human_review"] is True
    assert manual.model_metadata["review_state"] == "awaiting_user_review"
    assert store.get_resume_variant(JOB_ID, "v1") == before_v1
    assert store.get_resume_variant(JOB_ID, "v2") == before_v2


def test_manual_skill_policy_reaches_request_and_preserves_supported_terms(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    store.select_resume_variant(JOB_ID, "v2")
    before_v1 = store.get_resume_variant(JOB_ID, "v1")
    before_v2 = store.get_resume_variant(JOB_ID, "v2")
    runner = _FakeRunner(_skill_patch_response(proposed_matches=["Python"]))

    result = run_manual_resume_pass(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=runner,
        model_config=_config("manual_pass"),
    )

    assert len(runner.requests) == 1
    prompt = runner.requests[0].prompt
    assert all(term in prompt for term in SAFEGUARDED_SKILLS)
    assert "change only the existing category's jod_matched_items list" in prompt
    assert "category order, category names, primary and additional" in prompt
    assert "match_terms aliases" in prompt
    assert "operation replace_skill_matches" in prompt
    assert "Keep pruning and de-duplicating" not in prompt
    assert "replace only the rendered primary/additional" not in prompt
    assert "canonical_mro_skill_evidence" in prompt
    manual = store.get_resume_variant(JOB_ID, "manual")
    before_categories = before_v2.application_resume["core_technical_skills"][
        "bullet_points"
    ]
    manual_categories = manual.application_resume["core_technical_skills"][
        "bullet_points"
    ]
    manual_items = manual_categories[0]["items"]
    assert manual_items["primary"] == SAFEGUARDED_SKILLS
    assert manual_items["additional"] == ("Python",)
    assert manual_items["match_terms"]["DevOps"] == ("automation",)
    assert manual_categories[0]["jod_matched_items"] == ("Python",)
    assert [category["category"] for category in manual_categories] == [
        category["category"] for category in before_categories
    ]
    assert [category["items"] for category in manual_categories] == [
        category["items"] for category in before_categories
    ]
    assert manual.parent_variant_key == "v2"
    assert result.changed_count == 1
    assert store.get_application(JOB_ID).selected_resume_variant == "v2"
    assert store.get_application(JOB_ID).resume_variant_selection_mode == "manual"
    assert store.get_resume_variant(JOB_ID, "v1") == before_v1
    assert store.get_resume_variant(JOB_ID, "v2") == before_v2


def test_manual_skill_policy_rejects_match_from_another_category(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    before_v1 = store.get_resume_variant(JOB_ID, "v1")
    before_v2 = store.get_resume_variant(JOB_ID, "v2")
    with pytest.raises(ResumePatchError):
        run_manual_resume_pass(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=_FakeRunner(
                _skill_patch_response(proposed_matches=["warehouse alias"])
            ),
            model_config=_config("manual_pass"),
        )

    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "manual")
    assert store.get_resume_variant(JOB_ID, "v1") == before_v1
    assert store.get_resume_variant(JOB_ID, "v2") == before_v2


def test_manual_skill_policy_accepts_bounded_same_category_matches(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    proposed = [*SAFEGUARDED_SKILLS, "Python"]

    result = run_manual_resume_pass(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_skill_patch_response(proposed_matches=proposed)),
        model_config=_config("manual_pass"),
        dry_run=True,
    )

    assert result.changed_count == 1
    assert result.stored_variant is None
    category = result.candidate["core_technical_skills"]["bullet_points"][0]
    assert category["items"]["primary"] == SAFEGUARDED_SKILLS
    assert category["items"]["additional"] == ("Python",)
    assert category["jod_matched_items"] == tuple(proposed)


def test_manual_pass_inherits_exact_for_adjunct_fronting_validator(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)

    result = run_manual_resume_pass(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(
            _patch_response(
                current=V2_BULLET,
                proposed=MANUAL_FRONTED_BULLET,
            )
        ),
        model_config=_config("manual_pass"),
    )

    manual = store.get_resume_variant(JOB_ID, "manual")
    assert result.stored_variant == "manual"
    assert (
        manual.application_resume["professional_experience"]["jobs"][0][
            "bullet_points"
        ][0]["text"]
        == MANUAL_FRONTED_BULLET
    )
    assert MANUAL_FRONTED_BULLET.casefold().split() != MANUAL_BULLET.casefold().split()


def test_manual_dry_run_and_digest_conflict_leave_state_exact(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    before = store.get_workflow_snapshot(JOB_ID)
    response = _patch_response(current=V2_BULLET, proposed=MANUAL_BULLET)

    dry = run_manual_resume_pass(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(response),
        model_config=_config("manual_pass"),
        dry_run=True,
    )
    assert dry.stored_variant is None
    assert before.revision == store.get_workflow_snapshot(JOB_ID).revision
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "manual")

    def mutate_mro() -> None:
        paths.master_resume.write_text(
            yaml.safe_dump(_resume("Changed synthetic evidence.")),
            encoding="utf-8",
        )

    with pytest.raises(ResumeWorkflowConflictError):
        run_manual_resume_pass(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=_FakeRunner(response, mutate_mro),
            model_config=_config("manual_pass"),
        )
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_resume_variant(JOB_ID, "manual")


def test_manual_pass_preserves_concurrent_v2_pin_and_human_state(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)

    def mutate_human_state() -> None:
        store.select_resume_variant(JOB_ID, "v2")
        store.update_application_status(
            JOB_ID,
            applied_to="Rejected",
            date_applied="2043-05-06",
            notes="Synthetic manual checkpoint.",
        )
        store.archive([JOB_ID])

    result = run_manual_resume_pass(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(
            _patch_response(current=V2_BULLET, proposed=MANUAL_BULLET),
            mutate_human_state,
        ),
        model_config=_config("manual_pass"),
    )
    application = store.get_application(JOB_ID)
    assert result.stored_variant == "manual"
    assert application.selected_resume_variant == "v2"
    assert application.resume_variant_selection_mode == "manual"
    assert application.applied_to == "Rejected"
    assert application.date_applied == "2043-05-06"
    assert application.notes == "Synthetic manual checkpoint."
    assert application.archived_at is not None


def test_highlighting_updates_exact_override_and_preserves_pin_and_provenance(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    store.select_resume_variant(JOB_ID, "v1")
    before_v1 = store.get_resume_variant(JOB_ID, "v1")
    before_v2 = store.get_resume_variant(JOB_ID, "v2")
    highlighted = (
        "Built <strong>reliable Python automation</strong> for synthetic services."
    )

    result = highlight_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_highlight_response(V2_BULLET, highlighted)),
        model_config=_config("highlighting"),
        variant_override="v2",
    )

    after_app = store.get_application(JOB_ID)
    after_v1 = store.get_resume_variant(JOB_ID, "v1")
    after_v2 = store.get_resume_variant(JOB_ID, "v2")
    assert result.target_variant == "v2"
    assert after_app.selected_resume_variant == "v1"
    assert after_app.resume_variant_selection_mode == "manual"
    assert after_v1 == before_v1
    assert (
        after_v2.application_resume["professional_experience"]["jobs"][0][
            "bullet_points"
        ][0]["text"]
        == highlighted
    )
    for name in (
        "variant_label",
        "source",
        "parent_variant_key",
        "evidence_packet",
        "external_critique",
        "critique_prompt",
        "critique_response",
        "critique",
        "validation",
        "source_resume_html_path",
        "source_resume_path",
    ):
        assert getattr(after_v2, name) == getattr(before_v2, name)
    assert after_v2.model_metadata["generation"] == {"variant": "v2"}
    assert after_v2.model_metadata["highlighting"]["requires_human_review"] is True
    assert (
        after_v2.model_metadata["highlighting"]["mro_sha256"]
        == hashlib.sha256(paths.master_resume.read_bytes()).hexdigest()
    )
    assert (
        after_v2.model_metadata["highlighting"]["source_text_sha256"]
        == hashlib.sha256(paths.master_resume_text.read_bytes()).hexdigest()
    )


def test_highlighting_preserves_canonical_reader_evidence_errors(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    paths.master_resume.unlink()
    runner = _FakeRunner(
        _highlight_response(
            V2_BULLET,
            "Built <strong>reliable Python automation</strong> for synthetic services.",
        )
    )

    with pytest.raises(
        ResumeEvidenceError,
        match=r"\AResume evidence could not be loaded\.\Z",
    ) as captured:
        highlight_resume_for_job(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=runner,
            model_config=_config("highlighting"),
        )
    assert captured.value.__cause__ is None
    assert isinstance(captured.value.__context__, FileNotFoundError)
    assert runner.requests == []


def test_highlighting_dry_run_leaves_target_and_projection_exact(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    before = store.get_workflow_snapshot(JOB_ID)
    highlighted = (
        "Built <strong>reliable Python automation</strong> for synthetic services."
    )
    result = highlight_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(_highlight_response(V2_BULLET, highlighted)),
        model_config=_config("highlighting"),
        variant_override="v2",
        dry_run=True,
    )
    assert result.dry_run is True
    assert (
        result.candidate["professional_experience"]["jobs"][0]["bullet_points"][0][
            "text"
        ]
        == highlighted
    )
    assert store.get_workflow_snapshot(JOB_ID) == before


def test_pin_during_highlight_model_run_survives_conditional_write(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    highlighted = (
        "Built <strong>reliable Python automation</strong> for synthetic services."
    )

    def mutate_human_state() -> None:
        store.select_resume_variant(JOB_ID, "v1")
        store.update_application_status(
            JOB_ID,
            applied_to="Yes",
            date_applied="2043-05-07",
            notes="Synthetic highlight checkpoint.",
        )
        store.archive([JOB_ID])

    result = highlight_resume_for_job(
        store=store,
        paths=paths,
        job_id=JOB_ID,
        runner=_FakeRunner(
            _highlight_response(V2_BULLET, highlighted),
            mutate_human_state,
        ),
        model_config=_config("highlighting"),
    )

    assert result.target_variant == "v2"
    application = store.get_application(JOB_ID)
    assert application.selected_resume_variant == "v1"
    assert application.resume_variant_selection_mode == "manual"
    assert application.applied_to == "Yes"
    assert application.date_applied == "2043-05-07"
    assert application.notes == "Synthetic highlight checkpoint."
    assert application.archived_at is not None


def test_highlight_digest_drift_rejects_without_variant_change(
    tmp_path: Path,
    fake_rendering: None,
) -> None:
    private_sentinel = "private-highlighting-drift-sentinel"
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    before_snapshot = store.get_workflow_snapshot(JOB_ID)
    before_v1 = store.get_resume_variant(JOB_ID, "v1")
    before_v2 = store.get_resume_variant(JOB_ID, "v2")
    before_database = paths.database.read_bytes()
    before_mro = paths.master_resume.read_bytes()
    before_source = paths.master_resume_text.read_bytes()
    before_output = tuple(paths.output_dir.iterdir())
    before_temporary = tuple(paths.tmp_dir.iterdir())
    highlighted = (
        "Built <strong>reliable Python automation</strong> for synthetic services."
    )

    def mutate_source() -> None:
        paths.master_resume_text.write_text(
            private_sentinel,
            encoding="utf-8",
        )

    runner = _FakeRunner(
        _highlight_response(V2_BULLET, highlighted),
        mutate_source,
    )
    with pytest.raises(
        ResumeWorkflowConflictError,
        match=r"\AResume workflow input changed\.\Z",
    ) as captured:
        highlight_resume_for_job(
            store=store,
            paths=paths,
            job_id=JOB_ID,
            runner=runner,
            model_config=_config("highlighting"),
        )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert private_sentinel not in str(captured.value)
    assert private_sentinel not in repr(captured.value)
    assert len(runner.requests) == 1
    assert store.get_workflow_snapshot(JOB_ID) == before_snapshot
    assert store.get_resume_variant(JOB_ID, "v1") == before_v1
    assert store.get_resume_variant(JOB_ID, "v2") == before_v2
    assert paths.database.read_bytes() == before_database
    assert paths.master_resume.read_bytes() == before_mro
    assert paths.master_resume_text.read_bytes() == private_sentinel.encode()
    assert paths.master_resume_text.read_bytes() != before_source
    assert tuple(paths.output_dir.iterdir()) == before_output
    assert tuple(paths.tmp_dir.iterdir()) == before_temporary
    assert private_sentinel.encode() not in paths.database.read_bytes()
    assert private_sentinel not in repr(store.get_workflow_snapshot(JOB_ID))


@pytest.mark.parametrize(
    ("highlighted", "valid"),
    [
        ("Built <strong>Python automation</strong> for synthetic services.", True),
        ("Built <strong class='x'>Python</strong> automation.", False),
        ("Built <strong><strong>Python</strong></strong> automation.", False),
        ("Built <strong></strong> Python automation.", False),
        ("Built <strong>!!!</strong> Python automation.", False),
        (f"<strong>{V1_BULLET}</strong>", False),
        ("Built <b>Python</b> automation for synthetic services.", False),
        ("Built &amp; Python automation for synthetic services.", False),
        ("Built ＜strong＞Python＜/strong＞ automation for synthetic services.", False),
        ("Built <strong>Python\x00</strong> automation.", False),
        ("Built <strong>Python automation for synthetic services.", False),
        ("Built Python automation for synthetic services.</strong>", False),
        (
            (
                "Built <strong>Python</strong> <strong>automation</strong> for "
                "<strong>synthetic</strong> <strong>services</strong>."
            ),
            False,
        ),
    ],
)
def test_exact_wrapper_validator_matrix(highlighted: str, valid: bool) -> None:
    if valid:
        assert (
            validate_highlighted_text(
                original_text=V1_BULLET,
                highlighted_text=highlighted,
            )
            == 1
        )
    else:
        with pytest.raises(ResumeHighlightError):
            validate_highlighted_text(
                original_text=V1_BULLET,
                highlighted_text=highlighted,
            )


def test_existing_unicode_tag_lookalikes_are_rejected() -> None:
    original = "Built ＜strong＞Python＜/strong＞ automation for services."
    highlighted = (
        "Built ＜strong＞<strong>Python</strong>＜/strong＞ automation for services."
    )
    with pytest.raises(ResumeHighlightError):
        validate_highlighted_text(
            original_text=original,
            highlighted_text=highlighted,
        )


def test_highlight_response_requires_exact_one_per_target_atomically() -> None:
    resume = _resume(V1_BULLET)
    valid_text = "Built <strong>Python automation</strong> for synthetic services."
    valid = parse_highlight_response(_highlight_response(V1_BULLET, valid_text))
    candidate, stats = apply_highlight_response(resume, valid)
    assert stats.bullet_count == 1
    assert stats.strong_span_count == 1
    assert (
        candidate["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
        == valid_text
    )
    assert (
        resume["professional_experience"]["jobs"][0]["bullet_points"][0]["text"]
        == V1_BULLET
    )

    payload = json.loads(_highlight_response(V1_BULLET, valid_text))
    payload["updates"].append(copy.deepcopy(payload["updates"][0]))
    with pytest.raises(ResumeHighlightError):
        parse_highlight_response(json.dumps(payload))
    payload = json.loads(_highlight_response(V1_BULLET, valid_text))
    payload["updates"][0]["extra"] = True
    with pytest.raises(ResumeHighlightError):
        parse_highlight_response(json.dumps(payload))
    with pytest.raises(ResumeHighlightError):
        parse_highlight_response(
            '{"schema_version":"governed_resume_highlighting.v1","updates":[]}'
        )


def test_direct_highlight_object_rejects_active_fields_before_hashing() -> None:
    class ActiveString(str):
        activated = False

        def __hash__(self) -> int:
            type(self).activated = True
            raise AssertionError("active hash must not run")

        def __eq__(self, _other: object) -> bool:
            type(self).activated = True
            raise AssertionError("active equality must not run")

    response = HighlightResponse(
        updates=(
            HighlightUpdate(
                target_id=ActiveString("experience:1:bullet:1"),
                current_text=V1_BULLET,
                highlighted_text=(
                    "Built <strong>Python automation</strong> for synthetic services."
                ),
            ),
        )
    )
    with pytest.raises(ResumeHighlightError):
        apply_highlight_response(_resume(V1_BULLET), response)
    assert ActiveString.activated is False

    class ActiveSnapshot:
        activated = False

        def __getattribute__(self, _name: str) -> object:
            type(self).activated = True
            raise AssertionError("active attribute must not run")

    with pytest.raises(ResumeHighlightError):
        build_resume_highlight_prompt(
            snapshot=ActiveSnapshot(),  # type: ignore[arg-type]
            target_variant=None,  # type: ignore[arg-type]
            bullets=(),
        )
    assert ActiveSnapshot.activated is False


def test_target_policy_selected_override_chain_and_combined_precedence(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = _store_with_v1_v2(paths)
    store.upsert_resume_variant(
        JOB_ID,
        _write_variant("manual", MANUAL_BULLET, parent="v2"),
    )
    snapshot = store.get_workflow_snapshot(JOB_ID)
    assert resolve_highlight_target(snapshot) == "manual"
    assert resolve_highlight_target(snapshot, variant_override="v1") == "v1"
    assert resolve_highlight_target(snapshot, following_variant="v2") == "v2"
    assert (
        resolve_highlight_target(
            snapshot,
            combined_variants=("v2", "manual"),
        )
        == "manual"
    )
    with pytest.raises(ResumeHighlightError):
        resolve_highlight_target(
            snapshot,
            variant_override="v1",
            following_variant="v2",
        )


def test_collect_highlight_bullets_rejects_missing_duplicate_identities() -> None:
    missing = _resume(V1_BULLET)
    del missing["professional_experience"]["jobs"][0]["bullet_points"][0]["order"]
    with pytest.raises(ResumeHighlightError):
        collect_highlight_bullets(missing)
    duplicate = _resume(V1_BULLET)
    duplicate["professional_experience"]["jobs"][0]["bullet_points"].append(
        copy.deepcopy(
            duplicate["professional_experience"]["jobs"][0]["bullet_points"][0]
        )
    )
    with pytest.raises(ResumeHighlightError):
        collect_highlight_bullets(duplicate)


def test_highlight_company_job_order_and_span_bounds_are_exact() -> None:
    resume = _resume(V1_BULLET)
    second = copy.deepcopy(resume["professional_experience"]["jobs"][0])
    second["order"] = "2"
    second["line_1"]["company_name_text"] = "Fictional Harbor Group"
    second["bullet_points"][0]["order"] = "1"
    second["bullet_points"][0]["text"] = "Maintained synthetic harbor systems."
    resume["professional_experience"]["jobs"].append(second)

    selected = collect_highlight_bullets(
        resume,
        experience_company="harbor",
        experience_job_order="2",
    )
    assert tuple(item.target_id for item in selected) == ("experience:2:bullet:1",)

    highlighted = (
        "Maintained <strong>synthetic</strong> <strong>harbor</strong> systems."
    )
    response = json.dumps(
        {
            "schema_version": HIGHLIGHT_RESPONSE_SCHEMA_VERSION,
            "updates": [
                {
                    "target_id": "experience:2:bullet:1",
                    "current_text": "Maintained synthetic harbor systems.",
                    "highlighted_text": highlighted,
                }
            ],
        }
    )
    with pytest.raises(ResumeHighlightError):
        apply_highlight_response(
            resume,
            response,
            max_strong_spans_per_bullet=1,
            experience_company="harbor",
            experience_job_order="2",
        )


def test_manual_profiles_and_presence_aware_overrides_are_exact() -> None:
    assert {
        key.value: (value.model, value.reasoning_effort)
        for key, value in MANUAL_PASS_PROFILES.items()
    } == {
        "economy": ("gpt-5.6-terra", "high"),
        "regular": ("gpt-5.6-sol", "high"),
        "premium": ("gpt-5.6-sol", "xhigh"),
    }
    assert resolve_manual_pass_config().profile.key is ManualPassProfileKey.REGULAR
    model_only = resolve_manual_pass_config(
        profile="economy",
        workflow_model_override="synthetic-override",
    )
    assert model_only.model == "synthetic-override"
    assert model_only.reasoning_effort == "high"
    effort_only = resolve_manual_pass_config(
        profile="premium",
        workflow_reasoning_effort_override="",
    )
    assert effort_only.model == "gpt-5.6-sol"
    assert effort_only.reasoning_effort == ""
    independent = resolve_codex_model_config(
        default_model="highlight-model",
        default_reasoning_effort="medium",
        workflow="highlighting",
    )
    assert independent.workflow == "highlighting"
    assert independent.profile is None
    assert "gpt-5.6" not in repr(model_only)
    with pytest.raises(FrozenInstanceError):
        model_only.model = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        MANUAL_PASS_PROFILES[ManualPassProfileKey.REGULAR] = (  # type: ignore[index]
            MANUAL_PASS_PROFILES[ManualPassProfileKey.REGULAR]
        )


def _process_runner(
    tmp_path: Path,
    executor: _FakeExecutor,
) -> tuple[CodexProcessRunner, CodexProcessConfig]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = tmp_path / "synthetic-codex"
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o700)
    working = tmp_path / "working"
    temporary = tmp_path / "process-tmp"
    working.mkdir()
    temporary.mkdir()
    config = CodexProcessConfig(
        executable=executable.absolute(),
        argv=("exec", "--skip-git-repo-check"),
        working_directory=working.absolute(),
        tmp_dir=temporary.absolute(),
        child_environment={
            "LANG": "C.UTF-8",
            "SYNTHETIC_CHILD": "bounded",
        },
    )
    return CodexProcessRunner(config=config, executor=executor), config


def _request(
    *,
    attempts: int = 1,
    effort: str = "high",
    max_response_bytes: int = MAX_RESPONSE_BYTES,
) -> ModelRequest:
    return ModelRequest(
        prompt="Synthetic bounded prompt.",
        config=CodexModelConfig(
            model="synthetic-model",
            reasoning_effort=effort,
            workflow="synthetic_workflow",
        ),
        timeout_seconds=3,
        max_attempts=attempts,
        max_response_bytes=max_response_bytes,
    )


def test_fake_process_success_uses_exact_argv_env_and_cleans_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/private/ambient-marker")
    monkeypatch.setenv("PRIVATE_AMBIENT_MARKER", "must-not-leak")
    executor = _FakeExecutor(["success"])
    runner, config = _process_runner(tmp_path, executor)

    result = runner.run(_request())

    assert result.response == '{"synthetic":true}'
    assert result.model_metadata["attempt"] == 1
    call = executor.calls[0]
    assert call["env"] == {
        "LANG": "C.UTF-8",
        "SYNTHETIC_CHILD": "bounded",
    }
    assert call["shell"] is False
    assert call["check"] is False
    assert call["stdout"] == subprocess.DEVNULL
    assert call["stderr"] == subprocess.DEVNULL
    assert type(call["timeout"]) is float
    assert call["timeout"] == 3.0
    command = call["command"]
    assert command[:3] == (
        os.fspath(config.executable),
        "exec",
        "--skip-git-repo-check",
    )
    assert ("--ask-for-approval", "never") == (
        command[3],
        command[4],
    )
    assert "--sandbox" in command and "read-only" in command
    assert "--cd" in command and os.fspath(config.working_directory) in command
    assert "--output-last-message" in command
    assert "--model" in command and "synthetic-model" in command
    assert "-c" in command
    assert command[-1] == "-"
    assert list(config.tmp_dir.iterdir()) == []
    assert "Synthetic bounded prompt." not in repr(result)
    assert "PRIVATE_AMBIENT_MARKER" not in repr(config)


def test_empty_effort_omits_codex_override(tmp_path: Path) -> None:
    executor = _FakeExecutor(["success"])
    runner, _ = _process_runner(tmp_path, executor)
    runner.run(_request(effort=""))
    assert "-c" not in executor.calls[0]["command"]


def test_timeout_retries_then_succeeds_and_cleans_each_child(
    tmp_path: Path,
) -> None:
    executor = _FakeExecutor(["timeout", "success"])
    runner, config = _process_runner(tmp_path, executor)
    result = runner.run(_request(attempts=2))
    assert result.model_metadata["attempt"] == 2
    assert len(executor.calls) == 2
    assert list(config.tmp_dir.iterdir()) == []


@pytest.mark.parametrize("target", ["executable", "working", "tmp"])
def test_retry_revalidates_all_process_path_identities(
    tmp_path: Path,
    target: str,
) -> None:
    executor = _FakeExecutor([f"timeout_replace_{target}", "success"])
    runner, _ = _process_runner(tmp_path, executor)
    with pytest.raises(CodexConfigurationError):
        runner.run(_request(attempts=2))
    assert len(executor.calls) == 1


@pytest.mark.parametrize(
    ("action", "error_type"),
    [
        ("nonzero", CodexExecutionError),
        ("missing", CodexOutputError),
        ("oversized", CodexOutputError),
        ("cancel", CodexCancellationError),
    ],
)
def test_fake_process_failure_matrix_is_stable_and_cleans(
    tmp_path: Path,
    action: str,
    error_type: type[Exception],
) -> None:
    executor = _FakeExecutor([action])
    runner, config = _process_runner(tmp_path, executor)
    with pytest.raises(error_type) as captured:
        runner.run(_request())
    assert list(config.tmp_dir.iterdir()) == []
    state = " ".join((str(captured.value), repr(captured.value)))
    assert "synthetic private" not in state
    assert str(config.tmp_dir) not in state
    assert captured.value.__cause__ is None


def test_timeout_exhaustion_and_executor_error_are_content_free(
    tmp_path: Path,
) -> None:
    timeout_runner, _ = _process_runner(
        tmp_path / "timeout-case",
        _FakeExecutor(["timeout"]),
    )
    with pytest.raises(CodexTimeoutError):
        timeout_runner.run(_request())

    error_root = tmp_path / "error-case"
    error_root.mkdir()
    error_runner, _ = _process_runner(error_root, _FakeExecutor(["error"]))
    with pytest.raises(CodexExecutionError) as captured:
        error_runner.run(_request())
    assert "synthetic private executor detail" not in str(captured.value)


@pytest.mark.parametrize("action", ["output_symlink", "output_hardlink"])
def test_output_link_does_not_follow_or_remove_external_file(
    tmp_path: Path,
    action: str,
) -> None:
    external = tmp_path / "external.txt"
    external.write_text("preserve", encoding="utf-8")
    executor = _FakeExecutor([action], external=external)
    runner, _ = _process_runner(tmp_path / "runner", executor)
    with pytest.raises(CodexOutputError):
        runner.run(_request())
    assert external.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("action", ["replacement", "cleanup_failure"])
def test_replacement_and_cleanup_failure_report_distinctly_without_false_claim(
    tmp_path: Path,
    action: str,
) -> None:
    executor = _FakeExecutor([action])
    runner, config = _process_runner(tmp_path, executor)
    with pytest.raises(CodexCleanupError):
        runner.run(_request())
    assert any(config.tmp_dir.iterdir())


def test_symlink_and_nonnormalized_process_paths_fail_closed(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "executable"
    executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    executable.chmod(0o700)
    linked = tmp_path / "linked-executable"
    linked.symlink_to(executable)
    working = tmp_path / "working"
    temporary = tmp_path / "tmp"
    working.mkdir()
    temporary.mkdir()
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=linked,
            argv=("exec",),
            working_directory=working,
            tmp_dir=temporary,
            child_environment={},
        )
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=executable,
            argv=("exec",),
            working_directory=working,
            tmp_dir=temporary / ".." / "tmp",
            child_environment={},
        )


def test_runner_and_public_values_hide_content_and_reject_active_subclasses(
    tmp_path: Path,
) -> None:
    class ActiveString(str):
        def __str__(self) -> str:
            raise AssertionError("active conversion must not run")

    with pytest.raises(CodexConfigurationError):
        CodexModelConfig(
            model=ActiveString("private-model"),
            reasoning_effort="high",
        )
    executor = _FakeExecutor(["success"])
    runner, config = _process_runner(tmp_path, executor)
    request = _request()
    for value in (repr(runner), repr(config), repr(request)):
        assert "synthetic-model" not in value
        assert str(config.executable) not in value
        assert "SYNTHETIC_CHILD" not in value


def test_runner_revalidates_forged_exact_model_state_before_child_creation(
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

        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            type(self).activated = True
            raise AssertionError("active encoding must not run")

    executor = _FakeExecutor(["success"])
    runner, config = _process_runner(tmp_path, executor)
    missing_config = object.__new__(CodexModelConfig)
    active_config = object.__new__(CodexModelConfig)
    object.__setattr__(active_config, "model", ActiveString("synthetic-model"))
    object.__setattr__(active_config, "reasoning_effort", "high")
    object.__setattr__(active_config, "workflow", "synthetic_workflow")
    object.__setattr__(active_config, "profile", None)

    for forged_config in (missing_config, active_config):
        request = _request()
        object.__setattr__(request, "config", forged_config)
        with pytest.raises(CodexConfigurationError) as captured:
            runner.run(request)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert executor.calls == []
        assert list(config.tmp_dir.iterdir()) == []
    assert ActiveString.activated is False


def test_retry_revalidates_request_mutation_after_owned_child_cleanup(
    tmp_path: Path,
) -> None:
    request = _request(attempts=2)

    def mutate_request() -> None:
        object.__setattr__(request, "prompt", "Changed bounded prompt.")

    executor = _FakeExecutor(
        ["timeout_callback", "success"],
        callback=mutate_request,
    )
    runner, config = _process_runner(tmp_path, executor)

    with pytest.raises(CodexConfigurationError) as captured:
        runner.run(request)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert len(executor.calls) == 1
    assert list(config.tmp_dir.iterdir()) == []


@pytest.mark.parametrize(
    "action",
    ["missing_result_field", "active_result_field"],
)
def test_runner_rejects_forged_exact_process_results_and_cleans(
    tmp_path: Path,
    action: str,
) -> None:
    _ActiveReturnCode.activated = False
    runner, config = _process_runner(tmp_path, _FakeExecutor([action]))

    with pytest.raises(CodexExecutionError) as captured:
        runner.run(_request())
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert list(config.tmp_dir.iterdir()) == []
    assert _ActiveReturnCode.activated is False


@pytest.mark.parametrize("boundary", ["command", "output"])
def test_unexpected_post_child_boundary_failures_are_stable_and_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    executor = _FakeExecutor(["success"])
    runner, config = _process_runner(tmp_path, executor)

    def fail_boundary(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic private boundary detail")

    if boundary == "command":
        monkeypatch.setattr(codex_cli, "_build_command", fail_boundary)
    else:
        monkeypatch.setattr(codex_cli, "_read_owned_output", fail_boundary)

    with pytest.raises(CodexExecutionError) as captured:
        runner.run(_request())
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "synthetic private boundary detail" not in str(captured.value)
    assert list(config.tmp_dir.iterdir()) == []
    assert len(executor.calls) == (0 if boundary == "command" else 1)


def test_runner_rejects_hostile_mapping_proxies_without_activation(
    tmp_path: Path,
) -> None:
    class ActiveMapping(Mapping[str, str]):
        activated = False

        def __getitem__(self, _key: str) -> str:
            type(self).activated = True
            raise AssertionError("active mapping must not run")

        def __iter__(self) -> Any:
            type(self).activated = True
            raise AssertionError("active mapping must not run")

        def __len__(self) -> int:
            type(self).activated = True
            raise AssertionError("active mapping must not run")

    runner, config = _process_runner(tmp_path, _FakeExecutor(["success"]))
    hostile = MappingProxyType(ActiveMapping())
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=config.executable,
            argv=("exec",),
            working_directory=config.working_directory,
            tmp_dir=config.tmp_dir,
            child_environment=hostile,
        )
    with pytest.raises(CodexConfigurationError):
        ModelResult(
            response="bounded",
            model_metadata=hostile,  # type: ignore[arg-type]
        )
    assert ActiveMapping.activated is False
    assert "configured=True" in repr(runner)


def test_runner_bounds_timeout_prompt_environment_and_argv(
    tmp_path: Path,
) -> None:
    with pytest.raises(CodexConfigurationError):
        ModelRequest(
            prompt="bounded",
            config=CodexModelConfig(model="model", reasoning_effort="high"),
            timeout_seconds=10**10_000,
        )
    with pytest.raises(CodexConfigurationError):
        ModelRequest(
            prompt="unsafe\u0000prompt",
            config=CodexModelConfig(model="model", reasoning_effort="high"),
        )
    runner, config = _process_runner(tmp_path, _FakeExecutor(["success"]))
    del runner
    for argv in (
        ("exec", "--model"),
        ("exec", "--unknown"),
        ("exec", ""),
    ):
        with pytest.raises(CodexConfigurationError):
            CodexProcessConfig(
                executable=config.executable,
                argv=argv,
                working_directory=config.working_directory,
                tmp_dir=config.tmp_dir,
                child_environment={},
            )
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=config.executable,
            argv=("exec",),
            working_directory=config.working_directory,
            tmp_dir=config.tmp_dir,
            child_environment={f"KEY_{index}": "x" for index in range(65)},
        )
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=config.executable,
            argv=("exec",),
            working_directory=config.working_directory,
            tmp_dir=config.tmp_dir,
            child_environment={"SAFE": "unsafe\u0000value"},
        )


@pytest.mark.parametrize(
    "action",
    ["extra_output", "control_output", "fifo_output", "invalid_result"],
)
def test_runner_rejects_extra_control_or_invalid_process_output(
    tmp_path: Path,
    action: str,
) -> None:
    runner, config = _process_runner(tmp_path, _FakeExecutor([action]))
    error_type = CodexExecutionError if action == "invalid_result" else CodexOutputError
    with pytest.raises(error_type):
        runner.run(_request())
    assert list(config.tmp_dir.iterdir()) == []


def test_runner_rejects_missing_nonexecuting_and_symlink_directories(
    tmp_path: Path,
) -> None:
    runner, config = _process_runner(tmp_path, _FakeExecutor(["success"]))
    del runner
    config.executable.chmod(0o600)
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=config.executable,
            argv=("exec",),
            working_directory=config.working_directory,
            tmp_dir=config.tmp_dir,
            child_environment={},
        )
    config.executable.chmod(0o700)
    linked_working = tmp_path / "linked-working"
    linked_tmp = tmp_path / "linked-tmp"
    linked_working.symlink_to(config.working_directory, target_is_directory=True)
    linked_tmp.symlink_to(config.tmp_dir, target_is_directory=True)
    for working, temporary in (
        (linked_working, config.tmp_dir),
        (config.working_directory, linked_tmp),
    ):
        with pytest.raises(CodexConfigurationError):
            CodexProcessConfig(
                executable=config.executable,
                argv=("exec",),
                working_directory=working,
                tmp_dir=temporary,
                child_environment={},
            )
    config.executable.unlink()
    with pytest.raises(CodexConfigurationError):
        CodexProcessConfig(
            executable=config.executable,
            argv=("exec",),
            working_directory=config.working_directory,
            tmp_dir=config.tmp_dir,
            child_environment={},
        )


def test_tmp_root_replacement_between_validation_and_open_is_not_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _FakeExecutor(["success"])
    runner, config = _process_runner(tmp_path, executor)
    original_open = codex_cli.os.open
    displaced = config.tmp_dir.with_name("process-tmp-displaced")
    replaced = False

    def replacing_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == config.tmp_dir and dir_fd is None:
            config.tmp_dir.rename(displaced)
            config.tmp_dir.mkdir()
            (config.tmp_dir / "sentinel.txt").write_text(
                "preserve",
                encoding="utf-8",
            )
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(codex_cli.os, "open", replacing_open)
    with pytest.raises(CodexExecutionError):
        runner.run(_request())
    assert executor.calls == []
    assert (config.tmp_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserve"
    assert displaced.is_dir()


def test_child_replacement_during_creation_is_not_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, config = _process_runner(tmp_path, _FakeExecutor(["success"]))
    original_open = codex_cli.os.open
    replacement_name: str | None = None

    def replacing_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replacement_name
        if (
            replacement_name is None
            and type(path) is str
            and path.startswith(".codex-run-")
            and dir_fd is not None
        ):
            replacement_name = path
            os.rename(
                path,
                f"{path}.displaced",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.mkdir(path, dir_fd=dir_fd)
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            os.close(descriptor)
            (config.tmp_dir / path / "sentinel.txt").write_text(
                "preserve",
                encoding="utf-8",
            )
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(codex_cli.os, "open", replacing_open)
    with pytest.raises(CodexCleanupError):
        runner.run(_request())
    assert replacement_name is not None
    assert (config.tmp_dir / replacement_name / "sentinel.txt").read_text(
        encoding="utf-8"
    ) == "preserve"
    assert (config.tmp_dir / f"{replacement_name}.displaced").is_dir()


def test_command_construction_failure_cleans_owned_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, config = _process_runner(tmp_path, _FakeExecutor(["success"]))

    def reject_command(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        raise CodexConfigurationError("Codex command is invalid.")

    monkeypatch.setattr(codex_cli, "_build_command", reject_command)
    with pytest.raises(CodexConfigurationError):
        runner.run(_request())
    assert list(config.tmp_dir.iterdir()) == []
