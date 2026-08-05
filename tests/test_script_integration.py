from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

import career_agent_workbench.artifact_exports as artifact_exports
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateConflictError,
    ApplicationStateStore,
    ApplicationStateValidationError,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.ats import (
    AtsComponentScores,
    AtsDiagnostics,
    AtsProxyScore,
    AtsWeightedTerm,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths
from career_agent_workbench.errors import (
    LlmTimeoutError,
    ModelFailureSubtype,
    RetryableModelError,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "application_resume_generate_drafts.py",
    "application_resume_highlight_drafts.py",
    "application_resume_manual_pass.py",
    "application_resume_pass_one.py",
    "application_resume_regenerate_aros.py",
    "application_resume_store_first_draft.py",
    "application_resume_sync_drafts_to_aro.py",
    "render_resume_html.py",
)
STATE_OPTIONS = {
    "--workspace",
    "--database",
    "--output-dir",
    "--profile-dir",
    "--master-resume",
    "--master-resume-text",
    "--blacklist-path",
    "--tmp-dir",
}


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"p10_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", SCRIPT_NAMES)
@pytest.mark.parametrize("option", ["--help", "--version"])
def test_script_metadata_is_side_effect_free(
    name: str, option: str, tmp_path: Path
) -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), option],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_script_state_defaults_and_template_are_none(name: str) -> None:
    parser: argparse.ArgumentParser = _load_script(name).build_arg_parser()
    for action in parser._actions:
        if STATE_OPTIONS.intersection(action.option_strings):
            assert action.default is None
        if "--template" in action.option_strings:
            assert action.default is None
        if "--input" in action.option_strings and name == "render_resume_html.py":
            assert action.default is None
        if "--output" in action.option_strings and name == "render_resume_html.py":
            assert action.default is None


@pytest.mark.asyncio
async def test_first_draft_config_only_emits_defaults_without_boundaries(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    config = RuntimeConfig(paths=WorkspacePaths(), settings=Settings(), env_file=None)
    monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)
    monkeypatch.setattr(
        module,
        "build_llm_client",
        lambda *_a, **_k: pytest.fail("config-only created a model client"),
    )
    monkeypatch.setattr(
        module,
        "ApplicationStateStore",
        lambda *_a, **_k: pytest.fail("config-only created a state store"),
    )

    assert await module.main_async(["--config-only"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"config_only": True}
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert [event["stage"] for event in events] == [
        "v1_core",
        "v1_jod",
        "v1_experience",
    ]
    assert all(event["timeout_seconds"] == 300.0 for event in events)
    assert all(event["retry_count"] == 1 for event in events)
    assert all(event["total_attempts"] == 2 for event in events)
    assert all(event["sources"]["timeout"] == "default" for event in events)


def test_manual_and_highlight_config_only_defaults_are_model_free(
    monkeypatch,
    capsys,
) -> None:
    config = RuntimeConfig(paths=WorkspacePaths(), settings=Settings(), env_file=None)
    expectations = (
        ("application_resume_manual_pass.py", "manual", "gpt-5.6-sol", "regular"),
        ("application_resume_highlight_drafts.py", "highlight", "gpt-5.6-luna", None),
    )
    for name, stage, model, profile in expectations:
        module = _load_script(name)
        monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)
        monkeypatch.setattr(
            module,
            "build_codex_runner",
            lambda **_k: pytest.fail("config-only created a Codex runner"),
        )
        monkeypatch.setattr(
            module,
            "ApplicationStateStore",
            lambda *_a, **_k: pytest.fail("config-only created a state store"),
        )
        assert module.main(["--config-only"]) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {"config_only": True}
        event = json.loads(captured.err)
        assert event["stage"] == stage
        assert event["model"] == model
        assert event["timeout_seconds"] == 900.0
        assert event["retry_count"] == 1
        assert event["total_attempts"] == 2
        assert event.get("profile") == profile


@pytest.mark.parametrize(
    "name",
    ["application_resume_manual_pass.py", "application_resume_highlight_drafts.py"],
)
def test_codex_workflow_parser_defaults_defer_to_runtime_settings(name: str) -> None:
    args = _load_script(name).build_arg_parser().parse_args([])
    assert args.timeout_seconds is None
    assert args.retry_count is None


@pytest.mark.parametrize(
    ("name", "stage"),
    (
        ("application_resume_manual_pass.py", "manual"),
        ("application_resume_highlight_drafts.py", "highlight"),
    ),
)
def test_codex_config_only_reports_runtime_tuning_source(
    name: str,
    stage: str,
    monkeypatch,
    capsys,
) -> None:
    module = _load_script(name)
    config = RuntimeConfig(
        paths=WorkspacePaths(),
        settings=Settings(codex_timeout_seconds=777, codex_retries=2),
        env_file=None,
        setting_sources=(
            ("codex_retries", "private_dotenv"),
            ("codex_timeout_seconds", "process"),
        ),
    )
    monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)

    assert module.main(["--config-only"]) == 0
    captured = capsys.readouterr()
    event = json.loads(captured.err)
    assert event["stage"] == stage
    assert event["timeout_seconds"] == 777
    assert event["retry_count"] == 2
    assert event["sources"]["timeout"] == "process"
    assert event["sources"]["retry_count"] == "private_dotenv"


def test_render_script_resolves_configured_defaults_after_parsing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script("render_resume_html.py")
    resume = tmp_path / "resume.yml"
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    resume.write_text("name: Fictional Candidate\n", encoding="utf-8")
    calls = 0

    def fake_config(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return RuntimeConfig(
            paths=WorkspacePaths(master_resume=resume, tmp_dir=temporary),
            settings=Settings(),
            env_file=None,
        )

    monkeypatch.setattr(module, "load_command_config", fake_config)
    monkeypatch.setattr(module, "render_resume_html", lambda *_args, **_kwargs: "ok")
    assert module.main([]) == 0
    assert calls == 1
    assert (temporary / "resume.html").read_text(encoding="utf-8") == "ok"


def test_render_script_explicit_output_does_not_require_temporary_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script("render_resume_html.py")
    resume = tmp_path / "resume.yml"
    output = tmp_path / "explicit.html"
    resume.write_text("name: Fictional Candidate\n", encoding="utf-8")

    def fake_config(_args, *, required, **_kwargs):
        assert required == (module.WorkspaceMember.MASTER_RESUME,)
        return RuntimeConfig(
            paths=WorkspacePaths(master_resume=resume),
            settings=Settings(),
            env_file=None,
        )

    monkeypatch.setattr(module, "load_command_config", fake_config)
    monkeypatch.setattr(module, "render_resume_html", lambda *_args, **_kwargs: "ok")
    assert module.main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--dry-run"], 1),
        (["--dry-run", "--force"], 2),
        (["--dry-run", "--job-id", "fictional-existing"], 0),
        (["--dry-run", "--job-id", "fictional-existing", "--force"], 1),
        (["--dry-run", "--job-id", "fictional-missing"], 0),
        (["--dry-run", "--force", "--limit", "1"], 1),
    ],
)
async def test_first_draft_dry_run_counts_only_eligible_without_boundaries(
    monkeypatch,
    capsys,
    tmp_path: Path,
    arguments: list[str],
    expected: int,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    records = tuple(
        SimpleNamespace(job_id=value)
        for value in (
            "fictional-missing",
            "fictional-new",
            "fictional-existing",
            "fictional-invalid",
        )
    )
    snapshots = {
        "fictional-missing": SimpleNamespace(
            application=SimpleNamespace(
                prompt_job_description=None,
                job_description=None,
            ),
            variants=(),
        ),
        "fictional-new": SimpleNamespace(
            application=SimpleNamespace(
                prompt_job_description="Responsibilities: Build fictional systems.",
                job_description=None,
            ),
            variants=(),
        ),
        "fictional-existing": SimpleNamespace(
            application=SimpleNamespace(
                prompt_job_description="Responsibilities: Test fictional systems.",
                job_description=None,
            ),
            variants=(SimpleNamespace(variant_key="v1"),),
        ),
        "fictional-invalid": SimpleNamespace(
            application=SimpleNamespace(
                prompt_job_description="x" * 500_001,
                job_description=None,
            ),
            variants=(),
        ),
    }
    reads: list[str] = []

    class FakeStore:
        def __init__(self, _paths):
            pass

        def list_applications(self, scope, *, limit):
            assert scope == "active"
            assert limit == module.MAX_QUERY_RESULTS
            return records

        def get_workflow_snapshot(self, job_id):
            reads.append(job_id)
            return snapshots[job_id]

        def upsert_resume_variant_if_revision(self, *_args, **_kwargs):
            pytest.fail("dry-run attempted a state write")

    config = RuntimeConfig(
        paths=WorkspacePaths(
            database=tmp_path / "state.sqlite3",
            output_dir=tmp_path / "artifacts",
            master_resume=tmp_path / "resume.yml",
        ),
        settings=Settings(),
        env_file=None,
    )
    monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    for name in (
        "build_llm_client",
        "initialize_application_resume_object",
        "render_resume_html_from_mapping",
        "render_resume_pdf_from_html",
        "calculate_ats_diagnostics",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *_a, _name=name, **_k: pytest.fail(
                f"dry-run crossed {_name} boundary"
            ),
        )

    assert await module.main_async(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"candidates": expected, "dry_run": True}
    assert reads


def test_aro_regeneration_uses_canonical_query_bound(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_regenerate_aros.py")
    paths = WorkspacePaths(
        database=tmp_path / "state.sqlite3",
        output_dir=tmp_path / "artifacts",
        master_resume=tmp_path / "resume.yml",
    )

    class FakeStore:
        def __init__(self, configured_paths):
            assert configured_paths is paths

        def list_applications(self, scope, *, limit):
            assert scope == "active"
            assert limit == module.MAX_QUERY_RESULTS
            return ()

    monkeypatch.setattr(
        module,
        "load_command_config",
        lambda *_a, **_k: RuntimeConfig(
            paths=paths,
            settings=Settings(),
            env_file=None,
        ),
    )
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)

    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"processed": 0}


def test_highlighting_uses_canonical_query_bound(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_highlight_drafts.py")
    paths = WorkspacePaths(
        root=tmp_path,
        database=tmp_path / "state.sqlite3",
        output_dir=tmp_path / "artifacts",
        master_resume=tmp_path / "resume.yml",
        master_resume_text=tmp_path / "resume.txt",
        tmp_dir=tmp_path / "tmp",
    )

    class FakeStore:
        def __init__(self, configured_paths):
            assert configured_paths is paths

        def list_applications(self, scope, *, limit):
            assert scope == "active"
            assert limit == module.MAX_QUERY_RESULTS
            return ()

    monkeypatch.setattr(
        module,
        "load_command_config",
        lambda *_a, **_k: RuntimeConfig(
            paths=paths,
            settings=Settings(),
            env_file=None,
        ),
    )
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(module, "build_codex_runner", lambda **_k: object())
    monkeypatch.setattr(
        module,
        "with_model_request_policy",
        lambda runner, **_k: runner,
    )

    assert module.main(["--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "dry_run": True,
        "failed": 0,
        "processed": 0,
        "requires_human_review": True,
        "selection_changed": False,
    }


@pytest.mark.asyncio
async def test_first_draft_generation_contains_snapshot_failure_per_record(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    records = (
        SimpleNamespace(job_id="fictional-bad"),
        SimpleNamespace(job_id="fictional-later"),
    )
    later_snapshot = SimpleNamespace(
        application=SimpleNamespace(
            prompt_job_description="Responsibilities: Build fictional systems.",
            job_description=None,
        ),
        variants=(),
    )
    generated: list[str] = []

    class FakeStore:
        def __init__(self, _paths):
            pass

        def list_applications(self, *_args, **_kwargs):
            return records

        def get_workflow_snapshot(self, job_id):
            if job_id == "fictional-bad":
                raise RuntimeError(
                    "token=unsafe HTTP 429 /private/operator/path private resume text"
                )
            return later_snapshot

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    clients = (FakeClient(), FakeClient())
    client_iterator = iter(clients)

    async def fake_generate_one(**kwargs):
        generated.append(kwargs["candidate"].job_id)
        return True

    config = RuntimeConfig(
        paths=WorkspacePaths(
            database=tmp_path / "state.sqlite3",
            output_dir=tmp_path / "artifacts",
            master_resume=tmp_path / "resume.yml",
        ),
        settings=Settings(),
        env_file=None,
    )
    monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(
        module,
        "build_llm_client",
        lambda *_a, **_k: next(client_iterator),
    )
    monkeypatch.setattr(module, "_generate_one", fake_generate_one)

    assert await module.main_async([]) == 1
    rendered = capsys.readouterr().out
    assert json.loads(rendered) == {
        "failed": 1,
        "failure_diagnostics": [{"category": "unexpected", "stage": "candidate_read"}],
        "processed": 1,
    }
    assert "unsafe" not in rendered
    assert "HTTP 429" not in rendered
    assert "operator" not in rendered
    assert "resume text" not in rendered
    assert generated == ["fictional-later"]
    assert all(client.closed for client in clients)


def test_first_draft_failure_diagnostic_vocabularies_are_closed() -> None:
    module = _load_script("application_resume_generate_drafts.py")

    assert {item.value for item in module._FailureStage} == {
        "candidate_read",
        "resume_initialize",
        "core_request",
        "core_apply",
        "jod_request",
        "jod_apply",
        "experience_rewrite_request",
        "experience_rewrite_apply",
        "html_render",
        "pdf_render",
        "ats",
        "state_write",
        "artifact_export",
    }
    assert {item.value for item in module._FailureCategory} == {
        "model",
        "parse",
        "policy",
        "render",
        "ats",
        "state",
        "artifact",
        "local_io",
        "unexpected",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "expected_stage", "error_family", "expected_category"),
    [
        ("resume_initialize", "resume_initialize", "parse", "parse"),
        ("core_request", "core_request", "model", "model"),
        ("core_apply", "core_apply", "parse", "parse"),
        ("jod_request", "jod_request", "model", "model"),
        ("jod_apply", "jod_apply", "parse", "parse"),
        (
            "experience_rewrite_request",
            "experience_rewrite_request",
            "model",
            "model",
        ),
        (
            "experience_rewrite_apply",
            "experience_rewrite_apply",
            "parse",
            "parse",
        ),
        ("html_render", "html_render", "render", "render"),
        ("pdf_render", "pdf_render", "render", "render"),
        ("ats", "ats", "ats", "ats"),
        ("state_write", "state_write", "state", "state"),
        ("artifact_export", "artifact_export", "artifact", "artifact"),
    ],
)
async def test_first_draft_generation_sanitizes_each_stage_failure(
    monkeypatch,
    tmp_path: Path,
    boundary: str,
    expected_stage: str,
    error_family: str,
    expected_category: str,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    unsafe_text = "token=unsafe HTTP 503 /private/operator/path response body"
    errors = {
        "model": module.LlmError(unsafe_text),
        "parse": ValueError(unsafe_text),
        "render": module.ResumeRenderingError(unsafe_text),
        "ats": module.AtsError(unsafe_text),
        "state": module.ApplicationStateError(unsafe_text),
        "artifact": ValueError(unsafe_text),
    }
    error = errors[error_family]
    resume = {"professional_experience": {"jobs": []}}
    request_counts = {"core": 0, "jod": 0, "experience": 0}

    def result_or_failure(name: str, result):
        if boundary == name:
            raise error
        return result

    class FakeCoreClient:
        model = "fictional-core-model"

        async def generate_json(self, _prompt):
            request_counts["core"] += 1
            return result_or_failure("core_request", {})

    class FakeJodClient:
        async def generate_json(self, _prompt):
            request_counts["jod"] += 1
            return result_or_failure("jod_request", {})

        async def generate_text(self, _prompt):
            request_counts["experience"] += 1
            return result_or_failure("experience_rewrite_request", "rewritten")

    class FakeStore:
        def upsert_resume_variant_if_revision(self, *_args, **_kwargs):
            return result_or_failure("state_write", None)

    score = SimpleNamespace(
        overall_score=80,
        parsing_score=90,
        keyword_match_score=70,
        semantic_match_score=75,
        formatting_risk="low",
        missing_high_value_terms=(),
    )
    diagnostics = SimpleNamespace(score=score)
    monkeypatch.setattr(
        module,
        "initialize_application_resume_object",
        lambda *_a, **_k: result_or_failure("resume_initialize", resume),
    )
    monkeypatch.setattr(
        module,
        "build_core_skills_jod_match_prompt",
        lambda **_k: "core prompt",
    )
    monkeypatch.setattr(
        module,
        "apply_core_skill_jod_matches",
        lambda **_k: result_or_failure("core_apply", resume),
    )
    monkeypatch.setattr(
        module,
        "build_jod_requirements_target_prompt",
        lambda **_k: "jod prompt",
    )
    monkeypatch.setattr(
        module,
        "create_job_opening_description_object",
        lambda **_k: result_or_failure("jod_apply", {"requirements_targets": []}),
    )
    monkeypatch.setattr(
        module,
        "attach_job_opening_description_object",
        lambda **_k: resume,
    )
    monkeypatch.setattr(
        module,
        "experience_jobs_for_jod_bullet_rewrite",
        lambda _resume: ({"order": 1},),
    )
    monkeypatch.setattr(
        module,
        "build_experience_job_bullet_rewrite_prompt",
        lambda **_k: "experience prompt",
    )
    monkeypatch.setattr(
        module,
        "replace_experience_job_bullets_from_text_response",
        lambda **_k: result_or_failure("experience_rewrite_apply", resume),
    )
    monkeypatch.setattr(
        module,
        "render_resume_html_from_mapping",
        lambda **_k: result_or_failure("html_render", "<html></html>"),
    )
    monkeypatch.setattr(
        module,
        "render_resume_pdf_from_html",
        lambda _html: result_or_failure("pdf_render", b"synthetic-pdf"),
    )
    monkeypatch.setattr(
        module,
        "calculate_ats_diagnostics",
        lambda **_k: result_or_failure("ats", diagnostics),
    )
    monkeypatch.setattr(module, "asdict", lambda _value: {"synthetic": True})
    monkeypatch.setattr(
        module,
        "export_rendered_resume",
        lambda **_k: result_or_failure("artifact_export", None),
    )
    candidate = module._EligibleCandidate(
        job_id="fictional-job",
        snapshot=SimpleNamespace(revision=object()),
        description="Responsibilities: Build fictional systems.",
    )

    with pytest.raises(module._FirstDraftFailure) as raised:
        await module._generate_one(
            store=FakeStore(),
            paths=WorkspacePaths(master_resume=tmp_path / "resume.yml"),
            candidate=candidate,
            core_client=FakeCoreClient(),
            jod_client=FakeJodClient(),
            jod_model="fictional-model",
            template=None,
            artifact_dir=tmp_path / "exports",
            max_jod_chars=1_000,
            retries=2,
        )

    expected_diagnostic = {
        "category": expected_category,
        "stage": expected_stage,
    }
    if error_family == "model":
        expected_diagnostic["failure_subtype"] = "unexpected_model"
    assert module._failure_diagnostic(raised.value) == expected_diagnostic
    assert str(raised.value) == "First-draft record processing failed."
    assert "unsafe" not in str(raised.value)
    if boundary == "core_request":
        assert request_counts["core"] == 1
    if boundary == "jod_request":
        assert request_counts["jod"] == 1
    if boundary == "experience_rewrite_request":
        assert request_counts["experience"] == 1


@pytest.mark.asyncio
async def test_first_draft_materializes_real_ats_diagnostics_for_real_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    real_run_model_operation = module.run_model_operation
    deadline_calls: list[tuple[object, float | None]] = []

    async def recorded_run_model_operation(operation, **kwargs):
        deadline_calls.append((kwargs["stage"], kwargs.get("timeout_seconds")))
        return await real_run_model_operation(operation, **kwargs)

    monkeypatch.setattr(module, "run_model_operation", recorded_run_model_operation)
    job_id = "fictional-ats-state"
    description = "Responsibilities: Build reliable fictional systems."
    database = tmp_path / "state" / "applications.sqlite3"
    paths = WorkspacePaths(
        database=database,
        output_dir=tmp_path / "artifacts",
        master_resume=tmp_path / "profile" / "MASTER-RESUME.yml",
    )
    store = ApplicationStateStore(paths)
    store.initialize()
    store.seed_application(
        ApplicationMetadata(
            job_id=job_id,
            company="Fictional Systems Cooperative",
            job_title="Synthetic Reliability Engineer",
            job_url=f"https://jobs.example.test/openings/{job_id}",
            source="synthetic_test",
        ),
        source_text=description,
        prompt_text=description,
    )
    before = store.get_workflow_snapshot(job_id)
    diagnostics = AtsDiagnostics(
        score=AtsProxyScore(
            overall_score=83,
            parsing_score=97,
            keyword_match_score=79,
            semantic_match_score=81,
            formatting_risk="low",
            missing_high_value_terms=("bounded gap", "secondary gap"),
        ),
        component_scores=AtsComponentScores(
            overall_score=83,
            parsing_score=97,
            keyword_match_score=79,
            semantic_match_score=81,
            formatting_score=94,
            formatting_risk="low",
        ),
        matched_terms=(
            AtsWeightedTerm(term="python", weight=2.5),
            AtsWeightedTerm(term="automation", weight=1.5),
        ),
        unmatched_weighted_terms=(AtsWeightedTerm(term="bounded gap", weight=3.0),),
        repeated_phrase_terms=(AtsWeightedTerm(term="reliability", weight=1.25),),
        likely_noisy_phrase_matches=(AtsWeightedTerm(term="systems", weight=0.5),),
    )
    raw_diagnostics = asdict(diagnostics)
    assert type(raw_diagnostics["matched_terms"]) is tuple
    raw_write = ResumeVariantWrite(
        variant_key="v1",
        variant_label="Governed first draft",
        source="governed_first_draft",
        application_resume_yaml="basics:\n  name: Synthetic Candidate\n",
        ats=AtsFields(score=83),
        ats_diagnostics=raw_diagnostics,
    )
    with pytest.raises(ApplicationStateValidationError):
        store.upsert_resume_variant_if_revision(
            job_id,
            raw_write,
            expected_revision=before.revision,
        )
    assert store.get_workflow_snapshot(job_id).variants == ()

    resume = {
        "basics": {"name": "Synthetic Candidate"},
        "professional_experience": {"jobs": []},
    }

    class FakeCoreClient:
        model = "synthetic-core-model"

        async def generate_json(self, _prompt):
            return {}

    class FakeJodClient:
        async def generate_json(self, _prompt):
            return {}

        async def generate_text(self, _prompt):
            return "Synthetic rewritten evidence."

    monkeypatch.setattr(
        module,
        "initialize_application_resume_object",
        lambda *_a, **_k: resume,
    )
    monkeypatch.setattr(
        module,
        "build_core_skills_jod_match_prompt",
        lambda **_k: "synthetic core prompt",
    )
    monkeypatch.setattr(
        module,
        "apply_core_skill_jod_matches",
        lambda **_k: resume,
    )
    monkeypatch.setattr(
        module,
        "build_jod_requirements_target_prompt",
        lambda **_k: "synthetic JOD prompt",
    )
    monkeypatch.setattr(
        module,
        "create_job_opening_description_object",
        lambda **_k: {"requirements_targets": []},
    )
    monkeypatch.setattr(
        module,
        "attach_job_opening_description_object",
        lambda **_k: resume,
    )
    monkeypatch.setattr(
        module,
        "experience_jobs_for_jod_bullet_rewrite",
        lambda _resume: ({"order": 1},),
    )
    monkeypatch.setattr(
        module,
        "build_experience_job_bullet_rewrite_prompt",
        lambda **_k: "synthetic experience prompt",
    )
    monkeypatch.setattr(
        module,
        "replace_experience_job_bullets_from_text_response",
        lambda **_k: resume,
    )
    monkeypatch.setattr(
        module,
        "render_resume_html_from_mapping",
        lambda **_k: "<html><body>Synthetic resume</body></html>",
    )
    monkeypatch.setattr(
        module,
        "render_resume_pdf_from_html",
        lambda _html: b"synthetic-pdf",
    )
    monkeypatch.setattr(
        module,
        "calculate_ats_diagnostics",
        lambda **_k: diagnostics,
    )

    assert (
        await module._generate_one(
            store=store,
            paths=paths,
            candidate=module._EligibleCandidate(
                job_id=job_id,
                snapshot=store.get_workflow_snapshot(job_id),
                description=description,
            ),
            core_client=FakeCoreClient(),
            jod_client=FakeJodClient(),
            jod_model="synthetic-jod-model",
            template=None,
            artifact_dir=None,
            max_jod_chars=1_000,
            retries=0,
            timeout_seconds=12.5,
        )
        is True
    )

    after = store.get_workflow_snapshot(job_id)
    assert tuple(variant.variant_key for variant in after.variants) == ("v1",)
    assert after.application.selected_resume_variant == "v1"
    assert after.application.resume_variant_selection_mode == "auto"
    assert deadline_calls == [
        (module.WorkflowStage.V1_CORE, 12.5),
        (module.WorkflowStage.V1_JOD, 12.5),
        (module.WorkflowStage.V1_EXPERIENCE, 12.5),
    ]
    expected_diagnostics = json.loads(json.dumps(raw_diagnostics, allow_nan=False))
    with sqlite3.connect(database) as connection:
        stored_json = connection.execute(
            """
            SELECT ats_diagnostics_json
            FROM application_resume_variants
            WHERE job_id = ? AND variant_key = 'v1'
            """,
            (job_id,),
        ).fetchone()[0]
    persisted_diagnostics = json.loads(stored_json)
    assert persisted_diagnostics == expected_diagnostics
    assert type(persisted_diagnostics["score"]["missing_high_value_terms"]) is list
    assert type(persisted_diagnostics["matched_terms"]) is list
    assert type(persisted_diagnostics["unmatched_weighted_terms"]) is list
    assert type(persisted_diagnostics["repeated_phrase_terms"]) is list
    assert type(persisted_diagnostics["likely_noisy_phrase_matches"]) is list
    assert [item["term"] for item in persisted_diagnostics["matched_terms"]] == [
        "python",
        "automation",
    ]
    assert persisted_diagnostics["component_scores"] == {
        "formatting_risk": "low",
        "formatting_score": 94,
        "keyword_match_score": 79,
        "overall_score": 83,
        "parsing_score": 97,
        "semantic_match_score": 81,
    }

    with pytest.raises(ApplicationStateConflictError):
        store.upsert_resume_variant_if_revision(
            job_id,
            ResumeVariantWrite(
                variant_key="v1",
                variant_label="Governed first draft",
                source="governed_first_draft",
                application_resume_yaml="basics:\n  name: Stale Synthetic Candidate\n",
                ats=AtsFields(score=83),
                ats_diagnostics=expected_diagnostics,
            ),
            expected_revision=before.revision,
        )


@pytest.mark.asyncio
async def test_first_draft_retries_only_typed_timeout_initial_plus_two() -> None:
    module = _load_script("application_resume_generate_drafts.py")
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LlmTimeoutError("unsafe provider response")
        return "accepted"

    assert await module._with_retries(operation, retries=2) == "accepted"
    assert attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RetryableModelError(
            subtype=ModelFailureSubtype.TRANSIENT_HTTP,
            http_status=503,
            retry_after_seconds=0,
        ),
        RetryableModelError(
            subtype=ModelFailureSubtype.EMBEDDED_TRANSIENT,
            retry_after_seconds=0,
        ),
    ],
)
async def test_first_draft_retries_typed_api_transient_with_exact_boundaries(
    capsys: pytest.CaptureFixture[str],
    error: RetryableModelError,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    attempts = 0
    sleeps: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise error
        return "accepted"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    assert await module._with_retries(operation, retries=2, sleep=sleep) == "accepted"
    events = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]
    starts = [event for event in events if event["event"] == "attempt_start"]
    assert attempts == len(starts) == 3
    assert [event["attempt"] for event in starts] == [1, 2, 3]
    assert all(event["total_attempts"] == 3 for event in starts)
    decisions = [event for event in events if event["event"] == "retry_decision"]
    assert [event["retry"] for event in decisions] == [True, True]
    assert all(event["failure_subtype"] == error.subtype.value for event in decisions)
    assert sleeps == [0.0, 0.0]


@pytest.mark.asyncio
async def test_first_draft_does_not_retry_generic_model_failure() -> None:
    module = _load_script("application_resume_generate_drafts.py")
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise module.LlmError("unsafe provider response")

    with pytest.raises(module.LlmError):
        await module._with_retries(operation, retries=2)
    assert attempts == 1


@pytest.mark.asyncio
async def test_first_draft_success_payload_remains_compatible(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    snapshot = SimpleNamespace(
        application=SimpleNamespace(
            prompt_job_description="Responsibilities: Build fictional systems.",
            job_description=None,
        ),
        variants=(),
    )

    class FakeStore:
        def __init__(self, _paths):
            pass

        def list_applications(self, *_args, **_kwargs):
            return (SimpleNamespace(job_id="fictional-job"),)

        def get_workflow_snapshot(self, _job_id):
            return snapshot

    class FakeClient:
        async def aclose(self) -> None:
            pass

    clients = iter((FakeClient(), FakeClient()))
    config = RuntimeConfig(
        paths=WorkspacePaths(
            database=tmp_path / "state.sqlite3",
            output_dir=tmp_path / "artifacts",
            master_resume=tmp_path / "resume.yml",
        ),
        settings=Settings(),
        env_file=None,
    )
    monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(
        module,
        "build_llm_client",
        lambda *_a, **_k: next(clients),
    )

    async def successful_generation(**_kwargs):
        return True

    monkeypatch.setattr(module, "_generate_one", successful_generation)

    assert await module.main_async([]) == 0
    assert json.loads(capsys.readouterr().out) == {"failed": 0, "processed": 1}


@pytest.mark.asyncio
async def test_first_draft_fail_fast_cli_error_hides_record_exception(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_generate_drafts.py")
    unsafe_text = "token=unsafe HTTP 429 /private/operator/path response body"

    class FakeStore:
        def __init__(self, _paths):
            pass

        def list_applications(self, *_args, **_kwargs):
            return (SimpleNamespace(job_id="fictional-job"),)

        def get_workflow_snapshot(self, _job_id):
            raise RuntimeError(unsafe_text)

    class FakeClient:
        async def aclose(self) -> None:
            pass

    clients = iter((FakeClient(), FakeClient()))
    config = RuntimeConfig(
        paths=WorkspacePaths(
            database=tmp_path / "state.sqlite3",
            output_dir=tmp_path / "artifacts",
            master_resume=tmp_path / "resume.yml",
        ),
        settings=Settings(),
        env_file=None,
    )
    monkeypatch.setattr(module, "load_command_config", lambda *_a, **_k: config)
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(
        module,
        "build_llm_client",
        lambda *_a, **_k: next(clients),
    )

    with pytest.raises(SystemExit):
        await module.main_async(["--fail-fast"])

    stderr = capsys.readouterr().err
    assert "First-draft generation could not be completed." in stderr
    assert "unsafe" not in stderr
    assert "HTTP 429" not in stderr
    assert "operator" not in stderr
    assert "response body" not in stderr


def test_first_draft_import_uses_one_parsed_mapping_after_source_replacement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_store_first_draft.py")
    source = tmp_path / "first-draft.yml"
    original = {
        "name": "Fictional Original",
        "professional_experience": {"bullet_points": []},
    }
    replacement = {
        "name": "Fictional Replacement",
        "professional_experience": {"bullet_points": []},
    }
    source.write_text(yaml.safe_dump(original), encoding="utf-8")
    paths = WorkspacePaths(
        root=tmp_path,
        database=tmp_path / "state.sqlite3",
        output_dir=tmp_path / "artifacts",
    )
    template = tmp_path / "private-template.html"
    template.write_text("synthetic template", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, supplied_paths):
            assert supplied_paths is paths

        def get_workflow_snapshot(self, job_id):
            assert job_id == "fictional-job"
            return SimpleNamespace(
                application=SimpleNamespace(
                    prompt_job_description="Responsibilities: Build fictional systems.",
                    job_description=None,
                ),
                revision="fictional-revision",
            )

        def upsert_resume_variant_if_revision(
            self, job_id, variant, *, expected_revision
        ):
            captured["write"] = (job_id, variant, expected_revision)

    real_load = module.load_resume

    def replacing_load(path):
        parsed = real_load(path)
        path.write_text(yaml.safe_dump(replacement), encoding="utf-8")
        captured["parsed"] = parsed
        return parsed

    def fake_render(*, resume, template_path):
        captured["rendered_mapping"] = resume
        assert template_path == template
        return "<html>fictional original</html>"

    score = SimpleNamespace(
        overall_score=80,
        parsing_score=80,
        keyword_match_score=80,
        semantic_match_score=80,
        formatting_risk="low",
        missing_high_value_terms=(),
    )
    diagnostics = SimpleNamespace(score=score)
    monkeypatch.setattr(
        module,
        "load_command_config",
        lambda *_a, **_k: RuntimeConfig(
            paths=paths,
            settings=Settings(),
            env_file=None,
        ),
    )
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(module, "load_resume", replacing_load)
    monkeypatch.setattr(module, "render_resume_html_from_mapping", fake_render)
    monkeypatch.setattr(module, "render_resume_pdf_from_html", lambda html: b"pdf")
    monkeypatch.setattr(module, "calculate_ats_diagnostics", lambda **_k: diagnostics)
    monkeypatch.setattr(module, "asdict", lambda _value: {"synthetic": True})

    assert (
        module.main(
            [
                "--job-id",
                "fictional-job",
                "--input",
                str(source),
                "--template",
                str(template),
                "--output-yaml",
                "exports/review.yml",
                "--output-html",
                "exports/review.html",
                "--output-pdf",
                "exports/review.pdf",
            ]
        )
        == 0
    )
    job_id, variant, revision = captured["write"]
    assert job_id == "fictional-job"
    assert revision == "fictional-revision"
    assert captured["rendered_mapping"] == original
    assert yaml.safe_load(variant.application_resume_yaml) == original
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == replacement
    assert yaml.safe_load((tmp_path / "exports/review.yml").read_text()) == original
    assert (tmp_path / "exports/review.html").read_text() == (
        "<html>fictional original</html>"
    )
    assert (tmp_path / "exports/review.pdf").read_bytes() == b"pdf"


def test_pass_one_prompt_only_uses_stdout_or_explicit_private_output(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_pass_one.py")
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    master = workspace / "MASTER-RESUME.yml"
    master.write_text("name: Fictional Candidate\n", encoding="utf-8")
    jod = workspace / "trimmed-jod.txt"
    jod.write_text("Build synthetic systems.", encoding="utf-8")
    paths = WorkspacePaths(root=workspace, master_resume=master)
    monkeypatch.setattr(
        module,
        "load_command_config",
        lambda *_a, **_k: RuntimeConfig(
            paths=paths,
            settings=Settings(),
            env_file=None,
        ),
    )
    monkeypatch.setattr(
        module,
        "initialize_application_resume_object",
        lambda _path: {"synthetic": True},
    )
    monkeypatch.setattr(
        module,
        "build_core_skills_jod_match_prompt",
        lambda **_kwargs: "synthetic prompt only",
    )

    assert module.main(["--trimmed-jod", str(jod), "--prompt-only"]) == 0
    assert capsys.readouterr().out == "synthetic prompt only\n"

    assert (
        module.main(
            [
                "--trimmed-jod",
                str(jod),
                "--prompt-only",
                "--prompt-output",
                "prompts/pass-one.txt",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    assert (workspace / "prompts/pass-one.txt").read_text() == ("synthetic prompt only")


def test_rendered_resume_export_writes_only_yaml_html_pdf_under_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    template = workspace / "template.html"
    template.write_text("synthetic", encoding="utf-8")
    paths = WorkspacePaths(root=workspace)
    captured: dict[str, object] = {}

    def render(*, resume, template_path):
        captured["resume"] = resume
        captured["template"] = template_path
        return "<html>rendered synthetic</html>"

    monkeypatch.setattr(artifact_exports, "render_resume_html_from_mapping", render)
    monkeypatch.setattr(
        artifact_exports,
        "render_resume_pdf_from_html",
        lambda _html: b"synthetic-pdf",
    )
    result = artifact_exports.export_rendered_resume(
        paths=paths,
        output_dir=Path("rendered"),
        job_id="fictional-job",
        resume={"name": "Fictional Candidate"},
        template_path=template,
    )

    assert result.file_count == 3
    assert captured["template"] == template
    assert {item.name for item in (workspace / "rendered").iterdir()} == {
        "fictional-job.yml",
        "fictional-job.html",
        "fictional-job.pdf",
    }


def test_rendered_resume_export_materializes_governed_immutable_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "private-workspace"
    workspace.mkdir()
    paths = WorkspacePaths(root=workspace)
    resume = MappingProxyType(
        {
            "name": "Fictional Candidate",
            "skills": (MappingProxyType({"category": "Synthetic Systems"}),),
        }
    )
    captured: dict[str, object] = {}

    def render(*, resume, **_kwargs):
        captured["resume"] = resume
        return "<html>rendered synthetic</html>"

    monkeypatch.setattr(artifact_exports, "render_resume_html_from_mapping", render)
    monkeypatch.setattr(
        artifact_exports,
        "render_resume_pdf_from_html",
        lambda _html: b"synthetic-pdf",
    )

    artifact_exports.export_rendered_resume(
        paths=paths,
        output_dir=Path("rendered"),
        job_id="fictional-job",
        resume=resume,
    )

    assert captured["resume"] == {
        "name": "Fictional Candidate",
        "skills": [{"category": "Synthetic Systems"}],
    }
    assert (
        yaml.safe_load(
            (workspace / "rendered" / "fictional-job.yml").read_text(encoding="utf-8")
        )
        == captured["resume"]
    )


def test_private_output_resolution_rejects_symlink_components(tmp_path: Path) -> None:
    workspace = tmp_path / "private-workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(Exception) as caught:
        artifact_exports.export_rendered_resume(
            paths=WorkspacePaths(root=workspace),
            output_dir=Path("linked/rendered"),
            job_id="fictional-job",
            resume={"name": "Fictional Candidate"},
        )
    assert str(outside) not in str(caught.value)


def test_sync_drafts_uses_supported_page_and_preserves_selected_variant(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    module = _load_script("application_resume_sync_drafts_to_aro.py")
    paths = WorkspacePaths(
        database=tmp_path / "state.sqlite3",
        output_dir=tmp_path / "artifacts",
    )
    variant = SimpleNamespace(
        variant_key="manual",
        variant_label="Synthetic manual review",
        parent_variant_key="v2",
        application_resume={"basics": {"name": "Fictional Candidate"}},
        evidence_packet=MappingProxyType(
            {"status": "synthetic", "ids": ("evidence-1",)}
        ),
        external_critique=None,
        critique=MappingProxyType({"status": "bounded"}),
        validation=MappingProxyType({"accepted": ("synthetic evidence",)}),
        model_metadata=MappingProxyType(
            {"review_state": "accepted", "nested": MappingProxyType({"ok": True})}
        ),
    )
    record = SimpleNamespace(
        job_id="fictional-sync",
        selected_resume_variant="manual",
        prompt_job_description="Responsibilities: Build synthetic systems.",
        job_description=None,
    )
    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, configured_paths):
            assert configured_paths == paths

        def list_applications(self, scope, *, limit):
            assert scope == "active"
            captured["page_limit"] = limit
            return (record,)

        def get_workflow_snapshot(self, job_id):
            assert job_id == record.job_id
            return SimpleNamespace(
                application=record,
                variants=(variant,),
                revision="synthetic-revision",
            )

        def upsert_resume_variant_if_revision(
            self,
            job_id,
            write,
            *,
            expected_revision,
        ):
            captured["write"] = (job_id, write, expected_revision)

    score = SimpleNamespace(
        overall_score=81,
        parsing_score=82,
        keyword_match_score=79,
        semantic_match_score=80,
        formatting_risk="low",
        missing_high_value_terms=("bounded term",),
    )
    monkeypatch.setattr(
        module,
        "load_command_config",
        lambda *_a, **_k: RuntimeConfig(
            paths=paths,
            settings=Settings(),
            env_file=None,
        ),
    )
    monkeypatch.setattr(module, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(
        module,
        "render_resume_html_from_mapping",
        lambda **_k: "<html>synthetic resume</html>",
    )
    monkeypatch.setattr(
        module,
        "render_resume_pdf_from_html",
        lambda _html: b"synthetic-pdf",
    )
    monkeypatch.setattr(
        module,
        "calculate_ats_diagnostics",
        lambda **_k: SimpleNamespace(score=score),
    )
    monkeypatch.setattr(
        module,
        "asdict",
        lambda _value: {"synthetic_components": ("bounded",)},
    )

    assert module.main(["--job-id", record.job_id]) == 0
    assert json.loads(capsys.readouterr().out) == {"processed": 1}
    assert captured["page_limit"] == module.MAX_QUERY_RESULTS
    job_id, write, revision = captured["write"]
    assert job_id == record.job_id
    assert revision == "synthetic-revision"
    assert write.variant_key == variant.variant_key
    assert write.parent_variant_key == variant.parent_variant_key
    assert write.evidence_packet == {
        "status": "synthetic",
        "ids": ["evidence-1"],
    }
    assert write.critique == {"status": "bounded"}
    assert write.validation == {"accepted": ["synthetic evidence"]}
    assert write.model_metadata == {
        "review_state": "awaiting_user_review",
        "nested": {"ok": True},
        "render_sync": "packaged_template",
    }
    assert write.ats_diagnostics == {"synthetic_components": ["bounded"]}
