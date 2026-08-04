from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

import career_agent_workbench.artifact_exports as artifact_exports
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths

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
                raise RuntimeError("synthetic snapshot failure")
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
    assert json.loads(capsys.readouterr().out) == {"failed": 1, "processed": 1}
    assert generated == ["fictional-later"]
    assert all(client.closed for client in clients)


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
