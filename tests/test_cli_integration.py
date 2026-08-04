from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from career_agent_workbench import (
    cli_paths,
    jod_cleaner_audit,
    resume_refinement_cli,
    webapp,
)
from career_agent_workbench.application_state import (
    MAX_QUERY_RESULTS,
    ApplicationStateStore,
)
from career_agent_workbench.config import (
    RuntimeConfig,
    RuntimeOverrides,
    Settings,
    WorkspaceMember,
    WorkspacePaths,
    load_runtime_config,
)
from career_agent_workbench.codex_cli import CodexModelConfig, ModelRequest
from career_agent_workbench.errors import (
    LlmError,
    LlmTimeoutError,
    ModelFailureSubtype,
    NonRetryableModelError,
    RetryableModelError,
)
from career_agent_workbench.workflows import matching


STATE_OPTIONS = {
    "--workspace",
    "--database",
    "--output-dir",
    "--profile-dir",
    "--master-resume",
    "--master-resume-text",
    "--blacklist-path",
    "--tmp-dir",
    "--download-dir",
}


def _assert_state_defaults_are_none(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if STATE_OPTIONS.intersection(action.option_strings):
            assert action.default is None


@pytest.mark.parametrize(
    "builder",
    [
        matching.build_arg_parser,
        jod_cleaner_audit.build_arg_parser,
        resume_refinement_cli.build_arg_parser,
        webapp.build_arg_parser,
    ],
)
def test_console_state_path_defaults_are_none(builder) -> None:
    _assert_state_defaults_are_none(builder())


@pytest.mark.parametrize(
    "entrypoint",
    [matching.main, jod_cleaner_audit.main, resume_refinement_cli.main, webapp.main],
)
@pytest.mark.parametrize("option", ["--help", "--version"])
def test_console_metadata_exits_before_runtime_loading(
    monkeypatch, entrypoint, option: str
) -> None:
    monkeypatch.setattr(
        cli_paths,
        "load_runtime_config",
        lambda **_kwargs: pytest.fail("configuration was loaded for --help"),
    )
    with pytest.raises(SystemExit) as raised:
        entrypoint([option])
    assert raised.value.code == 0


def test_dotenv_workspace_and_explicit_member_precedence(tmp_path: Path) -> None:
    workspace = tmp_path / "fictional-workspace"
    workspace.mkdir()
    process_database = tmp_path / "process.sqlite3"
    explicit_database = tmp_path / "explicit.sqlite3"
    env_file = tmp_path / "bootstrap.env"
    private_env_file = workspace / ".env"
    env_file.write_text(
        "CAREER_AGENT_WORKBENCH_WORKSPACE=fictional-workspace\n"
        "CAREER_AGENT_WORKBENCH_PRIVATE_ENV_FILE=fictional-workspace/.env\n",
        encoding="utf-8",
    )
    private_env_file.write_text(
        "CAREER_AGENT_WORKBENCH_DATABASE=dotenv.sqlite3\n",
        encoding="utf-8",
    )
    os.chmod(private_env_file, 0o600)
    environment = {
        "CAREER_AGENT_WORKBENCH_ENV_FILE": str(env_file),
        "CAREER_AGENT_WORKBENCH_DATABASE": str(process_database),
    }
    configured = load_runtime_config(environ=environment, cwd=tmp_path)
    assert configured.paths.root == workspace
    assert configured.paths.master_resume == workspace / "profile" / "MASTER-RESUME.yml"
    assert configured.paths.database == process_database

    explicit = load_runtime_config(
        overrides=RuntimeOverrides(database=explicit_database),
        environ=environment,
        cwd=tmp_path,
    )
    assert explicit.paths.database == explicit_database
    assert explicit.paths.master_resume == workspace / "profile" / "MASTER-RESUME.yml"


def test_seed_job_legacy_derivations_are_bounded_and_explicit_wins(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated"
    profile = tmp_path / "candidate"
    args = argparse.Namespace(
        database=None,
        output_dir=output,
        master_resume=None,
        profile_dir=profile,
        master_resume_name="resume.yml",
    )
    assert cli_paths.seed_job_derived_paths(args) == (
        output / "tracking" / "applications.sqlite3",
        profile / "resume.yml",
    )

    explicit_database = tmp_path / "explicit.sqlite3"
    explicit_resume = tmp_path / "explicit.yml"
    args.database = explicit_database
    args.master_resume = explicit_resume
    assert cli_paths.seed_job_derived_paths(args) == (
        explicit_database,
        explicit_resume,
    )

    args.master_resume_name = ".."
    assert cli_paths.seed_job_derived_paths(args) == (
        explicit_database,
        explicit_resume,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "",
        ".",
        "..",
        "/",
        "\\",
        "nested/resume.yml",
        "nested\\resume.yml",
        "./resume.yml",
        "folder/../resume.yml",
    ],
)
def test_seed_job_rejects_non_filename_master_resume_names(
    filename: str,
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        database=None,
        output_dir=None,
        master_resume=None,
        profile_dir=tmp_path / "candidate",
        master_resume_name=filename,
    )
    with pytest.raises(cli_paths.CliConfigurationError) as raised:
        cli_paths.seed_job_derived_paths(args)
    assert str(raised.value) == "Command configuration is invalid."


def test_seed_job_rejects_absolute_master_resume_name(tmp_path: Path) -> None:
    args = argparse.Namespace(
        database=None,
        output_dir=None,
        master_resume=None,
        profile_dir=tmp_path / "candidate",
        master_resume_name=str(tmp_path / "resume.yml"),
    )
    with pytest.raises(cli_paths.CliConfigurationError):
        cli_paths.seed_job_derived_paths(args)


def test_command_loader_runs_once_and_requires_only_requested_members(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0
    expected = RuntimeConfig(
        paths=WorkspacePaths(database=tmp_path / "state.sqlite3"),
        settings=Settings(),
        env_file=None,
    )

    def fake_loader(**_kwargs):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(cli_paths, "load_runtime_config", fake_loader)
    args = argparse.Namespace(database=None)
    assert (
        cli_paths.load_command_config(args, required=(WorkspaceMember.DATABASE,))
        is expected
    )
    assert calls == 1


def test_missing_required_state_is_content_free(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_paths,
        "load_runtime_config",
        lambda **_kwargs: RuntimeConfig(
            paths=WorkspacePaths(), settings=Settings(), env_file=None
        ),
    )
    with pytest.raises(cli_paths.CliConfigurationError) as raised:
        cli_paths.load_command_config(
            argparse.Namespace(), required=(WorkspaceMember.DATABASE,)
        )
    assert str(raised.value) == "Command configuration is invalid."


@pytest.mark.asyncio
async def test_matching_console_composes_resolved_boundaries_without_io(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = WorkspacePaths(
        root=tmp_path,
        master_resume=tmp_path / "resume.yml",
        output_dir=tmp_path / "artifacts",
        database=tmp_path / "state.sqlite3",
    )
    config = RuntimeConfig(paths=paths, settings=Settings(), env_file=None)
    captured: dict[str, object] = {}

    def fake_load(args, **kwargs):
        captured["load_args"] = args
        captured["load_kwargs"] = kwargs
        return config

    class FakeStore:
        def __init__(self, supplied_paths):
            captured["store_paths"] = supplied_paths

        def initialize(self):
            captured["initialized"] = True

        def list_applications(self, *_args, **_kwargs):
            return ()

    class FakeProvider:
        def __init__(self, *, user_agent, timeout_seconds):
            captured["provider_settings"] = (user_agent, timeout_seconds)

        async def aclose(self):
            captured["provider_closed"] = True

    class FakeClient:
        async def aclose(self):
            captured["client_closed"] = True

    class FakeService:
        def __init__(self, *, provider, max_results):
            captured["service"] = (provider, max_results)

    class FakeWorkflow:
        def __init__(self, *, service, planner, store, paths):
            captured["workflow"] = (service, planner, store, paths)

        async def run(self, **kwargs):
            captured["run"] = kwargs
            return matching.MatchingWorkflowResult(
                query_outcomes=(),
                newly_seeded_job_ids=(),
                queries_planned=0,
                queries_searched=0,
                jobs_seen=0,
                jobs_seeded=0,
            )

    monkeypatch.setattr(matching, "load_command_config", fake_load)
    monkeypatch.setattr(matching, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(matching, "LinkedInPublicJobsProvider", FakeProvider)
    monkeypatch.setattr(
        matching, "build_llm_client", lambda *_args, **_kwargs: FakeClient()
    )
    monkeypatch.setattr(matching, "JobSearchService", FakeService)
    monkeypatch.setattr(matching, "MatchingWorkflow", FakeWorkflow)

    result = await matching.run_from_cli(matching.build_arg_parser().parse_args([]))
    assert result.jobs_seeded == 0
    assert captured["store_paths"] is paths
    assert captured["workflow"][3] is paths
    assert captured["initialized"] is True
    assert captured["provider_closed"] is True
    assert captured["client_closed"] is True


def test_jod_audit_console_composes_resolved_store_without_database(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    paths = WorkspacePaths(database=tmp_path / "state.sqlite3")
    config = RuntimeConfig(paths=paths, settings=Settings(), env_file=None)
    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, supplied_paths):
            captured["paths"] = supplied_paths

    monkeypatch.setattr(
        jod_cleaner_audit, "load_command_config", lambda *_a, **_k: config
    )
    monkeypatch.setattr(jod_cleaner_audit, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(
        jod_cleaner_audit,
        "audit_tracker_state",
        lambda store, **kwargs: (
            captured.update(store=store, kwargs=kwargs) or {"changed_rows": 0}
        ),
    )
    assert jod_cleaner_audit.main([]) == 0
    assert captured["paths"] is paths
    assert captured["kwargs"] == {
        "apply": False,
        "sample_limit": 10,
        "refresh_ats": True,
        "backup": None,
    }
    assert "changed_rows" in capsys.readouterr().out


def test_jod_audit_apply_refreshes_ats_after_one_default_backup(monkeypatch) -> None:
    stored: list[dict[str, object]] = []
    backups = 0
    record = SimpleNamespace(
        job_id="synthetic-job",
        job_title="Reliability Engineer",
        company="Example Cooperative",
        job_url="https://jobs.example.com/synthetic-job",
        source="example_public",
        job_description="Responsibilities: build reliable Python systems.",
        prompt_job_description="stale",
        resume_pdf=b"synthetic-pdf",
        selected_variant=None,
    )

    class FakeStore:
        def list_applications(self, scope, *, limit):
            assert scope == "all" and limit == MAX_QUERY_RESULTS
            return (record,)

        def store_jod(self, job_id, **kwargs):
            stored.append({"job_id": job_id, **kwargs})

    score = SimpleNamespace(
        overall_score=81,
        parsing_score=82,
        keyword_match_score=83,
        semantic_match_score=84,
        formatting_risk="low",
        missing_high_value_terms=("bounded",),
    )
    diagnostics = SimpleNamespace(score=score)
    monkeypatch.setattr(
        jod_cleaner_audit,
        "calculate_ats_diagnostics",
        lambda **_kwargs: diagnostics,
    )
    monkeypatch.setattr(
        jod_cleaner_audit,
        "asdict",
        lambda _value: {"synthetic": True},
    )

    def backup() -> None:
        nonlocal backups
        backups += 1

    result = jod_cleaner_audit.audit_tracker_state(
        FakeStore(),
        apply=True,
        sample_limit=10,
        backup=backup,
    )

    assert backups == 1
    assert result["backup_created"] is True
    assert result["ats_fields_changed"] == 1
    assert len(stored) == 1
    assert stored[0]["ats"].score == 81


def test_jod_audit_uses_real_store_result_bound(tmp_path: Path) -> None:
    paths = WorkspacePaths(database=(tmp_path / "state.sqlite3").absolute())
    store = ApplicationStateStore(paths)
    store.initialize()

    result = jod_cleaner_audit.audit_tracker_state(
        store,
        apply=False,
        sample_limit=10,
    )

    assert result == {
        "total_rows": 0,
        "usable_source_rows": 0,
        "changed_rows": 0,
        "applied_rows": 0,
        "sample_changed_count": 0,
        "ats_fields_changed": 0,
        "backup_created": False,
    }


def test_jod_sqlite_backup_is_read_only_and_user_only(tmp_path: Path) -> None:
    database = tmp_path / "synthetic.sqlite3"
    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE synthetic (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO synthetic VALUES (1)")
    before = database.read_bytes()

    jod_cleaner_audit._backup_sqlite_database(database)

    backups = tuple(tmp_path.glob("synthetic.sqlite3.backup-*"))
    assert len(backups) == 1
    assert database.read_bytes() == before
    assert backups[0].stat().st_mode & 0o777 == 0o600


def test_refinement_console_composes_resolved_runner_and_workflow(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    paths = WorkspacePaths(
        database=tmp_path / "state.sqlite3",
        output_dir=tmp_path / "artifacts",
        master_resume=tmp_path / "resume.yml",
        master_resume_text=tmp_path / "resume.txt",
    )
    settings = Settings(llm_api_key="fictional")
    config = RuntimeConfig(paths=paths, settings=settings, env_file=None)
    captured: dict[str, object] = {}

    class FakeStore:
        def __init__(self, supplied_paths):
            captured["store_paths"] = supplied_paths

        def list_applications(self, scope, *, limit):
            assert scope == "active"
            assert limit == MAX_QUERY_RESULTS
            return (SimpleNamespace(job_id="fictional-job"),)

    class FakeRunner:
        def __init__(self, supplied_settings, **kwargs):
            captured["runner"] = (supplied_settings, kwargs)

    def fake_refine(**kwargs):
        captured["workflow"] = kwargs
        return SimpleNamespace(stored_variant="v2")

    monkeypatch.setattr(
        resume_refinement_cli, "load_command_config", lambda *_a, **_k: config
    )
    monkeypatch.setattr(resume_refinement_cli, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(resume_refinement_cli, "_ConfiguredLlmRunner", FakeRunner)
    monkeypatch.setattr(resume_refinement_cli, "refine_resume_for_job", fake_refine)

    assert resume_refinement_cli.main(["--all-active"]) == 0
    assert captured["store_paths"] is paths
    assert captured["runner"][0] is settings
    assert captured["workflow"]["paths"] is paths
    assert captured["workflow"]["job_id"] == "fictional-job"
    assert "processed" in capsys.readouterr().out


def test_second_pass_config_only_is_model_and_state_free(
    monkeypatch,
    capsys,
) -> None:
    config = RuntimeConfig(paths=WorkspacePaths(), settings=Settings(), env_file=None)
    monkeypatch.setattr(
        resume_refinement_cli,
        "load_command_config",
        lambda *_a, **_k: config,
    )
    monkeypatch.setattr(
        resume_refinement_cli,
        "ApplicationStateStore",
        lambda *_a, **_k: pytest.fail("config-only created a state store"),
    )
    monkeypatch.setattr(
        resume_refinement_cli,
        "_ConfiguredLlmRunner",
        lambda *_a, **_k: pytest.fail("config-only created a model runner"),
    )

    assert resume_refinement_cli.main(["--config-only"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"config_only": True}
    event = json.loads(captured.err)
    assert event["stage"] == "v2_critique"
    assert event["model"] == "z-ai/glm-5.2"
    assert event["timeout_seconds"] == 600.0
    assert event["retry_count"] == 1
    assert event["total_attempts"] == 2


@pytest.mark.parametrize(
    ("failures", "expected_calls", "expected_attempt"),
    [
        ((LlmTimeoutError("synthetic"),), 2, 2),
        (
            (
                RetryableModelError(
                    subtype=ModelFailureSubtype.TRANSIENT_HTTP,
                    http_status=503,
                    retry_after_seconds=0,
                ),
            ),
            2,
            2,
        ),
        ((LlmError("synthetic"),), 1, None),
        (
            (
                NonRetryableModelError(
                    subtype=ModelFailureSubtype.PERMANENT_HTTP,
                    http_status=400,
                ),
            ),
            1,
            None,
        ),
    ],
)
def test_second_pass_uses_typed_retry_contract(
    monkeypatch,
    failures: tuple[Exception, ...],
    expected_calls: int,
    expected_attempt: int | None,
) -> None:
    class FakeClient:
        model = "synthetic-model"

        def __init__(self) -> None:
            self.calls = 0

        async def generate_text(self, _prompt):
            self.calls += 1
            if self.calls <= len(failures):
                raise failures[self.calls - 1]
            return '{"synthetic": true}'

        async def aclose(self):
            return None

    client = FakeClient()
    monkeypatch.setattr(
        resume_refinement_cli,
        "build_llm_client",
        lambda *_a, **_k: client,
    )
    runner = resume_refinement_cli._ConfiguredLlmRunner(
        Settings(),
        api_model="synthetic-model",
        retries=1,
        timeout_seconds=600,
    )
    request = ModelRequest(
        prompt="Synthetic prompt.",
        config=CodexModelConfig(
            model="synthetic-model",
            reasoning_effort="",
            workflow="refinement",
        ),
    )
    if expected_attempt is None:
        with pytest.raises(LlmError):
            runner.run(request)
    else:
        result = runner.run(request)
        assert result.model_metadata["attempt"] == expected_attempt
    assert client.calls == expected_calls


@pytest.mark.parametrize(
    "response",
    ["not-json", '{"score": NaN}', '{"score": Infinity}'],
)
def test_second_pass_invalid_json_stops_before_completion(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    response: str,
) -> None:
    class FakeClient:
        model = "synthetic-model"

        def __init__(self) -> None:
            self.calls = 0

        async def generate_text(self, _prompt):
            self.calls += 1
            return response

        async def aclose(self):
            return None

    client = FakeClient()
    monkeypatch.setattr(
        resume_refinement_cli,
        "build_llm_client",
        lambda *_a, **_k: client,
    )
    runner = resume_refinement_cli._ConfiguredLlmRunner(
        Settings(),
        api_model="synthetic-model",
        retries=3,
        timeout_seconds=600,
    )
    request = ModelRequest(
        prompt="Synthetic prompt.",
        config=CodexModelConfig(
            model="synthetic-model",
            reasoning_effort="",
            workflow="refinement",
        ),
    )

    with pytest.raises(NonRetryableModelError) as raised:
        runner.run(request)

    assert raised.value.subtype is ModelFailureSubtype.INVALID_GENERATION_JSON
    assert client.calls == 1
    events = [
        json.loads(line)
        for line in capsys.readouterr().err.splitlines()
        if line.strip()
    ]
    assert len([event for event in events if event["event"] == "attempt_start"]) == 1
    assert not any(event["event"] == "attempt_completion" for event in events)
    decision = next(event for event in events if event["event"] == "retry_decision")
    assert decision["retry"] is False
    assert decision["failure_subtype"] == "invalid_generation_json"


def test_second_pass_isolates_rows_and_artifact_exports(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    paths = WorkspacePaths(
        root=tmp_path,
        database=tmp_path / "state.sqlite3",
        output_dir=tmp_path / "output",
        master_resume=tmp_path / "resume.yml",
        master_resume_text=tmp_path / "resume.txt",
    )
    config = RuntimeConfig(paths=paths, settings=Settings(), env_file=None)
    attempts: list[str] = []

    class FakeStore:
        def __init__(self, _paths):
            pass

    class FakeRunner:
        def __init__(self, *_a, **_k):
            pass

    def refine(**kwargs):
        attempts.append(kwargs["job_id"])
        return SimpleNamespace(job_id=kwargs["job_id"], candidate={})

    def export(**kwargs):
        if kwargs["job_id"] == "fictional-first":
            raise OSError("synthetic private path")

    monkeypatch.setattr(
        resume_refinement_cli,
        "load_command_config",
        lambda *_a, **_k: config,
    )
    monkeypatch.setattr(resume_refinement_cli, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(resume_refinement_cli, "_ConfiguredLlmRunner", FakeRunner)
    monkeypatch.setattr(resume_refinement_cli, "refine_resume_for_job", refine)
    monkeypatch.setattr(resume_refinement_cli, "export_rendered_resume", export)
    monkeypatch.setattr(
        resume_refinement_cli,
        "resolve_private_workspace_path",
        lambda *_a, **_k: tmp_path / "exports",
    )

    assert (
        resume_refinement_cli.main(
            [
                "--job-id",
                "fictional-first",
                "--job-id",
                "fictional-later",
                "--artifact-dir",
                "exports",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["processed"] == 1
    assert payload["failed"] == 1
    assert attempts == ["fictional-first", "fictional-later"]
