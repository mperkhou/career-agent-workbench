from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

import pytest

from career_agent_workbench import (
    cli_paths,
    jod_cleaner_audit,
    resume_refinement_cli,
    webapp,
)
from career_agent_workbench.config import (
    RuntimeConfig,
    RuntimeOverrides,
    Settings,
    WorkspaceMember,
    WorkspacePaths,
    load_runtime_config,
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
    process_database = tmp_path / "process.sqlite3"
    explicit_database = tmp_path / "explicit.sqlite3"
    env_file = tmp_path / "runtime.env"
    env_file.write_text(
        "CAREER_AGENT_WORKBENCH_WORKSPACE=fictional-workspace\n"
        "CAREER_AGENT_WORKBENCH_DATABASE=dotenv.sqlite3\n",
        encoding="utf-8",
    )
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
    assert captured["kwargs"] == {"apply": False, "sample_limit": 10}
    assert "changed_rows" in capsys.readouterr().out


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

    assert resume_refinement_cli.main(["--job-id", "fictional-job"]) == 0
    assert captured["store_paths"] is paths
    assert captured["runner"][0] is settings
    assert captured["workflow"]["paths"] is paths
    assert captured["workflow"]["job_id"] == "fictional-job"
    assert "processed" in capsys.readouterr().out
