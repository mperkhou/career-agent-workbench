from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.fastmcp.exceptions import ToolError

from career_agent_workbench import config as config_module
from career_agent_workbench import providers
from career_agent_workbench.config import (
    RuntimeConfig,
    Settings,
    WorkspacePaths,
    load_runtime_config,
)
from career_agent_workbench.errors import ProviderError
from career_agent_workbench.models import (
    JobDetails,
    JobPosting,
    JobRawPayload,
    JobSearchResult,
)
from career_agent_workbench.workflows import matching

_MODULE = "career_agent_workbench.server"
_SECRET = "fictional-secret-sentinel"


class _FakePublicService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.failure: Exception | None = None

    async def search(self, query):
        self.calls.append(("search", query))
        if self.failure is not None:
            raise self.failure
        posting = JobPosting(
            job_id="123",
            title="Example Platform Engineer",
            company="Example Harbor Systems",
            location="Example Region",
            job_url="https://example.com/jobs/123",
        )
        return JobSearchResult(query=query, count=1, jobs=[posting])

    async def get_details(self, job_id_or_url: str):
        self.calls.append(("details", job_id_or_url))
        if self.failure is not None:
            raise self.failure
        return _details()

    async def get_raw_payload(self, job_id_or_url: str):
        self.calls.append(("raw", job_id_or_url))
        if self.failure is not None:
            raise self.failure
        return JobRawPayload(
            job_id="123",
            detail_url="https://example.com/jobs/123",
            status_code=200,
            content_type="text/html",
            payload_chars=19,
            payload="<p>Example role</p>",
            parsed=_details(),
        )


def _details() -> JobDetails:
    return JobDetails(
        job_id="123",
        title="Example Platform Engineer",
        company="Example Harbor Systems",
        location="Example Region",
        job_url="https://example.com/jobs/123",
        description="Build fictional public systems.",
    )


def _runtime(paths: WorkspacePaths | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        paths=WorkspacePaths() if paths is None else paths,
        settings=Settings(),
        env_file=None,
    )


async def _structured(server, name: str, arguments: dict[str, object]):
    response = await server.call_tool(name, arguments)
    if isinstance(response, tuple):
        return response[1]
    return response


def test_server_module_import_has_no_active_boundary(monkeypatch) -> None:
    sys.modules.pop(_MODULE, None)
    monkeypatch.setattr(
        config_module,
        "load_runtime_config",
        lambda **_kwargs: pytest.fail("configuration loaded during import"),
    )
    monkeypatch.setattr(
        providers,
        "LinkedInPublicJobsProvider",
        lambda **_kwargs: pytest.fail("provider constructed during import"),
    )
    imported = importlib.import_module(_MODULE)
    assert callable(imported.create_server)
    assert callable(imported.main)
    sys.modules.pop(_MODULE, None)


def test_server_loads_once_or_uses_exact_injected_runtime(monkeypatch) -> None:
    server_module = importlib.import_module(_MODULE)
    runtime = _runtime()
    service = _FakePublicService()
    calls = 0

    def fake_load():
        nonlocal calls
        calls += 1
        return runtime

    class FakeProvider:
        name = "linkedin_public"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(server_module, "load_runtime_config", fake_load)
    monkeypatch.setattr(server_module, "LinkedInPublicJobsProvider", FakeProvider)
    server_module.create_server(public_service=service)
    assert calls == 1
    created = server_module.create_server(runtime, public_service=service)
    assert calls == 1
    assert created.name == "Career Agent Workbench"


def test_default_server_constructs_no_workspace_or_request(monkeypatch) -> None:
    server_module = importlib.import_module(_MODULE)
    captured: dict[str, object] = {}

    class FakeProvider:
        name = "linkedin_public"

        def __init__(self, *, user_agent, timeout_seconds):
            captured["settings"] = (user_agent, timeout_seconds)

    monkeypatch.setattr(server_module, "LinkedInPublicJobsProvider", FakeProvider)
    created = server_module.create_server(_runtime())
    assert captured["settings"] == (
        Settings().user_agent,
        Settings().timeout_seconds,
    )
    assert asyncio.run(created.list_tools())


def test_tool_names_and_schemas_are_business_only() -> None:
    server_module = importlib.import_module(_MODULE)
    created = server_module.create_server(
        _runtime(), public_service=_FakePublicService()
    )
    tools = asyncio.run(created.list_tools())
    schemas = {tool.name: tool.inputSchema for tool in tools}
    assert set(schemas) == {
        "search_linkedin_jobs",
        "get_linkedin_job_details",
        "get_linkedin_job_raw_payload",
        "find_matching_linkedin_jobs",
    }
    assert set(schemas["find_matching_linkedin_jobs"]["properties"]) == {
        "location",
        "date_posted",
        "limit_per_query",
        "max_queries",
        "max_jobs",
    }
    assert set(schemas["get_linkedin_job_details"]["properties"]) == {"job_id_or_url"}
    assert set(schemas["get_linkedin_job_raw_payload"]["properties"]) == {
        "job_id_or_url"
    }
    search_fields = set(schemas["search_linkedin_jobs"]["properties"])
    assert search_fields == {
        "keywords",
        "location",
        "date_posted",
        "job_type",
        "workplace_type",
        "experience_level",
        "sort_by",
        "distance",
        "limit",
        "page",
        "exclude_job_ids",
    }
    exclusion_schema = schemas["search_linkedin_jobs"]["properties"]["exclude_job_ids"]
    assert exclusion_schema["anyOf"][0]["maxItems"] == 500
    rendered = json.dumps(schemas, sort_keys=True)
    assert _SECRET not in rendered
    for state_name in (
        "workspace",
        "database",
        "output_dir",
        "profile_dir",
        "master_resume",
        "blacklist",
        "tmp_dir",
    ):
        assert state_name not in rendered


def test_injected_offline_stdio_handshake_lists_exact_tools(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    async def exercise() -> tuple[str, ...]:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(project_root / "src"),
        }
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "career_agent_workbench.server"],
            env=environment,
            cwd=tmp_path,
        )
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await session.list_tools()
                return tuple(sorted(tool.name for tool in result.tools))

    assert asyncio.run(exercise()) == (
        "find_matching_linkedin_jobs",
        "get_linkedin_job_details",
        "get_linkedin_job_raw_payload",
        "search_linkedin_jobs",
    )


def test_exclusion_outer_bound_rejects_duplicates_before_service() -> None:
    server_module = importlib.import_module(_MODULE)
    service = _FakePublicService()
    created = server_module.create_server(_runtime(), public_service=service)
    with pytest.raises(ToolError):
        asyncio.run(
            created.call_tool(
                "search_linkedin_jobs",
                {
                    "keywords": "Python",
                    "location": "Example Region",
                    "exclude_job_ids": ["duplicate-123"] * 501,
                },
            )
        )
    assert service.calls == []


def test_mcp_entrypoint_dependency_and_sdist_membership() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))
    assert "mcp>=1.13,<2" in metadata["project"]["dependencies"]
    assert metadata["project"]["scripts"]["career-agent-workbench-mcp"] == (
        "career_agent_workbench.server:main"
    )
    included = set(metadata["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])
    assert {
        "/src/career_agent_workbench/server.py",
        "/src/career_agent_workbench/tools/__init__.py",
        "/src/career_agent_workbench/tools/jobs.py",
        "/src/career_agent_workbench/tools/matching.py",
        "/tests/test_mcp_integration.py",
    } <= included


def test_public_tools_work_with_completely_empty_workspace() -> None:
    server_module = importlib.import_module(_MODULE)
    service = _FakePublicService()
    runner_calls = 0

    async def runner(*_args, **_kwargs):
        nonlocal runner_calls
        runner_calls += 1
        raise AssertionError("matching must remain lazy")

    created = server_module.create_server(
        _runtime(), public_service=service, matching_runner=runner
    )

    async def exercise():
        search = await _structured(
            created,
            "search_linkedin_jobs",
            {"keywords": "Python", "location": "Example Region"},
        )
        details = await _structured(
            created, "get_linkedin_job_details", {"job_id_or_url": "123"}
        )
        raw = await _structured(
            created, "get_linkedin_job_raw_payload", {"job_id_or_url": "123"}
        )
        return search, details, raw

    search, details, raw = asyncio.run(exercise())
    assert search["count"] == 1
    assert details["job_id"] == "123"
    assert raw["payload_type"] == "html"
    assert [item[0] for item in service.calls] == ["search", "details", "raw"]
    assert runner_calls == 0


def test_missing_matching_paths_fail_before_runner_or_service() -> None:
    server_module = importlib.import_module(_MODULE)
    service = _FakePublicService()
    runner_calls = 0

    async def runner(*_args, **_kwargs):
        nonlocal runner_calls
        runner_calls += 1
        raise AssertionError("runner crossed missing configuration")

    created = server_module.create_server(
        _runtime(), public_service=service, matching_runner=runner
    )
    result = asyncio.run(_structured(created, "find_matching_linkedin_jobs", {}))
    assert result == {"error": "Matching tool configuration is invalid."}
    assert runner_calls == 0
    assert service.calls == []


def test_matching_receives_exact_runtime_bounds_and_returns_counts(tmp_path: Path):
    server_module = importlib.import_module(_MODULE)
    runtime = _runtime(
        WorkspacePaths(
            master_resume=tmp_path / "profile.yml",
            output_dir=tmp_path / "output",
            database=tmp_path / "state.sqlite3",
        )
    )
    captured: dict[str, object] = {}

    async def runner(received_runtime, *, bounds, service):
        captured.update(runtime=received_runtime, bounds=bounds, service=service)
        return matching.MatchingWorkflowResult(
            query_outcomes=(),
            newly_seeded_job_ids=(),
            queries_planned=3,
            queries_searched=2,
            jobs_seen=9,
            jobs_seeded=4,
        )

    service = _FakePublicService()
    created = server_module.create_server(
        runtime, public_service=service, matching_runner=runner
    )
    result = asyncio.run(
        _structured(
            created,
            "find_matching_linkedin_jobs",
            {
                "location": "Example Region",
                "date_posted": "past_month",
                "limit_per_query": 7,
                "max_queries": 4,
                "max_jobs": 8,
            },
        )
    )
    assert captured["runtime"] is runtime
    assert captured["service"] is service
    bounds = captured["bounds"]
    assert bounds.location == "Example Region"
    assert bounds.date_posted == "past_month"
    assert bounds.workplace_types == ("remote", "hybrid")
    assert bounds.experience_levels == ("associate", "mid_senior", "director")
    assert bounds.job_types == ("full_time", "contract")
    assert (bounds.limit_per_query, bounds.max_queries, bounds.max_jobs) == (7, 4, 8)
    assert result == {
        "queries_planned": 3,
        "queries_searched": 2,
        "jobs_seen": 9,
        "jobs_seeded": 4,
    }


def test_failures_instructions_schemas_and_logs_hide_secret(caplog, tmp_path: Path):
    server_module = importlib.import_module(_MODULE)
    service = _FakePublicService()
    service.failure = RuntimeError(_SECRET)

    async def runner(*_args, **_kwargs):
        raise RuntimeError(_SECRET)

    runtime = RuntimeConfig(
        paths=WorkspacePaths(
            master_resume=tmp_path / "profile.yml",
            output_dir=tmp_path / "output",
            database=tmp_path / "state.sqlite3",
        ),
        settings=Settings(llm_api_key=_SECRET),
        env_file=tmp_path / "secret.env",
    )
    created = server_module.create_server(
        runtime, public_service=service, matching_runner=runner
    )

    async def exercise():
        public = await _structured(
            created,
            "search_linkedin_jobs",
            {"keywords": "Python", "location": "Example Region"},
        )
        matched = await _structured(created, "find_matching_linkedin_jobs", {})
        schemas = [tool.inputSchema for tool in await created.list_tools()]
        return public, matched, schemas

    public, matched, schemas = asyncio.run(exercise())
    assert public == {"error": "Public job tool failed."}
    assert matched == {"error": "Matching tool failed."}
    evidence = " ".join(
        (
            created.instructions or "",
            json.dumps(schemas, sort_keys=True),
            caplog.text,
            json.dumps(public),
            json.dumps(matched),
        )
    )
    assert _SECRET not in evidence
    assert str(tmp_path) not in evidence

    service.failure = ProviderError("Public job search request failed.")
    expected = asyncio.run(
        _structured(
            created,
            "search_linkedin_jobs",
            {"keywords": "Python", "location": "Example Region"},
        )
    )
    assert expected == {"error": "Public job search request failed."}


def test_unrelated_cwd_dotenv_runtime_is_injected_once(
    monkeypatch, tmp_path: Path
) -> None:
    server_module = importlib.import_module(_MODULE)
    workspace = tmp_path / "fictional-workspace"
    env_file = tmp_path / "selected.env"
    env_file.write_text(
        f"CAREER_AGENT_WORKBENCH_WORKSPACE={workspace}\n", encoding="utf-8"
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    loads = 0

    def real_load_once():
        nonlocal loads
        loads += 1
        return load_runtime_config(
            environ={"CAREER_AGENT_WORKBENCH_ENV_FILE": str(env_file)},
            cwd=unrelated,
        )

    captured: dict[str, object] = {}

    async def runner(runtime, **_kwargs):
        captured["runtime"] = runtime
        return matching.MatchingWorkflowResult((), (), 0, 0, 0, 0)

    monkeypatch.setattr(server_module, "load_runtime_config", real_load_once)
    created = server_module.create_server(
        public_service=_FakePublicService(), matching_runner=runner
    )
    result = asyncio.run(_structured(created, "find_matching_linkedin_jobs", {}))
    runtime = captured["runtime"]
    assert loads == 1
    assert runtime.paths.root == workspace
    assert runtime.paths.master_resume == workspace / "profile/MASTER-RESUME.yml"
    assert runtime.paths.output_dir == workspace / "output"
    assert runtime.paths.database == workspace / "output/tracking/applications.sqlite3"
    assert result["jobs_seeded"] == 0


def test_main_runs_created_server_once(monkeypatch) -> None:
    server_module = importlib.import_module(_MODULE)
    calls = 0

    class FakeServer:
        def run(self):
            nonlocal calls
            calls += 1

    monkeypatch.setattr(server_module, "create_server", lambda: FakeServer())
    assert server_module.main() is None
    assert calls == 1


@pytest.mark.parametrize("workflow_fails", [False, True])
def test_runtime_composition_closes_only_constructed_capabilities(
    monkeypatch, tmp_path: Path, workflow_fails: bool
) -> None:
    paths = WorkspacePaths(
        master_resume=tmp_path / "profile.yml",
        output_dir=tmp_path / "output",
        database=tmp_path / "state.sqlite3",
    )
    runtime = _runtime(paths)
    captured: dict[str, object] = {"provider_closed": 0, "client_closed": 0}

    class FakeStore:
        def __init__(self, supplied_paths):
            assert supplied_paths is paths

        def initialize(self):
            captured["initialized"] = True

        def list_applications(self, *_args, **_kwargs):
            return (SimpleNamespace(job_id="existing-1"),)

    class FakeProvider:
        def __init__(self, **_kwargs):
            pass

        async def aclose(self):
            captured["provider_closed"] += 1

    class FakeClient:
        async def aclose(self):
            captured["client_closed"] += 1

    class FakeService:
        def __init__(self, *, provider, max_results):
            captured["service"] = (provider, max_results)

    class FakeWorkflow:
        def __init__(self, **kwargs):
            captured["workflow"] = kwargs

        async def run(self, **kwargs):
            captured["run"] = kwargs
            if workflow_fails:
                raise matching.MatchingWorkflowError("Matching workflow failed.")
            return matching.MatchingWorkflowResult((), (), 0, 0, 0, 0)

    monkeypatch.setattr(matching, "ApplicationStateStore", FakeStore)
    monkeypatch.setattr(matching, "LinkedInPublicJobsProvider", FakeProvider)
    monkeypatch.setattr(matching, "JobSearchService", FakeService)
    monkeypatch.setattr(matching, "build_llm_client", lambda *_a, **_k: FakeClient())
    monkeypatch.setattr(matching, "MatchingWorkflow", FakeWorkflow)
    bounds = matching.MatchingBounds(
        location="Example Region",
        date_posted="past_week",
        workplace_types=("remote",),
        experience_levels=("associate",),
        job_types=("full_time",),
    )

    if workflow_fails:
        with pytest.raises(matching.MatchingWorkflowError):
            asyncio.run(matching.run_matching_from_runtime(runtime, bounds=bounds))
    else:
        asyncio.run(matching.run_matching_from_runtime(runtime, bounds=bounds))
    assert captured["initialized"] is True
    assert captured["run"]["existing_job_ids"] == ("existing-1",)
    assert captured["provider_closed"] == 1
    assert captured["client_closed"] == 1

    workflow_fails = False

    class CallerOwned:
        async def aclose(self):
            pytest.fail("injected capability was closed")

    asyncio.run(
        matching.run_matching_from_runtime(
            runtime,
            bounds=bounds,
            service=CallerOwned(),
            planner=CallerOwned(),
        )
    )


def test_runtime_matching_uses_real_store_supported_query_bound(
    tmp_path: Path,
) -> None:
    master_resume = tmp_path / "profile.yml"
    master_resume.write_text(
        "professional_summary:\n  text: Builds reliable fictional public systems.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    paths = WorkspacePaths(
        master_resume=master_resume,
        output_dir=output_dir,
        database=tmp_path / "tracking" / "applications.sqlite3",
    )
    runtime = _runtime(paths)

    class InertPlanner:
        async def plan_queries(self, _request):
            return ()

    class InertService:
        async def search(self, _query):
            pytest.fail("empty planned work reached the public service")

        async def get_details(self, _job_id_or_url):
            pytest.fail("empty planned work reached the public service")

    bounds = matching.MatchingBounds(
        location="Example Region",
        date_posted="past_week",
        workplace_types=("remote",),
        experience_levels=("associate",),
        job_types=("full_time",),
    )
    result = asyncio.run(
        matching.run_matching_from_runtime(
            runtime,
            bounds=bounds,
            service=InertService(),
            planner=InertPlanner(),
        )
    )
    assert result == matching.MatchingWorkflowResult((), (), 0, 0, 0, 0)
    assert paths.database is not None and paths.database.is_file()


def test_cli_loads_once_and_delegates_to_runtime_seam(monkeypatch, tmp_path: Path):
    runtime = _runtime(
        WorkspacePaths(
            master_resume=tmp_path / "profile.yml",
            output_dir=tmp_path / "output",
            database=tmp_path / "state.sqlite3",
        )
    )
    captured: dict[str, object] = {"loads": 0}

    def fake_load(*_args, **_kwargs):
        captured["loads"] += 1
        return runtime

    async def fake_run(received, *, bounds, **_kwargs):
        captured.update(runtime=received, bounds=bounds)
        return matching.MatchingWorkflowResult((), (), 0, 0, 0, 0)

    monkeypatch.setattr(matching, "load_command_config", fake_load)
    monkeypatch.setattr(matching, "run_matching_from_runtime", fake_run)
    result = asyncio.run(
        matching.run_from_cli(matching.build_arg_parser().parse_args([]))
    )
    assert result.jobs_seeded == 0
    assert captured["loads"] == 1
    assert captured["runtime"] is runtime
    assert captured["bounds"].workplace_types == ("remote", "hybrid")
