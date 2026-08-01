from __future__ import annotations

import logging
from pathlib import Path

import pytest

from career_agent_workbench import webapp
from career_agent_workbench.application_state import ApplicationMetadata
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths


class _InlineThread:
    def __init__(self, *, target, args, daemon: bool) -> None:
        assert daemon is True
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def _runtime(paths: WorkspacePaths, *, secret: str = "") -> RuntimeConfig:
    return RuntimeConfig(
        paths=paths,
        settings=Settings(llm_api_key=secret),
        env_file=None,
    )


def _paths(tmp_path: Path, name: str = "workspace") -> WorkspacePaths:
    workspace = tmp_path / name
    return WorkspacePaths(
        root=workspace,
        database=workspace / "state" / "applications.sqlite3",
        output_dir=workspace / "artifacts-independent",
        profile_dir=workspace / "profile-independent",
        master_resume=workspace / "resume-independent.yml",
        master_resume_text=workspace / "resume-independent.txt",
        blacklist=workspace / "blacklist-independent.txt",
        tmp_dir=workspace / "temporary-independent",
    )


def _seed(app, *job_ids: str) -> None:
    store = app.extensions["career_agent_workbench"]["store"]
    for job_id in job_ids:
        store.upsert_application(
            ApplicationMetadata(
                job_id=job_id,
                company="Example Systems",
                job_title="Fictional Platform Engineer",
                job_url=f"https://example.com/jobs/{job_id}",
                source="synthetic",
            )
        )


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_web_metadata_exits_before_any_runtime_boundary(
    monkeypatch,
    tmp_path: Path,
    option: str,
) -> None:
    def fail(*_args, **_kwargs):
        pytest.fail("metadata crossed a runtime boundary")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(webapp, "load_command_config", fail)
    monkeypatch.setattr(webapp, "create_app", fail)
    monkeypatch.setattr(webapp.threading, "Thread", fail)
    monkeypatch.setattr(webapp.subprocess, "run", fail)
    with pytest.raises(SystemExit) as raised:
        webapp.main([option])
    assert raised.value.code == 0
    assert not tuple(tmp_path.iterdir())


def test_web_main_loads_once_and_injects_exact_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    runtime = _runtime(paths)
    project_root = tmp_path / "project"
    load_calls: list[object] = []
    create_calls: list[object] = []
    run_calls: list[dict[str, object]] = []

    class FakeApp:
        def run(self, **kwargs) -> None:
            run_calls.append(kwargs)

    def fake_load(args, *, required):
        load_calls.append((args, required))
        assert args.workspace == paths.root
        assert args.database == paths.database
        assert args.output_dir == paths.output_dir
        assert args.profile_dir == paths.profile_dir
        assert args.master_resume == paths.master_resume
        assert args.master_resume_text == paths.master_resume_text
        assert args.blacklist == paths.blacklist
        assert args.tmp_dir == paths.tmp_dir
        return runtime

    def fake_create(supplied_runtime, *, project_root):
        create_calls.append((supplied_runtime, project_root))
        return FakeApp()

    monkeypatch.setattr(webapp, "load_command_config", fake_load)
    monkeypatch.setattr(webapp, "create_app", fake_create)

    assert (
        webapp.main(
            [
                "--workspace",
                str(paths.root),
                "--database",
                str(paths.database),
                "--output-dir",
                str(paths.output_dir),
                "--profile-dir",
                str(paths.profile_dir),
                "--master-resume",
                str(paths.master_resume),
                "--master-resume-text",
                str(paths.master_resume_text),
                "--blacklist-path",
                str(paths.blacklist),
                "--tmp-dir",
                str(paths.tmp_dir),
                "--project-root",
                str(project_root),
                "--host",
                "127.0.0.2",
                "--port",
                "9876",
                "--debug",
            ]
        )
        == 0
    )
    assert len(load_calls) == 1
    assert create_calls == [(runtime, project_root)]
    assert run_calls == [{"host": "127.0.0.2", "port": 9876, "debug": True}]


def test_web_port_validation_is_bounded() -> None:
    parser = webapp.build_arg_parser()
    for value in ("0", "65536", "invalid"):
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(["--port", value])
        assert raised.value.code == 2


def test_create_app_uses_bound_store_and_renders_synthetic_tracker(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    runtime = _runtime(paths)
    app = webapp.create_app(runtime, project_root=project_root)
    extension = app.extensions["career_agent_workbench"]
    assert set(extension) == {
        "paths",
        "store",
        "executor",
        "project_root",
        "actions",
    }
    assert extension["paths"] is paths
    assert runtime.settings not in extension.values()
    assert runtime.env_file not in extension.values()
    extension["store"].assert_workspace_binding(paths)
    _seed(app, "fictional-job")

    response = app.test_client().get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "fictional-job" in page
    assert "Example Systems" in page
    assert "Fictional Platform Engineer" in page
    assert "https://example.com/jobs/fictional-job" in page
    assert app.test_client().get("/?scope=invalid").status_code == 400


def test_background_argv_uses_exact_paths_and_ignores_later_cwd(
    monkeypatch,
    caplog,
    tmp_path: Path,
) -> None:
    secret = "synthetic-secret-sentinel"
    paths = _paths(tmp_path)
    project_root = tmp_path / "bound-project"
    project_root.mkdir()
    (project_root / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    env_file = tmp_path / "synthetic.env"
    env_file.write_text(f"SECRET={secret}\n", encoding="utf-8")
    runtime = RuntimeConfig(
        paths=paths,
        settings=Settings(llm_api_key=secret),
        env_file=env_file,
    )
    commands: list[tuple[str, ...]] = []

    def executor(argv) -> int:
        commands.append(tuple(argv))
        return 0

    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(
        runtime,
        command_executor=executor,
        project_root=project_root,
    )
    _seed(app, "fictional-a", "fictional-b")
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    caplog.set_level(logging.DEBUG)

    response = app.test_client().post(
        "/actions/run",
        data={
            "target": "refine-draft-resumes",
            "job_id": ["fictional-a", "fictional-b"],
        },
    )
    assert response.status_code == 202
    expected = (
        "make",
        "-C",
        str(project_root.resolve()),
        "refine-draft-resumes",
        "JOB_IDS=fictional-a fictional-b",
        f"WORKSPACE={paths.root}",
        f"DATABASE={paths.database}",
        f"OUTPUT_DIR={paths.output_dir}",
        f"PROFILE_DIR={paths.profile_dir}",
        f"MASTER_RESUME={paths.master_resume}",
        f"MASTER_RESUME_TEXT={paths.master_resume_text}",
        f"BLACKLIST={paths.blacklist}",
        f"TMP_DIR={paths.tmp_dir}",
    )
    assert commands == [expected]

    status_text = app.test_client().get("/actions/status").get_data(as_text=True)
    page_text = app.test_client().get("/").get_data(as_text=True)
    assert secret not in repr(commands)
    assert secret not in status_text
    assert secret not in page_text
    assert secret not in caplog.text
    action = app.test_client().get("/actions/status").get_json()["actions"][0]
    assert action["status"] == "completed"
    assert action["return_code"] == 0


def test_action_rejections_precede_executor_and_failures_are_generic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    executor_calls: list[tuple[str, ...]] = []
    thread_count = 0

    class CountingThread(_InlineThread):
        def __init__(self, **kwargs) -> None:
            nonlocal thread_count
            thread_count += 1
            super().__init__(**kwargs)

    def failing_executor(argv) -> int:
        executor_calls.append(tuple(argv))
        raise RuntimeError("synthetic-private-exception")

    monkeypatch.setattr(webapp.threading, "Thread", CountingThread)
    app = webapp.create_app(
        _runtime(paths),
        command_executor=failing_executor,
        project_root=project_root,
    )
    _seed(app, "fictional-job")
    client = app.test_client()
    rejected = (
        {"target": "unsupported", "job_id": "fictional-job"},
        {"target": "refine-draft-resumes"},
        {"target": "refine-draft-resumes", "job_id": "missing-job"},
        {"target": "refine-draft-resumes", "job_id": "malformed job"},
    )
    for payload in rejected:
        response = client.post("/actions/run", data=payload)
        assert response.status_code == 400
        assert response.get_json() == {
            "message": "Action request is invalid.",
            "status": "rejected",
        }
    assert thread_count == 0
    assert executor_calls == []

    accepted = client.post(
        "/actions/run",
        data={"target": "refine-draft-resumes", "job_id": "fictional-job"},
    )
    assert accepted.status_code == 202
    assert thread_count == 1
    assert len(executor_calls) == 1
    status = client.get("/actions/status").get_json()["actions"][0]
    assert status["status"] == "failed"
    assert status["return_code"] == 1
    assert status["message"] == "Action failed."
    assert "synthetic-private-exception" not in repr(status)

    unavailable = webapp.create_app(
        _runtime(paths),
        command_executor=failing_executor,
        project_root=tmp_path / "no-makefile",
    )
    response = unavailable.test_client().post(
        "/actions/run",
        data={"target": "refine-draft-resumes", "job_id": "fictional-job"},
    )
    assert response.status_code == 503
    assert thread_count == 1


def test_action_registries_are_isolated_between_apps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    first = webapp.create_app(
        _runtime(_paths(tmp_path, "first")),
        command_executor=lambda _argv: 0,
        project_root=project_root,
    )
    second = webapp.create_app(
        _runtime(_paths(tmp_path, "second")),
        command_executor=lambda _argv: 0,
        project_root=project_root,
    )
    _seed(first, "fictional-job")

    response = first.test_client().post(
        "/actions/run",
        data={"target": "refine-draft-resumes", "job_id": "fictional-job"},
    )
    assert response.status_code == 202
    assert len(first.test_client().get("/actions/status").get_json()["actions"]) == 1
    assert second.test_client().get("/actions/status").get_json() == {"actions": []}
