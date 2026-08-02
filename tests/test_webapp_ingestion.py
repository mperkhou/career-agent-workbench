from __future__ import annotations

from pathlib import Path

from career_agent_workbench import webapp
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    AtsFields,
    ResumeVariantWrite,
)
from career_agent_workbench.config import RuntimeConfig, Settings, WorkspacePaths
from career_agent_workbench.generic_job_scraper import generic_job_id
from career_agent_workbench.models import JobDetails
from career_agent_workbench.webapp_actions import CommandExecution


class _InlineThread:
    def __init__(self, *, target, args, daemon: bool) -> None:
        assert daemon is True
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def _paths(tmp_path: Path, name: str = "workspace") -> WorkspacePaths:
    workspace = tmp_path / name
    return WorkspacePaths(
        root=workspace,
        database=workspace / "state" / "applications.sqlite3",
        output_dir=workspace / "artifacts",
        profile_dir=workspace / "profile",
        master_resume=workspace / "profile" / "MASTER-RESUME.yml",
        master_resume_text=workspace / "profile" / "MASTER-RESUME.txt",
        blacklist=workspace / "profile" / "blacklist.txt",
        tmp_dir=workspace / "tmp",
    )


def _runtime(paths: WorkspacePaths) -> RuntimeConfig:
    return RuntimeConfig(paths=paths, settings=Settings(), env_file=None)


def _view_form() -> dict[str, str]:
    return {
        "view_q": "platform",
        "view_status": "all",
        "view_scope": "all",
        "view_sort": "company",
        "view_direction": "asc",
    }


def _details(
    job_id: str,
    *,
    url: str,
    company: str = "Example Systems",
    title: str = "Fictional Platform Engineer",
    source: str = "linkedin_public",
) -> JobDetails:
    return JobDetails(
        job_id=job_id,
        title=title,
        company=company,
        listed_at="2026-08-01T12:00:00Z",
        job_url=url,
        source=source,
        description=(
            "Responsibilities: build fictional public platform services with Python, "
            "reliable APIs, careful testing, automation, and clear operational reviews. "
            "Qualifications: evidence-based engineering experience."
        ),
        seniority_level="mid_senior",
    )


def _generic_html(*, title: str = "Fictional Reliability Engineer") -> str:
    description = (
        "Design and operate fictional public services with Python, reliable APIs, "
        "automated testing, observability, documentation, and collaborative reviews. "
        "This synthetic posting contains enough bounded detail for the parser."
    )
    return f"""
    <html><head><script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "JobPosting",
      "title": "{title}",
      "description": "{description}",
      "datePosted": "2026-08-01",
      "hiringOrganization": {{"@type": "Organization", "name": "Example Harbor"}}
    }}
    </script></head><body><main>{description}</main></body></html>
    """


def test_linkedin_ingestion_deduplicates_and_reports_generic_partial_counts(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> JobDetails:
        calls.append(url)
        job_id = url.rstrip("/").rsplit("/", 1)[-1]
        if job_id == "456":
            raise RuntimeError("fictional provider failure with hidden details")
        return _details(job_id, url=url)

    app = webapp.create_app(
        _runtime(_paths(tmp_path)),
        project_root=tmp_path / "no-project",
        linkedin_details_fetcher=fetcher,
        generic_html_fetcher=lambda _url: _generic_html(),
    )
    response = app.test_client().post(
        "/applications/add/linkedin",
        data={
            **_view_form(),
            "linkedin_urls": (
                "https://www.linkedin.com/jobs/view/123/?trk=one\n"
                "https://www.linkedin.com/jobs/view/123\n"
                "https://www.linkedin.com/jobs/view/456\n"
                "https://example.com/not-linkedin"
            ),
        },
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "accepted": 1,
        "created": 1,
        "failed": 2,
        "message": "URL ingestion completed.",
        "refreshed": 0,
        "status": "partial",
    }
    assert calls == [
        "https://www.linkedin.com/jobs/view/123",
        "https://www.linkedin.com/jobs/view/456",
    ]
    record = app.extensions["career_agent_workbench"]["store"].get_application("123")
    assert record.company == "Example Systems"
    assert record.source == "linkedin_public"
    assert record.job_description is not None
    assert record.prompt_job_description is not None
    assert "hidden details" not in response.get_data(as_text=True)


def test_existing_url_ingestion_refreshes_only_metadata_and_jod(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    def fetcher(url: str) -> JobDetails:
        return _details(
            "123",
            url=url,
            company="Refreshed Example Systems",
            title="Refreshed Fictional Role",
        )

    app = webapp.create_app(
        _runtime(paths),
        project_root=tmp_path / "no-project",
        linkedin_details_fetcher=fetcher,
        generic_html_fetcher=lambda _url: _generic_html(),
    )
    store = app.extensions["career_agent_workbench"]["store"]
    store.seed_application(
        ApplicationMetadata(
            job_id="123",
            company="Original Example Systems",
            job_title="Original Fictional Role",
            job_url="https://www.linkedin.com/jobs/view/123",
            source="synthetic",
        ),
        source_text="Original fictional source description.",
        prompt_text="Original fictional prompt description.",
    )
    store.update_application_status(
        "123", applied_to="Yes", date_applied="2026-07-01", notes="Keep this note"
    )
    store.upsert_resume_variant(
        "123",
        ResumeVariantWrite(
            variant_key="v1",
            variant_label="Draft v1",
            source="synthetic",
            application_resume_yaml="name: Fictional Candidate\n",
            resume_html="<p>Keep this fictional resume</p>",
            resume_pdf=b"keep-this-fictional-pdf",
            ats=AtsFields(score=73),
        ),
    )
    store.select_resume_variant("123", "v1")
    store.store_clo("123", value={"body": "Keep this fictional CLO"})
    store.archive(("123",))
    before = store.get_application("123")
    variants_before = store.list_resume_variants("123")

    response = app.test_client().post(
        "/applications/add/linkedin",
        data={
            **_view_form(),
            "linkedin_urls": "https://www.linkedin.com/jobs/view/123",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["refreshed"] == 1
    after = store.get_application("123")
    assert after.company == "Refreshed Example Systems"
    assert after.job_title == "Refreshed Fictional Role"
    assert after.job_description != before.job_description
    assert after.prompt_job_description != before.prompt_job_description
    assert after.applied_to == before.applied_to
    assert after.date_applied == before.date_applied
    assert after.notes == before.notes
    assert after.archived_at == before.archived_at
    assert after.selected_resume_variant == before.selected_resume_variant
    assert after.resume_variant_selection_mode == before.resume_variant_selection_mode
    assert after.application_resume == before.application_resume
    assert after.resume_html == before.resume_html
    assert after.resume_pdf == before.resume_pdf
    assert after.ats == before.ats
    assert after.cover_letter == before.cover_letter
    assert store.list_resume_variants("123") == variants_before


def test_generic_ingestion_uses_injected_fetch_and_existing_pure_parser(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> str:
        calls.append(url)
        if "failed" in url:
            raise RuntimeError("synthetic HTTP details stay hidden")
        return _generic_html()

    app = webapp.create_app(
        _runtime(_paths(tmp_path)),
        project_root=tmp_path / "no-project",
        linkedin_details_fetcher=lambda url: _details("123", url=url),
        generic_html_fetcher=fetcher,
    )
    response = app.test_client().post(
        "/applications/add/other",
        data={
            **_view_form(),
            "other_urls": (
                "https://jobs.example.com/opening?utm_source=test\n"
                "https://jobs.example.com/opening\n"
                "https://jobs.example.com/failed"
            ),
        },
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "accepted": 1,
        "created": 1,
        "failed": 1,
        "message": "URL ingestion completed.",
        "refreshed": 0,
        "status": "partial",
    }
    assert calls == [
        "https://jobs.example.com/opening",
        "https://jobs.example.com/failed",
    ]
    store = app.extensions["career_agent_workbench"]["store"]
    record = store.get_application(generic_job_id("https://jobs.example.com/opening"))
    assert record.company == "Example Harbor"
    assert record.source == "generic_url"
    assert record.job_description is not None
    assert record.prompt_job_description is not None
    assert "synthetic HTTP details" not in response.get_data(as_text=True)


def test_seed_runs_in_app_scoped_background_action_with_bounded_make_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(
        _runtime(paths),
        project_root=project_root,
        command_executor=lambda argv: commands.append(tuple(argv)) or 0,
        linkedin_details_fetcher=lambda url: _details("123", url=url),
        generic_html_fetcher=lambda _url: _generic_html(),
    )
    client = app.test_client()
    page = client.get(
        "/applications/add",
        query_string={
            "q": "platform",
            "status": "all",
            "scope": "all",
            "sort": "company",
            "direction": "asc",
        },
    )
    assert page.status_code == 200
    page_text = page.get_data(as_text=True)
    assert "Return to preserved tracker view" in page_text
    assert 'data-background-form="seed"' in page_text
    assert 'data-ingestion-form="linkedin"' in page_text
    assert 'data-ingestion-form="generic"' in page_text
    assert page_text.count('name="run_v1"') == 3
    assert page_text.count('name="run_v2"') == 3
    assert page_text.count('name="run_manual"') == 3
    assert page_text.count('name="run_highlight"') == 3
    assert 'id="action-progress-bar"' in page_text

    response = client.post(
        "/applications/add/seed",
        data={
            **_view_form(),
            "location": "Fictional City",
            "date_posted": "past_week",
            "limit_per_query": "7",
            "max_queries": "4",
            "max_jobs": "3",
        },
    )
    assert response.status_code == 202
    assert response.get_json()["status"] == "accepted"
    assert commands == [
        (
            "make",
            "-C",
            str(project_root.resolve()),
            "seed-jobs",
            "LOCATION=Fictional City",
            "DATE_POSTED=past_week",
            "LIMIT_PER_QUERY=7",
            "MAX_QUERIES=4",
            "MAX_JOBS=3",
            f"WORKSPACE={paths.root}",
            f"DATABASE={paths.database}",
            f"OUTPUT_DIR={paths.output_dir}",
            f"PROFILE_DIR={paths.profile_dir}",
            f"MASTER_RESUME={paths.master_resume}",
            f"MASTER_RESUME_TEXT={paths.master_resume_text}",
            f"BLACKLIST={paths.blacklist}",
            f"TMP_DIR={paths.tmp_dir}",
        )
    ]
    action = client.get("/actions/status").get_json()["actions"][0]
    assert action["target"] == "Seed and match jobs"
    assert action["status"] == "completed"

    for overrides in (
        {"max_jobs": "0"},
        {"max_queries": "101"},
        {"limit_per_query": "not-a-number"},
        {"date_posted": "invalid"},
        {"location": "bad\nlocation"},
    ):
        payload = {
            **_view_form(),
            "location": "Fictional City",
            "date_posted": "past_week",
            "limit_per_query": "7",
            "max_queries": "4",
            "max_jobs": "3",
            **overrides,
        }
        assert client.post("/applications/add/seed", data=payload).status_code == 400
    assert len(commands) == 1


def test_invalid_ingestion_dependencies_fail_before_fetch_or_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fetch_calls: list[str] = []
    command_calls: list[tuple[str, ...]] = []
    thread_count = 0

    class CountingThread(_InlineThread):
        def __init__(self, **kwargs) -> None:
            nonlocal thread_count
            thread_count += 1
            super().__init__(**kwargs)

    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    monkeypatch.setattr(webapp.threading, "Thread", CountingThread)
    app = webapp.create_app(
        _runtime(_paths(tmp_path)),
        project_root=project,
        command_executor=lambda argv: command_calls.append(tuple(argv)) or 0,
        linkedin_details_fetcher=lambda url: (
            fetch_calls.append(url) or _details("123", url=url)
        ),
        generic_html_fetcher=lambda url: fetch_calls.append(url) or _generic_html(),
    )
    client = app.test_client()
    for route, payload in (
        (
            "/applications/add/linkedin",
            {
                "linkedin_urls": "https://www.linkedin.com/jobs/view/123",
                "run_v2": "1",
            },
        ),
        (
            "/applications/add/other",
            {
                "other_urls": "https://jobs.example.com/opening",
                "run_v1": "1",
                "run_manual": "1",
            },
        ),
        (
            "/applications/add/seed",
            {
                "location": "Fictional City",
                "date_posted": "past_week",
                "limit_per_query": "7",
                "max_queries": "4",
                "max_jobs": "3",
                "run_highlight": "1",
            },
        ),
    ):
        response = client.post(route, data={**_view_form(), **payload})
        assert response.status_code == 400
        assert response.get_json()["status"] == "rejected"
    assert fetch_calls == []
    assert command_calls == []
    assert thread_count == 0


def test_url_composition_runs_only_for_successfully_ingested_jobs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []

    def fetcher(url: str) -> str:
        if url.endswith("failed"):
            raise RuntimeError("synthetic private fetch detail")
        return _generic_html()

    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(
        _runtime(_paths(tmp_path)),
        project_root=project,
        command_executor=lambda argv: commands.append(tuple(argv)) or 0,
        generic_html_fetcher=fetcher,
    )
    response = app.test_client().post(
        "/applications/add/other",
        data={
            **_view_form(),
            "other_urls": (
                "https://jobs.example.com/survivor\nhttps://jobs.example.com/failed"
            ),
            "run_v1": "1",
            "run_v2": "1",
        },
    )
    assert response.status_code == 202
    assert response.get_json()["accepted"] == 1
    assert response.get_json()["failed"] == 1
    assert [command[3] for command in commands] == [
        "regenerate-draft-resumes",
        "refine-draft-resumes",
    ]
    survivor = generic_job_id("https://jobs.example.com/survivor")
    assert all(f"JOB_IDS={survivor}" in command for command in commands)
    assert not any("failed" in item for command in commands for item in command)


def test_seed_composition_uses_only_newly_seeded_jobs_and_selected_stages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("help:\n\t@true\n", encoding="utf-8")
    commands: list[tuple[str, ...]] = []
    app_holder: dict[str, object] = {}

    def executor(argv):
        command = tuple(argv)
        commands.append(command)
        if command[3] == "seed-jobs":
            store = app_holder["app"].extensions["career_agent_workbench"]["store"]
            store.upsert_application(
                ApplicationMetadata(
                    job_id="new-seeded-job",
                    company="Example Seed Systems",
                    job_title="Fictional Seed Engineer",
                    job_url="https://example.com/jobs/new-seeded-job",
                    source="synthetic",
                )
            )
            return CommandExecution(0, '{"jobs_seeded": 1}')
        return 0

    monkeypatch.setattr(webapp.threading, "Thread", _InlineThread)
    app = webapp.create_app(
        _runtime(paths),
        project_root=project,
        command_executor=executor,
    )
    app_holder["app"] = app
    store = app.extensions["career_agent_workbench"]["store"]
    store.upsert_application(
        ApplicationMetadata(
            job_id="existing-job",
            company="Example Existing Systems",
            job_title="Fictional Existing Engineer",
            job_url="https://example.com/jobs/existing-job",
            source="synthetic",
        )
    )
    response = app.test_client().post(
        "/applications/add/seed",
        data={
            **_view_form(),
            "location": "Fictional City",
            "date_posted": "past_week",
            "limit_per_query": "7",
            "max_queries": "4",
            "max_jobs": "3",
            "run_v1": "1",
            "run_highlight": "1",
        },
    )
    assert response.status_code == 202
    assert [command[3] for command in commands] == [
        "seed-jobs",
        "regenerate-draft-resumes",
        "highlight-draft-resumes",
    ]
    assert all("JOB_IDS=new-seeded-job" in command for command in commands[1:])
    assert not any("JOB_IDS=existing-job" in command for command in commands)


def test_url_inputs_are_bounded_and_all_failures_remain_count_only(
    tmp_path: Path,
) -> None:
    app = webapp.create_app(
        _runtime(_paths(tmp_path)),
        project_root=tmp_path / "no-project",
        linkedin_details_fetcher=lambda _url: (_ for _ in ()).throw(RuntimeError()),
        generic_html_fetcher=lambda _url: (_ for _ in ()).throw(RuntimeError()),
    )
    client = app.test_client()
    rejected = client.post(
        "/applications/add/linkedin",
        data={**_view_form(), "linkedin_urls": "https://example.com/not-linkedin"},
    )
    assert rejected.status_code == 400
    assert rejected.get_json() == {
        "message": "URL ingestion is invalid.",
        "status": "rejected",
    }

    too_many = "\n".join(f"https://jobs.example.com/{index}" for index in range(21))
    assert (
        client.post(
            "/applications/add/other",
            data={**_view_form(), "other_urls": too_many},
        ).status_code
        == 400
    )

    failed = client.post(
        "/applications/add/other",
        data={**_view_form(), "other_urls": "https://jobs.example.com/one"},
    )
    assert failed.status_code == 422
    assert failed.get_json() == {
        "accepted": 0,
        "created": 0,
        "failed": 1,
        "message": "URL ingestion completed.",
        "refreshed": 0,
        "status": "rejected",
    }
