from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateNotFoundError,
    ApplicationStateStore,
)
from career_agent_workbench.config import WorkspacePaths
from career_agent_workbench.models import (
    JobDetails,
    JobPosting,
    JobSearchQuery,
    JobSearchResult,
)
from career_agent_workbench.query_optimizer import StoredQueryOutcome
from career_agent_workbench.workflows.matching import (
    MatchingBlacklistError,
    MatchingBounds,
    MatchingConfigurationError,
    MatchingPlannerError,
    MatchingProgress,
    MatchingStateError,
    MatchingWorkflow,
)


def _paths(
    root: Path,
    *,
    database_name: str = "applications.sqlite3",
    blacklist: Path | None = None,
) -> WorkspacePaths:
    output = root / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    master_resume = root / "profile.yml"
    master_resume.write_text(
        """
professional_summary:
  text: Builds reliable fictional services.
core_technical_skills:
  bullet_points:
    - category: Platforms
      items:
        primary: [Python, Testing]
professional_experience:
  jobs:
    - company: Example Harbor Systems
      title: Platform Engineer
      bullet_points:
        - text: Built synthetic automation for reserved examples.
""".lstrip(),
        encoding="utf-8",
    )
    return WorkspacePaths(
        root=root,
        master_resume=master_resume,
        master_resume_text=root / "profile.txt",
        output_dir=output,
        database=root / database_name,
        blacklist=blacklist,
        tmp_dir=root / "tmp",
    )


def _bounds(*, max_jobs: int = 10) -> MatchingBounds:
    return MatchingBounds(
        location="Example Region",
        date_posted="past_week",
        workplace_types=("remote",),
        experience_levels=("mid_senior",),
        job_types=("full_time",),
        limit_per_query=10,
        max_queries=4,
        max_jobs=max_jobs,
    )


def test_matching_constructor_rejects_active_non_store_without_access(
    tmp_path: Path,
) -> None:
    class ActiveStore:
        calls = 0

        def __getattribute__(self, _name: str) -> object:
            type(self).calls += 1
            raise AssertionError("active store access must not run")

    with pytest.raises(MatchingConfigurationError):
        MatchingWorkflow(
            service=object(),  # type: ignore[arg-type]
            planner=object(),  # type: ignore[arg-type]
            store=ActiveStore(),  # type: ignore[arg-type]
            paths=_paths(tmp_path),
        )
    assert ActiveStore.calls == 0


def _query(keywords: str = "Platform Engineer") -> JobSearchQuery:
    return JobSearchQuery(
        keywords=keywords,
        location="Example Region",
        date_posted="past_week",
        workplace_type="remote",
        experience_level="mid_senior",
        job_type="full_time",
        limit=10,
    )


def _posting(job_id: str, company: str) -> JobPosting:
    return JobPosting(
        job_id=job_id,
        title="Platform Engineer",
        company=company,
        location="Remote",
        job_url=f"https://jobs.example.com/{job_id}",
        workplace_type="remote",
        source="example_public",
    )


def _details(
    job_id: str,
    company: str,
    *,
    workplace_type: str = "remote",
    seniority_level: str = "mid_senior",
    employment_type: str = "full_time",
    description: str | None = "Responsibilities: build safe fictional systems.",
) -> JobDetails:
    return JobDetails(
        job_id=job_id,
        title="Platform Engineer",
        company=company,
        location="Remote",
        job_url=f"https://jobs.example.com/{job_id}",
        workplace_type=workplace_type,
        seniority_level=seniority_level,
        employment_type=employment_type,
        description=description,
        source="example_public",
    )


class _Planner:
    def __init__(self, queries: object) -> None:
        self.queries = queries
        self.calls = 0

    async def plan_queries(self, request: object) -> object:
        self.calls += 1
        return self.queries


class _Service:
    def __init__(
        self,
        postings: tuple[JobPosting, ...],
        details: dict[str, JobDetails | Exception],
        *,
        search_error: Exception | None = None,
    ) -> None:
        self.postings = postings
        self.details = details
        self.search_error = search_error
        self.search_calls = 0
        self.detail_calls = 0
        self.queries: list[JobSearchQuery] = []

    async def search(self, query: JobSearchQuery) -> JobSearchResult:
        self.search_calls += 1
        self.queries.append(query)
        if self.search_error is not None:
            raise self.search_error
        return JobSearchResult(
            query=query,
            count=len(self.postings),
            jobs=list(self.postings),
            provider="example_public",
        )

    async def get_details(self, job_id_or_url: str) -> JobDetails:
        self.detail_calls += 1
        job_id = job_id_or_url.rstrip("/").rsplit("/", 1)[-1]
        value = self.details[job_id]
        if isinstance(value, Exception):
            raise value
        return value


def _store(paths: WorkspacePaths) -> ApplicationStateStore:
    store = ApplicationStateStore(paths)
    store.initialize()
    return store


def test_matching_uses_only_injected_inputs_and_seeds_ordered_rows(
    tmp_path: Path,
) -> None:
    blacklist = tmp_path / "blacklist.txt"
    blacklist.write_text("Blocked Example*\n", encoding="utf-8")
    paths = _paths(tmp_path, blacklist=blacklist)
    store = _store(paths)
    postings = (
        _posting("existing-1", "Existing Example"),
        _posting("blocked-1", "Blocked Example Labs"),
        _posting("onsite-1", "Onsite Example"),
        _posting("junior-1", "Junior Example"),
        _posting("contract-1", "Contract Example"),
        _posting("empty-1", "Empty Example"),
        _posting("missing-1", "Missing Example"),
        _posting("seed-1", "Seed Example"),
        _posting("seed-2", "Second Seed Example"),
    )
    details: dict[str, JobDetails | Exception] = {
        "blocked-1": _details("blocked-1", "Blocked Example Labs"),
        "onsite-1": _details(
            "onsite-1",
            "Onsite Example",
            workplace_type="on_site",
        ),
        "junior-1": _details(
            "junior-1",
            "Junior Example",
            seniority_level="entry_level",
        ),
        "contract-1": _details(
            "contract-1",
            "Contract Example",
            employment_type="contract",
        ),
        "empty-1": _details("empty-1", "Empty Example", description=None),
        "missing-1": RuntimeError("private detail marker"),
        "seed-1": _details("seed-1", "Seed Example"),
        "seed-2": _details("seed-2", "Second Seed Example"),
    }
    planner = _Planner((_query(),))
    service = _Service(postings, details)
    progress: list[MatchingProgress] = []
    workflow = MatchingWorkflow(
        service=service,
        planner=planner,
        store=store,
        paths=paths,
    )

    result = asyncio.run(
        workflow.run(
            bounds=_bounds(),
            history=(),
            supplemental_queries=(),
            existing_job_ids=("existing-1",),
            progress_callback=progress.append,
        )
    )

    assert result.newly_seeded_job_ids == ("seed-1", "seed-2")
    assert result.jobs_seeded == 2
    assert result.jobs_seen == len(postings)
    assert result.query_outcomes[0].skipped_existing == 1
    assert result.query_outcomes[0].skipped_blacklisted == 1
    assert result.query_outcomes[0].skipped_workplace_type == 1
    assert result.query_outcomes[0].skipped_experience_level == 1
    assert result.query_outcomes[0].skipped_job_type == 1
    assert result.query_outcomes[0].skipped_description == 1
    assert result.query_outcomes[0].details_missed == 1
    assert store.get_application("seed-1").job_description
    assert store.get_application("seed-1").prompt_job_description
    assert store.get_application("seed-2").job_description
    assert all(type(item) is MatchingProgress for item in progress)
    assert all("seed-1" not in repr(item) for item in progress)
    assert "seed-1" not in repr(result)
    assert "Platform Engineer" not in repr(result)


def test_store_binding_rejects_before_file_planner_or_provider_access(
    tmp_path: Path,
) -> None:
    store_paths = _paths(tmp_path / "store")
    store = _store(store_paths)
    other_paths = WorkspacePaths(
        master_resume=tmp_path / "missing.yml",
        output_dir=tmp_path / "other-output",
        database=tmp_path / "other.sqlite3",
    )
    planner = _Planner((_query(),))
    service = _Service((), {})

    workflow = MatchingWorkflow(
        service=service,
        planner=planner,
        store=store,
        paths=other_paths,
    )
    with pytest.raises(MatchingConfigurationError, match="configuration"):
        asyncio.run(workflow.run(bounds=_bounds()))

    assert planner.calls == 0
    assert service.search_calls == 0
    assert not (tmp_path / "other.sqlite3").exists()
    assert not (tmp_path / "other-output").exists()


@pytest.mark.parametrize(
    "setup",
    [
        lambda root: root / "missing.txt",
        lambda root: _symlink(root),
        lambda root: _blacklist(root, "x" * 513),
        lambda root: _blacklist(root, "Unsafe\\Pattern"),
        lambda root: _blacklist(root, "Unsafe\x00Pattern"),
    ],
)
def test_configured_blacklist_fails_closed(
    tmp_path: Path,
    setup: object,
) -> None:
    blacklist = setup(tmp_path)  # type: ignore[operator]
    paths = _paths(tmp_path / "workspace", blacklist=blacklist)
    store = _store(paths)
    planner = _Planner((_query(),))
    service = _Service((), {})
    workflow = MatchingWorkflow(
        service=service,
        planner=planner,
        store=store,
        paths=paths,
    )

    with pytest.raises(MatchingBlacklistError, match="blacklist") as caught:
        asyncio.run(workflow.run(bounds=_bounds()))
    assert caught.value.__cause__ is None
    assert planner.calls == 0
    assert service.search_calls == 0


def _blacklist(root: Path, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "configured-blacklist.txt"
    path.write_text(content, encoding="utf-8")
    return path


def _symlink(root: Path) -> Path:
    target = _blacklist(root, "Blocked Example")
    link = root / "blacklist-link"
    link.symlink_to(target)
    return link


def test_provider_and_detail_misses_continue_without_leaking_content(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = _store(paths)
    planner = _Planner((_query(),))
    provider_miss = MatchingWorkflow(
        service=_Service((), {}, search_error=RuntimeError("secret marker")),
        planner=planner,
        store=store,
        paths=paths,
    )
    result = asyncio.run(provider_miss.run(bounds=_bounds()))
    assert result.query_outcomes[0].provider_missed is True
    assert "secret marker" not in repr(result)

    detail_miss = MatchingWorkflow(
        service=_Service(
            (_posting("missing-1", "Missing Example"),),
            {"missing-1": RuntimeError("detail marker")},
        ),
        planner=_Planner((_query(),)),
        store=store,
        paths=paths,
    )
    result = asyncio.run(detail_miss.run(bounds=_bounds()))
    assert result.query_outcomes[0].details_missed == 1
    assert result.jobs_seeded == 0


def test_existing_filter_applies_to_canonical_detail_id(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = _store(paths)
    workflow = MatchingWorkflow(
        service=_Service(
            (_posting("card-id", "Existing Example"),),
            {"card-id": _details("existing-id", "Existing Example")},
        ),
        planner=_Planner((_query(),)),
        store=store,
        paths=paths,
    )
    result = asyncio.run(
        workflow.run(
            bounds=_bounds(),
            existing_job_ids=("existing-id",),
        )
    )
    assert result.newly_seeded_job_ids == ()
    assert result.query_outcomes[0].skipped_existing == 1
    with pytest.raises(ApplicationStateNotFoundError):
        store.get_application("existing-id")


def test_concurrent_matching_runs_report_only_transaction_confirmed_creation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    first_store = _store(paths)
    second_store = ApplicationStateStore(paths)
    detail_barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    class BarrierService(_Service):
        async def get_details(self, job_id_or_url: str) -> JobDetails:
            details = await super().get_details(job_id_or_url)
            detail_barrier.wait(timeout=5)
            return details

    def run(workflow: MatchingWorkflow) -> None:
        try:
            result: object = asyncio.run(
                workflow.run(
                    bounds=_bounds(),
                    existing_job_ids=(),
                )
            )
            with result_lock:
                results.append(result)
        except BaseException as error:  # noqa: BLE001 - capture worker failures.
            with result_lock:
                errors.append(error)

    posting = _posting("race-seed-1", "Example Race Cooperative")
    details = _details("race-seed-1", "Example Race Cooperative")
    workflows = tuple(
        MatchingWorkflow(
            service=BarrierService((posting,), {"race-seed-1": details}),
            planner=_Planner((_query(),)),
            store=store,
            paths=paths,
        )
        for store in (first_store, second_store)
    )
    threads = tuple(threading.Thread(target=run, args=(item,)) for item in workflows)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert sum(item.jobs_seeded for item in results) == 1
    assert sum(item.query_outcomes[0].seeded for item in results) == 1
    assert sum(item.query_outcomes[0].skipped_existing for item in results) == 1
    assert sorted(item.newly_seeded_job_ids for item in results) == [
        (),
        ("race-seed-1",),
    ]
    stored = first_store.get_application("race-seed-1")
    assert stored.job_description == details.description
    assert stored.prompt_job_description is not None


def test_matching_refreshes_a_post_detail_concurrent_insert_without_claiming_it(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    matching_store = _store(paths)
    inserting_store = ApplicationStateStore(paths)
    detail_fetched = threading.Event()
    release_detail = threading.Event()
    results: list[object] = []
    errors: list[BaseException] = []

    class GatedService(_Service):
        async def get_details(self, job_id_or_url: str) -> JobDetails:
            details = await super().get_details(job_id_or_url)
            detail_fetched.set()
            if not release_detail.wait(timeout=5):
                raise AssertionError("bounded synthetic detail gate timed out")
            return details

    details = _details("inserted-after-detail", "Example Insert Race Cooperative")
    workflow = MatchingWorkflow(
        service=GatedService(
            (_posting(details.job_id, details.company or ""),),
            {details.job_id: details},
        ),
        planner=_Planner((_query(),)),
        store=matching_store,
        paths=paths,
    )

    def run() -> None:
        try:
            results.append(
                asyncio.run(
                    workflow.run(
                        bounds=_bounds(),
                        existing_job_ids=(),
                    )
                )
            )
        except BaseException as error:  # noqa: BLE001 - capture worker failures.
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert detail_fetched.wait(timeout=5)
        inserted = inserting_store.seed_application(
            ApplicationMetadata(
                job_id=details.job_id,
                company="Example Earlier Insert Cooperative",
                job_title=details.title,
                job_url=str(details.job_url),
                source=details.source,
                date_matched="2034-01-02T03:04:05+00:00",
            ),
            source_text="Synthetic source before matching refresh",
            prompt_text="Synthetic prompt before matching refresh",
        )
        inserting_store.update_application_status(
            details.job_id,
            applied_to="Rejected",
            notes="Synthetic preserved concurrent note",
        )
        inserting_store.archive([details.job_id])
    finally:
        release_detail.set()
        thread.join(timeout=8)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    result = results[0]
    assert result.newly_seeded_job_ids == ()
    assert result.jobs_seeded == 0
    assert result.query_outcomes[0].seeded == 0
    assert result.query_outcomes[0].skipped_existing == 1
    refreshed = inserting_store.get_application(details.job_id)
    assert refreshed.company == details.company
    assert refreshed.job_description == details.description
    assert refreshed.prompt_job_description is not None
    assert refreshed.applied_to == "Rejected"
    assert refreshed.notes == "Synthetic preserved concurrent note"
    assert refreshed.archived_at is not None
    assert refreshed.date_matched == inserted.date_matched
    assert refreshed.imported_at == inserted.imported_at


def test_state_seed_failure_aborts_after_prior_atomic_candidate(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    seed_calls = 0

    def fail_second_seed(checkpoint: str) -> None:
        nonlocal seed_calls
        if checkpoint != "seed_before_metadata":
            return
        seed_calls += 1
        if seed_calls == 2:
            raise RuntimeError("state marker")

    actual = ApplicationStateStore(paths, failure_injector=fail_second_seed)
    actual.initialize()
    service = _Service(
        (
            _posting("seed-1", "Seed Example"),
            _posting("seed-2", "Second Seed Example"),
        ),
        {
            "seed-1": _details("seed-1", "Seed Example"),
            "seed-2": _details("seed-2", "Second Seed Example"),
        },
    )
    workflow = MatchingWorkflow(
        service=service,
        planner=_Planner((_query(),)),
        store=actual,
        paths=paths,
    )

    with pytest.raises(MatchingStateError, match="seed") as caught:
        asyncio.run(workflow.run(bounds=_bounds()))
    assert caught.value.__cause__ is None
    assert actual.get_application("seed-1").job_id == "seed-1"
    with pytest.raises(ApplicationStateNotFoundError):
        actual.get_application("seed-2")


def test_matching_uses_history_and_supplements_without_persisting_history(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    store = _store(paths)
    history = (
        StoredQueryOutcome(
            keywords="Reliability Engineer",
            location="Example Region",
            date_posted="past_week",
            workplace_type="remote",
            experience_level="mid_senior",
            job_type="full_time",
            sort_by="recent",
            limit=10,
            profile_match=0.5,
            query_score=0.8,
            results_returned=5,
            fresh_jobs_accepted=2,
        ),
    )
    workflow = MatchingWorkflow(
        service=_Service((), {}),
        planner=_Planner((_query(),)),
        store=store,
        paths=paths,
    )
    before = paths.database.read_bytes()
    result = asyncio.run(
        workflow.run(
            bounds=_bounds(),
            history=history,
            supplemental_queries=(_query("Automation Engineer"),),
        )
    )
    after = paths.database.read_bytes()

    assert result.queries_planned >= 2
    assert result.jobs_seeded == 0
    assert before == after


def test_history_reuse_has_no_hard_coded_remote_preference(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = _Service((), {})
    history = (
        StoredQueryOutcome(
            keywords="Reliability Engineer",
            location="Example Region",
            date_posted="past_week",
            workplace_type="on_site",
            experience_level="mid_senior",
            job_type="full_time",
            sort_by="recent",
            limit=10,
            profile_match=0.5,
            query_score=0.8,
            results_returned=5,
            fresh_jobs_accepted=2,
        ),
    )
    workflow = MatchingWorkflow(
        service=service,
        planner=_Planner(()),
        store=_store(paths),
        paths=paths,
    )
    bounds = MatchingBounds(
        location="Example Region",
        date_posted="past_week",
        workplace_types=("on_site", "remote"),
        experience_levels=("mid_senior",),
        job_types=("full_time",),
        max_queries=4,
    )
    asyncio.run(workflow.run(bounds=bounds, history=history))
    assert {query.workplace_type for query in service.queries} == {
        "on_site",
        "remote",
    }


@pytest.mark.parametrize(
    "planner_output",
    [
        {"queries": []},
        (_query(), "active"),  # type: ignore[list-item]
        tuple(_query(str(index)) for index in range(5)),
    ],
)
def test_planner_output_is_strict_and_content_free(
    tmp_path: Path,
    planner_output: object,
) -> None:
    paths = _paths(tmp_path)
    workflow = MatchingWorkflow(
        service=_Service((), {}),
        planner=_Planner(planner_output),
        store=_store(paths),
        paths=paths,
    )
    with pytest.raises(MatchingPlannerError, match="planner") as caught:
        asyncio.run(workflow.run(bounds=_bounds()))
    assert caught.value.__cause__ is None
    assert "active" not in str(caught.value)


def test_bounds_are_exact_builtins_and_do_not_coerce_active_objects() -> None:
    class ActiveTuple(tuple):
        def __iter__(self) -> object:
            raise AssertionError("must not execute")

    with pytest.raises(MatchingConfigurationError):
        MatchingBounds(
            location="Example Region",
            date_posted="past_week",
            workplace_types=ActiveTuple(("remote",)),  # type: ignore[arg-type]
            experience_levels=("mid_senior",),
            job_types=("full_time",),
        )


def test_active_history_and_query_fields_reject_without_execution(
    tmp_path: Path,
) -> None:
    class ActiveString(str):
        activated = False

        def casefold(self) -> str:
            type(self).activated = True
            raise AssertionError("active casefold must not run")

        def strip(self, *_args: object, **_kwargs: object) -> str:
            type(self).activated = True
            raise AssertionError("active strip must not run")

    paths = _paths(tmp_path)
    workflow = MatchingWorkflow(
        service=_Service((), {}),
        planner=_Planner(()),
        store=_store(paths),
        paths=paths,
    )
    active_history = StoredQueryOutcome(
        keywords=ActiveString("Reliability Engineer"),
        location="Example Region",
        date_posted="past_week",
        workplace_type="on_site",
        experience_level="mid_senior",
        job_type="full_time",
        sort_by="recent",
        limit=10,
        profile_match=0.5,
        query_score=0.8,
        results_returned=5,
        fresh_jobs_accepted=2,
    )
    with pytest.raises(MatchingConfigurationError):
        asyncio.run(workflow.run(bounds=_bounds(), history=(active_history,)))

    active_query = JobSearchQuery.model_construct(
        keywords=ActiveString("Platform Engineer"),
        location="Example Region",
        date_posted="past_week",
        workplace_type="remote",
        experience_level="mid_senior",
        job_type="full_time",
        sort_by="recent",
        distance=None,
        limit=10,
        page=0,
        exclude_job_ids=set(),
    )
    with pytest.raises(MatchingConfigurationError):
        asyncio.run(
            workflow.run(
                bounds=_bounds(),
                supplemental_queries=(active_query,),
            )
        )
    assert ActiveString.activated is False


def test_crossed_binding_precedes_hostile_provider_and_planner_attributes(
    tmp_path: Path,
) -> None:
    class HostileService:
        accessed = False

        @property
        def search(self) -> object:
            type(self).accessed = True
            raise AssertionError("provider access must not run")

        @property
        def get_details(self) -> object:
            type(self).accessed = True
            raise AssertionError("provider access must not run")

    class HostilePlanner:
        accessed = False

        @property
        def plan_queries(self) -> object:
            type(self).accessed = True
            raise AssertionError("planner access must not run")

    bound = _paths(tmp_path / "bound")
    store = _store(bound)
    crossed = WorkspacePaths(
        master_resume=tmp_path / "missing.yml",
        output_dir=tmp_path / "crossed-output",
        database=tmp_path / "crossed.sqlite3",
    )
    workflow = MatchingWorkflow(
        service=HostileService(),  # type: ignore[arg-type]
        planner=HostilePlanner(),  # type: ignore[arg-type]
        store=store,
        paths=crossed,
    )
    with pytest.raises(MatchingConfigurationError):
        asyncio.run(workflow.run(bounds=_bounds()))
    assert HostileService.accessed is False
    assert HostilePlanner.accessed is False


@pytest.mark.parametrize(
    "content",
    [
        "Blocked\x0bCompany",
        "Blocked\u0085Company",
        "Blocked\u2028Company",
        "x\n" * 2_001,
        "\n".join(f"pattern-{index}" for index in range(1_001)),
    ],
)
def test_blacklist_rejects_control_and_count_bounds(
    tmp_path: Path,
    content: str,
) -> None:
    blacklist = _blacklist(tmp_path, content)
    paths = _paths(tmp_path / "workspace", blacklist=blacklist)
    workflow = MatchingWorkflow(
        service=_Service((), {}),
        planner=_Planner((_query(),)),
        store=_store(paths),
        paths=paths,
    )
    with pytest.raises(MatchingBlacklistError):
        asyncio.run(workflow.run(bounds=_bounds()))


def test_blacklist_rejects_oversized_invalid_utf8_and_fifo(
    tmp_path: Path,
) -> None:
    cases: list[Path] = []
    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * 128_001)
    cases.append(oversized)
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    cases.append(invalid)
    fifo = tmp_path / "configured.fifo"
    os.mkfifo(fifo)
    cases.append(fifo)

    for index, blacklist in enumerate(cases):
        root = tmp_path / f"workspace-{index}"
        paths = _paths(root, blacklist=blacklist)
        workflow = MatchingWorkflow(
            service=_Service((), {}),
            planner=_Planner((_query(),)),
            store=_store(paths),
            paths=paths,
        )
        with pytest.raises(MatchingBlacklistError):
            asyncio.run(workflow.run(bounds=_bounds()))


def test_adversarial_blacklist_glob_is_bounded_and_deterministic(
    tmp_path: Path,
) -> None:
    blacklist = _blacklist(tmp_path, ("*a" * 10) + "b")
    paths = _paths(tmp_path / "workspace", blacklist=blacklist)
    company = "a" * 100
    workflow = MatchingWorkflow(
        service=_Service(
            (_posting("seed-1", company),),
            {"seed-1": _details("seed-1", company)},
        ),
        planner=_Planner((_query(),)),
        store=_store(paths),
        paths=paths,
    )
    result = asyncio.run(workflow.run(bounds=_bounds()))
    assert result.newly_seeded_job_ids == ("seed-1",)
    assert result.query_outcomes[0].skipped_blacklisted == 0
