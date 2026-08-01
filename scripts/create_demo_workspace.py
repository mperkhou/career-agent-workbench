"""Materialize one deterministic offline fictional demo workspace."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from career_agent_workbench.application_resume import (
    attach_job_opening_description_object,
    create_job_opening_description_object,
    initialize_application_resume_object,
)
from career_agent_workbench.application_state import (
    ApplicationMetadata,
    ApplicationStateStore,
    ResumeVariantWrite,
)
from career_agent_workbench.config import (
    RuntimeConfig,
    RuntimeOverrides,
    WorkspaceMember,
    load_runtime_config,
)
from career_agent_workbench.jod import clean_job_description_for_prompt
from career_agent_workbench.models import JobDetails
from career_agent_workbench.resume_rendering import render_resume_html_from_mapping

_ERROR = "Demo workspace could not be created."
_FIXED_TIME = datetime(2042, 4, 12, 12, 0, tzinfo=UTC)
_EXPECTED_SOURCE_FILES = frozenset(
    {
        Path("README.md"),
        Path(".blacklist"),
        Path("profile/MASTER-RESUME.yml"),
        Path("profile/MP-MASTER-RESUME.txt"),
        Path("jobs/demo-platform-engineer.json"),
    }
)
_MAX_SOURCE_BYTES = 2_000_000


class DemoWorkspaceError(Exception):
    """Stable error for this local fictional demo factory."""


def _resolved(path: Path, *, required: bool) -> Path:
    if not isinstance(path, Path):
        raise DemoWorkspaceError(_ERROR)
    try:
        return path.resolve(strict=required)
    except Exception:
        raise DemoWorkspaceError(_ERROR) from None


def _validate_source(source: Path) -> JobDetails:
    actual: set[Path] = set()
    try:
        for path in source.rglob("*"):
            if path.is_file() or path.is_symlink():
                actual.add(path.relative_to(source))
        if actual != _EXPECTED_SOURCE_FILES:
            raise DemoWorkspaceError(_ERROR)
        for relative in _EXPECTED_SOURCE_FILES:
            path = source / relative
            if path.is_symlink() or not path.is_file():
                raise DemoWorkspaceError(_ERROR)
            if not 0 < path.stat().st_size <= _MAX_SOURCE_BYTES:
                raise DemoWorkspaceError(_ERROR)
        initialize_application_resume_object(source / "profile/MASTER-RESUME.yml")
        payload = json.loads(
            (source / "jobs/demo-platform-engineer.json").read_text("utf-8")
        )
        job = JobDetails.model_validate(payload)
    except DemoWorkspaceError:
        raise
    except Exception:
        raise DemoWorkspaceError(_ERROR) from None
    if (
        job.job_id != "demo-platform-001"
        or job.company != "Nimbus Quay Example Labs"
        or job.title != "Demo Platform Engineer"
        or job.description is None
    ):
        raise DemoWorkspaceError(_ERROR)
    return job


def _copy_static_inputs(source: Path, runtime: RuntimeConfig) -> None:
    profile = runtime.paths.require(WorkspaceMember.PROFILE_DIR)
    master_resume = runtime.paths.require(WorkspaceMember.MASTER_RESUME)
    master_resume_text = runtime.paths.require(WorkspaceMember.MASTER_RESUME_TEXT)
    blacklist = runtime.paths.require(WorkspaceMember.BLACKLIST)
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        source / "profile/MASTER-RESUME.yml",
        master_resume,
    )
    shutil.copyfile(
        source / "profile/MP-MASTER-RESUME.txt",
        master_resume_text,
    )
    shutil.copyfile(source / ".blacklist", blacklist)


def _runtime(workspace: Path) -> RuntimeConfig:
    runtime = load_runtime_config(
        overrides=RuntimeOverrides(workspace=workspace),
        environ={},
        cwd=workspace,
        discover_dotenv=False,
    )
    for member in (
        WorkspaceMember.MASTER_RESUME,
        WorkspaceMember.MASTER_RESUME_TEXT,
        WorkspaceMember.OUTPUT_DIR,
        WorkspaceMember.DATABASE,
        WorkspaceMember.BLACKLIST,
    ):
        runtime.paths.require(member)
    return runtime


def _resume_for_job(runtime: RuntimeConfig, job: JobDetails) -> dict[str, object]:
    master_resume = runtime.paths.require(WorkspaceMember.MASTER_RESUME)
    resume = initialize_application_resume_object(master_resume)
    prompt_jod = clean_job_description_for_prompt(job.description or "")
    description = create_job_opening_description_object(
        trimmed_job_description=prompt_jod,
        requirements_response=[
            "Build reliable Python services with focused tests.",
            "Maintain SQLite-backed workflow state and useful observability.",
            "Document human review checkpoints and operational runbooks.",
        ],
    )
    return attach_job_opening_description_object(
        application_resume=resume,
        job_opening_description=description,
    )


def _cover_letter(job: JobDetails) -> dict[str, object]:
    return {
        "schema_version": "fictional_demo.cover_letter.v1",
        "job_id": job.job_id,
        "company": job.company,
        "title": job.title,
        "paragraphs": [
            "Avery Demo is interested in the fictional platform role.",
            "The supplied demo history supports Python, SQLite, testing, and observability work.",
        ],
        "requires_human_review": True,
    }


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _verify(
    *,
    store: ApplicationStateStore,
    job: JobDetails,
    resume: dict[str, object],
    cover_letter: dict[str, object],
    examples: dict[Path, str],
) -> None:
    record = store.get_application(job.job_id)
    variants = store.list_resume_variants(job.job_id)
    if (
        record.job_id != job.job_id
        or record.selected_resume_variant != "v1"
        or record.resume_variant_selection_mode != "auto"
        or len(variants) != 1
        or variants[0].variant_key != "v1"
        or _plain_json(variants[0].application_resume) != resume
        or _plain_json(record.application_resume) != resume
        or _plain_json(record.cover_letter) != cover_letter
        or record.applied_to != "No"
    ):
        raise DemoWorkspaceError(_ERROR)
    for path, expected in examples.items():
        try:
            if path.read_text("utf-8") != expected:
                raise DemoWorkspaceError(_ERROR)
        except DemoWorkspaceError:
            raise
        except Exception:
            raise DemoWorkspaceError(_ERROR) from None


def create_demo_workspace(source: Path, workspace: Path) -> RuntimeConfig:
    """Create or idempotently refresh one explicit fictional workspace."""

    source_path = _resolved(source, required=True)
    workspace_path = _resolved(workspace, required=False)
    if (
        source_path == workspace_path
        or source_path in workspace_path.parents
        or workspace_path in source_path.parents
    ):
        raise DemoWorkspaceError(_ERROR)
    try:
        job = _validate_source(source_path)
        runtime = _runtime(workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)
        _copy_static_inputs(source_path, runtime)
        paths = runtime.paths
        store = ApplicationStateStore(paths, utc_clock=lambda: _FIXED_TIME)
        store.initialize()

        source_jod = job.description or ""
        prompt_jod = clean_job_description_for_prompt(source_jod)
        store.seed_application(
            ApplicationMetadata(
                job_id=job.job_id,
                company=job.company or "",
                job_title=job.title,
                job_url=str(job.job_url),
                source=job.source,
                date_matched=_FIXED_TIME.date().isoformat(),
                date_posted=job.listed_at,
                experience_level=job.seniority_level,
            ),
            source_text=source_jod,
            prompt_text=prompt_jod,
        )
        resume = _resume_for_job(runtime, job)
        resume_yaml = yaml.safe_dump(resume, sort_keys=False, allow_unicode=False)
        resume_html = render_resume_html_from_mapping(resume=resume)
        store.upsert_resume_variant(
            job.job_id,
            ResumeVariantWrite(
                variant_key="v1",
                variant_label="Fictional demo v1",
                source="fictional_demo",
                application_resume_yaml=resume_yaml,
                resume_html=resume_html,
                validation={"requires_human_review": True, "fictional_demo": True},
            ),
        )
        cover_letter = _cover_letter(job)
        store.store_clo(job.job_id, value=cover_letter)

        output = paths.require(WorkspaceMember.OUTPUT_DIR) / "demo-examples"
        output.mkdir(parents=True, exist_ok=True)
        cover_json = json.dumps(cover_letter, indent=2, sort_keys=True) + "\n"
        examples = {
            output / "application-resume-v1.yml": resume_yaml,
            output / "cover-letter.json": cover_json,
            output / "resume-v1.html": resume_html,
        }
        for path, content in examples.items():
            path.write_text(content, encoding="utf-8")
        _verify(
            store=store,
            job=job,
            resume=resume,
            cover_letter=cover_letter,
            examples=examples,
        )
        return runtime
    except DemoWorkspaceError:
        raise
    except Exception:
        raise DemoWorkspaceError(_ERROR) from None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one offline fictional Career Agent Workbench demo."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        runtime = create_demo_workspace(args.source, args.workspace)
    except DemoWorkspaceError as exc:
        parser.error(str(exc))
    print(f"Fictional demo workspace ready: {runtime.paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
