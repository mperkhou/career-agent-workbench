# Career Agent Workbench

Career Agent Workbench is a local, human-governed portfolio application for
researching public job postings, organizing application state, and producing
reviewable résumé variants. The reusable Python application stays public-safe;
profile material, tracker data, model credentials, and generated documents stay
in a separate private workspace selected through one ignored `.env` file.

The project combines a provider-neutral public-job core, SQLite-backed state,
evidence-grounded résumé workflows, deterministic rendering and ATS diagnostics,
command-line and Make composition, a small Flask tracker/action adapter, and a
proportional FastMCP server. It does **not** authenticate to LinkedIn, access
private member data, submit applications, or contact employers.

## Terms

- **JOD** — Job Opening Description: normalized source and prompt-ready job text.
- **MRO** — Master Resume Object: structured, reusable evidence derived from the
  human-maintained résumé source.
- **ARO** — Application Resume Object: a job-specific résumé mapping derived
  from the MRO.
- **CLO** — Cover Letter Object: a separately reviewed, human-owned letter.
- **ATS** — deterministic local diagnostics for a résumé/JOD pair.
- **Variant** — an isolated `v1`, `v2`, or `manual` résumé version whose
  selection is explicit and reversible.

## Architecture: public code, private state

The repository contains application code, packaged templates, fictional demo
data, synthetic tests, and public documentation. An operator-owned workspace
contains the real profile, SQLite database, output, and temporary files. The
normal local configuration is one ignored `.env` value:

```dotenv
CAREER_AGENT_WORKBENCH_WORKSPACE=../career-agent-workbench-ops
```

Conventional workspace members resolve beneath that root. Exceptional layouts
can use the optional member overrides documented in [`.env.example`](.env.example).
Explicit CLI and Make values remain the highest-precedence overrides, and no
loader mutates the process environment.

```text
public checkout                        private workspace
career_agent_workbench/   --------->   profile/
scripts/ and Makefile                  output/tracking/applications.sqlite3
packaged templates                     output/ and tmp/
fictional examples                     .blacklist
```

## Evidence and human review

The MRO is built from a bounded YAML file plus its concise source text. Tailored
claims must remain grounded in that evidence. Refinement, manual review, and
highlighting recheck the configured evidence digests and use conditional
database writes so stale work cannot silently replace a newer variant.

![Master resume object build workflow](docs/assets/master-resume-object-build.svg)

The application workflow keeps `v1`, `v2`, and `manual` variants distinct.
Automatic preference is useful, but an explicit selection stays pinned until
the user changes it. External application submission is deliberately outside
the product boundary.

![Application résumé workflow](docs/assets/aro-application-workflow.svg)

This preserved diagram describes the mature/target workflow. Its GLM 5.2 and
Flask compare/select labels are not current `1.0.0` demo defaults or routes:
the demo uses the centrally configured general model unless explicitly
overridden and exposes the smaller three-route Flask tracker/action adapter.

## Fictional offline demo

The tracked [demo source](examples/demo-workspace/README.md) uses Avery Demo,
Nimbus Quay Example Labs, reserved domains, and fixed synthetic dates. It
contains no database or generated binary. Materialize it only into ignored,
disposable local state:

```bash
make demo
```

That target creates one fictional tracker row, one selected `v1` résumé object,
one cover-letter object, and readable YAML, JSON, and HTML examples under
`tmp/demo-workspace`. `make demo` invokes the Python demo materializer; the
materializer itself performs no network, provider, model, browser, or
child-process work.

For a normal development checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
make test PYTHON=.venv/bin/python
make lint PYTHON=.venv/bin/python
```

Edit only the ignored `.env` to point at a private workspace. Do not place real
profile or application state in this repository.

## Current interfaces

Installed commands:

```text
career-agent-workbench                  # help and version
career-agent-workbench-seed-jobs        # public-job matching and atomic seed
career-agent-workbench-audit-jods       # bounded local JOD audit
career-agent-workbench-refine-resume    # governed v1-to-v2 refinement
career-agent-workbench-webapp           # local Flask tracker/action adapter
career-agent-workbench-mcp              # FastMCP stdio server
```

The Makefile composes the reusable job-seeding, draft generation, ARO sync,
second-pass refinement, highlighting, manual-pass, local web, demo, test, and
lint workflows. Omitted state variables emit no repository-relative private
paths; `JOB_IDS=all` selects eligible active records where the target supports
batch work.

The Flask adapter is intentionally small: it lists configured tracker snapshots
and dispatches selected records to an allowlisted existing Make target while
reporting only bounded action status. The MCP adapter exposes three
workspace-independent public job tools plus one lazy matching tool that uses
the same centrally resolved runtime. Public-only tool startup requires no
workspace.

## Synthetic presentation

The annotated PNGs below are **synthetic documentation illustrations**. They
preserve the mature workflow's explanatory composition while using the tracked
fictional demo identity. Tracker/action/progress views describe the current
adapter's concepts; JOD, résumé, variant, and cover-letter views illustrate the
underlying stored objects and human-review boundaries, not additional public
Flask routes.

### Seed and review tracker state

![Add and seed jobs illustration](docs/assets/tracker-add-seed-annotated.png)

The Add/Seed image illustrates CLI/Make matching-and-seeding composition, not
a current Flask Add route.

![Application tracker illustration](docs/assets/tracker-main-annotated.png)

![Tracker actions illustration](docs/assets/tracker-actions-menu-annotated.png)

![Background progress illustration](docs/assets/background-progress-annotated.png)

### Inspect stored objects and variants

![JOD editor illustration](docs/assets/job-description-editor-annotated.png)

![JOD diff illustration](docs/assets/job-description-diff-annotated.png)

![Application résumé object illustration](docs/assets/resume-editor-annotated.png)

![Résumé variant review illustration](docs/assets/resume-variant-review-annotated.png)

![Cover-letter object illustration](docs/assets/cover-letter-editor-annotated.png)

## Operational limits

- LinkedIn integration uses guest-accessible public pages only. There is no
  authentication, private-member access, or application submission.
- Generic job support is parser-only; the application does not fetch arbitrary
  URLs.
- Public providers and optional model runners can fail, rate-limit, or change
  shape. Failures remain bounded and content-hidden, and retries are finite.
- No workflow contacts an employer. Human review and an external decision are
  required before any real application action.
- SQLite state, profile evidence, generated documents, credentials, and model
  configuration are local and operator-owned.
- This is a single-user portfolio application, not an enterprise deployment or
  multi-tenant service.

## Development and validation

- [Representative behavior coverage](docs/behavior-coverage.md)
- [Fictional demo workspace](examples/demo-workspace/README.md)
- [1.0.0 release notes](docs/release-notes/1.0.0.md)
- [Repository guidance](AGENTS.md)
- [MIT license](LICENSE)

The public-safety checker scans tracked and distribution candidates without
printing matched content. CI is read-only and proportional. Version `1.0.0` is
released only through the guarded publication workflow; repository tags and
GitHub Releases are the authoritative release record.
