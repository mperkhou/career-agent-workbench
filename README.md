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
normal local configuration uses two non-secret selectors in the ignored root
`.env`:

```dotenv
CAREER_AGENT_WORKBENCH_WORKSPACE=../career-agent-workbench-ops
CAREER_AGENT_WORKBENCH_PRIVATE_ENV_FILE=../career-agent-workbench-ops/.env
```

The bootstrap file is read only to find the workspace and private dotenv. The
private file is then loaded with exact mode `0600` where supported.
Conventional members—including the explicit configured `downloads/`
destination—resolve beneath the workspace. Optional overrides are documented
in [`.env.example`](.env.example). Explicit CLI/API values remain highest
precedence, and no loader mutates the process environment.

```text
public checkout                        private workspace
career_agent_workbench/   --------->   profile/
scripts/, skills/, Makefile            output/tracking/applications.sqlite3
packaged templates/static              output/, tmp/, and downloads/
fictional examples                     .blacklist and private .env
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

This diagram describes the governed workflow restored in `1.1.0`, including
workflow-specific GLM routing and local compare/select controls. Model calls
remain explicit, evidence-grounded, and outside offline tests.

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

For a normal development/operator checkout:

```bash
make install
make console-scripts
cp .env.example .env
make skill-link
make test
make lint
```

Edit only the ignored `.env` to point at a private workspace. Do not place real
profile or application state in this repository. `make install` creates the
public `.venv`, installs the editable `.[dev,browser]` package, and explicitly
installs Playwright Chromium; it does not install Ollama or call a provider.

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
lint workflows through public `.venv` executables. Omitted state variables emit
no repository-relative private paths; `JOB_IDS=all` selects eligible active
records where the target supports batch work.

The modular Flask adapter provides tracker filtering/lifecycle, Add/Seed and
bounded public-URL ingestion, allowlisted sequential resume actions with
progress, explicit ATS refresh, JOD and resume/CLO editing, variant comparison
and selection, inline/download artifacts, and explicit configured filesystem
copies. `GET` routes remain read-only. The MCP adapter exposes three
workspace-independent public job tools plus one lazy matching tool that uses
the same centrally resolved runtime. Public-only tool startup requires no
workspace.

## Local website, skills, and MCP

```bash
make start-website                 # browser stays closed
make start-website OPEN_BROWSER=1  # explicit opt-in
make stop-website
make restart-website
```

Lifecycle state and the bounded server log live under ignored `tmp/website/`.
Only the exact recorded child PID is stopped; stale records do not trigger port
or process scans. `make launch-website` remains the foreground command.

`make skill-link` idempotently links exactly the five tracked public skills
into the configured `CODEX_SKILLS_DIR` (normally `~/.codex/skills`) and refuses
to overwrite a non-symlink. The canonical MCP registration points to the
absolute public `.venv/bin/career-agent-workbench-mcp` executable. MCP stdio
arguments contain no private workspace values or tokens.

## Synthetic presentation

The annotated PNGs below are **synthetic documentation illustrations** using the
tracked fictional identity. They document the restored tracker, action,
progress, JOD, résumé, variant, and cover-letter surfaces without exposing a
private operator workspace.

### Seed and review tracker state

![Add and seed jobs illustration](docs/assets/tracker-add-seed-annotated.png)

The Add/Seed surface joins the existing guest provider and generic parser to
atomic state seeding and the same app-scoped background action registry.

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
- Generic URL ingestion uses a bounded public HTTP seam and normalized parser;
  tests inject responses and never use the network.
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
- [1.1.0 release notes](docs/release-notes/1.1.0.md)
- [Repository guidance](AGENTS.md)
- [MIT license](LICENSE)

The public-safety checker scans tracked and distribution candidates without
printing matched content. CI is read-only and proportional. Version `1.1.0`
remains a release candidate until the reviewed PR is merged normally and its
merge commit receives the separately authorized `v1.1.0` tag.
