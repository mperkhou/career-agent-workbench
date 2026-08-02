---
name: career-agent-workbench
description: Use when installing, configuring, operating, or developing the local Career Agent Workbench application, including its public CLI, Flask tracker, FastMCP server, external private workspace, and offline validation surfaces.
metadata:
  short-description: Operate Career Agent Workbench
---

# Career Agent Workbench

Follow the repository root `AGENTS.md` before changing code or private state.
This skill is the command-level reference for the public application. Real
profile data, tracker rows, credentials, and generated documents belong in the
external workspace selected by the ignored root `.env`.

## Setup

From the public repository root:

```bash
make install
make console-scripts
make skill-link
```

`make install` creates `.venv`, installs the editable `.[dev,browser]` package,
and installs Playwright Chromium explicitly. It does not install Ollama, call a
model, or alter the private workspace. Browser opening is off by default.

Create the ignored bootstrap `.env` from `.env.example`. It should contain only
the external workspace and private-dotenv selectors. Put application tokens and
machine-specific runtime settings in the private workspace `.env` with mode
`0600`.

## Operator commands

- `make start-website` starts the local tracker and records its exact child PID
  and bounded log beneath ignored `tmp/website/`.
- `make stop-website` stops only that recorded child; a stale record is removed
  without port scanning or broad process termination.
- `make restart-website` performs the bounded stop/start sequence.
- Set `OPEN_BROWSER=1` only when an automatic browser open is wanted.
- `make launch-website` remains the foreground server command.
- `make test`, `make lint`, and `make format-check` use `.venv`.

## Main workflows

- `make seed-jobs MAX_JOBS=<count> DATE_POSTED=<window>` plans and seeds bounded
  public-job results through the configured matching workflow.
- `make regenerate-draft-resumes JOB_IDS=<job-id>` creates or refreshes `v1`.
- `make refine-draft-resumes JOB_IDS=<job-id>` derives evidence-grounded `v2`.
- `make regenerate-resumes JOB_IDS=<job-id>` runs `v1` then `v2`.
- `make manual-pass-resumes JOB_IDS=<job-id> MANUAL_PASS_PROFILE=regular`
  stores the governed `manual` variant.
- `make highlight-draft-resumes JOB_IDS=<job-id>` highlights the intended
  selected variant under the independent highlighting policy.
- `make audit-jods` audits without applying changes; use the CLI's explicit
  apply/ATS options only after review.

Automatic selection prefers `manual > v2 > v1`; explicit selection stays
pinned. Status, notes, archive, JOD, ARO, CLO, and artifact edits use the local
tracker and state APIs. No workflow submits an application or contacts an
employer.

## MCP

The canonical stdio executable is:

```text
.venv/bin/career-agent-workbench-mcp
```

Register an absolute executable path in the local MCP client. The server lists
guest-accessible public job tools without a workspace and resolves private state
only for configured matching. Do not put tokens or workspace values in MCP
arguments.

## Validation and release

Use fictional temporary workspaces for tests. Before a commit, run the focused
tests plus `git diff --check`, Ruff check/format, and the public-tree safety
command. Before release closeout, also run the complete suite, distribution
build/scan/install smoke, offline Chromium rendering, and injected MCP stdio
handshake. Merge, tag, publishing, and external career actions require explicit
authority.
