# Agent Guidance

Use a Python-first architecture and public-safe synthetic evidence in this
repository.

- Never add real profile or application state, generated career artifacts,
  credentials, employer-internal material, or machine-specific paths.
- Keep private workspace state in the root-anchored ignored directories and
  preserve the root `.env` protections.
- Keep edits scoped, add focused tests, and align the version, changelog, and
  release note for committed changes.
- Do not use destructive Git operations or discard work without explicit
  authority.
- Preserve human-review checkpoints and require explicit authority before
  actions that publish, contact third parties, or change external state.

## Local operator contract

- Run `make install` from the repository root to create the ignored public
  `.venv`, install `.[dev,browser]`, and install Playwright Chromium. Do not use
  the legacy repository environment for public operation.
- The ignored root `.env` is bootstrap-only. It selects the external workspace
  and its private dotenv; tokens and machine-specific settings belong only in
  that external mode-`0600` file.
- Use `make start-website`, `make stop-website`, and `make restart-website` for
  PID-owned local lifecycle. Browser opening is opt-in through
  `OPEN_BROWSER=1`; never stop processes by port scanning.
- `make skill-link` manages exactly the five public repository skills. It must
  not replace unrelated personal skills or overwrite a non-symlink.
- The canonical MCP executable is the absolute public
  `.venv/bin/career-agent-workbench-mcp`; retain a clearly named legacy
  registration until the final cutover gate.

## Validation and release

- Focused changes require their tests, Ruff check/format, `git diff --check`,
  and public-tree safety. Release readiness additionally requires the complete
  suite, wheel/sdist build and scan, isolated wheel smoke, offline Chromium,
  browser workbench, and injected MCP stdio checks.
- Release PRs stay unmerged and untagged until their final gate and explicit
  user approval. Preserve normal reviewed commits; do not squash or rebase them.
