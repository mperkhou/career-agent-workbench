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
