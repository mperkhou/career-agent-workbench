---
name: manual-resume-passthrough
description: Use when the user asks for a manual resume pass, manual passthrough, Jack and Jill review, ATS refinement pass, or the next explicit job row in Career Agent Workbench. Guides a grounded manual variant while preserving v1, v2, lineage, and explicit selection.
---

# Manual Resume Passthrough

Follow root `AGENTS.md`. Resolve all application and profile state through the
ignored bootstrap/private workspace; never pass private content or identifiers
through process arguments, public logs, tests, or commits.

## Preconditions

- The exact application row exists with usable JOD and evidence state.
- `v1` and `v2` exist when the governed manual workflow depends on them.
- The configured MRO and human source are current.
- The requested profile is exactly `economy`, `regular`, or `premium`.

Profiles are execution policy, not variant keys. `regular` is the normal
`gpt-5.6-sol`/`high` policy. All profiles write the same `manual` row, preserve
`v1` and `v2`, and leave explicit selection unchanged. Automatic preference is
`manual > v2 > v1`.

## Workflow

1. Inspect the coherent workflow snapshot, v2 evidence/critique, ATS
   diagnostics, selected JOD, MRO evidence, current selection, and lineage.
2. Treat ATS/model suggestions as review signals. Improve truthful wording,
   ordering, density, and supported terminology; do not invent evidence.
3. Run only the exact requested row:

   ```bash
   make manual-pass-resumes JOB_IDS=<job-id> MANUAL_PASS_PROFILE=regular
   ```

4. Confirm the stored `manual` variant derives from v2, includes governed YAML,
   HTML, PDF, ATS, evidence, validation, and model metadata, and leaves all
   non-target variants unchanged.
5. Review `/resumes/<job-id>/variants`; move an explicit selection only after
   human review.
6. Render every PDF page to images and inspect legibility, clipping, overlap,
   wrapping, and semantic completeness. Recalculate ATS only through the
   explicit local action when needed.

Highlighting is separate and keeps its independent model/reasoning policy. A
manual profile must never configure highlighting.

## Output

Report the application only as authorized by the user, ATS movement, remaining
unsupported/missing terms, artifact presence, lineage/selection invariants,
visual-review outcome, and commands used. Do not modify applied status, notes,
MRO evidence, or unrelated rows unless separately authorized.
