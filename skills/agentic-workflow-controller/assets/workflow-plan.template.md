# __VERSION__ __WORKFLOW_ID__

Release: `__VERSION__`

Objective: __OBJECTIVE__

## Operating Rules

- Follow root `AGENTS.md` before editing, validating, committing, pushing, or
  closing out release work.
- Treat this committed plan as the canonical workflow plan.
- Use runtime state in `tmp/agentic-workflows/__WORKFLOW_ID__/tracker.json` as
  the local cursor and evidence log, not as a second implementation plan.
- Treat P steps as implementation phases and G steps as reassessment gates.
- P/G naming is a convention. Add no gates, one gate, or multiple gates based
  on risk and validation needs.
- At each G step, compare completed work against the remaining plan and either
  continue, amend the future steps, or pause for user input.
- Gate amendments may change only incomplete/future work. When a gate amends the
  plan, update this committed file, add a same-version `CHANGELOG.md` bullet,
  commit those changes on the workflow branch, and rebind the tracker digest.
- Do not mutate live tracker DB rows, generated artifacts, remote branches,
  PRs, tags, scheduled jobs, or external systems unless explicitly authorized by
  the workflow or user.
- Store only small sanitized tracker state, route prompts, route summaries, and
  artifact manifests in the repository. Store heavy or sensitive evidence under
  `$TMPDIR/career-agent-workbench-agentic/__WORKFLOW_ID__/`.

## Steps

### P01 - Readiness Check

Confirm branch, worktree, release version, and intended scope before editing.

### G01 - Scope Reassessment

Review P01 evidence against the rest of the plan. Continue, amend, or pause.

## Pause Conditions

- The remaining plan no longer matches the implementation evidence.
- The next step would change product behavior beyond the stated objective.
- The next step would mutate live tracker data or generated artifacts without
  explicit approval.
- Validation failure requires a broader refactor than the plan authorized.
