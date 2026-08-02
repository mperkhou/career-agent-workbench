---
name: agentic-workflow-init
description: Use when Codex should bootstrap a new committed agentic workflow plan before implementation, including repository readiness, SemVer, a canonical plan, changelog bootstrap, ignored tracker initialization, digest binding, and a controller kickoff prompt. Do not use to execute an initialized workflow.
---

# Agentic Workflow Init

Follow root `AGENTS.md`. This skill creates control state only; it does not
implement product work, mutate application rows/artifacts, push, open a PR,
merge, tag, release, or contact an external system.

## Bootstrap

1. Verify the repository root, worktree, current branch, remote baseline, and
   operator authority.
2. Confirm one target SemVer and create the feature branch.
3. Create the committed canonical plan at
   `docs/agentic-workflows/<version>-<slug>.md` using the controller plan asset.
4. Add the matching top changelog heading and bootstrap bullet.
5. Commit the plan and changelog before implementation.
6. Initialize ignored runtime state bound to the committed plan digest,
   revision, branch, bootstrap commit, current P step, and pause conditions.
7. Return a kickoff prompt for `$agentic-workflow-controller`.

Use the public environment helper:

```bash
.venv/bin/python skills/agentic-workflow-controller/scripts/workflow_state.py \
  init <workflow-id> --version <X.Y.Z> --objective "<objective>" \
  --current-step P01 --branch <branch> \
  --plan-path docs/agentic-workflows/<X.Y.Z>-<slug>.md \
  --plan-revision 1 --bootstrap-commit <commit>
```

Runtime state belongs under ignored `tmp/agentic-workflows/<workflow-id>/` and
must remain a cursor/evidence log, not a competing plan. Keep sensitive or
heavy evidence outside the repository tree. User authority remains required
for publication, destructive work, private application mutation, and release
closeout.
