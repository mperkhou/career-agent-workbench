---
name: agentic-workflow-controller
description: Use when Codex should execute, resume, inspect, or reassess an existing committed agentic workflow with P steps, G gates, digest-bound runtime tracking, evidence routes, validation, pause conditions, and release closeout state. Use agentic-workflow-init for new workflows.
---

# Agentic Workflow Controller

Follow root `AGENTS.md`. The committed plan is authoritative; the ignored
tracker is only a bound cursor and evidence log. This controller never turns a
tracker flag into permission to publish, mutate private application state, or
perform release closeout.

## Resume an initialized workflow

1. Read root guidance, the complete committed plan, and the complete tracker.
2. Validate the tracker and plan digest before relying on its cursor:

   ```bash
   .venv/bin/python skills/agentic-workflow-controller/scripts/workflow_state.py \
     validate tmp/agentic-workflows/<workflow-id>/tracker.json
   ```

3. Identify the current P step or G gate and record `begin` before mutation.
4. Execute only the authorized step and path/state boundary.
5. Collect sanitized evidence before `complete` or `gate`.
6. Pause on plan-digest drift, scope change, failed invariant, missing
   authority, private-state risk, or release action requiring approval.

The helper supports `status`, `inspect`, `validate`, `begin`, `complete`,
`gate`, `pause`, `resume`, `rebind-plan`, artifact manifests, and evidence-route
lifecycle commands. It does not push, merge, tag, call providers, mutate live
tracker rows, or operate as a daemon.

## Evidence routes and artifacts

Use read-only evidence routes for bounded independent checks such as code-path
audits, schema comparisons, package metadata, or rendered-layout inspection.
The main implementor owns edits, live/private mutation, commits, PR changes,
and gate decisions.

Store only small sanitized control evidence beneath ignored
`tmp/agentic-workflows/<workflow-id>/`. Put databases, PDFs, screenshots, raw
logs, token stores, credentials, runtime homes, and other heavy/private
artifacts beneath a disposable system temporary directory such as
`$TMPDIR/career-agent-workbench-agentic/<workflow-id>/`.

## Plan amendments and closeout

Gate amendments affect incomplete work only. Update the committed plan and
matching changelog entry together, commit them normally, then use
`rebind-plan`. Preserve completed history. Never amend, rebase, squash, merge,
tag, publish, deploy, or contact an external party without the plan's explicit
boundary and current user authority.
