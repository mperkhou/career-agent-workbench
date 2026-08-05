# Representative Behavior Coverage

This matrix links the portfolio's current user-visible behavior and real trust
boundaries to focused synthetic coverage. It is deliberately representative,
not a test catalog or a legacy-count comparison.

| Behavior or boundary | Current tests | What they prove |
|---|---|---|
| Central configuration and path precedence | `tests/test_config.py`, `tests/test_cli_integration.py` | Two-phase bootstrap/private-dotenv discovery, download/workspace members, exact precedence, permissions, and content-free failures. |
| Public models, provider, parser, and service | `tests/test_public_models_services.py`, `tests/test_public_provider_core.py`, `tests/test_public_parsing_ranking.py` | Bounded public schemas, guest-only provider behavior, parser-only generic jobs, and injected service limits. |
| Matching, filtering, and atomic seed | `tests/test_matching_workflow_core.py`, `tests/test_application_state.py` | Injected planning/search, policy filters, existing-row handling, and transactional metadata plus JOD persistence. |
| Tracker variants, explicit selection, and human review | `tests/test_application_state.py` | Distinct v1/v2/manual lineage, automatic preference, pinned explicit selection, CAS, and review-required workflow state. |
| Application resume, rendering, and ATS | `tests/test_application_resume.py`, `tests/test_resume_html_template.py`, `tests/test_ats.py` | Defensive resume transformations, autoescaped packaged rendering, bounded PDF behavior, and deterministic ATS diagnostics. |
| Evidence-grounded v2 refinement and conditional writes | `tests/test_refinement_workflow_core.py` | Canonical evidence derivation, strict grounded patches, digest rechecks, workflow revisions, and no-write failure paths. |
| Model execution, JSON recovery, and retry ownership | `tests/test_public_llm_clients.py`, `tests/test_workflow_retry.py`, `tests/test_cli_integration.py` | One internal API attempt, per-attempt deadlines, typed transient recovery, tolerant bare/fenced/wrapped JSON normalization, retryable generated-JSON syntax failures, and nonretryable valid-JSON validation failures. |
| Manual pass and highlighting boundaries | `tests/test_manual_highlighting_workflows.py` | Manual-from-v2 lineage, exact highlight targets, selection preservation, evidence drift rejection, and human review. |
| CLI, scripts, and Make composition | `tests/test_cli_integration.py`, `tests/test_script_integration.py`, `tests/test_makefile_workflows.py` | Post-parse runtime resolution, compatibility handling, path-free defaults, explicit job selection, and mocked composition. |
| Archive-first Flask tracker and workflows | `tests/test_archived_flask_source.py`, `tests/test_webapp_ingestion.py` | Bound inline frontend literals, 31-route activation, byte-stable reads, canonical add/select/delete state, variant-targeted CAS edit/revert, configured downloads, and bounded public URL ingestion. |
| JOD, resume, variant, CLO, and artifact state | `tests/test_application_state.py`, `tests/test_cover_letter_rendering.py`, `tests/test_archived_flask_source.py` | Exact artifact targeting, lineage-aware selection, CAS editing/revert/sync, JOD-plus-ATS projection, CLO sanitation, deterministic PDFs, and guarded copies. |
| MCP public tools and lazy configured matching | `tests/test_mcp_integration.py` | Workspace-free public tools, path-free schemas, startup-once configuration, lazy matching, owned-capability cleanup, and an offline stdio handshake/tool list. |
| Fictional demo workspace and materializer | `tests/test_demo_workspace.py` | From-scratch source semantics, one offline row/v1/CLO, readable examples, Flask presentation, and idempotence. |
| Operator runtime and exact skills | `tests/test_operator_helpers.py`, `tests/test_makefile_workflows.py`, `tests/test_guidance.py` | Public-venv commands, explicit Chromium installation, exact PID ownership, stale-state handling, browser opt-in, exact-five idempotent links, and current guidance/assets. |
| Release/package alignment | `tests/test_release_metadata.py`, `tests/test_public_safety.py`, `tests/test_smoke.py` | Version agreement, explicit restored sdist members, recursive wheel resources/modules, six entry points, and public artifact/install safety. |

## Deliberate exclusions

- Real legacy fixtures, profile data, and tracker state are denied; this demo
  replaces them with newly authored fictional content.
- One small demo does not need a fixture-loader or global `conftest.py`
  abstraction.
- Private fixtures, authenticated LinkedIn, application submission, multi-user
  deployment, and generalized security architecture remain outside this
  release. The bound archived inline frontend is now the public Flask baseline.
