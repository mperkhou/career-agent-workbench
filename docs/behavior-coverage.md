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
| Manual pass and highlighting boundaries | `tests/test_manual_highlighting_workflows.py` | Manual-from-v2 lineage, exact highlight targets, selection preservation, evidence drift rejection, and human review. |
| CLI, scripts, and Make composition | `tests/test_cli_integration.py`, `tests/test_script_integration.py`, `tests/test_makefile_workflows.py` | Post-parse runtime resolution, compatibility handling, path-free defaults, explicit job selection, and mocked composition. |
| Tracker, ingestion, lifecycle, and progress | `tests/test_webapp.py`, `tests/test_webapp_tracker.py`, `tests/test_webapp_ingestion.py`, `tests/test_webapp_actions.py` | Read-only GET behavior, preserved views/selections, atomic public URL seeding, exact action ordering, bounded polling, and explicit ATS refresh. |
| JOD, resume, variant, CLO, and artifact workbench | `tests/test_webapp_artifacts.py`, `tests/test_webapp_editors.py`, `tests/test_webapp_cover_letters.py`, `tests/test_cover_letter_rendering.py` | Exact artifact targeting, lineage-aware comparison, CAS editing/revert/sync, JOD-plus-ATS writes, sanitation, deterministic PDFs, and guarded copies. |
| MCP public tools and lazy configured matching | `tests/test_mcp_integration.py` | Workspace-free public tools, path-free schemas, startup-once configuration, lazy matching, owned-capability cleanup, and an offline stdio handshake/tool list. |
| Fictional demo workspace and materializer | `tests/test_demo_workspace.py` | From-scratch source semantics, one offline row/v1/CLO, readable examples, Flask presentation, and idempotence. |
| Operator runtime and exact skills | `tests/test_operator_helpers.py`, `tests/test_makefile_workflows.py`, `tests/test_guidance.py` | Public-venv commands, explicit Chromium installation, exact PID ownership, stale-state handling, browser opt-in, exact-five idempotent links, and current guidance/assets. |
| Release/package alignment | `tests/test_release_metadata.py`, `tests/test_public_safety.py`, `tests/test_smoke.py` | Version agreement, explicit restored sdist members, recursive wheel resources/modules, six entry points, and public artifact/install safety. |

## Deliberate exclusions

- Real legacy fixtures, profile data, and tracker state are denied; this demo
  replaces them with newly authored fictional content.
- One small demo does not need a fixture-loader or global `conftest.py`
  abstraction.
- Exact legacy markup/count parity, private fixtures, authenticated LinkedIn,
  application submission, multi-user deployment, and generalized security
  architecture remain outside this release.
