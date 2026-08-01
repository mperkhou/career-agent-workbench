# Representative Behavior Coverage

This matrix links the portfolio's current user-visible behavior and real trust
boundaries to focused synthetic coverage. It is deliberately representative,
not a test catalog or a legacy-count comparison.

| Behavior or boundary | Current tests | What they prove |
|---|---|---|
| Central configuration and path precedence | `tests/test_config.py`, `tests/test_cli_integration.py` | Dotenv discovery, independent workspace members, explicit precedence, and content-free missing-state failures. |
| Public models, provider, parser, and service | `tests/test_public_models_services.py`, `tests/test_public_provider_core.py`, `tests/test_public_parsing_ranking.py` | Bounded public schemas, guest-only provider behavior, parser-only generic jobs, and injected service limits. |
| Matching, filtering, and atomic seed | `tests/test_matching_workflow_core.py`, `tests/test_application_state.py` | Injected planning/search, policy filters, existing-row handling, and transactional metadata plus JOD persistence. |
| Tracker variants, explicit selection, and human review | `tests/test_application_state.py` | Distinct v1/v2/manual lineage, automatic preference, pinned explicit selection, CAS, and review-required workflow state. |
| Application resume, rendering, and ATS | `tests/test_application_resume.py`, `tests/test_resume_html_template.py`, `tests/test_ats.py` | Defensive resume transformations, autoescaped packaged rendering, bounded PDF behavior, and deterministic ATS diagnostics. |
| Evidence-grounded v2 refinement and conditional writes | `tests/test_refinement_workflow_core.py` | Canonical evidence derivation, strict grounded patches, digest rechecks, workflow revisions, and no-write failure paths. |
| Manual pass and highlighting boundaries | `tests/test_manual_highlighting_workflows.py` | Manual-from-v2 lineage, exact highlight targets, selection preservation, evidence drift rejection, and human review. |
| CLI, scripts, and Make composition | `tests/test_cli_integration.py`, `tests/test_script_integration.py`, `tests/test_makefile_workflows.py` | Post-parse runtime resolution, compatibility handling, path-free defaults, explicit job selection, and mocked composition. |
| Flask foreground/background path propagation | `tests/test_webapp.py` | One injected runtime, one bound store, allowlisted Make argv, app-scoped actions, and content-free status. |
| MCP public tools and lazy configured matching | `tests/test_mcp_integration.py` | Workspace-free public tools, path-free schemas, startup-once configuration, lazy matching, and owned-capability cleanup. |
| Fictional demo workspace and materializer | `tests/test_demo_workspace.py` | From-scratch source semantics, one offline row/v1/CLO, readable examples, Flask presentation, and idempotence. |
| Release metadata alignment | `tests/test_release_metadata.py`, `tests/test_smoke.py` | Version agreement, current release-note presence, explicit P13 sdist membership, and public package identity. |

## Deliberate exclusions and later ownership

- Real legacy fixtures, profile data, and tracker state are denied; this demo
  replaces them with newly authored fictional content.
- One small demo does not need a fixture-loader or global `conftest.py`
  abstraction.
- Guidance and agentic-workflow documentation tests wait for P16-owned public
  documentation.
- P13 added no hostile-object, allocator, address-reuse, exotic-subclass,
  implementation-detail, or exact legacy-count parity tests; existing focused
  trust-boundary tests remain represented in the matrix above.
- P14 owns CI, privacy, and package guardrails; P16 owns the root README and
  approved assets; G18 owns distribution builds.
