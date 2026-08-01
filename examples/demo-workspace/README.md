# Fictional Demo Workspace Source

This tracked directory contains static, newly authored fictional inputs for a
small offline Career Agent Workbench demonstration. The person, employers,
role, identifiers, dates, and contact details are invented, and all URLs use
reserved example domains.

Run `make demo` to copy and materialize these inputs into the root-ignored
`tmp/demo-workspace` directory, or supply a different disposable/private
`DEMO_WORKSPACE`. Use the materialized workspace only as local disposable or
private state.

The tracked source intentionally contains no SQLite database, `output/` or
`tmp/` tree, PDF, model/provider traffic, browser output, or generated binary
artifact. The demo factory creates only one tracker row, one v1 resume object,
one cover-letter object, and three readable text examples in the explicit
workspace.
