---
name: master-resume-yaml
description: Use before public-job search planning or resume and cover-letter tailoring when creating, rebuilding, or refining the private workspace MASTER-RESUME.yml from its human-maintained resume text source.
metadata:
  short-description: Build the private master resume object
---

# Master Resume YAML

Follow root `AGENTS.md` and resolve the external workspace through the normal
bootstrap/private-dotenv contract. This skill edits private profile evidence;
it never copies that evidence into the public repository or a test fixture.

## Inputs and target

- Human source: the configured `MASTER_RESUME_TEXT` member, conventionally
  `profile/MP-MASTER-RESUME.txt` inside the private workspace.
- Target MRO: the configured `MASTER_RESUME` member, conventionally
  `profile/MASTER-RESUME.yml`.
- Renderer: `scripts/render_resume_html.py` and the packaged resume template.

Read both sources before editing. Preserve intentional existing YAML structure
unless the user explicitly requests a rebuild.

## Required shape

Keep renderer-owned sections such as header, summary, Core Technical Skills,
experience, education, certifications, and portfolio as mappings with explicit
render state. Core-skill categories contain factual primary/additional items and
empty job-specific match lists. Experience bullets retain source text, category
links, supported skill links, and neutral zero match counters.

Every linked skill must exist in its declared Core Technical Skills category.
Preserve meaningful compound phrases. Add missing evidence-supported skills to
the appropriate category before linking them; never invent tools, scope,
metrics, certifications, or responsibilities.

## Passes

1. Reconcile source sections and the bare YAML mapping.
2. Normalize Core Technical Skills categories without job-specific matches.
3. Link every experience bullet to supported categories and skills.
4. Confirm neutral `jod_match_count` and
   `bullet_point_total_match_count` values.
5. Render and visually review a private preview.

The MRO remains neutral. Per-job JOD matching and pruning belong in ARO
generation, not this skill.

## Validation and handoff

Use `.venv/bin/python` to parse YAML and render a preview into the configured
private `tmp` directory. Run focused resume-renderer tests and `make lint` for
code changes. Report only the selected source/target classes, bullet/category
counts, invariant booleans, and validation results; do not paste private
content, paths, or identifiers into public evidence.
