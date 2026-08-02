SYSTEM_PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PYTHON ?= $(VENV_PYTHON)
SEED_JOBS ?= $(VENV)/bin/career-agent-workbench-seed-jobs
AUDIT_JODS ?= $(VENV)/bin/career-agent-workbench-audit-jods
REFINE_RESUME ?= $(VENV)/bin/career-agent-workbench-refine-resume
WEBAPP ?= $(VENV)/bin/career-agent-workbench-webapp
CODEX_SKILLS_DIR ?= $(HOME)/.codex/skills
CONSOLE_SCRIPTS := career-agent-workbench career-agent-workbench-audit-jods career-agent-workbench-refine-resume career-agent-workbench-seed-jobs career-agent-workbench-webapp career-agent-workbench-mcp

HOST ?= 127.0.0.1
PORT ?= 8765
OPEN_BROWSER ?=

WORKSPACE ?=
DATABASE ?=
OUTPUT_DIR ?=
PROFILE_DIR ?=
MASTER_RESUME ?=
MASTER_RESUME_TEXT ?=
RESUME_TEMPLATE ?=
BLACKLIST ?=
TMP_DIR ?=
DEMO_WORKSPACE ?= tmp/demo-workspace

JOB_IDS ?= all
LOCATION ?= United States
DATE_POSTED ?= past_week
LIMIT_PER_QUERY ?= 10
MAX_QUERIES ?= 6
MAX_JOBS ?= 10
JOD_MODEL ?=
CORE_SKILL_MODEL ?=
FIRST_DRAFT_LLM_TIMEOUT_SECONDS ?= 300
FIRST_DRAFT_LLM_RETRIES ?= 1
FIRST_DRAFT_FORCE ?=
SECOND_PASS_MODEL ?=
SECOND_PASS_TIMEOUT_SECONDS ?= 600
SECOND_PASS_RETRIES ?= 1
CODEX_COMMAND ?= codex
CODEX_MODEL ?=
CODEX_REASONING_EFFORT ?=
CODEX_TIMEOUT_SECONDS ?= 900
CODEX_RETRIES ?= 1
MANUAL_PASS_PROFILE ?= regular
MANUAL_PASS_CODEX_MODEL ?=
MANUAL_PASS_CODEX_REASONING_EFFORT ?=
HIGHLIGHT_CODEX_MODEL ?=
HIGHLIGHT_CODEX_REASONING_EFFORT ?=
HIGHLIGHT_RESUME_VARIANT ?=
HIGHLIGHT_MAX_STRONG_SPANS_PER_BULLET ?=
HIGHLIGHT_EXPERIENCE_COMPANY ?=
HIGHLIGHT_EXPERIENCE_JOB_ORDER ?=

workspace_flag = $(if $(strip $(WORKSPACE)),--workspace "$(WORKSPACE)")
database_flag = $(if $(strip $(DATABASE)),--database "$(DATABASE)")
output_flag = $(if $(strip $(OUTPUT_DIR)),--output-dir "$(OUTPUT_DIR)")
profile_flag = $(if $(strip $(PROFILE_DIR)),--profile-dir "$(PROFILE_DIR)")
master_flag = $(if $(strip $(MASTER_RESUME)),--master-resume "$(MASTER_RESUME)")
master_text_flag = $(if $(strip $(MASTER_RESUME_TEXT)),--master-resume-text "$(MASTER_RESUME_TEXT)")
template_flag = $(if $(strip $(RESUME_TEMPLATE)),--template "$(RESUME_TEMPLATE)")
blacklist_flag = $(if $(strip $(BLACKLIST)),--blacklist-path "$(BLACKLIST)")
tmp_flag = $(if $(strip $(TMP_DIR)),--tmp-dir "$(TMP_DIR)")
batch_job_flags = $(if $(filter all,$(strip $(JOB_IDS))),,$(foreach id,$(strip $(JOB_IDS)),--job-id "$(id)"))
refine_job_flags = $(if $(filter all,$(strip $(JOB_IDS))),--all-active,$(if $(strip $(JOB_IDS)),$(foreach id,$(strip $(JOB_IDS)),--job-id "$(id)"),--all-active))
manual_job_flags = $(if $(filter all,$(strip $(JOB_IDS))),,$(foreach id,$(strip $(JOB_IDS)),--job-id "$(id)"))
manual_model_flag = $(if $(strip $(MANUAL_PASS_CODEX_MODEL)),--codex-model "$(MANUAL_PASS_CODEX_MODEL)",$(if $(strip $(CODEX_MODEL)),--codex-model "$(CODEX_MODEL)"))
manual_effort_flag = $(if $(strip $(MANUAL_PASS_CODEX_REASONING_EFFORT)),--codex-reasoning-effort "$(MANUAL_PASS_CODEX_REASONING_EFFORT)",$(if $(strip $(CODEX_REASONING_EFFORT)),--codex-reasoning-effort "$(CODEX_REASONING_EFFORT)"))
first_draft_force_flag = $(if $(filter 1 true,$(strip $(FIRST_DRAFT_FORCE))),--force)
highlight_span_flag = $(if $(strip $(HIGHLIGHT_MAX_STRONG_SPANS_PER_BULLET)),--max-strong-spans-per-bullet "$(HIGHLIGHT_MAX_STRONG_SPANS_PER_BULLET)")
highlight_company_flag = $(if $(strip $(HIGHLIGHT_EXPERIENCE_COMPANY)),--experience-company "$(HIGHLIGHT_EXPERIENCE_COMPANY)")
highlight_job_order_flag = $(if $(strip $(HIGHLIGHT_EXPERIENCE_JOB_ORDER)),--experience-job-order "$(HIGHLIGHT_EXPERIENCE_JOB_ORDER)")
open_browser_flag = $(if $(filter 1 true,$(strip $(OPEN_BROWSER))),--open-browser)

.PHONY: help install venv install-browser console-scripts skill-link env test lint format-check seed-jobs audit-jods \
	generate-draft-resumes regenerate-draft-resumes regenerate-resumes \
	regenerate-resume-variants regenerate-aro-objects sync-draft-to-aro \
	refine-draft-resumes second-pass-refinement highlight-draft-resumes \
	manual-pass-resumes launch-website start-website stop-website restart-website demo

help:
	@echo "Public-safe setup, operator, CLI, demo, test, and lint targets"

install: venv install-browser

venv: $(VENV)/.installed

$(VENV)/.installed: pyproject.toml
	$(SYSTEM_PYTHON) -m venv "$(VENV)"
	$(VENV_PYTHON) -m pip install -e ".[dev,browser]"
	@touch "$(VENV)/.installed"

install-browser: venv
	$(VENV_PYTHON) -m playwright install chromium

console-scripts:
	@for command in $(CONSOLE_SCRIPTS); do \
		test -x "$(VENV)/bin/$$command" || exit 1; \
	done

skill-link:
	$(VENV_PYTHON) scripts/workbench_operator.py skills link --destination "$(CODEX_SKILLS_DIR)"

env:
	$(PYTHON) --version

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts

format-check:
	$(PYTHON) -m ruff format --check src tests scripts

seed-jobs:
	$(SEED_JOBS) $(workspace_flag) $(database_flag) $(output_flag) $(profile_flag) $(master_flag) $(blacklist_flag) --location "$(LOCATION)" --date-posted "$(DATE_POSTED)" --limit-per-query "$(LIMIT_PER_QUERY)" --max-queries "$(MAX_QUERIES)" --max-jobs "$(MAX_JOBS)"

audit-jods:
	$(AUDIT_JODS) $(workspace_flag) $(database_flag) $(output_flag)

generate-draft-resumes:
	$(PYTHON) scripts/application_resume_generate_drafts.py $(workspace_flag) $(database_flag) $(output_flag) $(master_flag) $(template_flag) $(batch_job_flags) $(if $(strip $(CORE_SKILL_MODEL)),--api-model "$(CORE_SKILL_MODEL)") $(if $(strip $(JOD_MODEL)),--jod-model "$(JOD_MODEL)") --llm-timeout-seconds "$(FIRST_DRAFT_LLM_TIMEOUT_SECONDS)" --llm-retries "$(FIRST_DRAFT_LLM_RETRIES)"

regenerate-draft-resumes:
	$(PYTHON) scripts/application_resume_generate_drafts.py $(workspace_flag) $(database_flag) $(output_flag) $(master_flag) $(template_flag) $(batch_job_flags) $(first_draft_force_flag) $(if $(strip $(CORE_SKILL_MODEL)),--api-model "$(CORE_SKILL_MODEL)") $(if $(strip $(JOD_MODEL)),--jod-model "$(JOD_MODEL)") --llm-timeout-seconds "$(FIRST_DRAFT_LLM_TIMEOUT_SECONDS)" --llm-retries "$(FIRST_DRAFT_LLM_RETRIES)"

regenerate-resumes: regenerate-draft-resumes refine-draft-resumes

regenerate-resume-variants: regenerate-resumes

regenerate-aro-objects:
	$(PYTHON) scripts/application_resume_regenerate_aros.py $(workspace_flag) $(database_flag) $(output_flag) $(master_flag) $(batch_job_flags)

sync-draft-to-aro:
	$(PYTHON) scripts/application_resume_sync_drafts_to_aro.py $(workspace_flag) $(database_flag) $(output_flag) $(batch_job_flags)

refine-draft-resumes:
	$(REFINE_RESUME) $(workspace_flag) $(database_flag) $(output_flag) $(master_flag) $(master_text_flag) $(template_flag) $(refine_job_flags) $(if $(strip $(SECOND_PASS_MODEL)),--api-model "$(SECOND_PASS_MODEL)") --api-timeout-seconds "$(SECOND_PASS_TIMEOUT_SECONDS)" --api-retries "$(SECOND_PASS_RETRIES)"

second-pass-refinement: refine-draft-resumes

highlight-draft-resumes:
	$(PYTHON) scripts/application_resume_highlight_drafts.py $(workspace_flag) $(database_flag) $(output_flag) $(master_flag) $(master_text_flag) $(tmp_flag) $(template_flag) $(batch_job_flags) --codex-command "$(CODEX_COMMAND)" $(if $(strip $(HIGHLIGHT_CODEX_MODEL)),--codex-model "$(HIGHLIGHT_CODEX_MODEL)",$(if $(strip $(CODEX_MODEL)),--codex-model "$(CODEX_MODEL)")) $(if $(strip $(HIGHLIGHT_CODEX_REASONING_EFFORT)),--codex-reasoning-effort "$(HIGHLIGHT_CODEX_REASONING_EFFORT)",$(if $(strip $(CODEX_REASONING_EFFORT)),--codex-reasoning-effort "$(CODEX_REASONING_EFFORT)")) $(if $(strip $(HIGHLIGHT_RESUME_VARIANT)),--variant-key "$(HIGHLIGHT_RESUME_VARIANT)") $(highlight_span_flag) $(highlight_company_flag) $(highlight_job_order_flag) --timeout-seconds "$(CODEX_TIMEOUT_SECONDS)" --retry-count "$(CODEX_RETRIES)"

manual-pass-resumes:
	$(if $(strip $(manual_job_flags)),$(PYTHON) scripts/application_resume_manual_pass.py $(workspace_flag) $(database_flag) $(output_flag) $(master_flag) $(master_text_flag) $(tmp_flag) $(template_flag) $(manual_job_flags) --codex-command "$(CODEX_COMMAND)" --manual-pass-profile "$(MANUAL_PASS_PROFILE)" $(manual_model_flag) $(manual_effort_flag) --timeout-seconds "$(CODEX_TIMEOUT_SECONDS)" --retry-count "$(CODEX_RETRIES)",@echo "Set JOB_IDS to one or more explicit job IDs." >&2; exit 2)

launch-website:
	$(WEBAPP) $(workspace_flag) $(database_flag) $(output_flag) $(profile_flag) $(master_flag) $(master_text_flag) $(blacklist_flag) $(tmp_flag) --host "$(HOST)" --port "$(PORT)"

start-website:
	$(VENV_PYTHON) scripts/workbench_operator.py website start --host "$(HOST)" --port "$(PORT)" $(open_browser_flag)

stop-website:
	$(VENV_PYTHON) scripts/workbench_operator.py website stop

restart-website: stop-website
	$(MAKE) start-website HOST="$(HOST)" PORT="$(PORT)" OPEN_BROWSER="$(OPEN_BROWSER)"

demo:
	$(PYTHON) scripts/create_demo_workspace.py --source examples/demo-workspace --workspace "$(DEMO_WORKSPACE)"
