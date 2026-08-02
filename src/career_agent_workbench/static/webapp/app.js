"use strict";

(() => {
  const pollIntervalMilliseconds = 1500;
  const feedback = document.querySelector("#web-feedback");
  const panel = document.querySelector("#action-progress-panel");
  const details = document.querySelector("#action-progress-details");
  const progress = document.querySelector("#action-progress");
  const progressBar = document.querySelector("#action-progress-bar");
  const title = document.querySelector("#action-title");
  const failures = document.querySelector("#action-failures");
  const messages = document.querySelector("#action-messages");
  const collapseButton = document.querySelector("#action-collapse");
  const retryButton = document.querySelector("#action-retry");
  const dismissButton = document.querySelector("#action-dismiss");
  const terminalStates = new Set(["completed", "failed"]);
  terminalStates.add("partial");
  let currentActionId = null;

  function feedbackText(payload, fallback) {
    const parts = [payload.message || fallback];
    if (Number.isInteger(payload.accepted)) {
      parts.push(`${payload.accepted} accepted`);
    }
    if (Number.isInteger(payload.failed)) {
      parts.push(`${payload.failed} failed`);
    }
    return parts.join(" · ");
  }

  function renderFeedback(payload, fallback) {
    if (feedback) {
      feedback.textContent = feedbackText(payload, fallback);
      feedback.dataset.status = payload.status || "rejected";
    }
  }

  function replaceList(list, values, formatter) {
    if (!list) {
      return;
    }
    list.replaceChildren();
    for (const value of values || []) {
      const item = document.createElement("li");
      item.textContent = formatter(value);
      list.append(item);
    }
  }

  function renderAction(action) {
    if (!panel || !progress || !action) {
      return;
    }
    currentActionId = action.id || currentActionId;
    panel.hidden = false;
    if (title) {
      title.textContent = action.target || "Action status";
    }
    const stage = action.current_stage ? ` · ${action.current_stage}` : "";
    const currentJob = action.current_job_id ? ` · job ${action.current_job_id}` : "";
    const jobs = action.total_jobs
      ? ` · ${action.completed_jobs}/${action.total_jobs} jobs completed`
      : "";
    const atsCounts = action.updated_count || action.skipped_count
      ? ` · ${action.updated_count} updated, ${action.skipped_count} skipped`
      : "";
    progress.textContent = `${action.status}${stage}${currentJob}${jobs}${atsCounts} · ${action.message}`;
    progress.dataset.status = action.status;
    if (progressBar) {
      const maximum = Math.max(1, Number(action.progress_total) || 1);
      const value = Math.min(maximum, Math.max(0, Number(action.progress_current) || 0));
      progressBar.max = maximum;
      progressBar.value = value;
      progressBar.textContent = `${Math.round((value / maximum) * 100)}%`;
    }
    replaceList(
      failures,
      action.failed_work,
      (item) => `${item.job_id} · stage ${item.stage_index}: ${item.stage}`,
    );
    replaceList(messages, action.messages, (item) => item);
    if (retryButton) {
      retryButton.hidden = !action.retryable;
    }
    if (dismissButton) {
      dismissButton.hidden = !terminalStates.has(action.status);
    }
  }

  async function fetchActions() {
    const response = await fetch("/actions/status", {
      headers: {"Accept": "application/json"},
    });
    if (!response.ok) {
      throw new Error("status unavailable");
    }
    const payload = await response.json();
    return payload.actions || [];
  }

  async function pollAction(actionId, refreshUrl) {
    try {
      const actions = await fetchActions();
      const action = actions.find((item) => item.id === actionId);
      if (!action) {
        renderFeedback({}, "Action status is unavailable.");
        return;
      }
      renderAction(action);
      if (terminalStates.has(action.status)) {
        window.setTimeout(() => {
          window.location.assign(refreshUrl || document.body.dataset.refreshUrl || "/");
        }, 500);
        return;
      }
      window.setTimeout(() => pollAction(actionId, refreshUrl), pollIntervalMilliseconds);
    } catch {
      renderFeedback({}, "Action status is unavailable.");
    }
  }

  async function renderPreferredAction() {
    if (!panel) {
      return;
    }
    try {
      const actions = await fetchActions();
      const action = actions.find((item) => !terminalStates.has(item.status)) || actions[0];
      if (!action) {
        panel.hidden = true;
        return;
      }
      renderAction(action);
      if (!terminalStates.has(action.status)) {
        window.setTimeout(
          () => pollAction(action.id, document.body.dataset.refreshUrl),
          pollIntervalMilliseconds,
        );
      }
    } catch {
      renderFeedback({}, "Action status is unavailable.");
    }
  }

  async function submitForm(form) {
    renderFeedback({status: "submitting"}, "Submitting request.");
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: {"Accept": "application/json"},
      });
      const payload = await response.json();
      renderFeedback(payload, response.ok ? "Request accepted." : "Request rejected.");
      if (response.ok && payload.action_id) {
        renderAction({
          id: payload.action_id,
          target: "Action queued",
          status: "queued",
          current_stage: null,
          completed_jobs: 0,
          total_jobs: 0,
          updated_count: 0,
          skipped_count: 0,
          progress_current: 0,
          progress_total: 1,
          failed_work: [],
          messages: ["Action queued."],
          retryable: false,
          message: "Action queued.",
        });
        window.setTimeout(
          () => pollAction(payload.action_id, payload.refresh_url),
          pollIntervalMilliseconds,
        );
      }
    } catch {
      renderFeedback({}, "Request failed.");
    }
  }

  collapseButton?.addEventListener("click", () => {
    if (!details) {
      return;
    }
    details.hidden = !details.hidden;
    collapseButton.setAttribute("aria-expanded", String(!details.hidden));
    collapseButton.textContent = details.hidden ? "Expand" : "Collapse";
  });

  retryButton?.addEventListener("click", async () => {
    if (!currentActionId) {
      return;
    }
    const response = await fetch(`/actions/${currentActionId}/retry`, {
      method: "POST",
      headers: {"Accept": "application/json"},
    });
    const payload = await response.json();
    renderFeedback(payload, response.ok ? "Retry accepted." : "Retry rejected.");
    if (response.ok && payload.action_id) {
      window.setTimeout(
        () => pollAction(payload.action_id, document.body.dataset.refreshUrl),
        pollIntervalMilliseconds,
      );
    }
  });

  dismissButton?.addEventListener("click", async () => {
    if (!currentActionId) {
      return;
    }
    const response = await fetch(`/actions/${currentActionId}/dismiss`, {
      method: "POST",
      headers: {"Accept": "application/json"},
    });
    if (response.ok && panel) {
      panel.hidden = true;
      currentActionId = null;
      await renderPreferredAction();
    }
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest(
      "form[data-action-form], form[data-background-form], form[data-ingestion-form]",
    );
    if (!form) {
      return;
    }
    if (form.matches("[data-action-form]") && event.submitter?.hasAttribute("formaction")) {
      return;
    }
    event.preventDefault();
    submitForm(form);
  });

  renderPreferredAction();
})();
