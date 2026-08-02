"use strict";

(() => {
  const pollIntervalMilliseconds = 1500;
  const feedback = document.querySelector("#web-feedback");
  const progress = document.querySelector("#action-progress");
  const terminalStates = new Set(["completed", "failed"]);

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

  function renderAction(action) {
    if (!progress || !action) {
      return;
    }
    const stage = action.current_stage ? ` · ${action.current_stage}` : "";
    const counts = action.total_stages
      ? ` · ${action.completed_stages}/${action.total_stages} stages`
      : "";
    const atsCounts = action.updated_count || action.skipped_count
      ? ` · ${action.updated_count} updated, ${action.skipped_count} skipped`
      : "";
    progress.textContent = `${action.status}${stage}${counts}${atsCounts} · ${action.message}`;
    progress.dataset.status = action.status;
  }

  async function pollAction(actionId, refreshUrl) {
    try {
      const response = await fetch("/actions/status", {
        headers: {"Accept": "application/json"},
      });
      if (!response.ok) {
        renderFeedback({}, "Action status is unavailable.");
        return;
      }
      const payload = await response.json();
      const action = (payload.actions || []).find((item) => item.id === actionId);
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
          status: "queued",
          current_stage: null,
          completed_stages: 0,
          total_stages: 0,
          updated_count: 0,
          skipped_count: 0,
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
})();
