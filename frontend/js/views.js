// View builders: chats, the timeline, and every appended section.

import {
  TOOL_INFO,
  downloadText,
  formatBytes,
  formatDuration,
  formatNumber,
  h,
  hostOf,
  icon,
  pad,
  profileLabel,
  slugify,
  stopReasonLabel,
  truncate,
} from "./dom.js";
import { renderMarkdown, safeUrl } from "./markdown.js";

// ---------------------------------------------------------------- chat + timeline

const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Play the entrance animation once, only for content that arrives while its chat is
// on screen; content that arrived in a background chat is simply there when opened.
// The class is removed afterwards because display:none would replay it on re-show.
function enter(element, visible, settleMs = 650) {
  if (!visible || reducedMotion()) return element;
  element.classList.add("is-entering");
  const onEnd = (event) => {
    if (event.target !== element) return; // child stagger animations bubble too
    element.removeEventListener("animationend", onEnd);
    setTimeout(() => element.classList.remove("is-entering"), settleMs);
  };
  element.addEventListener("animationend", onEnd);
  return element;
}

export class Chat {
  constructor(id) {
    this.root = h("div", { class: "chat", id, hidden: true });
    this.inner = h("div", { class: "chat-inner" });
    this.root.append(this.inner);
    this.timeline = null;
    this.pinned = true;
    this.savedScroll = 0;
    this.origin = performance.now();
    this.autoScrollUntil = 0;
    this.root.addEventListener(
      "scroll",
      () => {
        // Ignore the intermediate positions of our own smooth scrolling.
        if (performance.now() < this.autoScrollUntil) return;
        const { scrollHeight, scrollTop, clientHeight } = this.root;
        this.pinned = scrollHeight - scrollTop - clientHeight < 160;
      },
      { passive: true },
    );
  }

  clock() {
    return (performance.now() - this.origin) / 1000;
  }

  show() {
    this.root.hidden = false;
    if (this.pinned) this.scrollToEnd();
    else this.root.scrollTop = this.savedScroll;
  }

  hide() {
    this.savedScroll = this.root.scrollTop;
    this.root.hidden = true;
  }

  appendSection(element) {
    this.timeline = null; // the next step starts a new timeline below this section
    this.inner.append(enter(element, !this.root.hidden));
    this.follow();
    return element;
  }

  step(options) {
    if (!this.timeline) {
      this.timeline = h("div", { class: "timeline" });
      this.inner.append(enter(this.timeline, !this.root.hidden, 0));
    }
    const item = new TimelineItem(this.clock(), options);
    this.timeline.append(enter(item.root, !this.root.hidden, 0));
    this.follow();
    return item;
  }

  follow() {
    if (this.pinned && !this.root.hidden) requestAnimationFrame(() => this.scrollToEnd(true));
  }

  scrollToEnd(smooth = false) {
    const behavior = smooth && !reducedMotion() ? "smooth" : "auto";
    if (behavior === "smooth") this.autoScrollUntil = performance.now() + 700;
    this.root.scrollTo({ top: this.root.scrollHeight, behavior });
  }
}

function describeArgs(args) {
  if (!args || typeof args !== "object") return "";
  if (typeof args.query === "string") return `"${truncate(args.query, 96)}"`;
  if (typeof args.url === "string") return truncate(args.url, 96);
  return truncate(JSON.stringify(args), 96);
}

function prettyOutput(content) {
  try {
    return JSON.stringify(JSON.parse(content), null, 2);
  } catch {
    return content;
  }
}

class TimelineItem {
  constructor(seconds, { status = "info", text = "", note, tool }) {
    this.textEl = h("div", { class: "tl-text" });
    this.body = h("div", { class: "tl-body" }, this.textEl);
    this.root = h(
      "div",
      { class: `tl-item is-${status}` },
      h("span", { class: "tl-dot" }),
      this.body,
      h("span", { class: "tl-time" }, `+${seconds.toFixed(1)}s`),
    );
    if (tool) {
      const args = describeArgs(tool.args);
      this.textEl.classList.add("tl-text-tool");
      this.textEl.append(h("span", { class: "tl-tool" }, tool.name), h("span", { class: "tl-args", title: args }, args));
    } else {
      this.textEl.textContent = text;
    }
    if (note) this.setNote(note);
  }

  update({ status, text, note } = {}) {
    if (status) {
      // Swap only the status class so an entrance animation in progress keeps running.
      for (const name of [...this.root.classList]) {
        if (name.startsWith("is-") && name !== "is-entering") this.root.classList.remove(name);
      }
      this.root.classList.add(`is-${status}`);
    }
    if (text !== undefined) this.textEl.textContent = text;
    if (note) this.setNote(note);
    return this;
  }

  setNote(note) {
    const value = String(note ?? "").trim();
    if (!value) return;
    this.noteEl?.remove();
    this.noteEl =
      value.length > 260
        ? h("details", { class: "tl-note" }, h("summary", {}, truncate(value, 180)), h("div", { class: "tl-note-full" }, value))
        : h("div", { class: "tl-note" }, value);
    this.body.append(this.noteEl);
  }

  // Tool output stays collapsed; its <pre> is only built when first opened.
  setOutput(content, ok) {
    this.update({ status: ok ? "done" : "error" });
    const text = String(content ?? "");
    const details = h("details", { class: "tl-output" }, h("summary", {}, `${ok ? "Output" : "Error"} · ${formatBytes(text.length)}`));
    details.addEventListener("toggle", () => {
      if (details.open && !details.querySelector("pre")) details.append(h("pre", {}, prettyOutput(text)));
    });
    this.body.append(details);
  }
}

// ---------------------------------------------------------------- sections

export function section({ icon: iconName, kicker, title, tone = "violet", meta = null, actions = [], className = "" }) {
  const metaEl = h("div", { class: "section-meta" }, meta);
  const head = h(
    "header",
    { class: "section-head" },
    h("div", { class: `section-icon tone-${tone}` }, icon(iconName)),
    h("div", { class: "section-titles" }, h("div", { class: "kicker" }, kicker), h("h2", { class: "section-title" }, title)),
    h("div", { class: "section-side" }, metaEl, ...actions),
  );
  const body = h("div", { class: "section-body" });
  return { root: h("section", { class: `card ${className}`.trim() }, head, body), body, metaEl };
}

function badge(text, tone = "muted") {
  return h("span", { class: `badge tone-${tone}` }, text);
}

function exportButton(filename, getText) {
  return h(
    "button",
    { class: "btn btn-small", type: "button", onclick: () => downloadText(filename, getText()) },
    icon("download"),
    "Export Markdown",
  );
}

// -- research configuration

let helpCount = 0;

// A "?" button that shows its content on hover, keyboard focus, or click (touch).
function help(content, label) {
  helpCount += 1;
  const id = `help-${helpCount}`;
  const pop = h("span", { class: "help-pop", role: "tooltip", id }, content);
  const button = h("button", { class: "help-btn", type: "button", "aria-label": label, "aria-describedby": id }, "?");
  const root = h("span", { class: "help" }, button, pop);
  button.addEventListener("click", () => root.classList.toggle("is-open"));
  button.addEventListener("blur", () => root.classList.remove("is-open"));
  return { root, set: (next) => pop.replaceChildren(next) };
}

function modeDetails(name, profile) {
  const models = [...new Set(Object.values(profile.models).map((tier) => tier.model))];
  const cheap = name === "fast";
  const row = (term, detail) => [h("dt", {}, term), h("dd", {}, detail)];
  return h(
    "dl",
    { class: "help-list" },
    row("Cost", cheap ? "Low" : "High"),
    row("Latency", cheap ? "Low" : "High"),
    row("Models", models.join(", ")),
    row("Passes", `Up to ${profile.max_passes} · ${profile.max_tool_calls_per_pass} tool calls each`),
    row("Tools", profile.allowed_tools.map((tool) => TOOL_INFO[tool]?.label ?? tool).join(", ")),
    row("Deadline", `${formatDuration(profile.deadline_seconds)} per analyst`),
  );
}

export function configSection({ meta, values, onStart, onReset }) {
  const card = section({ icon: "sliders", kicker: "Choose a topic, team size and depth", title: "Research configuration", tone: "cyan" });
  const topic = h("textarea", {
    class: "input topic-input",
    rows: 2,
    maxlength: 500,
    placeholder: "What should the analysts research?",
    "aria-label": "Research topic",
  });
  topic.value = values.topic ?? "";
  const count = h("input", {
    class: "input count-input",
    type: "number",
    min: meta.min_analysts,
    max: meta.max_analysts,
    step: 1,
    "aria-label": "Number of analysts",
  });
  count.value = values.maxAnalysts ?? 3;
  const stepCount = (delta) => {
    count.value = Math.min(meta.max_analysts, Math.max(meta.min_analysts, (Number(count.value) || 0) + delta));
  };
  const minus = h("button", { class: "btn btn-icon", type: "button", "aria-label": "Fewer analysts", onclick: () => stepCount(-1) }, icon("minus"));
  const plus = h("button", { class: "btn btn-icon", type: "button", "aria-label": "More analysts", onclick: () => stepCount(1) }, icon("plus"));
  const countHelp = help(
    `How many independent perspectives should the research use? Select ${meta.min_analysts}–${meta.max_analysts} analysts; they all run in parallel.`,
    "About perspectives",
  );

  // Research mode: one button that toggles between the profiles.
  const names = Object.keys(meta.profiles);
  let mode = names.includes(values.profile) ? values.profile : meta.default;
  const modeIcon = h("span", { class: "mode-icon" });
  // Every label shares one grid cell, so the button is as wide as the longest and never resizes.
  const modeName = h(
    "span",
    { class: "mode-name" },
    names.map((name) => h("span", { "data-for": name }, profileLabel(name))),
  );
  const modeButton = h("button", { class: "mode-toggle", type: "button" }, modeIcon, modeName, icon("swap", "icon mode-swap"));
  const modeHelp = help("", "About research modes");
  function setMode(next) {
    mode = next;
    const label = profileLabel(next);
    modeIcon.replaceChildren(icon(next === "fast" ? "zap" : "layers"));
    const other = names[(names.indexOf(next) + 1) % names.length];
    modeButton.setAttribute("aria-label", `${label}. Click to switch to ${profileLabel(other)}.`);
    modeButton.dataset.mode = next;
    modeHelp.set(modeDetails(next, meta.profiles[next]));
  }
  modeButton.addEventListener("click", () => setMode(names[(names.indexOf(mode) + 1) % names.length]));
  setMode(mode);

  const error = h("div", { class: "form-error", role: "alert" });
  const start = h("button", { class: "btn btn-primary", type: "submit" }, "Start research");
  const reset = h("button", { class: "btn", type: "button", hidden: true, onclick: () => onReset() }, icon("restart"), "New research");

  const form = h(
    "form",
    { class: "config-form", novalidate: true },
    h("label", { class: "field" }, h("span", { class: "field-label" }, "Research topic"), topic),
    h(
      "div",
      { class: "config-row" },
      h(
        "div",
        { class: "field" },
        h("span", { class: "field-label" }, "Perspectives", countHelp.root),
        h("div", { class: "stepper" }, minus, count, plus),
      ),
      h("div", { class: "mode-row" }, modeButton, modeHelp.root),
      h("div", { class: "form-actions" }, reset, start),
    ),
    error,
  );
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const maxAnalysts = Number(count.value);
    if (!topic.value.trim()) return showError("Enter a research topic.");
    if (!Number.isInteger(maxAnalysts) || maxAnalysts < meta.min_analysts || maxAnalysts > meta.max_analysts) {
      return showError(`Perspectives must be a whole number from ${meta.min_analysts} to ${meta.max_analysts}.`);
    }
    showError("");
    onStart({ topic: topic.value.trim(), maxAnalysts, profile: mode });
  });
  function showError(message) {
    error.textContent = message;
  }
  card.body.append(form);

  const controls = [topic, count, minus, plus, start, modeButton];
  const setLocked = (locked) => {
    controls.forEach((control) => {
      control.disabled = locked;
    });
    start.hidden = locked;
    reset.hidden = !locked;
    card.root.classList.toggle("is-locked", locked);
  };
  return {
    root: card.root,
    lock: () => setLocked(true),
    unlock: () => setLocked(false),
    showError,
  };
}

// -- human feedback

export function analystGrid(analysts) {
  return h(
    "div",
    { class: "analyst-grid" },
    analysts.map((analyst) =>
      h(
        "article",
        { class: "analyst-card" },
        h("div", { class: "analyst-card-name" }, analyst.name),
        h("div", { class: "analyst-card-role" }, analyst.role),
        h("div", { class: "analyst-card-aff" }, analyst.affiliation),
        h("p", { class: "analyst-card-desc" }, analyst.description),
      ),
    ),
  );
}

const FEEDBACK_STATES = {
  pending: ["Awaiting your review", "warn"],
  submitting: ["Submitting…", "muted"],
  changes: ["Changes requested", "violet"],
  approved: ["Approved & executing", "cyan"],
  limit: ["Revision limit reached · proceeding", "warn"],
  stopped: ["Run stopped", "muted"],
};

export function feedbackSection({ round, payload, analysts, onRevise, onApprove }) {
  const remaining = payload.revisions_remaining ?? null;
  const state = h("span", { class: "badge" });
  const card = section({
    icon: "userCheck",
    kicker: `Round ${round}`,
    title: "Human feedback",
    tone: "cyan",
    meta: state,
    className: "feedback-card",
  });
  const textarea = h("textarea", {
    class: "input",
    rows: 2,
    placeholder:
      remaining === 0
        ? "Revision limit reached. Approve to start research with this panel."
        : "Describe changes to the panel, e.g. “add a regulatory perspective”…",
    "aria-label": "Operator feedback",
  });
  const revise = h("button", { class: "btn", type: "button", disabled: true }, icon("message"), "Request changes");
  const approve = h("button", { class: "btn btn-primary", type: "button" }, icon("checkCircle"), "Approve & start research");
  const decision = h("div", { class: "decision", hidden: true });
  textarea.addEventListener("input", () => {
    revise.disabled = !textarea.value.trim() || remaining === 0;
  });
  revise.addEventListener("click", () => onRevise(textarea.value.trim()));
  approve.addEventListener("click", () => onApprove());

  card.body.append(
    h(
      "div",
      { class: "review-intro" },
      h("h3", { class: "review-title" }, "Review your research panel"),
      h(
        "p",
        { class: "review-sub" },
        `${analysts.length} analyst${analysts.length === 1 ? "" : "s"} will investigate the topic in parallel. ` +
          "Approve to start, or describe what to change and a new panel will be drafted.",
      ),
    ),
    analystGrid(analysts),
    h(
      "div",
      { class: "operator" },
      h("div", { class: "field-label" }, icon("lock", "icon icon-inline"), "Operator feedback"),
      textarea,
      decision,
      h(
        "div",
        { class: "form-actions" },
        h(
          "span",
          { class: "hint" },
          remaining === null ? "" : `${remaining} revision${remaining === 1 ? "" : "s"} remaining`,
        ),
        revise,
        approve,
      ),
    ),
  );

  function setState(name, detail) {
    const [label, tone] = FEEDBACK_STATES[name] ?? FEEDBACK_STATES.pending;
    state.className = `badge tone-${tone}`;
    state.textContent = label;
    if (name !== "pending") {
      textarea.disabled = true;
      revise.disabled = true;
      approve.disabled = true;
      card.root.classList.add("is-resolved");
    }
    if (detail) {
      decision.hidden = false;
      decision.textContent = detail;
    }
  }
  setState("pending");
  return {
    root: card.root,
    setState,
    fill(text) {
      textarea.value = text;
      textarea.dispatchEvent(new Event("input"));
    },
  };
}

// -- progress tracker

export function progressTracker({ total, profile, allowedTools, onStop, onRestart }) {
  const status = h("span", { class: "badge tone-cyan" }, "Running");
  const stop = h("button", { class: "btn btn-small", type: "button", onclick: () => onStop() }, icon("stop"), "Stop");
  const restart = h("button", { class: "btn btn-small", type: "button", onclick: () => onRestart() }, icon("restart"), "Restart");
  const card = section({
    icon: "activity",
    kicker: "All analysts, then the final report",
    title: "Progress tracker",
    tone: "violet",
    meta: status,
    actions: [stop, restart],
    className: "tracker-card",
  });
  const percent = h("span", { class: "tracker-percent" }, "0%");
  const bar = h("div", { class: "bar-fill" });
  const phase = h("span", { class: "tracker-phase" }, "Starting analysts…");
  const elapsed = h("span", { class: "tracker-elapsed" }, "0.0s");
  const stats = {
    analysts: h("strong", {}, `0/${total}`),
    calls: h("strong", {}, "0"),
    tokens: h("strong", {}, "0"),
    findings: h("strong", {}, "0"),
  };
  const toolCounts = {};
  const totalCalls = h("strong", {}, "0");
  const toolCards = Object.entries(TOOL_INFO).map(([name, info]) => {
    toolCounts[name] = h("span", { class: "tool-count" }, allowedTools.includes(name) ? "0" : "off");
    return h(
      "div",
      { class: `tool-card${allowedTools.includes(name) ? "" : " is-off"}` },
      h("div", { class: "tool-top" }, icon(info.icon), toolCounts[name]),
      h("div", { class: "tool-label" }, info.label),
      h("div", { class: "tool-sub" }, allowedTools.includes(name) ? info.sub : `Off in ${profileLabel(profile)}`),
    );
  });
  card.body.append(
    h(
      "div",
      { class: "tracker-overall" },
      h("div", { class: "tracker-row" }, h("span", { class: "tracker-label" }, "Overall completion"), percent),
      h("div", { class: "bar", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100 }, bar),
      h("div", { class: "tracker-row tracker-sub" }, phase, elapsed),
    ),
    h(
      "div",
      { class: "stat-row" },
      h("div", { class: "stat" }, h("span", {}, "Analysts done"), stats.analysts),
      h("div", { class: "stat" }, h("span", {}, "LLM calls"), stats.calls),
      h("div", { class: "stat" }, h("span", {}, "Input tokens"), stats.tokens),
      h("div", { class: "stat" }, h("span", {}, "Findings"), stats.findings),
    ),
    h("div", { class: "tracker-row tools-head" }, h("span", { class: "kicker" }, "Tool calls"), h("span", { class: "tools-total" }, "Total calls: ", totalCalls)),
    h("div", { class: "tool-grid" }, toolCards),
  );
  return {
    root: card.root,
    update({ fraction, phaseText, elapsedSeconds, done, llmCalls, inputTokens, findings, tools }) {
      const value = Math.round(fraction * 100);
      percent.textContent = `${value}%`;
      bar.style.setProperty("width", `${(fraction * 100).toFixed(1)}%`);
      bar.parentElement.setAttribute("aria-valuenow", value);
      phase.textContent = phaseText;
      elapsed.textContent = formatDuration(elapsedSeconds);
      stats.analysts.textContent = `${done}/${total}`;
      stats.calls.textContent = formatNumber(llmCalls);
      stats.tokens.textContent = formatNumber(inputTokens);
      stats.findings.textContent = formatNumber(findings);
      let sum = 0;
      for (const [name, element] of Object.entries(toolCounts)) {
        sum += tools[name] ?? 0;
        if (allowedTools.includes(name)) element.textContent = formatNumber(tools[name] ?? 0);
      }
      totalCalls.textContent = formatNumber(sum);
    },
    setStatus(text, tone, finished = false) {
      status.className = `badge tone-${tone}`;
      status.textContent = text;
      stop.disabled = finished;
    },
  };
}

// -- analyst chat sections

export function analystHeader(analyst) {
  const chips = h("div", { class: "chips" }, badge("Queued", "muted"));
  const root = h(
    "section",
    { class: "card analyst-hero" },
    h("div", { class: "badge tone-cyan hero-badge" }, "Analyst"),
    h("h1", { class: "hero-name" }, analyst.name),
    h("div", { class: "hero-role" }, `${analyst.role} · ${analyst.affiliation}`),
    h("p", { class: "hero-desc" }, analyst.description),
    chips,
  );
  return {
    root,
    setLimits(limits, profile) {
      chips.replaceChildren(
        badge(profileLabel(profile), "violet"),
        badge(`≤${limits.max_passes} passes`, "muted"),
        badge(`≤${limits.max_tool_calls_per_pass} tool calls / pass`, "muted"),
        badge(`${formatDuration(limits.deadline_seconds)} deadline`, "muted"),
        badge(limits.allowed_tools.map((tool) => TOOL_INFO[tool]?.label ?? tool).join(" · "), "muted"),
      );
    },
  };
}

export function planSection(subQuestions, researchPass, maxPasses) {
  const card = section({
    icon: "branch",
    kicker: "Questions for this pass",
    title: "Research plan",
    meta: badge(`Pass ${researchPass}${maxPasses ? `/${maxPasses}` : ""} · ${subQuestions.length} questions`, "muted"),
  });
  card.body.append(
    h(
      "div",
      { class: "sq-grid" },
      subQuestions.map((question, index) =>
        h("article", { class: "sq-card" }, h("div", { class: "sq-id" }, `SQ-${pad(index + 1)}`), h("p", {}, question)),
      ),
    ),
  );
  return card.root;
}

const SOURCE_ICONS = { wikipedia: "book", arxiv: "flask", pubmed: "cross", scraped_page: "globe" };

export function findingsSection(findings, firstNumber, researchPass) {
  const card = section({
    icon: "checkCircle",
    kicker: "Evidence gathered in this pass",
    title: "Key findings",
    tone: "cyan",
    meta: badge(`Pass ${researchPass} · ${findings.length} new`, "muted"),
  });
  if (!findings.length) {
    card.body.append(h("p", { class: "empty" }, "No new evidence was extracted in this pass."));
    return card.root;
  }
  card.body.append(
    h(
      "div",
      { class: "finding-list" },
      findings.map((finding, index) => {
        const href = safeUrl(finding.source_url);
        const source = h(
          "div",
          { class: "finding-source" },
          icon(SOURCE_ICONS[finding.source_type] ?? "search"),
          h("div", { class: "finding-source-text" }, h("div", { class: "finding-source-title" }, finding.source_title || "Source")),
          href
            ? h("a", { class: "finding-link", href, target: "_blank", rel: "noopener noreferrer" }, hostOf(href), icon("external", "icon icon-inline"))
            : h("span", { class: "finding-link" }, finding.source_url),
        );
        return h(
          "article",
          { class: "finding" },
          h(
            "div",
            { class: "finding-tags" },
            h("span", { class: "tag tag-accent" }, `E${pad(firstNumber + index)}`),
            h("span", { class: "tag" }, (finding.source_type || "source").replace(/_/g, " ")),
          ),
          h("div", { class: "finding-question" }, `Answers: ${finding.sub_question}`),
          h("div", { class: "label" }, "Claim"),
          h("p", { class: "finding-claim" }, finding.claim),
          finding.excerpt ? h("blockquote", { class: "finding-excerpt" }, finding.excerpt) : null,
          source,
        );
      }),
    ),
  );
  return card.root;
}

export function reviewSection(evaluation, researchPass, maxPasses) {
  const complete = Boolean(evaluation?.is_complete);
  const finalPass = maxPasses && researchPass >= maxPasses;
  const verdict = complete ? ["Sufficient · write next", "ok"] : finalPass ? ["Gaps remain · pass limit", "warn"] : [`Gaps found · pass ${researchPass + 1} next`, "warn"];
  const card = section({
    icon: "listCheck",
    kicker: "Is the evidence enough?",
    title: "Research review",
    meta: badge(`Pass ${researchPass}`, "muted"),
  });
  const gaps = evaluation?.coverage_gaps ?? [];
  card.body.append(
    h(
      "div",
      { class: "review-grid" },
      h(
        "div",
        { class: "review-col" },
        h("div", { class: "label label-warn" }, icon("flag", "icon icon-inline"), "Identified coverage gaps"),
        gaps.length
          ? h("ul", { class: "gap-list" }, gaps.map((gap) => h("li", {}, gap)))
          : h("p", { class: "empty" }, "No material coverage gaps identified."),
      ),
      h(
        "div",
        { class: "review-col review-feedback" },
        h("div", { class: "label label-cyan" }, icon("message", "icon icon-inline"), "Evaluator feedback"),
        h("div", { class: "verdict" }, badge(verdict[0], verdict[1])),
        h("p", {}, evaluation?.feedback || "—"),
      ),
    ),
  );
  return card.root;
}

export function perspectiveSection(analyst, draft, profile) {
  const meta = h("div", { class: "doc-meta" });
  const card = section({
    icon: "terminal",
    kicker: "Written from this analyst's findings",
    title: "Analyst perspective",
    tone: "cyan",
    actions: [exportButton(`${slugify(analyst.name)}-perspective.md`, () => draft)],
  });
  const content = h("div", { class: "md" });
  content.innerHTML = renderMarkdown(draft); // escaped by the renderer
  card.body.append(meta, content);
  const setMeta = ({ stopReason, findings, duration } = {}) => {
    meta.replaceChildren(
      h("span", {}, `Author: ${analyst.name}`),
      h("span", {}, `Mode: ${profileLabel(profile)}`),
      h("span", {}, `Stop: ${stopReasonLabel(stopReason)}`),
      findings !== undefined ? h("span", {}, `Findings: ${findings}`) : null,
      duration !== undefined ? h("span", {}, `Duration: ${formatDuration(duration)}`) : null,
    );
  };
  setMeta();
  return { root: card.root, setMeta };
}

export function finalReportSection(report) {
  const card = section({
    icon: "fileText",
    kicker: "Combined from every analyst",
    title: "Final report",
    tone: "cyan",
    actions: [exportButton("research-report.md", () => report)],
    className: "report-card",
  });
  const content = h("div", { class: "md md-report" });
  content.innerHTML = renderMarkdown(report); // escaped by the renderer
  card.body.append(content);
  return card.root;
}

export function runStatsSection(runStats) {
  const rows = runStats?.per_analyst ?? [];
  const card = section({
    icon: "chart",
    kicker: `${profileLabel(runStats?.model_profile)} run`,
    title: "Run statistics",
    meta: badge(`${runStats?.completed ?? 0}/${runStats?.analysts ?? rows.length} analysts completed`, rows.length && runStats?.failed?.length ? "warn" : "ok"),
    className: "stats-card",
  });
  const tile = (label, value) => h("div", { class: "stat" }, h("span", {}, label), h("strong", {}, value));
  card.body.append(
    h(
      "div",
      { class: "stat-row stat-row-wide" },
      tile("Mean analyst time", formatDuration(runStats?.mean_analyst_seconds)),
      tile("Slowest analyst", formatDuration(runStats?.max_analyst_seconds)),
      tile("LLM calls", formatNumber(runStats?.llm_calls)),
      tile("Input tokens", formatNumber(runStats?.input_tokens)),
      tile("Findings", formatNumber(rows.reduce((sum, row) => sum + (row.findings ?? 0), 0))),
    ),
  );
  if (rows.length) {
    card.body.append(
      h(
        "div",
        { class: "md-table" },
        h(
          "table",
          { class: "stats-table" },
          h("thead", {}, h("tr", {}, ["Analyst", "Status", "Stop reason", "Duration", "LLM calls", "Input tokens", "Findings"].map((label) => h("th", {}, label)))),
          h(
            "tbody",
            {},
            rows.map((row) =>
              h(
                "tr",
                {},
                h("td", {}, h("div", { class: "cell-name" }, row.analyst), h("div", { class: "cell-sub" }, row.role ?? "")),
                h("td", {}, h("span", { class: `badge tone-${row.status === "completed" ? "ok" : "err"}` }, row.status)),
                h("td", {}, stopReasonLabel(row.stop_reason)),
                h("td", {}, formatDuration(row.duration_seconds)),
                h("td", {}, formatNumber(row.llm_calls)),
                h("td", {}, formatNumber(row.input_tokens)),
                h("td", {}, formatNumber(row.findings)),
              ),
            ),
          ),
        ),
      ),
    );
  }
  return card.root;
}

export function errorSection(message) {
  const card = section({ icon: "alert", kicker: "The run stopped early", title: "Something went wrong", tone: "err", className: "error-card" });
  card.body.append(h("p", {}, message));
  return card.root;
}

// ---------------------------------------------------------------- sidebar

export function orchestratorNav(onSelect) {
  const dot = h("span", { class: "dot" });
  const text = h("div", { class: "nav-sub" }, "Configure a research run");
  const bar = h("div", { class: "bar-fill" });
  const root = h(
    "button",
    { class: "nav-card nav-orchestrator is-active", type: "button", onclick: () => onSelect() },
    h("div", { class: "nav-top" }, h("span", { class: "nav-title" }, "Research Orchestrator"), dot),
    text,
    h("div", { class: "bar bar-thin" }, bar),
  );
  return {
    root,
    update({ statusText, tone, fraction, attention }) {
      text.textContent = statusText;
      dot.className = `dot tone-${tone}`;
      bar.style.setProperty("width", `${Math.round((fraction ?? 0) * 100)}%`);
      root.classList.toggle("has-attention", Boolean(attention));
    },
  };
}

export function analystNav(analyst, onSelect) {
  const pct = h("span", { class: "nav-pct" }, "0%");
  const statusDot = h("span", { class: "dot tone-muted" });
  const statusText = h("span", { class: "nav-status-text" }, "Queued");
  const root = h(
    "button",
    { class: "nav-card nav-analyst", type: "button", onclick: () => onSelect() },
    h("div", { class: "nav-top" }, h("span", { class: "nav-title" }, analyst.name), pct),
    h("div", { class: "nav-sub" }, analyst.role),
    h("div", { class: "nav-status" }, statusDot, statusText),
  );
  return {
    root,
    update({ fraction, text, tone }) {
      pct.textContent = `${Math.round(fraction * 100)}%`;
      statusText.textContent = text;
      statusDot.className = `dot tone-${tone}`;
    },
  };
}
