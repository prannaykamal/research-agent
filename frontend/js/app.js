// Research Orchestrator UI controller: wires stream actions to views and progress.

import { LangGraphClient } from "./api.js";
import { DemoClient } from "./demo.js";
import { TOOL_INFO, formatDuration, profileLabel, stopReasonLabel, truncate } from "./dom.js";
import { createInterpreter } from "./protocol.js";
import * as progress from "./progress.js";
import * as views from "./views.js";

const params = new URLSearchParams(location.search);
const DEMO = params.has("demo");
const SPEED = Math.max(0.1, Number(params.get("speed")) || 1);
const INITIAL_VIEW = params.get("view");
const API_BASE = (params.get("api") || (location.protocol.startsWith("http") ? location.origin : "http://127.0.0.1:2024")).replace(/\/+$/, "");

// Used only when /app/profiles.json is unavailable (e.g. the page is opened from disk).
const FALLBACK_META = {
  default: "quality",
  min_analysts: 1,
  max_analysts: 10,
  profiles: {
    quality: {
      models: { heavy: { model: "gemini-3.1-pro-preview" }, medium: { model: "gemini-3.8-flash" } },
      max_passes: 3,
      max_tool_calls_per_pass: 6,
      max_researcher_turns: 3,
      deadline_seconds: 400,
      allowed_tools: Object.keys(TOOL_INFO),
    },
    fast: {
      models: { heavy: { model: "gemini-3.8-flash" }, medium: { model: "gemini-3.5-flash-lite" } },
      max_passes: 3,
      max_tool_calls_per_pass: 6,
      max_researcher_turns: 3,
      deadline_seconds: 300,
      allowed_tools: ["tavily_search", "wikipedia", "arxiv", "pubmed"],
    },
  },
};

const STAGE_LABELS = {
  planner_node: "Planning",
  researcher_node: "Researching",
  tools: "Running tools",
  extract_findings: "Extracting evidence",
  evaluate_research: "Evaluating",
  writer_node: "Writing",
};

const PHASES = {
  config: ["Configure a research run", "muted"],
  generating: ["Generating analyst panel…", "cyan"],
  awaiting_feedback: ["Awaiting your review", "warn"],
  starting: ["Starting research…", "cyan"],
  researching: [() => `Supervising ${app.analysts.length} parallel analysts`, "cyan"],
  synthesizing: ["Synthesizing final report…", "violet"],
  complete: ["Report ready", "ok"],
  error: ["Run failed", "err"],
  cancelled: ["Run stopped", "muted"],
};
const ACTIVE_PHASES = new Set(["generating", "starting", "researching", "synthesizing"]);

const els = {
  main: document.getElementById("main"),
  orchestratorSlot: document.getElementById("orchestrator-slot"),
  analystList: document.getElementById("analyst-list"),
  analystCount: document.getElementById("analyst-count"),
  analystEmpty: document.getElementById("analyst-empty"),
  connection: document.getElementById("connection"),
  thread: document.getElementById("breadcrumb-thread"),
  banner: document.getElementById("demo-banner"),
};

const app = {
  phase: "config",
  values: null,
  threadId: null,
  runId: null,
  abort: null,
  interpreter: null,
  rounds: [],
  latestAnalysts: [],
  analysts: [],
  synthesis: {},
  final: false,
  tools: {},
  pending: {},
  tracker: null,
  ticker: null,
  renderQueued: false,
  overall: 0,
  researchStartedAt: null,
  finishedAt: null,
  active: "orchestrator",
  attention: false,
};

let client;
let meta = FALLBACK_META;
let orchestrator;
let config;
const orchestratorNav = views.orchestratorNav(() => select("orchestrator"));

// ---------------------------------------------------------------- setup

async function init() {
  els.orchestratorSlot.replaceWith(orchestratorNav.root);
  let recording = null;
  if (DEMO) {
    recording = await (await fetch("demo/sample-run.json")).json();
    client = new DemoClient(recording, SPEED);
    els.banner.hidden = false;
  } else {
    client = new LangGraphClient(API_BASE, params.get("assistant") || "deep_agent");
  }
  meta = await loadMeta();
  buildOrchestrator(
    recording
      ? { topic: recording.input.topic, maxAnalysts: recording.input.max_analysts, profile: recording.input.model_profile }
      : { topic: "", maxAnalysts: 3, profile: meta.profiles.fast ? "fast" : meta.default },
  );
  checkConnection();
  render();
  if (recording) {
    setTimeout(
      () => startRun({ topic: recording.input.topic, maxAnalysts: recording.input.max_analysts, profile: recording.input.model_profile }),
      600 / SPEED,
    );
  }
}

async function loadMeta() {
  try {
    const response = await fetch(`${API_BASE}/app/profiles.json`);
    if (response.ok) return await response.json();
  } catch {
    // fall through to the built-in copy
  }
  return FALLBACK_META;
}

async function checkConnection() {
  const host = DEMO ? "demo replay" : new URL(API_BASE).host;
  try {
    await client.health();
    els.connection.className = "connection tone-ok";
    els.connection.textContent = DEMO ? "Demo replay · no server calls" : `Connected · ${host}`;
  } catch {
    els.connection.className = "connection tone-err";
    els.connection.textContent = `Server unreachable · ${host}`;
  }
}

function buildOrchestrator(values) {
  orchestrator?.root.remove();
  orchestrator = new views.Chat("chat-orchestrator");
  els.main.append(orchestrator.root);
  config = views.configSection({ meta, values, onStart: startRun, onReset: resetToConfig });
  orchestrator.appendSection(config.root);
  select("orchestrator");
}

function clearRun() {
  app.abort?.abort();
  stopTicker();
  for (const analyst of app.analysts) {
    analyst.chat.root.remove();
    analyst.nav.root.remove();
  }
  Object.assign(app, {
    threadId: null,
    runId: null,
    abort: null,
    interpreter: null,
    rounds: [],
    latestAnalysts: [],
    analysts: [],
    synthesis: {},
    final: false,
    tools: {},
    pending: {},
    tracker: null,
    overall: 0,
    researchStartedAt: null,
    finishedAt: null,
    attention: false,
  });
  setTeamLabel(0);
  els.thread.textContent = "";
}

// ---------------------------------------------------------------- run lifecycle

async function startRun(values) {
  app.values = values;
  clearRun();
  config.lock();
  orchestrator.origin = performance.now();
  setPhase("generating");
  app.pending.generate = orchestrator.step({
    status: "running",
    text: `Generating ${values.maxAnalysts} analyst persona${values.maxAnalysts === 1 ? "" : "s"} · ${profileLabel(values.profile)}`,
  });
  try {
    app.threadId = await client.createThread();
  } catch (error) {
    return fail(`Could not reach the LangGraph server at ${API_BASE} (${error.message}). Start it with \`uv run langgraph dev\`.`);
  }
  els.thread.textContent = ` / ${app.threadId.slice(0, 8)}`;
  await stream({ input: { topic: values.topic, max_analysts: values.maxAnalysts, model_profile: values.profile } });
}

async function stream(payload) {
  const controller = new AbortController();
  app.abort = controller;
  app.interpreter ??= createInterpreter();
  try {
    for await (const event of client.streamRun(app.threadId, payload, { signal: controller.signal })) {
      if (controller.signal.aborted) return;
      for (const action of app.interpreter(event)) dispatch(action);
    }
  } catch (error) {
    if (controller.signal.aborted || error.name === "AbortError") return;
    fail(`The stream failed: ${error.message}`);
    return;
  } finally {
    if (app.abort === controller) app.abort = null;
  }
  if (!controller.signal.aborted) streamClosed();
}

function streamClosed() {
  if (["awaiting_feedback", "complete", "error", "cancelled"].includes(app.phase)) return;
  fail("The run ended without a final report. Check the LangGraph server logs for details.");
}

function dispatch(action) {
  const handler = HANDLERS[action.kind];
  if (handler) handler(action);
  scheduleRender();
}

const HANDLERS = {
  run_started: ({ runId }) => {
    app.runId = runId;
  },
  run_error: ({ message }) => fail(message),
  analysts_generated: ({ analysts }) => {
    app.latestAnalysts = analysts;
  },
  feedback_requested: onFeedbackRequested,
  feedback_resolved: onFeedbackResolved,
  analyst_started: onAnalystStarted,
  analyst_step: onAnalystStep,
  plan: onPlan,
  researcher_turn: onResearcherTurn,
  tools_dispatched: onToolsDispatched,
  tool_results: onToolResults,
  findings: onFindings,
  review: onReview,
  review_skipped: onReviewSkipped,
  draft: onDraft,
  analyst_completed: onAnalystCompleted,
  synthesis_step: onSynthesisStep,
  final_report: onFinalReport,
};

// ---------------------------------------------------------------- human feedback

function onFeedbackRequested({ payload }) {
  const analysts = payload.analysts?.length ? payload.analysts : app.latestAnalysts;
  app.pending.generate?.update({ status: "done", text: `Generated ${analysts.length} analyst persona${analysts.length === 1 ? "" : "s"}` });
  app.pending.generate = null;
  const round = { number: app.rounds.length + 1, analysts, decision: null };
  round.waitItem = orchestrator.step({ status: "waiting", text: "Human feedback requested · review the analyst panel" });
  round.view = views.feedbackSection({
    round: round.number,
    payload,
    analysts,
    onRevise: (text) => revise(round, text),
    onApprove: () => approve(round),
  });
  orchestrator.appendSection(round.view.root);
  app.rounds.push(round);
  setPhase("awaiting_feedback");
  if (DEMO) scheduleDemoDecision(round);
}

function revise(round, text) {
  if (round.decision || app.phase !== "awaiting_feedback" || !text) return;
  round.decision = { type: "revise", text };
  round.view.setState("changes", `Requested: “${text}”`);
  round.waitItem.update({ status: "done" });
  orchestrator.step({ status: "info", text: `Changes requested · “${truncate(text, 140)}”` });
  setPhase("generating");
  app.pending.generate = orchestrator.step({ status: "running", text: "Regenerating the analyst panel with your feedback" });
  stream({ command: { resume: text } });
}

function approve(round) {
  if (round.decision || app.phase !== "awaiting_feedback") return;
  round.decision = { type: "approve" };
  round.view.setState("submitting");
  round.waitItem.update({ status: "done" });
  orchestrator.step({ status: "done", text: "Panel approved · starting research" });
  setPhase("starting");
  stream({ command: { resume: "approved" } });
}

function onFeedbackResolved({ feedback }) {
  if (feedback) return; // the panel is regenerated and the next interrupt appends a new round
  const round = app.rounds[app.rounds.length - 1];
  if (round?.decision?.type === "revise") {
    round.view.setState("limit");
    app.pending.generate?.update({ status: "skipped", text: "Revision limit reached · proceeding with the current panel" });
    app.pending.generate = null;
  } else {
    round?.view.setState("approved");
  }
  spawnAnalysts(round?.analysts ?? app.latestAnalysts);
}

function scheduleDemoDecision(round) {
  const decision = client.decision(round.number);
  setTimeout(() => {
    if (round.decision) return;
    if (decision === "approved") return approve(round);
    round.view.fill(decision);
    setTimeout(() => revise(round, decision), 900 / SPEED);
  }, 1800 / SPEED);
}

// ---------------------------------------------------------------- analysts

function spawnAnalysts(list) {
  const profileName = app.values.profile;
  app.analysts = list.map((info, index) => {
    const chat = new views.Chat(`chat-analyst-${index}`);
    els.main.append(chat.root);
    const header = views.analystHeader(info);
    chat.appendSection(header.root);
    const analyst = {
      index,
      info,
      chat,
      header,
      nav: views.analystNav(info, () => select(index)),
      progress: progress.newAnalystProgress(),
      status: "queued",
      stage: null,
      limits: null,
      profile: profileName,
      pending: {},
      requested: [],
      toolItems: new Map(),
      findingsCount: 0,
      usage: { llmCalls: 0, inputTokens: 0 },
      stats: null,
      perspective: null,
    };
    analyst.pending.start = chat.step({ status: "waiting", text: "Waiting for a free analyst slot…" });
    els.analystList.append(analyst.nav.root);
    return analyst;
  });
  setTeamLabel(list.length);
  app.tracker = views.progressTracker({
    total: list.length,
    profile: profileName,
    allowedTools: meta.profiles[profileName]?.allowed_tools ?? Object.keys(TOOL_INFO),
    onStop: stop,
    onRestart: restart,
  });
  orchestrator.appendSection(app.tracker.root);
  orchestrator.step({ status: "done", text: `Spawned ${list.length} analyst${list.length === 1 ? "" : "s"} · all running in parallel` });
  app.researchStartedAt = performance.now();
  setPhase("researching");
  const requested = /^analyst-(\d+)$/.exec(INITIAL_VIEW ?? "");
  if (requested && app.analysts[Number(requested[1])]) select(Number(requested[1]));
}

function setTeamLabel(count) {
  els.analystCount.textContent = count ? `${count} in parallel` : "";
  els.analystCount.hidden = !count;
  els.analystEmpty.hidden = Boolean(count);
}

function analystAt(index) {
  return app.analysts[index] ?? null;
}

function applyUsage(analyst, usage) {
  if (usage) analyst.usage = usage;
}

const plural = (count, word, many = `${word}s`) => `${count} ${count === 1 ? word : many}`;

function onAnalystStarted({ index, profile, limits }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  analyst.status = "running";
  analyst.limits = limits;
  analyst.profile = profile;
  analyst.chat.origin = performance.now();
  progress.markStarted(analyst.progress, limits, performance.now());
  analyst.header.setLimits(limits, profile);
  analyst.pending.start?.update({
    status: "done",
    text: `Started · up to ${plural(limits.max_passes, "pass", "passes")} · ${formatDuration(limits.deadline_seconds)} deadline`,
  });
}

function onAnalystStep({ index, node, researchPass, turn, toolOutputs, stopReason }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  analyst.stage = node;
  progress.markStep(analyst.progress, node, researchPass);
  if (node === "writer_node") analyst.status = "writing";
  const text =
    {
      planner_node:
        researchPass > 1
          ? `Planning pass ${researchPass} · targeting the evaluator's coverage gaps`
          : "Planning pass 1 · decomposing the analyst's objectives into sub-questions",
      researcher_node: `Researcher reasoning · pass ${researchPass}, turn ${turn}`,
      extract_findings: `Extracting evidence from ${plural(toolOutputs ?? 0, "tool output")}`,
      evaluate_research: `Evaluating coverage of ${plural(analyst.findingsCount, "finding")}`,
      writer_node: `Writing the perspective · ${stopReasonLabel(stopReason)}`,
    }[node] ?? node;
  analyst.pending[node] = analyst.chat.step({ status: "running", text });
}

function finish(analyst, key, update) {
  const item = analyst.pending[key];
  analyst.pending[key] = null;
  if (item) item.update(update);
  return Boolean(item);
}

function onPlan({ index, subQuestions, usage }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  progress.markPlan(analyst.progress);
  applyUsage(analyst, usage);
  const researchPass = analyst.progress.pass;
  finish(analyst, "planner_node", { status: "done", text: `Plan ready · ${plural(subQuestions.length, "sub-question")} for pass ${researchPass}` });
  analyst.chat.appendSection(views.planSection(subQuestions, researchPass, analyst.limits?.max_passes));
}

function onResearcherTurn({ index, turn, reasoning, toolCalls, usage }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  progress.markResearcherTurn(analyst.progress);
  applyUsage(analyst, usage);
  analyst.requested = toolCalls;
  const tools = [...new Set(toolCalls.map((call) => call.name))].join(", ");
  const summary = toolCalls.length
    ? `requested ${plural(toolCalls.length, "tool call")} · ${tools}`
    : "no further tool calls · evidence is sufficient for this pass";
  const update = { status: "done", text: `Researcher turn ${turn ?? analyst.progress.researchTurns}: ${summary}`, note: reasoning };
  if (!finish(analyst, "researcher_node", update)) analyst.chat.step(update);
}

function onToolsDispatched({ index, calls }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  analyst.stage = "tools";
  const allowed = new Set(calls.map((call) => call.id));
  const dropped = analyst.requested.filter((call) => !allowed.has(call.id));
  if (dropped.length) {
    analyst.chat.step({
      status: "skipped",
      text: `Guardrail dropped ${plural(dropped.length, "call")} (${[...new Set(dropped.map((call) => call.name))].join(", ")}) · tool not allowed or per-pass limit reached`,
    });
  }
  for (const call of calls) analyst.toolItems.set(call.id, analyst.chat.step({ status: "running", tool: call }));
}

const TOOL_FAILURE = /^(Webpage scrape|Wikipedia search|arXiv search|PubMed search) failed|^Error[: ]/;

function onToolResults({ index, results }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  for (const result of results) {
    const item = analyst.toolItems.get(result.id) ?? analyst.chat.step({ tool: { name: result.name, args: {} } });
    analyst.toolItems.delete(result.id);
    item.setOutput(result.content, result.status !== "error" && !TOOL_FAILURE.test(result.content));
    app.tools[result.name] = (app.tools[result.name] ?? 0) + 1;
  }
  progress.markToolResults(analyst.progress, results.length);
}

function onFindings({ index, findings, usage }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  progress.markFindings(analyst.progress);
  applyUsage(analyst, usage);
  const researchPass = analyst.progress.pass;
  const done = finish(analyst, "extract_findings", { status: "done", text: `Extracted ${plural(findings.length, "new finding")} · pass ${researchPass}` });
  if (!done) analyst.chat.step({ status: "skipped", text: `No tool output in pass ${researchPass} · nothing to extract` });
  analyst.chat.appendSection(views.findingsSection(findings, analyst.findingsCount + 1, researchPass));
  analyst.findingsCount += findings.length;
}

function onReview({ index, evaluation, usage }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  progress.markEvaluated(analyst.progress);
  applyUsage(analyst, usage);
  const gaps = evaluation?.coverage_gaps?.length ?? 0;
  finish(analyst, "evaluate_research", {
    status: "done",
    text: evaluation?.is_complete ? "Verdict · evidence is sufficient" : `Verdict · ${plural(gaps, "coverage gap")} identified`,
  });
  analyst.chat.appendSection(views.reviewSection(evaluation, analyst.progress.pass, analyst.limits?.max_passes));
}

function onReviewSkipped({ index }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  progress.markEvaluated(analyst.progress);
  analyst.chat.step({ status: "skipped", text: "Evaluation skipped · final pass or a budget was reached; writing next" });
}

function onDraft({ index, draft, stopReason, usage }) {
  const analyst = analystAt(index);
  if (!analyst) return;
  progress.markDone(analyst.progress);
  applyUsage(analyst, usage);
  analyst.status = "done";
  finish(analyst, "writer_node", { status: "done", text: "Perspective written" });
  analyst.perspective = views.perspectiveSection(analyst.info, draft, analyst.profile);
  analyst.perspective.setMeta({ stopReason, findings: analyst.findingsCount });
  analyst.chat.appendSection(analyst.perspective.root);
}

function onAnalystCompleted({ stats }) {
  const analyst = analystAt(stats.analyst_index) ?? app.analysts.find((item) => item.info.name === stats.analyst);
  if (!analyst) return;
  analyst.stats = stats;
  analyst.usage = { llmCalls: stats.llm_calls, inputTokens: stats.input_tokens };
  const ok = stats.status === "completed";
  if (ok) {
    analyst.status = "done";
    progress.markDone(analyst.progress);
    analyst.perspective?.setMeta({ stopReason: stats.stop_reason, findings: stats.findings, duration: stats.duration_seconds });
  } else {
    analyst.status = "failed";
    progress.markDone(analyst.progress, { failed: true });
    for (const key of Object.keys(analyst.pending)) finish(analyst, key, { status: "error" });
    analyst.chat.step({ status: "error", text: `Research failed · ${stats.stop_reason}` });
  }
  orchestrator.step({
    status: ok ? "done" : "error",
    text: `${analyst.info.name} ${ok ? "completed" : "failed"} in ${formatDuration(stats.duration_seconds)} · ${stopReasonLabel(stats.stop_reason)}`,
  });
  if (app.analysts.every((item) => item.stats)) {
    setPhase("synthesizing");
    app.pending.synthesis = orchestrator.step({ status: "running", text: "Synthesizing the final report · body, introduction and conclusion in parallel" });
  }
}

const SYNTHESIS_LABELS = {
  write_report: "Report body written",
  write_introduction: "Introduction written",
  write_conclusion: "Conclusion written",
};

function onSynthesisStep({ step }) {
  app.synthesis[step] = true;
  if (SYNTHESIS_LABELS[step]) orchestrator.step({ status: "done", text: SYNTHESIS_LABELS[step] });
}

function onFinalReport({ report, runStats }) {
  app.final = true;
  app.synthesis.finalize_report = true;
  app.finishedAt = performance.now();
  app.pending.synthesis?.update({ status: "done" });
  app.pending.synthesis = null;
  orchestrator.step({ status: "done", text: "Final report assembled" });
  orchestrator.appendSection(views.finalReportSection(report));
  if (runStats) orchestrator.appendSection(views.runStatsSection(runStats));
  app.tracker?.setStatus("Complete", "ok", true);
  if (app.active !== "orchestrator") app.attention = true;
  setPhase("complete");
}

// ---------------------------------------------------------------- controls

async function stop() {
  if (!ACTIVE_PHASES.has(app.phase)) return;
  const { threadId, runId } = app;
  app.abort?.abort();
  app.finishedAt = performance.now();
  setPhase("cancelled");
  app.tracker?.setStatus("Stopped", "muted", true);
  orchestrator.step({ status: "error", text: "Run stopped by the operator" });
  for (const analyst of app.analysts) {
    if (analyst.progress.done) continue;
    for (const key of Object.keys(analyst.pending)) finish(analyst, key, { status: "skipped" });
    analyst.status = "stopped";
    analyst.chat.step({ status: "skipped", text: "Run stopped by the operator" });
  }
  if (threadId && runId) {
    try {
      await client.cancelRun(threadId, runId);
    } catch (error) {
      orchestrator.step({ status: "error", text: `Cancel request failed: ${error.message}` });
    }
  }
}

async function restart() {
  const values = app.values;
  await stop();
  buildOrchestrator(values);
  startRun(values);
}

async function resetToConfig() {
  const values = app.values;
  await stop();
  clearRun();
  setPhase("config");
  buildOrchestrator(values);
}

function fail(message) {
  if (app.phase === "error") return;
  app.finishedAt = performance.now();
  for (const item of Object.values(app.pending)) item?.update({ status: "error" });
  app.pending = {};
  app.tracker?.setStatus("Error", "err", true);
  orchestrator.appendSection(views.errorSection(message));
  setPhase("error");
}

// ---------------------------------------------------------------- views + rendering

function select(view) {
  const target = view === "orchestrator" ? orchestrator : app.analysts[view]?.chat;
  if (!target) return;
  const current = app.active === "orchestrator" ? orchestrator : app.analysts[app.active]?.chat;
  if (current && current !== target) current.hide();
  target.show();
  app.active = view;
  if (view === "orchestrator") app.attention = false;
  orchestratorNav.root.classList.toggle("is-active", view === "orchestrator");
  app.analysts.forEach((analyst) => analyst.nav.root.classList.toggle("is-active", analyst.index === view));
  scheduleRender();
}

function setPhase(phase) {
  app.phase = phase;
  if (phase === "researching" || phase === "synthesizing") startTicker();
  else if (!ACTIVE_PHASES.has(phase)) stopTicker();
  scheduleRender();
}

// Live progress only needs a once-a-second refresh for the deadline floor and clocks.
function startTicker() {
  app.ticker ??= setInterval(scheduleRender, 1000);
}

function stopTicker() {
  clearInterval(app.ticker);
  app.ticker = null;
}

function scheduleRender() {
  if (app.renderQueued) return;
  app.renderQueued = true;
  requestAnimationFrame(render);
}

function analystStatus(analyst) {
  const passes = analyst.limits?.max_passes ?? "?";
  switch (analyst.status) {
    case "failed":
      return { text: "Failed", tone: "err" };
    case "stopped":
      return { text: "Stopped", tone: "muted" };
    case "done":
      return { text: `Completed${analyst.stats ? ` · ${formatDuration(analyst.stats.duration_seconds)}` : ""}`, tone: "ok" };
    case "writing":
      return { text: "Writing perspective", tone: "violet" };
    case "running":
      return { text: `Pass ${analyst.progress.pass}/${passes} · ${STAGE_LABELS[analyst.stage] ?? "Starting"}`, tone: "cyan" };
    default:
      return { text: "Queued · waiting for a slot", tone: "muted" };
  }
}

function render() {
  app.renderQueued = false;
  const now = performance.now();
  for (const analyst of app.analysts) {
    analyst.nav.update({ fraction: progress.advance(analyst.progress, now), ...analystStatus(analyst) });
  }
  if (app.analysts.length) {
    const fraction = progress.overallFraction(
      { analysts: app.analysts.map((analyst) => analyst.progress), synthesis: app.synthesis, final: app.final },
      now,
    );
    app.overall = Math.max(app.overall, fraction);
  }
  const [label, tone] = PHASES[app.phase];
  orchestratorNav.update({
    statusText: typeof label === "function" ? label() : label,
    tone,
    fraction: app.overall,
    attention: app.attention,
  });
  if (app.tracker) {
    const done = app.analysts.filter((analyst) => analyst.progress.done).length;
    const elapsed = ((app.finishedAt ?? now) - (app.researchStartedAt ?? now)) / 1000;
    const phaseText =
      {
        researching: `Researching · ${done}/${app.analysts.length} analysts complete`,
        synthesizing: "Synthesizing report · body, introduction and conclusion",
        complete: `Complete in ${formatDuration(elapsed)}`,
        cancelled: "Stopped by the operator",
        error: "Run failed",
      }[app.phase] ?? "Starting analysts…";
    app.tracker.update({
      fraction: app.overall,
      phaseText,
      elapsedSeconds: elapsed,
      done,
      llmCalls: app.analysts.reduce((sum, analyst) => sum + (analyst.usage.llmCalls ?? 0), 0),
      inputTokens: app.analysts.reduce((sum, analyst) => sum + (analyst.usage.inputTokens ?? 0), 0),
      findings: app.analysts.reduce((sum, analyst) => sum + analyst.findingsCount, 0),
      tools: app.tools,
    });
  }
}

init().catch((error) => {
  console.error(error);
  els.connection.className = "connection tone-err";
  els.connection.textContent = `UI failed to start: ${error.message}`;
});
