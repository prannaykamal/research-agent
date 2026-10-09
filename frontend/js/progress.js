// Progress model. Pure: no DOM, testable in Node.
//
// Analyst progress (0..1)
//   90% research, split evenly across the profile's maximum passes. Within a pass:
//   plan 15%, research (researcher turns + tools) 50%, extraction 20%, evaluation 15%.
//   The research share advances by the larger of turns used / turn cap and
//   tool calls used / tool-call cap, capped at 95% until extraction starts.
//   When the Writer starts, unused passes are skipped and progress jumps to 90%;
//   the finished draft (or a failure) is 100%.
//   Time floor: research is also at least elapsed / deadline, because the
//   deadline caps research time. Shown progress never moves backwards.
//
// Overall progress (0..1)
//   85% research (mean of analyst progress) + 15% synthesis
//   (report 60%, introduction 15%, conclusion 15%, final assembly 10%).
//   The final report is 100%.

export const STAGE_WEIGHTS = Object.freeze({ plan: 0.15, research: 0.5, extract: 0.2, evaluate: 0.15 });
export const RESEARCH_SHARE = 0.9;
export const PHASE_WEIGHTS = Object.freeze({ research: 0.85, synthesis: 0.15 });
export const SYNTHESIS_WEIGHTS = Object.freeze({
  write_report: 0.6,
  write_introduction: 0.15,
  write_conclusion: 0.15,
  finalize_report: 0.1,
});
const ACTIVE_STAGE_CAP = 0.95;

export function newAnalystProgress() {
  return {
    limits: null,
    startedAt: null,
    pass: 1,
    completedPasses: 0,
    stagesDone: new Set(),
    researchTurns: 0,
    passToolCalls: 0,
    writing: false,
    done: false,
    failed: false,
    shown: 0,
  };
}

export function markStarted(progress, limits, now) {
  progress.limits = limits;
  progress.startedAt = now;
}

export function markStep(progress, node, researchPass) {
  if (node === "planner_node" && researchPass && researchPass > progress.pass) {
    progress.pass = researchPass;
    progress.stagesDone = new Set();
    progress.researchTurns = 0;
    progress.passToolCalls = 0;
  }
  if (node === "writer_node") progress.writing = true;
}

export function markPlan(progress) {
  progress.stagesDone.add("plan");
}

export function markResearcherTurn(progress) {
  progress.researchTurns += 1;
}

export function markToolResults(progress, count) {
  progress.passToolCalls += count;
}

export function markFindings(progress) {
  progress.stagesDone.add("plan");
  progress.stagesDone.add("research");
  progress.stagesDone.add("extract");
}

// A finished pass moves into completedPasses; in-pass state resets for the next one.
export function markEvaluated(progress) {
  progress.completedPasses = Math.max(progress.completedPasses, progress.pass);
  progress.stagesDone = new Set();
  progress.researchTurns = 0;
  progress.passToolCalls = 0;
}

export function markDone(progress, { failed = false } = {}) {
  progress.done = true;
  progress.failed = failed;
  progress.shown = 1;
}

function clamp(value, low = 0, high = 1) {
  return Math.min(high, Math.max(low, value));
}

export function analystFraction(progress, now) {
  if (progress.done) return 1;
  const limits = progress.limits;
  if (!limits || progress.startedAt == null) return 0;
  const maxPasses = Math.max(1, limits.max_passes || 1);

  let passFraction = 0;
  for (const stage of progress.stagesDone) passFraction += STAGE_WEIGHTS[stage] ?? 0;
  if (progress.stagesDone.has("plan") && !progress.stagesDone.has("research")) {
    const turns = progress.researchTurns / Math.max(1, limits.max_researcher_turns || 1);
    const tools = progress.passToolCalls / Math.max(1, limits.max_tool_calls_per_pass || 1);
    passFraction += STAGE_WEIGHTS.research * Math.min(ACTIVE_STAGE_CAP, Math.max(turns, tools));
  }
  const research = clamp((progress.completedPasses + clamp(passFraction)) / maxPasses);
  let value = RESEARCH_SHARE * research;
  if (progress.writing) value = Math.max(value, RESEARCH_SHARE);

  const elapsedSeconds = Math.max(0, (now - progress.startedAt) / 1000);
  const timeFloor = RESEARCH_SHARE * clamp(elapsedSeconds / Math.max(1, limits.deadline_seconds || 1));
  return Math.min(0.99, Math.max(value, timeFloor));
}

// Monotonic: returns and stores the progress to display.
export function advance(progress, now) {
  progress.shown = Math.max(progress.shown, analystFraction(progress, now));
  return progress.shown;
}

export function overallFraction({ analysts, synthesis, final }, now) {
  if (final) return 1;
  if (!analysts.length) return 0;
  const research = analysts.reduce((sum, progress) => sum + advance(progress, now), 0) / analysts.length;
  let synthesized = 0;
  for (const [step, weight] of Object.entries(SYNTHESIS_WEIGHTS)) {
    if (synthesis[step]) synthesized += weight;
  }
  return Math.min(0.99, PHASE_WEIGHTS.research * research + PHASE_WEIGHTS.synthesis * synthesized);
}
