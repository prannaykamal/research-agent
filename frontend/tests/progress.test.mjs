import assert from "node:assert/strict";
import test from "node:test";

import * as progress from "../js/progress.js";

const LIMITS = { max_passes: 2, max_tool_calls_per_pass: 4, max_researcher_turns: 2, deadline_seconds: 150 };

function started(now = 0) {
  const state = progress.newAnalystProgress();
  progress.markStarted(state, LIMITS, now);
  return state;
}

test("an analyst that has not started is at zero", () => {
  assert.equal(progress.analystFraction(progress.newAnalystProgress(), 0), 0);
});

test("stage progress is monotonic through a two-pass analyst", () => {
  const state = started();
  const seen = [];
  const sample = () => seen.push(progress.advance(state, 1));
  progress.markStep(state, "planner_node", 1);
  progress.markPlan(state);
  sample();
  progress.markResearcherTurn(state);
  progress.markToolResults(state, 3);
  sample();
  progress.markFindings(state);
  sample();
  progress.markEvaluated(state);
  sample();
  assert.equal(seen.at(-1), 0.45); // one of two passes = half of the 90% research share
  progress.markStep(state, "planner_node", 2);
  progress.markPlan(state);
  sample();
  progress.markStep(state, "writer_node", null);
  sample();
  progress.markDone(state);
  sample();
  for (let i = 1; i < seen.length; i += 1) assert.ok(seen[i] >= seen[i - 1], `step ${i} moved backwards`);
  assert.equal(seen.at(-2), progress.RESEARCH_SHARE); // the writer skips unused passes
  assert.equal(seen.at(-1), 1);
});

test("research is advanced by the larger of turn and tool budget use, capped below the next stage", () => {
  const state = started();
  progress.markPlan(state);
  progress.markToolResults(state, 4); // the whole per-pass tool budget
  const expected = progress.RESEARCH_SHARE * ((0.15 + 0.5 * 0.95) / 2);
  assert.ok(Math.abs(progress.analystFraction(state, 0) - expected) < 1e-9);
});

test("the deadline acts as a time floor for research", () => {
  const state = started(0);
  assert.equal(progress.analystFraction(state, 75_000), progress.RESEARCH_SHARE * 0.5);
  // Past the deadline only the Writer remains, so time alone never exceeds the research share.
  assert.equal(progress.analystFraction(state, 10_000_000), progress.RESEARCH_SHARE);
});

test("shown progress never decreases", () => {
  const state = started(0);
  assert.equal(progress.advance(state, 75_000), 0.45);
  state.startedAt = 75_000; // e.g. a clock adjustment
  assert.equal(progress.advance(state, 75_000), 0.45);
});

test("overall progress weights research and synthesis and completes on the final report", () => {
  const done = started();
  progress.markDone(done);
  const idle = started(0);
  const research = (1 + 0) / 2;
  assert.equal(progress.overallFraction({ analysts: [done, idle], synthesis: {}, final: false }, 0), 0.85 * research);
  const allDone = [done, done];
  const synthesis = { write_report: true, write_introduction: true };
  assert.ok(Math.abs(progress.overallFraction({ analysts: allDone, synthesis, final: false }, 0) - (0.85 + 0.15 * 0.75)) < 1e-9);
  assert.equal(progress.overallFraction({ analysts: allDone, synthesis, final: true }, 0), 1);
});
