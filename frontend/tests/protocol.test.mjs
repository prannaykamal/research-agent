// Replays the recorded stub-server session (real langgraph-api output) through the
// interpreter and progress model the UI uses.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import * as progress from "../js/progress.js";
import { createInterpreter, messageText } from "../js/protocol.js";

const recording = JSON.parse(readFileSync(new URL("../demo/sample-run.json", import.meta.url), "utf8"));

function interpretRuns() {
  const interpret = createInterpreter();
  return recording.runs.map((run) => run.events.flatMap((event) => interpret(event)));
}

const kinds = (actions, kind) => actions.filter((action) => action.kind === kind);

test("each run starts with run metadata", () => {
  for (const actions of interpretRuns()) assert.equal(actions[0].kind, "run_started");
});

test("every human_feedback interrupt produces a new feedback round", () => {
  const [first, revision, approval] = interpretRuns();
  const firstRound = kinds(first, "feedback_requested");
  assert.equal(firstRound.length, 1);
  assert.equal(firstRound[0].payload.analysts.length, recording.input.max_analysts);

  assert.equal(kinds(revision, "feedback_resolved")[0].feedback, recording.runs[1].resume);
  const secondRound = kinds(revision, "feedback_requested");
  assert.equal(secondRound.length, 1, "the revised panel is a new round");
  const names = (round) => round.payload.analysts.map((analyst) => analyst.name).join("|");
  assert.notEqual(names(secondRound[0]), names(firstRound[0]));

  assert.equal(kinds(approval, "feedback_resolved")[0].feedback, null);
  assert.equal(kinds(approval, "feedback_requested").length, 0);
});

test("every analyst streams plan, findings, review and draft in order", () => {
  const approval = interpretRuns()[2];
  const count = recording.input.max_analysts;
  const started = kinds(approval, "analyst_started");
  assert.deepEqual(started.map((action) => action.index).sort(), [...Array(count).keys()]);
  for (const action of started) assert.ok(action.limits.max_passes >= 1 && action.limits.deadline_seconds > 0);

  for (let index = 0; index < count; index += 1) {
    const mine = approval.filter((action) => action.index === index);
    const order = mine.map((action) => action.kind);
    assert.equal(order[0], "analyst_started");
    assert.ok(order.includes("plan"));
    assert.ok(order.includes("findings"));
    assert.ok(order.includes("review") || order.includes("review_skipped"));
    assert.equal(order.filter((kind) => kind === "draft").length, 1);
    assert.ok(order.lastIndexOf("findings") < order.indexOf("draft"));
    assert.ok(order.indexOf("plan") < order.indexOf("findings"));

    const dispatched = new Set(kinds(mine, "tools_dispatched").flatMap((action) => action.calls.map((call) => call.id)));
    for (const result of kinds(mine, "tool_results").flatMap((action) => action.results)) {
      assert.ok(dispatched.has(result.id), `tool result ${result.id} matches a dispatched call`);
    }
    for (const findings of kinds(mine, "findings")) {
      for (const finding of findings.findings) assert.match(finding.source_url, /^https?:\/\//);
    }
  }
  assert.equal(kinds(approval, "analyst_completed").length, count);
  assert.deepEqual(kinds(approval, "synthesis_step").map((action) => action.step).sort(), [
    "write_conclusion",
    "write_introduction",
    "write_report",
  ]);
  const final = kinds(approval, "final_report");
  assert.equal(final.length, 1);
  assert.ok(final[0].report.length > 0);
  assert.equal(final[0].runStats.analysts, count);
});

test("progress replayed from the session is monotonic and reaches 100%", () => {
  const approval = interpretRuns()[2];
  const analysts = Array.from({ length: recording.input.max_analysts }, () => progress.newAnalystProgress());
  const synthesis = {};
  let final = false;
  let shown = 0;
  let now = 0;
  for (const action of approval) {
    now += 500;
    const state = analysts[action.index];
    switch (action.kind) {
      case "analyst_started":
        progress.markStarted(state, action.limits, now);
        break;
      case "analyst_step":
        progress.markStep(state, action.node, action.researchPass);
        break;
      case "plan":
        progress.markPlan(state);
        break;
      case "researcher_turn":
        progress.markResearcherTurn(state);
        break;
      case "tool_results":
        progress.markToolResults(state, action.results.length);
        break;
      case "findings":
        progress.markFindings(state);
        break;
      case "review":
      case "review_skipped":
        progress.markEvaluated(state);
        break;
      case "draft":
        progress.markDone(state);
        break;
      case "synthesis_step":
        synthesis[action.step] = true;
        break;
      case "final_report":
        final = true;
        break;
      default:
        break;
    }
    const value = Math.max(shown, progress.overallFraction({ analysts, synthesis, final }, now));
    assert.ok(value >= shown);
    shown = value;
  }
  assert.equal(shown, 1);
});

test("message text flattens Gemini content parts", () => {
  assert.equal(messageText([{ type: "text", text: "a" }, "b", { type: "image" }]), "ab");
  assert.equal(messageText(null), "");
});

test("a rejected topic becomes a form error; any other failure stays a run error", () => {
  const interpret = createInterpreter();
  const message = '"hi" is not a research topic.';
  assert.deepEqual(interpret({ event: "error", data: { error: "UnresearchableTopicError", message } }), [
    { kind: "topic_rejected", message },
  ]);
  assert.deepEqual(interpret({ event: "error", data: { error: "ValueError", message: "boom" } }), [
    { kind: "run_error", message: "boom" },
  ]);
});
