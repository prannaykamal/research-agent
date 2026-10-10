import assert from "node:assert/strict";
import test from "node:test";

import { SAMPLE_TOPICS, pickTopics } from "../js/suggestions.js";

test("the suggestion pool holds 20-30 distinct questions", () => {
  assert.ok(SAMPLE_TOPICS.length >= 20 && SAMPLE_TOPICS.length <= 30);
  assert.equal(new Set(SAMPLE_TOPICS.map((topic) => topic.toLowerCase())).size, SAMPLE_TOPICS.length);
  assert.ok(SAMPLE_TOPICS.every((topic) => topic.endsWith("?")));
});

test("three distinct questions are picked, and a reshuffle avoids the last set", () => {
  for (let run = 0; run < 50; run += 1) {
    const first = pickTopics(3);
    assert.equal(first.length, 3);
    assert.equal(new Set(first).size, 3);
    assert.ok(first.every((topic) => SAMPLE_TOPICS.includes(topic)));
    const next = pickTopics(3, { exclude: first });
    assert.ok(next.every((topic) => !first.includes(topic)));
  }
});

test("the pick is random rather than a fixed slice", () => {
  const firsts = new Set(Array.from({ length: 40 }, () => pickTopics(3)[0]));
  assert.ok(firsts.size > 5);
});
