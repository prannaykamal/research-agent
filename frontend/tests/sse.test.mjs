import assert from "node:assert/strict";
import test from "node:test";

import { createSSEParser } from "../js/sse.js";

function parse(chunks) {
  const events = [];
  const parser = createSSEParser((event) => events.push(event));
  for (const chunk of chunks) parser.push(chunk);
  parser.end();
  return events;
}

// Shape of langgraph-api output: CRLF separators and heartbeat comments.
const RAW =
  'event: metadata\r\ndata: {"run_id":"r1","attempt":1}\r\n\r\n' +
  ": heartbeat\r\n\r\n" +
  'event: updates|conduct_research:abc\r\ndata: {"planner_node":{"sub_questions":["q"]}}\r\n\r\n' +
  'event: custom|conduct_research:abc\r\ndata: {"type":"analyst_step"}\r\n\r\n';

const EXPECTED = [
  { event: "metadata", data: '{"run_id":"r1","attempt":1}', id: null },
  { event: "updates|conduct_research:abc", data: '{"planner_node":{"sub_questions":["q"]}}', id: null },
  { event: "custom|conduct_research:abc", data: '{"type":"analyst_step"}', id: null },
];

test("parses CRLF events and skips heartbeats", () => {
  assert.deepEqual(parse([RAW]), EXPECTED);
});

test("is independent of chunk boundaries", () => {
  for (let size = 1; size <= 17; size += 1) {
    const chunks = [];
    for (let start = 0; start < RAW.length; start += size) chunks.push(RAW.slice(start, start + size));
    assert.deepEqual(parse(chunks), EXPECTED, `chunk size ${size}`);
  }
});

test("joins multi-line data and accepts LF and CR endings", () => {
  const events = parse(["data: line one\ndata: line two\n\nevent: x\rdata: y\r\r"]);
  assert.deepEqual(events, [
    { event: "message", data: "line one\nline two", id: null },
    { event: "x", data: "y", id: null },
  ]);
});

test("flushes a final event without a trailing blank line", () => {
  assert.deepEqual(parse(["event: end\r\ndata: {}"]), [{ event: "end", data: "{}", id: null }]);
});
