import assert from "node:assert/strict";
import test from "node:test";

import { renderInline, renderMarkdown, safeUrl } from "../js/markdown.js";

test("escapes raw HTML instead of rendering it", () => {
  const html = renderMarkdown('<script>alert(1)</script> <img src=x onerror="alert(1)">');
  assert.ok(!html.includes("<script"));
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;script&gt;"));
});

test("only http(s) links become anchors", () => {
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("data:text/html,hi"), null);
  const html = renderInline("[click](javascript:alert(1)) and [ok](https://example.org/a)");
  assert.ok(!html.includes('href="javascript'));
  assert.ok(html.includes('<a href="https://example.org/a" target="_blank" rel="noopener noreferrer">ok</a>'));
});

test("autolinks bare URLs without trailing punctuation or emphasis damage", () => {
  const html = renderInline("See https://en.wikipedia.org/wiki/Grid_energy_storage. Also _this_.");
  assert.ok(html.includes('href="https://en.wikipedia.org/wiki/Grid_energy_storage"'));
  assert.ok(html.includes(">https://en.wikipedia.org/wiki/Grid_energy_storage</a>."));
  assert.ok(html.includes("<em>this</em>"));
});

test("renders emphasis, code and attribute-safe quotes", () => {
  const html = renderInline('**bold** *it* `a<b>` "q"');
  assert.equal(html, "<strong>bold</strong> <em>it</em> <code>a&lt;b&gt;</code> &quot;q&quot;");
});

test("renders headings, rules, quotes and code fences", () => {
  const html = renderMarkdown("# Title\n\n---\n\n> quoted\n\n```\n<raw>\n```");
  assert.ok(html.includes("<h1>Title</h1>"));
  assert.ok(html.includes("<hr>"));
  assert.ok(html.includes("<blockquote><p>quoted</p></blockquote>"));
  assert.ok(html.includes("<pre><code>&lt;raw&gt;</code></pre>"));
});

test("renders pipe tables", () => {
  const html = renderMarkdown("| A | B |\n| --- | --- |\n| 1 | **2** |");
  assert.ok(html.includes("<thead><tr><th>A</th><th>B</th></tr></thead>"));
  assert.ok(html.includes("<td>1</td><td><strong>2</strong></td>"));
});

test("renders nested and ordered lists", () => {
  const html = renderMarkdown("- one\n  - nested\n- two\n\n1. first\n2. second");
  assert.ok(html.includes("<ul><li>one<ul><li>nested</li></ul></li><li>two</li></ul>"));
  assert.ok(html.includes("<ol><li>first</li><li>second</li></ol>"));
});

test("keeps dollar signs inside nested lists", () => {
  const html = renderMarkdown("- costs\n  - $5 per run");
  assert.ok(html.includes("$5 per run"));
});
