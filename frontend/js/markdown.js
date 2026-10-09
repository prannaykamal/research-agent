// Small, safe Markdown renderer for model-written reports.
// All input is HTML-escaped first; only http(s) links are produced; no raw HTML
// passes through. Supports headings, paragraphs, emphasis, code, links, lists
// (nested by indentation), blockquotes, rules and pipe tables.

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const PLACEHOLDER = /\u0000([CLA])(\d+)\u0000/g;
const BARE_URL = /https?:\/\/[^\s<>"'`)\]\u0000]+/g;

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (char) => ESCAPES[char]);
}

export function safeUrl(url) {
  try {
    const parsed = new URL(String(url).trim());
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

function anchor(href, labelHtml) {
  return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${labelHtml}</a>`;
}

function emphasis(html) {
  return html
    .replace(/\*\*(?=\S)([^*]+?)\*\*/g, (_, text) => `<strong>${text}</strong>`)
    .replace(/__(?=\S)([^_]+?)__/g, (_, text) => `<strong>${text}</strong>`)
    .replace(/~~(?=\S)([^~]+?)~~/g, (_, text) => `<del>${text}</del>`)
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]+?)\*(?!\w)/g, (_, lead, text) => `${lead}<em>${text}</em>`)
    .replace(/(^|[^_\w])_(?=\S)([^_\n]+?)_(?!\w)/g, (_, lead, text) => `${lead}<em>${text}</em>`);
}

export function renderInline(raw) {
  const codes = [];
  const links = [];
  const autolinks = [];
  let text = String(raw ?? "").replace(/`([^`\n]+)`/g, (_, code) => `\u0000C${codes.push(code) - 1}\u0000`);
  text = text.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (match, label, url) => {
    const href = safeUrl(url);
    return href ? `\u0000L${links.push({ label, href }) - 1}\u0000` : match;
  });
  text = text.replace(BARE_URL, (url) => {
    const trimmed = url.replace(/[.,;:!?]+$/, "");
    const href = safeUrl(trimmed);
    if (!href) return url;
    return `\u0000A${autolinks.push({ label: trimmed, href }) - 1}\u0000${url.slice(trimmed.length)}`;
  });

  let html = emphasis(escapeHtml(text));
  // Links first (their labels may contain code placeholders), then code.
  html = html.replace(PLACEHOLDER, (match, type, index) => {
    if (type === "L") return anchor(links[index].href, emphasis(escapeHtml(links[index].label)));
    if (type === "A") return anchor(autolinks[index].href, escapeHtml(autolinks[index].label));
    return match;
  });
  return html.replace(PLACEHOLDER, (match, type, index) =>
    type === "C" ? `<code>${escapeHtml(codes[index])}</code>` : match,
  );
}

const FENCE = /^\s*(```|~~~)/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const QUOTE = /^\s{0,3}>\s?/;
const BULLET = /^(\s*)[-*+]\s+(.*)$/;
const NUMBERED = /^(\s*)(\d+)[.)]\s+(.*)$/;
const TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

const isBlank = (line) => /^\s*$/.test(line);
const isListItem = (line) => BULLET.test(line) || NUMBERED.test(line);
const isTableStart = (line, next) => line.includes("|") && next !== undefined && TABLE_RULE.test(next) && next.includes("-");

function startsBlock(line, next) {
  return FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || isListItem(line) || isTableStart(line, next);
}

function splitRow(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|") && !row.endsWith("\\|")) row = row.slice(0, -1);
  return row.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

function renderTable(header, rows) {
  const head = header.map((cell) => `<th>${renderInline(cell)}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${header.map((_, column) => `<td>${renderInline(row[column] ?? "")}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="md-table"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderList(lines) {
  const items = [];
  for (const line of lines) {
    const bullet = line.match(BULLET);
    const numbered = bullet ? null : line.match(NUMBERED);
    if (bullet) items.push({ indent: bullet[1].length, ordered: false, text: bullet[2] });
    else if (numbered) items.push({ indent: numbered[1].length, ordered: true, text: numbered[3] });
    else if (items.length) items[items.length - 1].text += ` ${line.trim()}`;
  }
  let position = 0;
  function build(indent) {
    const tag = items[position].ordered ? "ol" : "ul";
    const parts = [];
    while (position < items.length && items[position].indent >= indent) {
      const item = items[position];
      if (item.indent > indent) {
        const nested = build(item.indent);
        if (parts.length) parts[parts.length - 1] = parts[parts.length - 1].replace(/<\/li>$/, () => `${nested}</li>`);
        else parts.push(`<li>${nested}</li>`);
        continue;
      }
      parts.push(`<li>${renderInline(item.text)}</li>`);
      position += 1;
    }
    return `<${tag}>${parts.join("")}</${tag}>`;
  }
  let html = "";
  while (position < items.length) html += build(items[position].indent);
  return html;
}

export function renderMarkdown(source) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) {
      i += 1;
      continue;
    }
    if (FENCE.test(line)) {
      const marker = line.match(FENCE)[1];
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith(marker)) code.push(lines[i++]);
      i += 1;
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = line.match(HEADING);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }
    if (RULE.test(line)) {
      html.push("<hr>");
      i += 1;
      continue;
    }
    if (QUOTE.test(line)) {
      const inner = [];
      while (i < lines.length && QUOTE.test(lines[i])) inner.push(lines[i++].replace(QUOTE, ""));
      html.push(`<blockquote>${renderMarkdown(inner.join("\n"))}</blockquote>`);
      continue;
    }
    if (isTableStart(line, lines[i + 1])) {
      const header = splitRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && !isBlank(lines[i])) rows.push(splitRow(lines[i++]));
      html.push(renderTable(header, rows));
      continue;
    }
    if (isListItem(line)) {
      const block = [];
      while (i < lines.length && (isListItem(lines[i]) || (block.length && /^\s{2,}\S/.test(lines[i])))) {
        block.push(lines[i++]);
      }
      html.push(renderList(block));
      continue;
    }
    const paragraph = [line.trim()];
    i += 1;
    while (i < lines.length && !isBlank(lines[i]) && !startsBlock(lines[i], lines[i + 1])) paragraph.push(lines[i++].trim());
    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
  }
  return html.join("\n");
}
