// Incremental Server-Sent Events parser (text/event-stream).
// Handles LF, CRLF and CR line endings split across arbitrary chunk boundaries,
// comment lines (heartbeats) and multi-line data fields.

export function createSSEParser(onEvent) {
  let buffer = "";
  let eventName = "";
  let dataLines = [];
  let lastEventId = null;

  function dispatch() {
    if (dataLines.length === 0) {
      eventName = "";
      return;
    }
    onEvent({ event: eventName || "message", data: dataLines.join("\n"), id: lastEventId });
    eventName = "";
    dataLines = [];
  }

  function processLine(line) {
    if (line === "") return dispatch();
    if (line.startsWith(":")) return; // comment / heartbeat
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
    else if (field === "id") lastEventId = value;
  }

  return {
    push(chunk) {
      buffer += chunk;
      let start = 0;
      for (let i = 0; i < buffer.length; i += 1) {
        const ch = buffer[i];
        if (ch !== "\n" && ch !== "\r") continue;
        if (ch === "\r" && i === buffer.length - 1) break; // a "\n" may follow in the next chunk
        processLine(buffer.slice(start, i));
        if (ch === "\r" && buffer[i + 1] === "\n") i += 1;
        start = i + 1;
      }
      buffer = buffer.slice(start);
    },
    end() {
      if (buffer) {
        processLine(buffer.replace(/\r$/, ""));
        buffer = "";
      }
      dispatch();
    },
  };
}

// Yield parsed SSE events from a fetch() Response body.
export async function* readEventStream(response, signal) {
  const queue = [];
  const parser = createSSEParser((event) => queue.push(event));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const abort = () => reader.cancel().catch(() => {});
  signal?.addEventListener("abort", abort, { once: true });
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
      while (queue.length) yield queue.shift();
    }
    parser.push(decoder.decode());
    parser.end();
    while (queue.length) yield queue.shift();
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}
