// Minimal LangGraph API client. The UI only ever holds one streaming request per
// run: no polling, and the run keeps going if the page disconnects.

import { readEventStream } from "./sse.js";

// Only node updates and small custom progress events. "messages" mode is avoided on
// purpose: it switches every model call to token streaming. "values" sends the full
// state after every step.
export const STREAM_OPTIONS = Object.freeze({
  stream_mode: ["updates", "custom"],
  stream_subgraphs: true,
  on_disconnect: "continue",
});

export class LangGraphClient {
  constructor(baseUrl, assistantId = "deep_agent") {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.assistantId = assistantId;
  }

  async request(path, init = {}) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(init.headers || {}) },
    });
    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.json()).detail ?? "";
      } catch {
        // non-JSON error body
      }
      const message = typeof detail === "string" ? detail : JSON.stringify(detail);
      throw new Error(`${response.status} ${response.statusText}${message ? `: ${message}` : ""}`);
    }
    return response;
  }

  async health() {
    const response = await this.request("/ok");
    return response.json();
  }

  async createThread() {
    const response = await this.request("/threads", { method: "POST", body: "{}" });
    return (await response.json()).thread_id;
  }

  // Yields {event, data} with data already JSON-decoded.
  async *streamRun(threadId, payload, { signal } = {}) {
    const body = JSON.stringify({ assistant_id: this.assistantId, ...STREAM_OPTIONS, ...payload });
    const response = await this.request(`/threads/${threadId}/runs/stream`, {
      method: "POST",
      body,
      signal,
      headers: { accept: "text/event-stream" },
    });
    for await (const event of readEventStream(response, signal)) {
      let data = null;
      try {
        data = event.data ? JSON.parse(event.data) : null;
      } catch {
        data = event.data;
      }
      yield { event: event.event, data };
    }
  }

  async cancelRun(threadId, runId) {
    await this.request(`/threads/${threadId}/runs/${runId}/cancel?wait=false`, { method: "POST" });
  }
}
