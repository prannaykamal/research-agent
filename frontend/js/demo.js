// Replays a recorded session (scripts/record_demo.py) through the same code path
// as a live run. No network calls beyond loading the recording.

const MAX_GAP_SECONDS = 2.5;

function sleep(ms, signal) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

export class DemoClient {
  constructor(recording, speed = 1) {
    this.recording = recording;
    this.speed = speed;
    this.runIndex = 0;
  }

  get input() {
    return this.recording.input;
  }

  // The recorded human decision for feedback round `round` (1-based).
  decision(round) {
    return this.recording.runs[round]?.resume ?? "approved";
  }

  async health() {
    return { ok: true, demo: true };
  }

  async createThread() {
    this.runIndex = 0;
    return "demo-0000-replay";
  }

  async *streamRun(_threadId, _payload, { signal } = {}) {
    const run = this.recording.runs[this.runIndex];
    this.runIndex += 1;
    if (!run) return;
    let previous = 0;
    for (const event of run.events) {
      const gap = Math.min(MAX_GAP_SECONDS, Math.max(0, event.t - previous)) / this.speed;
      previous = event.t;
      if (gap > 0) await sleep(gap * 1000, signal);
      if (signal?.aborted) return;
      yield { event: event.event, data: event.data };
    }
  }

  async cancelRun() {}
}
