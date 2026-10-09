"""Record a full stub-server session as the UI's offline demo replay.

Drives the real LangGraph API exactly like the UI does (start, one revision,
approval) and writes every SSE event with its arrival time to
``frontend/demo/sample-run.json``. Start the stub server first::

    uv run python scripts/stub_server.py
    uv run python scripts/record_demo.py
"""

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "frontend" / "demo" / "sample-run.json"
STREAM_OPTIONS = {
    "assistant_id": "deep_agent",
    "stream_mode": ["updates", "custom"],
    "stream_subgraphs": True,
    "on_disconnect": "continue",
}
RUN_INPUT = {
    "topic": "Enterprise adoption of multi-agent research assistants",
    "max_analysts": 3,
    "model_profile": "fast",
}
DECISIONS = ["Add a perspective on legal and regulatory risk.", "approved"]


def stream_run(client: httpx.Client, thread_id: str, body: dict) -> list[dict]:
    events: list[dict] = []
    started = time.monotonic()
    name, data = None, []
    with client.stream("POST", f"/threads/{thread_id}/runs/stream", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line == "":
                if name and data:
                    events.append(
                        {
                            "t": round(time.monotonic() - started, 3),
                            "event": name,
                            "data": json.loads("\n".join(data)),
                        }
                    )
                name, data = None, []
            elif line.startswith(":"):
                continue
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="http://127.0.0.1:2025")
    args = parser.parse_args()

    with httpx.Client(base_url=args.server, timeout=None) as client:
        thread_id = client.post("/threads", json={}).raise_for_status().json()["thread_id"]
        runs = [{"resume": None, "events": stream_run(client, thread_id, {**STREAM_OPTIONS, "input": RUN_INPUT})}]
        for decision in DECISIONS:
            body = {**STREAM_OPTIONS, "command": {"resume": decision}}
            runs.append({"resume": decision, "events": stream_run(client, thread_id, body)})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "note": "Recorded from scripts/stub_server.py; all content is placeholder data.",
                "input": RUN_INPUT,
                "runs": runs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    total = sum(len(run["events"]) for run in runs)
    print(f"Recorded {total} events across {len(runs)} runs -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
