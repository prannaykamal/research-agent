# Bounded Adaptive Map-Reduce Research Assistant

A LangGraph research workflow that creates analyst personas, obtains human approval,
and runs one bounded evidence-gathering subgraph for each analyst, all in parallel.

## Architecture

```text
question → requirements → create_analysts → human_feedback → Send(conduct_research) × N   (all N run at once)

Planner → Researcher ⇄ ToolNode → Extract findings → Evaluator → Planner or Writer

completed analyst drafts → report → introduction + conclusion → final report + run_stats
```

The question is first split into the requirements a complete answer must cover, and
every requirement is assigned to at least one analyst (a single analyst owns them all).
Each analyst subgraph has isolated state. The Planner creates research questions, the
Researcher alone invokes Tavily, Wikipedia, arXiv, PubMed, or the webpage scraper, the
Evaluator identifies evidence gaps, and the Writer synthesizes only structured findings.
The introduction and conclusion are written from the finished report body, so they
inherit its reconciled figures and hedges.

## Run input

| Field | Values | Default |
| --- | --- | --- |
| `topic` | Research topic | required |
| `max_analysts` | Integer, **1–10** (**1–5** for `quality`) | required |
| `model_profile` | `quality` or `fast` | `quality` |

## Model profiles

| | `quality`: high cost, high latency | `fast`: lower cost, lower latency |
| --- | --- | --- |
| Heavy (Planner, Evaluator) | `gemini-3.1-pro-preview`, thinking high | `gemini-3.8-flash`, thinking low |
| Panel (question requirements, analyst personas) | `gemini-3.8-flash` | `gemini-3.8-flash`, thinking low |
| Medium (Researcher, extraction, report) | `gemini-3.8-flash` | `gemini-3.5-flash-lite`, thinking minimal |
| Writer (analyst section) | `gemini-3.8-flash`, thinking low, ≤600 words | `gemini-3.5-flash-lite`, ≤500 words |
| Light (introduction, conclusion) | `gemini-3.8-flash` | `gemini-3.5-flash-lite` |
| Research passes / tool calls per pass (min–max) / Researcher turns | 3 / 4–6 / 3 (4 below the floor) | 3 / 4–6 / 3 (4 below the floor) |
| Tools | all five | Tavily, Wikipedia, arXiv, PubMed |
| Analyst deadline | 400 s | 300 s |

Override any model with `GEMINI_<PROFILE>_<TIER>_MODEL`, its thinking level with
`GEMINI_<PROFILE>_<TIER>_THINKING` (`minimal`, `low`, `medium`, `high` or `none`), and its
request timeout with `GEMINI_<PROFILE>_<TIER>_REQUEST_TIMEOUT_SECONDS`, for example
`GEMINI_FAST_MEDIUM_MODEL`.

## Parallelism and quotas

Analysts run as parallel `Send` branches on a dedicated 10-thread executor. Every model
call passes two rolling one-minute budgets:

1. **The analyst's slot:** a fixed 1/10 share of each model's quota. An analyst
   gets the same throughput whether it runs alone or with nine others, so 1 and 10
   analysts take the same time per analyst.
2. **The model's global quota:** shared by every tier and run that uses the model ID,
   because Gemini quotas are per project and per model.

Reservations count input tokens only (Gemini's TPM counts input tokens), including bound
tool and structured-output schemas. After each call the estimate is replaced with the
provider-reported usage. The limiters reserve 80% of each quota.

Defaults mirror published Tier 1 limits. Set your project's real limits from AI Studio:

```bash
GEMINI_QUOTA_GEMINI_3_1_PRO_PREVIEW_TPM=2000000
GEMINI_QUOTA_GEMINI_3_1_PRO_PREVIEW_RPM=150
GEMINI_QUOTA_GEMINI_3_8_FLASH_TPM=1000000
GEMINI_QUOTA_GEMINI_3_8_FLASH_RPM=1000
GEMINI_QUOTA_GEMINI_3_5_FLASH_LITE_TPM=4000000
GEMINI_QUOTA_GEMINI_3_5_FLASH_LITE_RPM=4000
```

At startup a warning is logged if a quota is too small for ten unthrottled analysts.
The warning includes the TPM and RPM you would need.

## Guardrails

### Per analyst (one `conduct_research` subgraph)

- A research pass is counted only after its tool-use phase finishes.
- Both profiles research to the same depth: at most 3 passes, 6 tool calls per pass and 3 Researcher turns per pass.
- Search floor: a Researcher that stops before 4 tool calls in a pass is sent back, with the requirements that still lack evidence. Below the floor a pass may take up to 4 Researcher turns instead of 3, so an analyst searching once per turn still reaches it. Cheaper research models otherwise stop after one or two searches.
- Tool calls outside the profile's tool list are dropped before execution.
- A deadline (400 s / 300 s, `quality` / `fast`), an input-token budget (150k), and an LLM-call budget (25). When any of them runs out, no new research starts; evidence already gathered is extracted and the section is written.
- Tool output is capped at 12,000 characters per call, Tavily included. The Researcher sees outputs clipped to 2,000 / 1,500 characters. Extraction sees up to 4,000 / 3,000 characters per tool and 24,000 / 18,000 per pass.
- Raw tool messages are cleared before the next Planner pass; only structured evidence remains durable.
- Findings require a source URL and are deterministically deduplicated. At most 4 per pass and 12 per analyst, with claims ≤300 and excerpts ≤500 characters.
- If the Planner or Evaluator returns unusable structured output, a deterministic fallback is used instead of crashing.
- The Evaluator rates every owned requirement `supported`, `thin` or `missing`, and cannot end research while any is thin, missing, or without a finding, as long as budget remains. A "changed since X" requirement stays thin until the evidence includes a value from around X.
- Every system prompt starts with today's date, and forward-looking questions get a "current status as of today" requirement, so reports do not describe past events as future ones.
- Findings carry a `source_date` (from arXiv and PubMed metadata, the URL path, or the source text), and the Evaluator and Writer see whether each source is under a year old. A current-status requirement stays `thin` without recent evidence, and writers give status claims "as of" the source's date, saying when the newest evidence is old.
- Analyst personas must be three or four sentences; a panel with a description under 40 words gets one corrective retry.
- Every source gets a deterministic trust tier from its domain (`trusted`, `verify`, `low`). When findings exceed capacity, higher tiers are kept, and the Writer must hedge any figure only `low`-tier sources support. Tavily never returns video, social or Simple-English Wikipedia pages.
- Failed tool calls are counted per tool in `analyst_stats.tool_errors`. A keyed service rejecting its API key (Tavily 401/403) fails the analyst immediately with `stop_reason: "auth: <tool>"` instead of letting it research on a degraded evidence base.
- Each analyst section is capped by a word target and at 8,000 characters.
- A 429 or 5xx error is retried once through the limiters. Any remaining failure affects only that analyst: the report is still written and a coverage note names the missing perspectives.
- The scraper only fetches public HTTP(S) hosts. Every redirect hop is resolved via DNS and rejected if it points to a private, loopback, link-local, reserved or multicast address.

### System-wide

- The topic must name something to research. The question-split call also judges the input, leniently, before any analyst is created and with no extra model call. Greetings, lone numbers or words, calculations and one-line facts are refused. The UI then shows a "Research topic is not relevant" section with three example questions, drawn at random from the evaluation questions; choosing one fills the topic box.
- `max_analysts` must be between 1 and 10 for Quick Research and between 1 and 5 for Deep Research, whose analysts cost far more and would mostly split the same requirements. The generated panel is trimmed to exactly that number.
- At most 3 analyst-panel revisions; after that, the current panel is used.
- At most 10 analysts run at once across all runs in the process.
- TPM and RPM limits are enforced per model ID. Provider-side retries are disabled, so every attempt passes the limiters.
- Synthesis input is bounded: ten capped sections always fit one report request.
- When the analyst sections cite but the synthesized report cites fewer than 3 of their URLs inline, the report is rewritten once on the panel model. Cheaper report writers otherwise moved every citation into `## Sources`.
- Every inline citation URL is added to `## Sources` if the report writer left it out, and a trailing horizontal rule in the body is dropped so the report never shows two in a row.
- `## Evidence limits` loses entries that only say there are no limits or repeat a body sentence, and the heading goes when nothing remains.
- Coverage notes are appended to the report for failed analysts, rejected API keys, and runs where more than half of all tool calls failed. Requirements with no supporting evidence by finding label or Evaluator verdict are recorded in `run_stats.uncovered_requirements` and passed to the report writer, which checks them against the sections and states real gaps under `## Evidence limits`, so a note can never contradict the report.
- `run_stats` in the final state reports each analyst's status, stop reason, duration, LLM calls and input tokens.

## Setup

```bash
uv sync --all-groups
```

Create `.env` with `GOOGLE_API_KEY` and `TAVILY_API_KEY`. Optionally set the
`GEMINI_QUOTA_*` limits above, then start LangGraph Studio:

```bash
uv run langgraph dev
```

The primary graph is `deep_agent` in `langgraph.json`. The research UI is served by the
same server at **http://127.0.0.1:2024/app/**.

## Research UI

A single-page UI (`frontend/`, no build step) served by the LangGraph server itself through
`http.app` in `langgraph.json` (`src/webapp.py`).

**Research Orchestrator (main chat).** Configure the topic, the number of perspectives
(1–10, or 1–5 in Deep Research; switching modes clamps the count), and the research mode: the toggle switches between **Quick Research** (`fast`) and
**Deep Research** (`quality`), and its `?` lists each mode's cost, latency, models, passes,
tools and deadline. Sections are appended as the graph runs:

- **Human feedback** is appended every time `human_feedback` interrupts, so a revised
  panel appears as a new round below the previous one. Earlier rounds keep the decision
  that was made.
- **Progress tracker** is appended on approval, when `conduct_research` fans out. At the
  same moment one chat per analyst appears in the sidebar.
- **Final report** is appended when `finalize_report` finishes, with a Markdown export,
  followed by a separate **Run statistics** section (per-analyst status, stop reason,
  duration, LLM calls, input tokens and findings). The chat follows new content as it
  arrives, except here: it stops at the report, with its heading at the top of the view,
  and leaves Run statistics below for the reader to scroll to.

**Analyst chats.** Each analyst gets its own chat that appends sections as its subgraph runs:

| Node finishes | Section appended |
| --- | --- |
| `planner_node` | Research plan (sub-questions for the pass) |
| `extract_findings` | Key findings (claims, excerpts, sources) |
| `evaluate_research` | Research review (coverage gaps and feedback) |
| `writer_node` | Analyst perspective (final draft, Markdown export) |

Between sections, a compact timeline shows reasoning steps, the Researcher's text, tool
calls (expand to see each output), guardrail actions, and skipped steps.

**Progress.** Each analyst's progress is 90% research, split evenly across the profile's
maximum passes. Within a pass, planning counts 15%, research (turns and tool calls) 50%,
extraction 20% and evaluation 15%. When the Writer starts, unused passes are skipped and
progress jumps to 90%; the finished draft is 100%. Research progress is never less than
elapsed time divided by the analyst deadline, and displayed progress never moves
backwards. Overall progress is 85% the average of the analysts and 15% synthesis (report
60%, introduction 15%, conclusion 15%, final assembly 10%), and reaches 100% with the
final report.

**Latency.** The UI is a passive observer and adds no work to a run:

- It streams only `updates` and small `custom` progress events. It never requests
  `messages`, which would switch every model call to token streaming, or `values`, which
  sends the full state after every step.
- The progress events come from LangGraph's stream writer, which does nothing when no
  client is streaming `custom` events. There are no extra nodes or LLM calls.
- Runs use `on_disconnect: "continue"`, so closing or reloading the tab never cancels a
  run, and the page does no polling.
- `tests/test_ui_stream.py` checks that a 10-analyst run takes the same time whether or not
  it is being streamed.

**Offline development.** `scripts/stub_server.py` runs the real graph and API with canned
models and tools: no API keys, cost or network. Content is placeholder text.

```bash
uv run python scripts/stub_server.py         # UI at http://127.0.0.1:2025/app/
```

`http://127.0.0.1:2025/app/?demo` (or `/app/?demo` on any server) replays a recorded
session from `frontend/demo/sample-run.json`; add `&speed=4` to replay faster. Regenerate the
recording with `uv run python scripts/record_demo.py` while the stub server is running.

## Tests

```bash
uv run pytest
```

`tests/test_parallelism.py` runs the full graph with stub models (fixed latency) through
the real limiters and checks that 10 analysts take as long per analyst as 1.
`tests/test_ui_stream.py` checks the event contract the UI relies on and that streaming adds
no latency. When Node.js is installed, `tests/test_frontend.py` also runs the frontend unit
tests (`node --test "frontend/tests/*.test.mjs"`): SSE parsing, safe Markdown rendering,
progress math, and a replay of the recorded session.
