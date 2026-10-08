# Bounded Adaptive Map-Reduce Research Assistant

A LangGraph research workflow that creates analyst personas, obtains human approval,
and runs one bounded evidence-gathering subgraph for each analyst, all in parallel.

## Architecture

```text
create_analysts → human_feedback → Send(conduct_research) × N   (all N run at once)

Planner → Researcher ⇄ ToolNode → Extract findings → Evaluator → Planner or Writer

completed analyst drafts → report + introduction + conclusion → final report + run_stats
```

Each analyst subgraph has isolated state. The Planner creates research questions, the
Researcher alone invokes Tavily, Wikipedia, arXiv, PubMed, or the webpage scraper, the
Evaluator identifies evidence gaps, and the Writer synthesizes only structured findings.

## Run input

| Field | Values | Default |
| --- | --- | --- |
| `topic` | Research topic | required |
| `max_analysts` | Integer, **1–10** | required |
| `model_profile` | `quality` or `fast` | `quality` |

## Model profiles

| | `quality`: high cost, high latency | `fast`: low cost, low latency |
| --- | --- | --- |
| Heavy (Planner, Evaluator) | `gemini-3.1-pro-preview`, thinking high | `gemini-3.8-flash`, thinking low |
| Medium (analysts, Researcher, extraction, report) | `gemini-3.8-flash` | `gemini-3.5-flash-lite`, thinking minimal |
| Writer (analyst section) | `gemini-3.8-flash`, thinking low, ≤600 words | `gemini-3.5-flash-lite`, ≤350 words |
| Light (introduction, conclusion) | `gemini-3.8-flash` | `gemini-3.5-flash-lite` |
| Research passes / tool calls per pass / Researcher turns | 3 / 6 / 3 | 2 / 4 / 2 |
| Tools | all five | Tavily, Wikipedia, arXiv, PubMed |
| Analyst deadline | 400 s | 150 s |

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
- At most 3 / 2 research passes, 6 / 4 tool calls per pass, and 3 / 2 Researcher turns per pass (`quality` / `fast`).
- Tool calls outside the profile's tool list are dropped before execution.
- A deadline (400 s / 150 s), an input-token budget (150k / 60k), and an LLM-call budget (25 / 12). When any of them runs out, no new research starts; evidence already gathered is extracted and the section is written.
- Tool output is capped at 12,000 characters per call, Tavily included. The Researcher sees outputs clipped to 2,000 / 1,500 characters. Extraction sees up to 4,000 / 3,000 characters per tool and 24,000 / 12,000 per pass.
- Raw tool messages are cleared before the next Planner pass; only structured evidence remains durable.
- Findings require a source URL and are deterministically deduplicated. At most 4 per pass and 12 per analyst, with claims ≤300 and excerpts ≤500 characters.
- If the Planner or Evaluator returns unusable structured output, a deterministic fallback is used instead of crashing.
- Each analyst section is capped by a word target and at 8,000 characters.
- A 429 or 5xx error is retried once through the limiters. Any remaining failure affects only that analyst: the report is still written and a coverage note names the missing perspectives.
- The scraper only fetches public HTTP(S) hosts. Every redirect hop is resolved via DNS and rejected if it points to a private, loopback, link-local, reserved or multicast address.

### System-wide

- `max_analysts` must be between 1 and 10. The generated panel is trimmed to exactly that number.
- At most 3 analyst-panel revisions; after that, the current panel is used.
- At most 10 analysts run at once across all runs in the process.
- TPM and RPM limits are enforced per model ID. Provider-side retries are disabled, so every attempt passes the limiters.
- Synthesis input is bounded: ten capped sections always fit one report request.
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

The primary graph is `deep_agent` in `langgraph.json`.

## Tests

```bash
uv run pytest
```

`tests/test_parallelism.py` runs the full graph with stub models (fixed latency) through
the real limiters and checks that 10 analysts take as long per analyst as 1.
