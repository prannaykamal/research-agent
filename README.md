# Bounded Adaptive Map-Reduce Research Assistant

A LangGraph research workflow that creates analyst personas, obtains human approval,
and runs one bounded evidence-gathering subgraph for each analyst in parallel.

## Architecture

```text
create_analysts → human_feedback → Send(conduct_research)

Planner → Researcher ⇄ ToolNode → Evaluator → Planner or Writer

completed analyst drafts → report + introduction + conclusion → final report
```

Each analyst subgraph has isolated state. The Planner creates research questions, the
Researcher alone invokes Tavily, Wikipedia, arXiv, PubMed, or the webpage scraper, the
Evaluator identifies evidence gaps, and the Writer synthesizes only structured findings.

## Guardrails

- A research pass is counted only after its tool-use phase finishes.
- Each analyst can complete at most three research passes.
- Each pass can execute at most six tool calls.
- Raw tool messages are cleared before the next Planner pass; only structured evidence
  remains durable.
- Findings require a source URL and are deterministically deduplicated.


## Gemini model routing

Gemini models are routed by task tier and all calls of a tier share a rolling
per-minute token budget, including parallel analyst runs:

| Tier | Model default | Nodes |
| --- | --- | --- |
| Heavy | `gemini-3.1-pro-preview` | Planner (high), Planner correction (medium), Evaluator (high) |
| Medium | `gemini-3.8-flash` | Analyst generation, Researcher, finding extraction, Writer, report synthesis |
| Light | `gemini-3.8-flash` | Introduction, conclusion |

The defaults reserve 80% of each configured token-per-minute quota. Set each
`GEMINI_<TIER>_TPM_LIMIT` to its real Google project quota; the shared limiter
then keeps concurrent calls below that budget. Each provider request has a 45-second timeout and one attempt by default, so provider throttling surfaces as a trace error instead of a long retry wait. Override the timeout with `GEMINI_<TIER>_REQUEST_TIMEOUT_SECONDS`. Model IDs can be overridden with
`GEMINI_<TIER>_MODEL` if Google exposes a different deployment identifier.

## Setup

```bash
uv sync --all-groups
```

Create `.env` with `GOOGLE_API_KEY` and `TAVILY_API_KEY`. Optionally set each `GEMINI_<TIER>_TPM_LIMIT` to the matching Google project quota, then start LangGraph Studio:

```bash
uv run langgraph dev
```

The primary graph is `deep_agent` in `langgraph.json`.

## Tests

```bash
uv run pytest
```
