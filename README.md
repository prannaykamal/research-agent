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

## Setup

```bash
uv sync --all-groups
```

Create `.env` with `OPENAI_API_KEY` and `TAVILY_API_KEY`, then start LangGraph Studio:

```bash
uv run langgraph dev
```

The primary graph is `deep_agent` in `langgraph.json`.

## Tests

```bash
uv run pytest
```
