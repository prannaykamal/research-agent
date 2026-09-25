# LangGraph Research Assistant

> A multi-agent research workflow that creates analyst personas, runs parallel interview loops with web search, and synthesizes the results into a cited report using LangGraph, LangChain, OpenAI, and Tavily.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-LLM%20Workflows-1C3C3C)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-0EA5E9)](https://www.tavily.com/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9)](https://docs.astral.sh/uv/)

---

## What Is This?

This project is a LangGraph-powered research assistant inspired by analyst interview workflows. Given a research topic, the system generates a panel of analyst personas, gives the user a chance to approve or revise them, sends each analyst through an interview loop, gathers web context with Tavily, and combines the resulting memos into a final Markdown report.

The project demonstrates several practical agent patterns:

- Human-in-the-loop review before the research process begins.
- Structured analyst generation with Pydantic models.
- Parallel interview fan-out using LangGraph's `Send` API.
- Search-grounded question answering with Tavily.
- Map-reduce style report writing across analyst sections.
- LangGraph graph configuration for local development and Studio inspection.

| File | Responsibility |
|---|---|
| `src/agent.py` | Analyst-generation graph with human feedback interrupt |
| `src/answering_questions.py` | Interview subgraph for question generation, search, answer generation, and section writing |
| `src/deep_agent.py` | Full research graph that creates analysts, runs interviews, and finalizes the report |
| `src/utils/nodes.py` | Graph node implementations for analysts, interviews, search, and report writing |
| `src/utils/edges.py` | Conditional routing logic and interview fan-out |
| `src/utils/states.py` | TypedDict and message-state definitions used by the graphs |
| `src/utils/objects.py` | Pydantic models for analyst personas and structured outputs |
| `src/utils/prompts.py` | Prompt templates for analyst creation, interviews, search, and report synthesis |
| `src/utils/models.py` | OpenAI chat model initialization |
| `langgraph.json` | LangGraph CLI/Studio graph registration |

---

## Screenshot

### Full Research Graph

![LangGraph research workflow](<screenshots/Screenshot 2026-09-08 101749.png>)

The full graph creates analyst personas, pauses for human feedback, runs interview subgraphs, writes report sections, and then produces the final introduction, body, conclusion, and source list.

---

## Feature List

### Analyst Generation

- Generates a configurable number of analyst personas for a research topic.
- Uses structured output through `Perspectives` and `Analyst` Pydantic models.
- Captures analyst name, role, affiliation, description, and persona text.
- Supports human review through a LangGraph interrupt before continuing.

### Interview Workflow

- Each analyst starts with a topic-aware opening question.
- The expert answer is grounded in retrieved context.
- The interview loop continues until the maximum number of expert turns is reached or the analyst closes the interview.
- Finished interviews are saved as transcripts and converted into short report sections.

### Web Research

- Tavily search is used to retrieve external context for analyst questions.
- Search queries are generated from the conversation state rather than hard-coded.
- Retrieved documents are formatted with source URLs for downstream citation.

### Report Synthesis

- Analyst sections are gathered into a consolidated report body.
- Introduction and conclusion are generated as separate graph nodes.
- Final output is assembled into a Markdown report with preserved citations and sources.

---

## How It Works

1. **The user provides a topic.** The graph receives a research topic and a maximum number of analysts.

2. **Analyst personas are generated.** `create_analysts` uses the OpenAI model with structured output to create distinct expert perspectives.

3. **Human feedback is requested.** The graph interrupts before `human_feedback`, allowing the user to approve the analysts or request changes.

4. **Interviews run in parallel.** When approved, `initiate_all_interviews` fans out one interview subgraph per analyst.

5. **Each interview searches the web.** The interview graph generates search queries from the current conversation and retrieves context through Tavily.

6. **Expert answers are generated.** The model answers using only the retrieved context and includes source references.

7. **Sections are written.** Each completed interview becomes a concise report section.

8. **The final report is assembled.** The graph writes the introduction, report body, conclusion, and final Markdown output.

---

## Architecture

```text
Research topic
    -> create_analysts
        -> Analyst personas
    -> human_feedback
        -> regenerate analysts when feedback is provided
        -> continue when approved
    -> conduct_interview for each analyst
        -> ask_question
        -> search_web
        -> search_web2
        -> answer_question
        -> repeat until max turns
        -> save_interview
        -> write_section
    -> write_report
    -> write_introduction
    -> write_conclusion
    -> finalize_report
    -> final Markdown report
```

Registered graphs in `langgraph.json`:

| Graph | Entry Point | Purpose |
|---|---|---|
| `create_analysts` | `src/agent.py:graph` | Generate and review analyst personas |
| `asnwer_question` | `src/answering_questions.py:question_answer_graph` | Run a single interview workflow |
| `deep_agent` | `src/deep_agent.py:graph` | Run the complete research and reporting workflow |

---

## Project Structure

```text
LangGraph_Assistant/
|
+-- main.py
+-- langgraph.json
+-- pyproject.toml
+-- uv.lock
+-- README.md
+-- screenshots/
|   `-- Screenshot 2026-09-08 101749.png
+-- src/
|   +-- agent.py
|   +-- answering_questions.py
|   +-- deep_agent.py
|   +-- __init__.py
|   `-- utils/
|       +-- edges.py
|       +-- models.py
|       +-- nodes.py
|       +-- objects.py
|       +-- prompts.py
|       +-- states.py
|       +-- tools.py
|       `-- __init__.py
`-- .env
```

---

## Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Required by `pyproject.toml` |
| `uv` | Used for dependency installation and command execution |
| OpenAI API key | Required for `ChatOpenAI` |
| Tavily API key | Required for web search |

Check your local versions:

```bash
python --version
uv --version
```

If `uv` is not installed:

```bash
pip install uv
```

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/LangGraph_Assistant.git
cd LangGraph_Assistant
```

### 2. Install dependencies

```bash
uv sync
```

This creates the virtual environment and installs the locked dependencies from `uv.lock`.

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Keep `.env` local and do not commit real API keys.

### 4. Run with LangGraph

Start the LangGraph development server:

```bash
uv run langgraph dev
```

Then open the local LangGraph Studio URL shown in your terminal. The configured graphs are loaded from `langgraph.json`.

---

## Example Inputs

### Analyst Generation

Use the `create_analysts` graph with state like:

```json
{
  "topic": "How autonomous AI agents are changing software engineering teams",
  "max_analysts": 3
}
```

### Full Research Report

Use the `deep_agent` graph with state like:

```json
{
  "topic": "The business impact of retrieval-augmented generation in enterprise support teams",
  "max_analysts": 3
}
```

The full graph returns a `final_report` field containing the generated Markdown report.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---:|---|
| `OPENAI_API_KEY` | Yes | Authenticates the OpenAI chat model used by LangChain |
| `TAVILY_API_KEY` | Yes | Authenticates Tavily web search |

---

## Key Design Decisions

| Choice | Reason |
|---|---|
| LangGraph state machines | Makes the agent workflow explicit, inspectable, and easier to debug |
| Human feedback interrupt | Allows analyst personas to be reviewed before expensive research begins |
| Structured analyst objects | Keeps persona generation predictable and type-safe |
| Interview subgraph | Encapsulates repeated question, search, answer, and section-writing behavior |
| Parallel fan-out | Lets each analyst pursue their own research direction independently |
| Separate report nodes | Keeps introduction, body, conclusion, and final assembly easy to iterate on |

---

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| Language | Python 3.12+ | Application runtime |
| Graph orchestration | LangGraph | Stateful graph execution, interrupts, and fan-out |
| LLM framework | LangChain | Message handling, structured output, and model integration |
| LLM provider | OpenAI | Analyst generation, interview responses, and report writing |
| Web research | Tavily | Search API for retrieval-grounded answers |
| Configuration | dotenv | Loads local environment variables from `.env` |
| Package management | uv | Dependency resolution and command execution |

---

## Limitations

- The project requires valid OpenAI and Tavily API keys.
- Report quality depends on the specificity of the topic and the quality of Tavily search results.
- Generated reports should be reviewed before being used for business, academic, legal, medical, or financial decisions.
- The current `main.py` is a minimal placeholder; the primary workflow is exposed through LangGraph.
- The interview graph currently includes two similar search nodes, `search_web` and `search_web2`, for parallel retrieval.

---

## Future Improvements

- Add a polished CLI runner for invoking the full research graph from the terminal.
- Add automated tests for graph routing, structured output parsing, and report assembly.
- Add configurable model selection through environment variables.
- Deduplicate the two Tavily search node implementations.
- Persist reports to Markdown files or a lightweight database.
- Add richer source handling with normalized citations and source metadata.
- Add a Streamlit or FastAPI interface for non-technical users.

---

## Security Notes

- Never commit `.env` or real API keys.
- Rotate keys immediately if they are accidentally pushed to a public repository.
- Treat generated reports as research drafts, not final professional advice.

---

## License

MIT. Use it, extend it, and build on it.
=======
# LangGraph_Research_Assistant_Agent
This project is to create personal assistant using LangGraph from Scratch
>>>>>>> 374985914f0c6354ffc0a77d282f9297b3518c22
