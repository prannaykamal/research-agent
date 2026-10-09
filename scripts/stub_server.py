"""Offline development server: the real graph and API with canned models and tools.

Runs the same LangGraph API server as ``langgraph dev`` (including the UI at
``/app``) but replaces every Gemini model and research tool with deterministic
stubs, so the UI can be exercised with no API keys, cost, or network access::

    uv run python scripts/stub_server.py              # http://127.0.0.1:2025/app/
    uv run python scripts/stub_server.py --latency 0  # no simulated latency

All generated content is placeholder text citing example.org sources.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("GOOGLE_API_KEY", "stub-google-key")
os.environ.setdefault("TAVILY_API_KEY", "tvly-stub-key")
# Keep placeholder runs out of the LangSmith project configured in .env.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_tavily import TavilySearch  # noqa: E402

import src.analyst_research as research  # noqa: E402
import src.utils.nodes as nodes  # noqa: E402
import src.utils.tools as tools  # noqa: E402
from src.utils.models import RateLimitedRunnable  # noqa: E402
from src.utils.objects import (  # noqa: E402
    Analyst,
    Perspectives,
    ResearchEvaluation,
    ResearchFinding,
    ResearchFindingBatch,
    ResearchPlan,
)
from src.utils.profiles import get_profile  # noqa: E402

LATENCY_SCALE = 1.0

PERSONAS = [
    ("Dr. Elena Vance", "Systems Architecture & Scalability", "Distributed Systems Lab",
     "Throughput, state persistence, and failure recovery at scale."),
    ("Marcus Thorne", "Enterprise Security & Compliance", "Office of the CISO",
     "Access control, data boundaries, and audit requirements."),
    ("Priya Patel", "Financial TCO & Unit Economics", "FinOps Council",
     "Cost drivers, budgeting, and return on investment."),
    ("Dr. Kenji Mori", "Reliability Engineering", "SRE Guild",
     "Observability, incident response, and service-level objectives."),
    ("Amara Okafor", "Product & Adoption Strategy", "Digital Transformation Office",
     "User adoption, change management, and measurable outcomes."),
    ("Lucas Ferreira", "Data Governance", "Data Stewardship Board",
     "Lineage, retention, and data-quality controls."),
    ("Dr. Sofia Lindqvist", "Applied AI Research", "Institute for Machine Reasoning",
     "Model capability limits and evaluation methodology."),
    ("Rahul Mehta", "Platform Operations", "Cloud Infrastructure Group",
     "Deployment topology, capacity planning, and automation."),
    ("Omar Haddad", "Human Factors & Safety", "Human-Centred Design Studio",
     "Human oversight, operator workload, and safe failure modes."),
    ("Dr. Hannah Weiss", "Energy & Sustainability", "Sustainable Computing Initiative",
     "Energy use, carbon intensity, and efficiency."),
]
REGULATORY_PERSONA = ("Grace Liu", "Legal & Regulatory Affairs", "Regulatory Policy Unit",
                      "Regulatory exposure, liability, and contractual obligations.")


def _pause(low: float, high: float) -> None:
    if LATENCY_SCALE > 0:
        time.sleep(random.uniform(low, high) * LATENCY_SCALE)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "source"


def _last_json(messages: list[Any]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            try:
                return json.loads(message.content)
            except (TypeError, ValueError):
                return {}
    return {}


def _system_text(messages: list[Any]) -> str:
    return next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")


def _usage(messages: list[Any], output: str = "") -> dict[str, int]:
    input_tokens = max(1, sum(len(str(getattr(m, "content", m))) for m in messages) // 4)
    output_tokens = max(1, len(output) // 4)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def _panel(messages: list[Any]) -> Perspectives:
    system = _system_text(messages)
    count = int(re.search(r"Create exactly (\d+)", system).group(1))
    feedback = re.search(r"Editorial feedback: (.*)", system)
    personas = list(PERSONAS)
    if feedback and feedback.group(1).strip():
        personas = personas[1:] + personas[:1]  # a visibly different panel
        if re.search(r"regulat|legal|law|complian", feedback.group(1), re.I):
            personas.insert(min(count - 1, 1), REGULATORY_PERSONA)
    return Perspectives(
        analysts=[
            Analyst(name=name, role=role, affiliation=affiliation, description=description)
            for name, role, affiliation, description in personas[:count]
        ]
    )


def _plan(messages: list[Any]) -> ResearchPlan:
    data = _last_json(messages)
    topic = data.get("topic", "the topic")
    analyst = data.get("analyst", {})
    role = analyst.get("role", "the analyst")
    research_pass = len(data.get("question_history", [])) // 3 + 1
    if research_pass == 1:
        questions = [
            f"What measurable evidence exists on {topic} from a {role} perspective?",
            f"Which constraints most limit {topic} for {analyst.get('affiliation', 'adopters')}?",
            f"What do recent case studies report about outcomes of {topic}?",
        ]
    else:
        feedback = data.get("feedback") or "remaining coverage gaps"
        questions = [
            f"What quantitative benchmarks address: {feedback[:80]}?",
            f"How do independent sources validate the earlier {role} findings?",
            f"What failure cases or counter-evidence exist for {topic}?",
        ]
    return ResearchPlan(sub_questions=questions)


def _researcher(messages: list[Any]) -> AIMessage:
    if any(isinstance(message, ToolMessage) for message in messages):
        return AIMessage(
            content=(
                "The gathered sources cover each sub-question with at least one "
                "independent reference, so this pass has enough evidence."
            )
        )
    brief = _last_json(messages)
    questions = brief.get("current_sub_questions") or [brief.get("topic", "research")]
    calls = [
        {"name": "tavily_search", "args": {"query": questions[0]}},
        {"name": "wikipedia", "args": {"query": brief.get("topic", questions[0])}},
        {"name": "arxiv", "args": {"query": questions[-1]}},
    ]
    if len(brief.get("analyst", {}).get("name", "")) % 2 == 0:
        calls.append({"name": "pubmed", "args": {"query": questions[1 % len(questions)]}})
    return AIMessage(
        content=(
            "Starting broad: a web search for recent reports, an encyclopedia "
            "overview for definitions, then academic sources for measured results."
        ),
        tool_calls=[
            {**call, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"} for call in calls
        ],
    )


def _sources_from_outputs(outputs: list[dict[str, str]]) -> list[dict[str, str]]:
    sources = []
    for output in outputs:
        try:
            payload = json.loads(output["content"])
        except (TypeError, ValueError):
            continue
        items = payload.get("results", []) if isinstance(payload, dict) else payload
        for item in items:
            url = item.get("url") or item.get("source_url")
            if url:
                sources.append(
                    {
                        "title": item.get("title") or item.get("source_title", "Source"),
                        "url": url,
                        "excerpt": item.get("content") or item.get("excerpt", ""),
                        "type": item.get("source_type", "web"),
                    }
                )
    return sources


def _findings(messages: list[Any]) -> ResearchFindingBatch:
    data = _last_json(messages)
    questions = data.get("sub_questions") or ["Research question"]
    sources = _sources_from_outputs(data.get("tool_outputs", []))
    findings = [
        ResearchFinding(
            sub_question=questions[index % len(questions)],
            claim=(
                f"{source['title']} reports a measurable effect relevant to "
                f"'{questions[index % len(questions)][:60]}…' (placeholder claim)."
            ),
            source_title=source["title"],
            source_url=source["url"],
            excerpt=source["excerpt"][:220],
            source_type=source["type"],
        )
        for index, source in enumerate(sources[:4])
    ]
    return ResearchFindingBatch(findings=findings)


def _evaluation(messages: list[Any]) -> ResearchEvaluation:
    data = _last_json(messages)
    research_pass = max(1, (len(data.get("question_history", [])) + 2) // 3)
    name = data.get("analyst", {}).get("name", "")
    if research_pass >= 2 or len(name) % 3 == 0:
        return ResearchEvaluation(
            is_complete=True,
            feedback="Evidence now covers each assigned objective with independent sources.",
        )
    return ResearchEvaluation(
        is_complete=False,
        coverage_gaps=[
            "No quantitative benchmark compares outcomes before and after adoption.",
            "Sources are mostly vendor-authored; independent validation is missing.",
        ],
        feedback="Target independent benchmarks and at least one counter-example in the next pass.",
    )


def _section(messages: list[Any]) -> str:
    data = _last_json(messages)
    analyst = data.get("analyst", {})
    findings = data.get("research_findings", [])
    lines = [
        f"### {analyst.get('role', 'Analyst')}: key takeaways",
        "",
        f"From the perspective of {analyst.get('affiliation', 'this team')}, the evidence "
        f"gathered across {len(findings)} findings points to three themes. This is "
        "placeholder text generated by the offline stub server.",
        "",
    ]
    for finding in findings[:5]:
        lines.append(f"- **{finding['source_title']}**: {finding['claim']} ([source]({finding['source_url']}))")
    lines += [
        "",
        "| Factor | Evidence strength | Note |",
        "| --- | --- | --- |",
        "| Measured outcomes | Moderate | Mostly single-site studies |",
        "| Independent validation | Limited | Few third-party audits |",
        "",
        "_Limits: placeholder evidence; replace the stub server with real models for actual research._",
    ]
    return "\n".join(lines)


def _report(messages: list[Any]) -> str:
    system = _system_text(messages)
    urls = sorted(set(re.findall(r"https://example\.org/[^\s)\]]+", system)))
    body = [
        "## Insights",
        "",
        "Across the analyst sections, three themes recur: measurable gains are real but "
        "concentrated in well-scoped workflows; governance and security controls decide "
        "whether pilots reach production; and total cost depends more on operations than on "
        "model pricing. This is placeholder synthesis from the offline stub server.",
        "",
        "### Where the perspectives agree",
        "",
        "- Bounded, auditable workflows outperform open-ended autonomy.",
        "- Independent benchmarks are scarce and should be commissioned early.",
        "",
        "## Sources",
    ]
    body += [f"- {url}" for url in urls[:12]]
    return "\n".join(body)


def _prose(messages: list[Any]) -> str:
    system = _system_text(messages)
    human = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
    if "analyst report section" in system:
        return _section(messages)
    if "technical writer" in system:
        return _report(messages)
    topic = re.search(r"report on (.*?)\. Based only", system, re.S)
    title = topic.group(1).strip() if topic else "Research report"
    if "introduction" in human.lower():
        return (
            f"# {title}\n\n## Introduction\n\nThis report combines several analyst "
            "perspectives gathered in parallel. It is placeholder text from the offline stub server."
        )
    return (
        "## Conclusion\n\nThe perspectives converge on scoped, well-governed deployments. "
        "Placeholder text from the offline stub server."
    )


class StubModel:
    """Answers each node's call shape with canned content after simulated latency."""

    def __init__(self, latency: tuple[float, float], schema=None, include_raw=False, tools_bound=False):
        self.latency = latency
        self.schema = schema
        self.include_raw = include_raw
        self.tools_bound = tools_bound

    def with_structured_output(self, schema, include_raw: bool = False, **_kwargs):
        return StubModel(self.latency, schema, include_raw, self.tools_bound)

    def bind_tools(self, _tools, **_kwargs):
        return StubModel(self.latency, self.schema, self.include_raw, True)

    def invoke(self, messages, *_args, **_kwargs):
        _pause(*self.latency)
        builders = {
            Perspectives: _panel,
            ResearchPlan: _plan,
            ResearchFindingBatch: _findings,
            ResearchEvaluation: _evaluation,
        }
        if self.schema in builders:
            parsed = builders[self.schema](messages)
            if not self.include_raw:
                return parsed
            raw = AIMessage(content=parsed.model_dump_json(), usage_metadata=_usage(messages))
            return {"parsed": parsed, "raw": raw, "parsing_error": None}
        if self.tools_bound:
            message = _researcher(messages)
        else:
            message = AIMessage(content=_prose(messages))
        message.usage_metadata = _usage(messages, str(message.content))
        return message


_stub_models: dict[str, SimpleNamespace] = {}


def stub_models(name: str | None = None) -> SimpleNamespace:
    profile = get_profile(name)
    if profile.name not in _stub_models:
        _stub_models[profile.name] = SimpleNamespace(
            heavy=RateLimitedRunnable(StubModel((0.8, 1.6)), profile.heavy.model),
            medium=RateLimitedRunnable(StubModel((0.4, 1.0)), profile.medium.model),
            writer=RateLimitedRunnable(StubModel((0.8, 1.5)), profile.writer.model),
            light=RateLimitedRunnable(StubModel((0.3, 0.6)), profile.light.model),
        )
    return _stub_models[profile.name]


def _stub_results(kind: str, query: str, count: int = 2) -> list[dict[str, str]]:
    slug = _slug(query)
    return [
        {
            "source_title": f"{kind.title()} reference {index + 1}: {query[:50]}",
            "source_url": f"https://example.org/{kind}/{slug}-{index + 1}",
            "excerpt": (
                f"Placeholder {kind} excerpt {index + 1} about '{query[:70]}'. Reported metrics, "
                "methods, and limitations would appear here in a real run."
            ),
            "source_type": kind,
        }
        for index in range(count)
    ]


def _tavily(self, query: str = "", **_kwargs):
    _pause(0.3, 0.8)
    return {
        "query": query,
        "results": [
            {"title": item["source_title"], "url": item["source_url"], "content": item["excerpt"], "score": 0.9}
            for item in _stub_results("web", query, 3)
        ],
    }


def _document_tool(kind: str):
    def run(self, query: str, run_manager=None) -> str:
        _pause(0.2, 0.6)
        return json.dumps(_stub_results(kind, query))

    return run


def install_stubs() -> None:
    research.get_models = stub_models
    nodes.get_models = stub_models
    TavilySearch._run = _tavily
    tools.WikipediaEvidenceTool._run = _document_tool("wikipedia")
    tools.ArxivEvidenceTool._run = _document_tool("arxiv")
    tools.PubMedEvidenceTool._run = _document_tool("pubmed")
    tools._fetch_public_page = lambda url: (
        url,
        "<html><title>Placeholder page</title><body>Placeholder scraped text.</body></html>",
    )


def main() -> None:
    global LATENCY_SCALE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2025)
    parser.add_argument("--latency", type=float, default=1.0, help="Scale simulated latency (0 disables).")
    args = parser.parse_args()
    LATENCY_SCALE = args.latency
    install_stubs()

    from langgraph_api.cli import run_server

    print(f"Stub server UI: http://{args.host}:{args.port}/app/")
    run_server(
        host=args.host,
        port=args.port,
        reload=False,
        graphs={"deep_agent": "./src/deep_agent.py:graph"},
        http={"app": "./src/webapp.py:app"},
        open_browser=False,
        allow_blocking=True,
    )


if __name__ == "__main__":
    main()
