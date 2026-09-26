from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.utils.models import llm
from src.utils.objects import Perspectives
from src.utils.prompts import (
    analyst_instructions,
    intro_conclusion_instructions,
    report_writer_instructions,
)
from src.utils.states import GenerateAnalystsState, ResearchGraphState


def create_analysts(state: GenerateAnalystsState):
    structured_llm = llm.with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=state["topic"],
        human_analyst_feedback=state.get("human_analyst_feedback", ""),
        max_analysts=state["max_analysts"],
    )
    analysts = structured_llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the analyst panel."),
        ]
    )
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    feedback = interrupt(
        {
            "question": "Are these analysts okay?",
            "analysts": [
                analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
                for analyst in state.get("analysts", [])
            ],
            "instructions": "Return revision feedback, or empty/perfect/continue to approve.",
        }
    )
    if not isinstance(feedback, str):
        return {"human_analyst_feedback": None}
    cleaned = feedback.strip()
    if not cleaned or cleaned.lower() in {"perfect", "continue", "approved", "yes"}:
        return {"human_analyst_feedback": None}
    return {"human_analyst_feedback": cleaned}


def dummy(_: GenerateAnalystsState):
    return {}


def write_report(state: ResearchGraphState):
    sections = "\n\n".join(state["sections"])
    instructions = report_writer_instructions.format(topic=state["topic"], context=sections)
    report = llm.invoke(
        [SystemMessage(content=instructions), HumanMessage(content="Write the report.")]
    )
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    sections = "\n\n".join(state["sections"])
    instructions = intro_conclusion_instructions.format(
        topic=state["topic"], formatted_str_sections=sections
    )
    intro = llm.invoke(
        [SystemMessage(content=instructions), HumanMessage(content="Write the introduction.")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    sections = "\n\n".join(state["sections"])
    instructions = intro_conclusion_instructions.format(
        topic=state["topic"], formatted_str_sections=sections
    )
    conclusion = llm.invoke(
        [SystemMessage(content=instructions), HumanMessage(content="Write the conclusion.")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    content = state["content"].strip()
    if content.startswith("## Insights"):
        content = content.removeprefix("## Insights").lstrip()
    body, separator, sources = content.partition("\n## Sources\n")
    final_report = (
        f"{state['introduction']}\n\n---\n\n{body}\n\n---\n\n{state['conclusion']}"
    )
    if separator and sources.strip():
        final_report += f"\n\n## Sources\n{sources.strip()}"
    return {"final_report": final_report}
