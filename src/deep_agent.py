from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from src.analyst_research import aconduct_research, conduct_research
from src.utils.edges import initiate_all_research
from src.utils.nodes import (
    create_analysts,
    finalize_report,
    human_feedback,
    write_conclusion,
    write_introduction,
    write_report,
)
from src.utils.states import ResearchGraphState, ResearchInput


builder = StateGraph(ResearchGraphState, input_schema=ResearchInput)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
# Sync and async variants: under the LangGraph server the async one runs every
# analyst on a dedicated executor so all of them progress in parallel.
builder.add_node(
    "conduct_research",
    RunnableLambda(conduct_research, afunc=aconduct_research, name="conduct_research"),
)
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback",
    initiate_all_research,
    {"create_analysts": "create_analysts", "conduct_research": "conduct_research"},
)
builder.add_edge("conduct_research", "write_report")
builder.add_edge("conduct_research", "write_introduction")
builder.add_edge("conduct_research", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

graph = builder.compile()
