"""Wire the four agents into a LangGraph state machine.

    planner -> retrieve -> synthesize -> critic --(faithful?)--> END
                               ^                     |
                               +-----(retry)---------+

The conditional edge after the critic is the self-correcting loop.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from . import nodes
from .state import GraphState


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("planner", nodes.planner)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("critic", nodes.critic)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "critic")
    graph.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {"retry": "synthesize", "end": END},
    )

    return graph.compile()


def answer_question(question: str) -> dict:
    """Run the full pipeline for one question and return a JSON-friendly result."""
    final_state = build_graph().invoke({"question": question})
    return {
        "question": question,
        "answer": final_state.get("answer", ""),
        "sub_questions": final_state.get("sub_questions", []),
        "citations": final_state.get("citations", []),
        "faithful": final_state.get("faithful", False),
        "iterations": final_state.get("iterations", 0),
        "critique": final_state.get("critique", ""),
    }
