"""The state object that flows through the LangGraph nodes.

LangGraph threads one typed dict through every node; each node reads what it needs
and returns the keys it updates. Keeping it explicit here makes the graph readable.
"""

from __future__ import annotations

from typing import TypedDict

from ..schema import Chunk


class GraphState(TypedDict, total=False):
    question: str  # the original user question
    sub_questions: list[str]  # planner output
    contexts: list[Chunk]  # retrieved + reranked evidence
    answer: str  # synthesizer output (with [n] citations)
    citations: list[dict]  # source map for the [n] markers
    critique: str  # critic's feedback when the answer isn't faithful
    faithful: bool  # critic's verdict
    iterations: int  # synthesize/critic loops so far
