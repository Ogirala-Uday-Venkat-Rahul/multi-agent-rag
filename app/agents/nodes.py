"""The four agents, one per node.

planner    -> break a hard question into focused sub-questions
retrieve   -> hybrid-retrieve evidence for each sub-question, dedupe
synthesize -> write a cited answer grounded only in the evidence
critic     -> self-check faithfulness; on failure the graph loops back to synthesize

The critic loop is what raises faithfulness and cuts made-up citations: a separate
pass judges the draft against the evidence before anything is returned.
"""

from __future__ import annotations

import json
import re

from ..config import settings
from ..llm import complete
from ..retrieval import Retriever
from ..schema import Chunk
from ..vectorstore import get_store
from .state import GraphState


def _extract_json(text: str):
    """LLMs wrap JSON in prose or fences; pull out the first {...} or [...]."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# --- planner -------------------------------------------------------------------

PLANNER_SYSTEM = (
    "You decompose a research question into a few focused sub-questions that can each "
    "be answered from a document search. Return ONLY a JSON array of strings. If the "
    "question is already simple, return it unchanged as a single-element array."
)


def planner(state: GraphState) -> GraphState:
    question = state["question"]
    raw = complete(
        PLANNER_SYSTEM,
        f"Question: {question}\nReturn at most {settings.max_subquestions} sub-questions.",
        max_tokens=400,
    )
    parsed = _extract_json(raw)
    if isinstance(parsed, list) and parsed:
        sub_questions = [str(q) for q in parsed][: settings.max_subquestions]
    else:
        sub_questions = [question]  # degrade gracefully to the original question
    return {"sub_questions": sub_questions, "iterations": 0}


# --- retrieve ------------------------------------------------------------------


def retrieve(state: GraphState) -> GraphState:
    retriever = Retriever(get_store())
    seen: dict[str, Chunk] = {}
    for sub_q in state["sub_questions"]:
        for chunk in retriever.retrieve(sub_q):
            seen.setdefault(chunk.id, chunk)  # dedupe across sub-questions
    return {"contexts": list(seen.values())}


# --- synthesize ----------------------------------------------------------------

SYNTH_SYSTEM = (
    "You answer the question using ONLY the numbered sources provided. Cite every claim "
    "with its source number in square brackets, e.g. [1]. If the sources do not contain "
    "the answer, say so plainly rather than guessing. Be concise and direct."
)


def _format_sources(contexts: list[Chunk]) -> str:
    lines = []
    for i, chunk in enumerate(contexts, start=1):
        lines.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n".join(lines)


def synthesize(state: GraphState) -> GraphState:
    contexts = state["contexts"]
    sources_block = _format_sources(contexts)
    user = f"Question: {state['question']}\n\nSources:\n{sources_block}"
    if state.get("critique"):
        user += f"\n\nA previous draft was rejected for this reason: {state['critique']}\nWrite an improved, fully grounded answer."
    answer = complete(SYNTH_SYSTEM, user, max_tokens=800)
    citations = [{"n": i + 1, "source": c.source} for i, c in enumerate(contexts)]
    return {"answer": answer, "citations": citations}


# --- critic --------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You are a strict fact-checker. Given a question, an answer, and the sources the answer "
    "was supposed to use, judge whether every claim in the answer is supported by the sources. "
    'Return ONLY JSON: {"faithful": true|false, "score": 0.0-1.0, "critique": "..."}. '
    "score is the fraction of claims that are supported. Flag any claim not in the sources."
)


def critic(state: GraphState) -> GraphState:
    sources_block = _format_sources(state["contexts"])
    user = (
        f"Question: {state['question']}\n\nAnswer:\n{state['answer']}\n\nSources:\n{sources_block}"
    )
    raw = complete(CRITIC_SYSTEM, user, max_tokens=400)
    parsed = _extract_json(raw) or {}
    score = float(parsed.get("score", 0.0)) if isinstance(parsed, dict) else 0.0
    faithful = bool(parsed.get("faithful", False)) and score >= settings.faithfulness_threshold
    return {
        "faithful": faithful,
        "critique": "" if faithful else str(parsed.get("critique", "Unsupported claims found.")),
        "iterations": state.get("iterations", 0) + 1,
    }


def route_after_critic(state: GraphState) -> str:
    """Loop back to synthesize on an unfaithful answer, until the retry budget runs out."""
    if state.get("faithful"):
        return "end"
    if state.get("iterations", 0) >= settings.max_iterations:
        return "end"  # give up gracefully; return the best draft with its critique attached
    return "retry"
