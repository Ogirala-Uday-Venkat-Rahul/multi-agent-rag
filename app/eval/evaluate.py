"""A small RAGAS-style evaluation harness using an LLM as judge.

The metrics mirror what RAGAS reports, computed here with the same LLM provider so
the whole thing stays free and dependency-light:

  - faithfulness        : are the answer's claims grounded in the retrieved context?
  - answer_relevancy    : does the answer actually address the question?
  - context_precision   : was the retrieved context relevant (not padding)?

Run against a golden set (questions + optional reference answers). In production you
would swap this judge for the RAGAS library; the contract (a score per metric per
example, plus an aggregate) is the same, which is what lets you gate CI on it.
"""

from __future__ import annotations

import json
import re

from ..agents.graph import build_graph
from ..llm import complete
from ..schema import Chunk

JUDGE_SYSTEM = (
    "You are an evaluation judge for a RAG system. Score each metric from 0.0 to 1.0. "
    'Return ONLY JSON: {"faithfulness": x, "answer_relevancy": x, "context_precision": x}.'
)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _score_example(question: str, answer: str, contexts: list[Chunk]) -> dict:
    context_block = "\n\n".join(f"- {c.text}" for c in contexts)
    user = (
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRetrieved context:\n{context_block}\n\n"
        "faithfulness = fraction of answer claims supported by the context. "
        "answer_relevancy = how well the answer addresses the question. "
        "context_precision = fraction of the retrieved context that is relevant."
    )
    raw = complete(JUDGE_SYSTEM, user, max_tokens=200)
    parsed = _extract_json(raw)
    return {
        "faithfulness": float(parsed.get("faithfulness", 0.0)),
        "answer_relevancy": float(parsed.get("answer_relevancy", 0.0)),
        "context_precision": float(parsed.get("context_precision", 0.0)),
    }


def evaluate(golden: list[dict]) -> dict:
    """golden: list of {"question": str}. Returns per-example scores + aggregates."""
    graph = build_graph()
    per_example = []
    for item in golden:
        state = graph.invoke({"question": item["question"]})
        scores = _score_example(
            item["question"], state.get("answer", ""), state.get("contexts", [])
        )
        per_example.append({"question": item["question"], **scores})

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    aggregate = {
        m: round(sum(e[m] for e in per_example) / len(per_example), 3) for m in metrics
    } if per_example else {}
    return {"aggregate": aggregate, "examples": per_example}


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/golden.json"
    with open(path, encoding="utf-8") as f:
        golden_set = json.load(f)
    report = evaluate(golden_set)
    print(json.dumps(report, indent=2))
