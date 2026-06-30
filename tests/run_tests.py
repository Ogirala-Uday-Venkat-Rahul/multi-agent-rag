"""End-to-end smoke test that needs no API key.

Exercises the real local pipeline (embeddings, Chroma, BM25, RRF, cross-encoder
rerank) against the sample doc, then runs the full LangGraph with a stubbed LLM so
the agent wiring and the critic retry loop are tested without any provider call.

Run:  python -m tests.run_tests
"""

from __future__ import annotations

import os
import tempfile

# Configure an isolated store BEFORE importing app modules (settings reads env at import).
_TMP = tempfile.mkdtemp(prefix="rag-test-")
os.environ["CHROMA_PATH"] = os.path.join(_TMP, "chroma")
os.environ["COLLECTION_NAME"] = "test_docs"
os.environ["VECTOR_STORE"] = "chroma"
os.environ["LLM_PROVIDER"] = "groq"  # never actually called — we stub the LLM

import app.agents.nodes as nodes  # noqa: E402
from app.agents.graph import answer_question  # noqa: E402
from app.ingest import ingest_path  # noqa: E402
from app.retrieval import Retriever  # noqa: E402
from app.vectorstore import get_store  # noqa: E402

_CRITIC_ALWAYS_FAILS = False


def _fake_complete(system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """Deterministic stand-in for llm.complete, branching on each node's system prompt."""
    if "decompose" in system:
        return '["What is hybrid retrieval?", "Why combine vector search and BM25?"]'
    if "fact-checker" in system:
        if _CRITIC_ALWAYS_FAILS:
            return '{"faithful": false, "score": 0.2, "critique": "Claim [2] is unsupported."}'
        return '{"faithful": true, "score": 0.95, "critique": ""}'
    # synthesizer
    return "Hybrid retrieval combines vector search for meaning and BM25 for exact terms [1]."


nodes.complete = _fake_complete  # monkeypatch the name nodes imported


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    # 1. Ingestion
    added = ingest_path("data/sample")
    check("ingest sample docs", added > 0, f"{added} chunks")
    check("store count matches", get_store().count() == added)

    # 2. Hybrid retrieval with real local models (no API key)
    retriever = Retriever(get_store())
    hits = retriever.retrieve("why use both vector search and keyword search?")
    check("retrieval returns results", len(hits) > 0, f"{len(hits)} chunks")
    check("retrieval respects top_k_rerank", len(hits) <= 4)
    check("chunks carry text + source", all(h.text and h.source for h in hits))
    check("reranker scored chunks", any(h.score != 0.0 for h in hits))

    # 3. Full graph, happy path (stubbed LLM)
    result = answer_question("Why does hybrid retrieval combine vector search and BM25?")
    check("graph returns an answer", bool(result["answer"]))
    check("planner produced sub-questions", len(result["sub_questions"]) >= 1)
    check("answer is marked faithful", result["faithful"] is True)
    check("citations present", len(result["citations"]) > 0)
    check("iterations within budget", result["iterations"] >= 1)

    # 4. Critic retry loop terminates when an answer can't be grounded
    global _CRITIC_ALWAYS_FAILS
    _CRITIC_ALWAYS_FAILS = True
    result2 = answer_question("An unanswerable question with no grounding.")
    check("loop is bounded (returns despite failing critic)", bool(result2["answer"]))
    check("hit max iterations", result2["iterations"] >= 2, f"{result2['iterations']} iters")
    check("unfaithful answer flagged", result2["faithful"] is False)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
