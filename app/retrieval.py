"""Hybrid retrieval: semantic (vector) + keyword (BM25), fused, then reranked.

Why hybrid: vector search captures meaning ("car" ~ "automobile") but misses exact
terms (a part number, an acronym); BM25 nails exact terms but misses paraphrase.
Fusing both with Reciprocal Rank Fusion gives a candidate set that is strong on
both axes, and the cross-encoder rerank then picks the few chunks that actually
answer the query.
"""

from __future__ import annotations

import re

from .config import settings
from .embeddings import rerank
from .schema import Chunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _reciprocal_rank_fusion(result_lists: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """Combine ranked lists by summing 1/(k + rank). Rank-based, so it doesn't need
    the two retrievers' scores to be on the same scale (they aren't)."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[chunk.id] = chunk
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [by_id[cid] for cid in ordered_ids]


class Retriever:
    """Built once per request from the current store snapshot.

    BM25 needs the whole corpus in memory to compute term statistics, so we pull
    `all_documents()` up front. For a small/medium knowledge base that's fine; at
    larger scale you'd push keyword search into Postgres full-text or OpenSearch.
    """

    def __init__(self, store) -> None:
        self._store = store
        self._docs = store.all_documents()
        self._bm25 = None
        if self._docs:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(d.text) for d in self._docs])

    def _keyword_search(self, query: str, k: int) -> list[Chunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def retrieve(self, query: str) -> list[Chunk]:
        vector_hits = self._store.query(query, settings.top_k_vector)
        keyword_hits = self._keyword_search(query, settings.top_k_bm25)
        fused = _reciprocal_rank_fusion([vector_hits, keyword_hits])
        shortlist = fused[: settings.top_k_vector + settings.top_k_bm25]
        reranked = rerank(query, shortlist)
        return reranked[: settings.top_k_rerank]
