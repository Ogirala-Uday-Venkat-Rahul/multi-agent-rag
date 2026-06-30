"""Local embedding and cross-encoder reranking models.

Both load lazily and are cached for the process lifetime. They run on CPU and
need no API key, which is what keeps the deployed system free.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings
from .schema import Chunk


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Normalized so cosine == dot product."""
    vectors = _embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    return _embedder().get_sentence_embedding_dimension()


def rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Re-score chunks against the query with a cross-encoder, best first.

    The cross-encoder reads (query, chunk) together, so it judges relevance far
    more precisely than the bi-encoder vectors used for first-pass retrieval —
    at the cost of running per candidate, which is why it only sees the shortlist.
    """
    if not chunks:
        return []
    pairs = [[query, c.text] for c in chunks]
    scores = _reranker().predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    return sorted(chunks, key=lambda c: c.score, reverse=True)
