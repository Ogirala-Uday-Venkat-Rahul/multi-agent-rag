"""Vector store backends behind a common interface.

Two backends, selected by VECTOR_STORE:
  - chroma   : embedded, file-backed, zero setup. Good for local dev and demos.
  - pgvector : Postgres + pgvector. What you'd point at a Supabase/Neon free tier
               in production (and what the resume describes the deployed version using).

Both expose: add(chunks), query(text, k), all_documents(), count().
`query` does vector (semantic) search; keyword search lives in retrieval.py so the
hybrid logic is independent of which backend stores the vectors.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings
from .embeddings import embed_query, embed_texts, embedding_dim
from .schema import Chunk


class ChromaStore:
    def __init__(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._col = self._client.get_or_create_collection(
            settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        # upsert (not add) so re-ingesting a changed doc updates it instead of
        # warning-and-ignoring an existing id.
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed_texts([c.text for c in chunks]),
            metadatas=[c.metadata or {} for c in chunks],
        )

    def query(self, text: str, k: int) -> list[Chunk]:
        res = self._col.query(query_embeddings=[embed_query(text)], n_results=k)
        out: list[Chunk] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for i in range(len(ids)):
            # cosine distance -> similarity
            out.append(Chunk(id=ids[i], text=docs[i], metadata=metas[i] or {}, score=1.0 - dists[i]))
        return out

    def all_documents(self) -> list[Chunk]:
        res = self._col.get()
        return [
            Chunk(id=res["ids"][i], text=res["documents"][i], metadata=res["metadatas"][i] or {})
            for i in range(len(res["ids"]))
        ]

    def count(self) -> int:
        return self._col.count()


class PgVectorStore:
    def __init__(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        if not settings.database_url:
            raise ValueError("VECTOR_STORE=pgvector requires DATABASE_URL")
        self._table = settings.collection_name
        self._conn = psycopg.connect(settings.database_url, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "  id text PRIMARY KEY,"
            "  text text NOT NULL,"
            "  metadata jsonb NOT NULL DEFAULT '{}',"
            f"  embedding vector({embedding_dim()})"
            ")"
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        from psycopg.types.json import Jsonb

        embeddings = embed_texts([c.text for c in chunks])
        with self._conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    f"INSERT INTO {self._table} (id, text, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                    (chunk.id, chunk.text, Jsonb(chunk.metadata or {}), emb),
                )

    def query(self, text: str, k: int) -> list[Chunk]:
        query_emb = embed_query(text)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS similarity "
                f"FROM {self._table} ORDER BY embedding <=> %s LIMIT %s",
                (query_emb, query_emb, k),
            )
            rows = cur.fetchall()
        return [Chunk(id=r[0], text=r[1], metadata=r[2] or {}, score=float(r[3])) for r in rows]

    def all_documents(self) -> list[Chunk]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT id, text, metadata FROM {self._table}")
            rows = cur.fetchall()
        return [Chunk(id=r[0], text=r[1], metadata=r[2] or {}) for r in rows]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return int(cur.fetchone()[0])


@lru_cache(maxsize=1)
def get_store():
    """The configured store, built once and reused."""
    backend = settings.vector_store.lower()
    if backend == "chroma":
        return ChromaStore()
    if backend == "pgvector":
        return PgVectorStore()
    raise ValueError(f"Unknown VECTOR_STORE: {backend!r}")
