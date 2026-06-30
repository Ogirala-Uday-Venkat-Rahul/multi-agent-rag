---
title: Multi-Agent RAG Research Assistant
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Agent RAG Research Assistant

A research assistant that breaks a hard question into parts, retrieves evidence for
each part with hybrid search and cross-encoder reranking, writes an answer that cites
its sources, and self-checks that answer for faithfulness before returning it.

Built to run end to end on free infrastructure: local embedding and reranking models
(CPU, no API key), a free LLM provider (Groq or Gemini), and either an embedded
ChromaDB store or Postgres + pgvector.

## How it works

```
question
   │
   ▼
 planner ──► retrieve ──► synthesize ──► critic ──(faithful?)──► answer + citations
                            ▲                │
                            └───── retry ────┘   (loops if claims aren't grounded)
```

- **planner** decomposes the question into focused sub-questions.
- **retrieve** runs hybrid search (vector + BM25), fuses with Reciprocal Rank Fusion,
  and reranks the shortlist with a cross-encoder. Evidence is deduped across sub-questions.
- **synthesize** writes an answer grounded only in the retrieved passages, with `[n]` citations.
- **critic** checks every claim against the evidence; an unfaithful draft loops back to
  synthesize until it passes or the retry budget runs out.

The agents are wired together with LangGraph; the LLM and vector store are pluggable
behind small interfaces.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then set LLM_PROVIDER and the matching API key

python -m scripts.ingest_docs data/sample           # index the sample doc
uvicorn app.main:app --reload                        # serve the API
```

Ask a question:

```bash
curl -X POST localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "Why does hybrid retrieval combine vector search and BM25?"}'
```

## Configuration

All configuration is environment-driven (see `.env.example`). The defaults are chosen
to run for free:

| Setting | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` / `gemini` (free) or `openai` / `anthropic` (paid) |
| `VECTOR_STORE` | `chroma` | `chroma` (local) or `pgvector` (set `DATABASE_URL`) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | runs locally on CPU |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | cross-encoder, runs locally on CPU |

## Evaluation

```bash
python -m app.eval.evaluate data/golden.json
```

Reports faithfulness, answer relevancy, and context precision per question plus an
aggregate — RAGAS-style metrics computed with the configured LLM as judge.

## Free hosting

- **LLM** — Groq or Google Gemini free tier
- **Vector store** — Supabase or Neon free Postgres (both ship pgvector)
- **App** — Hugging Face Spaces or Render free tier
- **Embeddings / reranker** — run in-process, no external service

See `docs/ARCHITECTURE.md` for the design and the rationale behind each choice.
