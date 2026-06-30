# Architecture

## Layers

```
app/
  config.py        env-driven settings, loaded once
  schema.py        Chunk dataclass shared across the pipeline
  llm.py           provider abstraction (groq/gemini/openai/anthropic)
  embeddings.py    local embedder + cross-encoder reranker (cached)
  vectorstore.py   ChromaStore / PgVectorStore behind one interface
  ingest.py        load + chunk documents, write to the store
  retrieval.py     hybrid search (vector + BM25) + RRF + rerank
  agents/
    state.py       the typed state passed through the graph
    nodes.py       planner / retrieve / synthesize / critic
    graph.py       LangGraph wiring + answer_question()
  eval/
    evaluate.py    RAGAS-style LLM-judge harness
  main.py          FastAPI surface
```

The design principle: each external dependency (LLM vendor, vector backend, embedding
model) sits behind a small interface, so swapping one never touches the agent logic.
That is what makes the same codebase run on a free stack or a paid one by changing
environment variables only.

## The pipeline, stage by stage

**Planning.** A hard question often bundles several sub-questions. Answering each
separately retrieves more targeted evidence than one broad query would. The planner is
an LLM call that returns a JSON array of sub-questions; if it fails or the question is
simple, it degrades to the original question unchanged.

**Retrieval.** For each sub-question:

1. *Vector search* over the store finds semantically similar chunks (handles paraphrase).
2. *BM25* over the same corpus finds exact-term matches (handles acronyms, identifiers).
3. *Reciprocal Rank Fusion* merges the two ranked lists. RRF is rank-based, so it doesn't
   need the two retrievers' scores to share a scale — they don't.
4. A *cross-encoder reranker* reads each shortlisted chunk together with the query and
   reorders by true relevance. It is accurate but per-candidate expensive, so it only
   sees the fused shortlist, never the whole corpus.

Evidence is deduped by chunk id across sub-questions before synthesis.

**Synthesis.** The model is given the numbered sources and instructed to answer using
only those sources, citing each claim with `[n]`. If a previous draft was rejected, the
critic's feedback is included so the rewrite targets the specific problem.

**Critique (the self-correcting loop).** A separate LLM pass judges whether every claim
in the draft is supported by the evidence and returns a faithfulness score. Below the
threshold, the graph loops back to synthesis with the critique attached. The loop is
bounded by `MAX_ITERATIONS` so a stubbornly ungroundable question still returns the best
draft (with its critique) rather than looping forever.

## Why these components

- **LangGraph** models the pipeline as an explicit state machine with a conditional edge.
  The critic→synthesize loop is a cycle, which a plain linear chain can't express cleanly.
- **ChromaDB → pgvector.** Chroma is embedded and needs zero setup, which is ideal for
  local development and demos. pgvector is the production target: it lives in a Postgres
  you already operate (Supabase/Neon free tier), so retrieval and your relational data
  share one database and one backup story.
- **Local embeddings + reranker.** Running `sentence-transformers` in-process keeps the
  hot path free and removes a network hop. The trade-off is RAM and CPU, which is why the
  deployment target is Hugging Face Spaces (16 GB) rather than a 512 MB free dyno.
- **Provider abstraction.** The resume version uses Claude / GPT-4; the free version uses
  Groq or Gemini. Because every model call goes through `llm.complete()`, switching is a
  one-line env change and the agent code is identical either way.

## What I'd change at higher scale

- Push BM25 into Postgres full-text search or OpenSearch so the corpus doesn't have to be
  loaded into memory per request.
- Cache embeddings and rerank scores for repeated queries.
- Stream the synthesis step to the client for lower perceived latency.
- Persist eval runs and gate CI on a faithfulness regression threshold.
