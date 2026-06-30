# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation grounds a language model's answers in an external
knowledge base instead of relying only on what the model memorized during training.
At query time the system retrieves the most relevant passages from a document store
and passes them to the model as context, so the model can cite real sources and stay
current without being retrained.

## Why hybrid retrieval

Dense vector search embeds text into vectors and finds passages with similar meaning,
so it matches paraphrases ("automobile" against "car"). It can miss exact tokens such
as a part number or an acronym. Sparse keyword search such as BM25 ranks passages by
exact term overlap, so it nails those literal matches but misses paraphrase. Hybrid
retrieval runs both and fuses the results, which is more robust than either alone.

## Reranking

First-pass retrieval is fast but approximate. A cross-encoder reranker reads the query
and a candidate passage together and scores how well the passage answers the query.
It is more accurate than the bi-encoder used for first-pass search, but it must run
once per candidate, so it is only applied to a small shortlist.

## Reducing hallucinations

A self-checking step improves faithfulness. After drafting an answer, a separate pass
verifies that each claim is supported by the retrieved passages; unsupported claims are
sent back for revision. Evaluation frameworks such as RAGAS measure faithfulness,
answer relevancy, and context precision so regressions are caught before release.
