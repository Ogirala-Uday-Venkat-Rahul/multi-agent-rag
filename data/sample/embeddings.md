# Embeddings and Vector Search

An embedding model maps a piece of text to a fixed-length vector of numbers so that
texts with similar meaning land close together in vector space. Sentence-transformer
models such as all-MiniLM-L6-v2 produce 384-dimensional vectors and run on a CPU, which
makes them cheap to host.

## Cosine similarity

To find passages similar to a query, the query is embedded and compared against stored
document vectors. Cosine similarity measures the angle between two vectors and ignores
their magnitude, so it compares direction (meaning) rather than length. When vectors are
normalized to unit length, cosine similarity equals their dot product, which is faster to
compute at scale.

## Approximate nearest neighbour

Comparing a query against every stored vector is exact but slow once there are many
documents. Approximate nearest neighbour indexes such as HNSW trade a small amount of
recall for a large speedup, returning the closest vectors in roughly logarithmic time.
Both ChromaDB and pgvector use HNSW-style indexes under the hood.

## Bi-encoder vs cross-encoder

The model that produces these vectors is a bi-encoder: the query and the document are
embedded separately, so document vectors can be precomputed and reused. A cross-encoder
instead reads the query and a candidate document together and outputs a single relevance
score. It is more accurate because it can model interactions between the two texts, but it
cannot be precomputed and must run per candidate at query time.
