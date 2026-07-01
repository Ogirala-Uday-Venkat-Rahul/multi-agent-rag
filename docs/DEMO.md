# Demo Guide — Multi-Agent RAG Research Assistant

A short runbook for presenting the live project. Covers what to open, what to say, and
which questions to ask so the demo shows off the parts that matter.

## Links

- **Web app (present this):** https://ouvrahul-multi-agent-rag.hf.space
- **Interactive API docs (fallback / technical audience):** https://ouvrahul-multi-agent-rag.hf.space/docs
- **Space page:** https://huggingface.co/spaces/ouvrahul/multi-agent-rag (click the **App** tab)
- **Source:** https://github.com/Ogirala-Uday-Venkat-Rahul/multi-agent-rag

## Before you present (2 minutes)

The Space runs on Hugging Face's free tier, which **sleeps after inactivity** and
cold-starts on the next visit (a rebuild plus a one-time model download — slow first hit).

1. Open the live app link a few minutes early so it's awake.
2. Confirm it's healthy: open `.../health` — it should return
   `{"status":"ok", ...,"indexed_chunks":3}`. The app auto-seeds the sample docs on
   startup, so a freshly-woken Space is already populated. (If you ever see `0`, it's
   still finishing boot — give it a few seconds and refresh.)
3. Leave the web app open. That's the interface you'll drive.

## What this is, in one sentence

A research assistant that breaks a hard question into parts, retrieves evidence for each
with hybrid search and reranking, writes an answer that **cites its sources**, and
**self-checks** that answer for faithfulness before returning it.

## The 30-second architecture pitch

> A question comes in. A **planner** splits it into focused sub-questions. For each one,
> **hybrid retrieval** — dense vector search for meaning plus BM25 keyword search for exact
> terms, fused with Reciprocal Rank Fusion and reranked by a cross-encoder — pulls the few
> most relevant passages. A **synthesizer** writes an answer grounded only in those
> passages, with `[n]` citations. Then a **critic** checks every claim against the evidence
> and, if anything isn't supported, sends it back for a bounded rewrite. Four cooperating
> roles wired as a graph with a feedback loop — that's the "multi-agent" part.

## Running the demo

On the web app: type a question (or click one of the example chips) and press **Ask**.
The answer renders with a **faithful** badge and iteration count, the sub-questions the
planner asked, and the source list. Cmd/Ctrl+Enter also submits.

(Technical audience? The `/docs` page does the same thing over raw HTTP: expand
**POST `/ask`** -> **Try it out** -> edit the JSON body -> **Execute**.)

### Question 1 — the happy path (shows decomposition + citations)

```json
{ "question": "Why does hybrid retrieval combine vector search and BM25?" }
```

Point at the response and narrate:
- `sub_questions` — the planner decomposed one question into four. Retrieval ran per
  sub-question, not once on the whole thing.
- `answer` + `citations` — every claim carries a `[n]` that maps to a real source file.
- `faithful: true`, `iterations: 1` — the critic verified the answer on the first pass.

### Question 2 — the money shot (shows the faithfulness guardrail)

```json
{ "question": "Why does hybrid retrieval combine vector search and BM25, and how are the results merged?" }
```

The sample documents explain the *why* but not the *how* of merging. Watch the answer:
it grounds the first half with a citation, then says **"the sources do not contain
information on how the results are merged"** rather than inventing it.

Say this out loud: *"The merge method is actually Reciprocal Rank Fusion, and the model
likely knows that from pretraining — but it refused to state what the retrieved evidence
didn't support. It answered the part the sources covered and declined the part they
didn't, in the same response. That's the anti-hallucination guarantee working, not a
prompt trick."*

This is the single best thing to demo — it proves the grounding is real.

### Question 3 — optional, a comparison question

```json
{ "question": "How do embeddings power semantic search, and what are their limitations?" }
```

Shows the assistant handling a two-part question and pulling from more than one source
document.

## If someone asks "why not just one LLM call?"

Because a single call can't do two things this does: **decompose** the question before
retrieving (so each search is focused), and **verify** its own answer and retry if a
claim isn't grounded (the critic -> synthesizer loop). Those are the reasons it's a
graph of roles, not one prompt.

## Honest caveats to have ready (they build credibility)

- **Free-tier latency.** Embeddings and reranking run on a local CPU model, and the LLM
  is a free Gemini tier — slower than a paid GPU/API stack. The architecture is identical
  to a paid deployment; only the env vars and the bill change.
- **Ephemeral storage.** The demo re-indexes sample docs on demand; a production build
  points at a persistent pgvector store (Supabase/Neon free tier ship it).
- **"Multi-agent" is shorthand.** These are cooperating LLM roles in a state machine, not
  autonomous agents running their own tool loops. Said plainly, it lands as honest rather
  than overclaimed.

## Troubleshooting during a live demo

| Symptom | Cause | Fix |
|---|---|---|
| First request hangs ~30–60s | Cold start / model download | Warm it up before presenting |
| `indexed_chunks: 0` | Still seeding on boot | Wait a few seconds and refresh; auto-seeding runs at startup |
| Answer says "not in the sources" | Working as designed | That's the faithfulness guardrail — feature, not bug |
