# LLM Agents and Orchestration

An LLM agent is a system that uses a language model to decide what to do across multiple
steps, rather than producing a single response. A common pattern decomposes a task into
stages — planning, acting, and checking — where each stage has its own prompt and role.

## Why split work into roles

Giving each stage a narrow job makes the system easier to reason about and to evaluate.
A planner that only decomposes a question, a retriever that only gathers evidence, a
writer that only drafts an answer, and a verifier that only checks it can each be prompted
and tested in isolation. The verifier can return control to the writer when an answer is
wrong, forming a loop that a single straight-through call cannot express.

## State machines

Orchestration frameworks model this flow as a graph of nodes connected by edges, with a
shared state object passed from node to node. A conditional edge lets the graph branch —
for example, ending when an answer passes verification or looping back when it fails. A
retry budget bounds the loop so it cannot run forever.

## Grounding and verification

Agents that answer from documents are kept honest by grounding every claim in retrieved
evidence and by a separate verification step that checks each claim against that evidence.
This is what reduces fabricated answers in retrieval-augmented systems.
