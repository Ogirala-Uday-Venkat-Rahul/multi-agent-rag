# Deploying to Hugging Face Spaces (free)

The repo ships a `Dockerfile` already set up for a Hugging Face **Docker Space**
(non-root user, port 7860). HF's free CPU tier gives 16 GB RAM, which is what the
local embedding + reranker models need.

> I can't create your HF account or push to it for you — those need your login.
> Everything below is the exact sequence; it's ~5 minutes.

## 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Name it (e.g. `multi-agent-rag`), License optional.
3. **Space SDK: Docker** → **Blank** template.
4. Hardware: **CPU basic (free)**.
5. Create.

This creates a git repo for the Space with a starter `README.md`. The Space README
**must** begin with this frontmatter (HF reads `app_port` from it to route traffic):

```yaml
---
title: Multi-Agent RAG Research Assistant
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

## 2. Add your secret

In the Space: **Settings → Variables and secrets**.

- Add a **secret** `GROQ_API_KEY` = your key. (Or `GOOGLE_API_KEY` if using Gemini.)
- Add a **variable** `LLM_PROVIDER` = `groq` (or `gemini`).

HF injects these as environment variables, which is exactly what `config.py` reads.
Nothing else is required — `VECTOR_STORE` defaults to `chroma`, which works in-container.

## 3. Push the code to the Space

From the project folder (`D:\multi-agent-rag`), add the Space as a remote and push.
Replace `<user>` and `<space>`:

```bash
git init                       # if not already a git repo
git add .
git commit -m "Multi-agent RAG research assistant"

git remote add space https://huggingface.co/spaces/<user>/<space>
git push space main --force    # --force only on the first push, to overwrite the starter README
```

If git asks for a password, use a Hugging Face **access token** (Settings → Access
Tokens → New token, role *write*) as the password.

> Keep the frontmatter from step 1 at the top of the `README.md` you push, or HF won't
> know which port to route to. (The `.dockerignore` keeps `.env` and local caches out of
> the image.)

## 4. Watch it build

The Space rebuilds on push. First build takes a few minutes (installing PyTorch). When
it's running:

- `https://<user>-<space>.hf.space/health` → should return `{"status":"ok", ...}`
- The embedding/reranker models download on the **first** `/ask` or `/ingest`, so the
  first call is slow (~30–60s) and subsequent ones are fast.

## 5. Seed some documents

The container starts with an empty store. Index something:

```bash
curl -X POST https://<user>-<space>.hf.space/ingest \
  -H "content-type: application/json" \
  -d '{"path": "data/sample"}'
```

Then ask:

```bash
curl -X POST https://<user>-<space>.hf.space/ask \
  -H "content-type: application/json" \
  -d '{"question": "Why does hybrid retrieval combine vector search and BM25?"}'
```

## Note on persistence

A free Space's filesystem is **ephemeral** — the Chroma store resets when the Space
restarts, so you'd re-run `/ingest`. For durable storage, switch `VECTOR_STORE=pgvector`
and point `DATABASE_URL` at a free **Supabase** or **Neon** Postgres (both ship pgvector).
That keeps the index across restarts and is the production-shaped setup.

## Alternative: Render

The same `Dockerfile` deploys to Render (or Railway/Fly). Render's free web service is
512 MB RAM, which is too tight once both local models are resident — bump to the smallest
paid instance, or use a hosted embedding API. HF Spaces' 16 GB free tier is the reason
it's the recommended host here.
