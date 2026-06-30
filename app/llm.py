"""Thin provider abstraction so the rest of the app never imports a vendor SDK.

Every backend is reached through `complete(system, user)`. Providers are
lazy-imported inside their branch, so you only need the SDK for the one you use.
"""

from __future__ import annotations

from .config import resolved_model, settings


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Single-turn completion. Returns the model's text answer as a plain string."""
    provider = settings.llm_provider.lower()
    model = resolved_model()
    if not model:
        raise ValueError(f"No model resolved for provider '{provider}'")

    if provider == "groq":
        return _groq(system, user, model, max_tokens, temperature)
    if provider == "gemini":
        return _gemini(system, user, model, max_tokens, temperature)
    if provider == "openai":
        return _openai(system, user, model, max_tokens, temperature)
    if provider == "anthropic":
        return _anthropic(system, user, model, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _groq(system: str, user: str, model: str, max_tokens: int, temperature: float) -> str:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _gemini(system: str, user: str, model: str, max_tokens: int, temperature: float) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key or None)
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=temperature,
        # Gemini 2.5 models "think" by spending output tokens on hidden reasoning
        # before the visible answer. For our structured planner/critic JSON that
        # silently truncates the output, so we turn thinking off.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    resp = client.models.generate_content(model=model, contents=user, config=config)
    return resp.text or ""


def _openai(system: str, user: str, model: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic

    # Note: the Opus 4.x family rejects temperature / top_p, so we don't pass them.
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
