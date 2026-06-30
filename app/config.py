"""Central configuration, loaded from environment / .env once at startup."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider
    llm_provider: str = "groq"
    llm_model: str = ""  # blank -> provider default (see DEFAULT_MODELS)
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Local models (CPU, free)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Vector store
    vector_store: str = "chroma"  # "chroma" | "pgvector"
    chroma_path: str = "./.chroma"
    database_url: str = ""
    collection_name: str = "documents"

    # Pipeline knobs
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k_vector: int = 10
    top_k_bm25: int = 10
    top_k_rerank: int = 4
    max_subquestions: int = 4
    max_iterations: int = 2
    faithfulness_threshold: float = 0.7


settings = Settings()

# Sensible per-provider defaults so the user only sets a key, not a model id.
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-4-8",
}


def resolved_model() -> str:
    """The model id to call: explicit override, else the provider default."""
    return settings.llm_model or DEFAULT_MODELS.get(settings.llm_provider.lower(), "")
