from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # ---- LLM -----------------------------------------------------------------------------------
    openai_api_key: str               # your OpenAI secret key
    llm_model: str = "gpt-4o-mini"   # Which model to use for answering questions

    # ---- Embeddings ----------------------------------------------------------------------------
    # "huggingface" = free, run on your CPU, 384-dimensional vectors
    # "openai"      = paid, runs in the cloud, 1536-dimensional vectors
    # Important: once you index documents with one provider, you cannot
    # switch without deleting the collection and re-indexing from scratch.
    embedding_provider: str = "openai"

    # ---- Qdrant --------------------------------------------------------------------------------
    qdrant_url: str                          # e.g. https://0a547433-5701-4785-bd...
    qdrant_api_key: str                      # From Qdrant cloud dashboard
    qdrant_collection: str = "rag_uploads"  # Name of the collection we create/use

    # ---- File upload ---------------------------------------------------------------------------
    upload_dir: str = "uploads_tmp"  # Temporary folder for files while being processed
    max_upload_size_mb: int = 50     # Reject files larger than this (50MB default)

    # ✅ FIX 1: Was 'class config' (lowercase) — Pydantic requires 'class Config'
    class Config:
        env_file = ".env"          # Read from .env file in the project root
        env_file_encoding = "utf-8"
        extra = "ignore"           # Silently ignore unknown env vars

# ✅ FIX 2: Was INSIDE the class body — moved here to module level
settings = Settings()
