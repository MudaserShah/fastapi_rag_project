from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # ---- LLM -----------------------------------------------------------------------------------
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"

    # ---- Embeddings ----------------------------------------------------------------------------
    embedding_provider: str = "openai"

    # ---- Qdrant --------------------------------------------------------------------------------
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "rag_uploads"

    # ---- File Storage --------------------------------------------------------------------------
    upload_dir: str = "uploads_tmp"          # Temporary — deleted after processing
    uploads_original_dir: str = "uploads_original"  # ✅ NEW: permanent original files
    markdown_files_dir: str = "markdown_files"       # ✅ NEW: markdown versions
    max_upload_size_mb: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
