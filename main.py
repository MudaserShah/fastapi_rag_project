# main.py
# Run: uvicorn main:app --reload --port 8000

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag_chain import get_embeddings, get_qdrant_client
from app.api import app_state
from app.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model...")
    app_state["embeddings"]    = get_embeddings(settings.embedding_provider)

    logger.info("Connecting to Qdrant...")
    app_state["qdrant_client"] = get_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)

    logger.info("Connecting to OpenAI LLM...")
    app_state["llm"] = ChatOpenAI(
        model       = settings.llm_model,
        temperature = 0,
        api_key     = settings.openai_api_key,
    )

    # Create all required directories
    for directory in [
        settings.upload_dir,
        settings.uploads_original_dir,   # ✅ NEW
        settings.markdown_files_dir,     # ✅ NEW
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Ready | embeddings={settings.embedding_provider} "
        f"| collection={settings.qdrant_collection}"
    )
    yield
    app_state.clear()
    logger.info("Server shut down.")


app = FastAPI(
    title       = "RAG API",
    description = "Upload files → Ask questions → Get answers with source references",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
