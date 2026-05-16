import json
import logging
import os
from pathlib import Path
from typing import Annotated, AsyncGenerator, List

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.config import settings
from app.file_loader import is_supported, load_file
from app.markdown_generator import generate_markdown, save_markdown, save_original
from app.models import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    FileUploadResult,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.rag_chain import (
    compute_doc_id,
    delete_indexed_document,
    index_documents,
    list_indexed_documents,
    query_with_references,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state — populated once at startup by lifespan in main.py
# ─────────────────────────────────────────────────────────────────────────────
app_state: dict = {}

def get_client() -> QdrantClient:
    return app_state["qdrant_client"]

def get_emb():
    return app_state["embeddings"]

def get_llm() -> ChatOpenAI:
    return app_state["llm"]


# =============================================================================
# UPLOAD HANDLER
# =============================================================================

async def upload_files_handler(
    files      : Annotated[List[UploadFile], File(description="PDF, DOCX, MD, CSV, TXT")],
    client     : QdrantClient = Depends(get_client),
    embeddings                = Depends(get_emb),
) -> UploadResponse:
    """
    For each file:
      1. Validate type & size
      2. Save original bytes permanently  →  uploads_original/{doc_id}_{name}
      3. Load with LangChain loaders
      4. Generate markdown               →  markdown_files/{doc_id}.md
      5. Chunk + embed + upsert to Qdrant
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: List[FileUploadResult] = []
    docs_to_index: dict = {}

    # ── Phase 1: validate & load ──────────────────────────────────────────────
    for upload in files:
        file_name = upload.filename or "unknown"
        doc_id    = compute_doc_id(file_name)

        if not is_supported(file_name):
            results.append(FileUploadResult(
                file_name=file_name, doc_id=doc_id,
                file_type=Path(file_name).suffix.lstrip("."),
                status="error", chunks_created=0,
                error=f"Unsupported type '{Path(file_name).suffix}'. Accepted: PDF, DOCX, MD, TXT, CSV",
            ))
            continue

        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)

        if size_mb > settings.max_upload_size_mb:
            results.append(FileUploadResult(
                file_name=file_name, doc_id=doc_id,
                file_type=Path(file_name).suffix.lstrip("."),
                status="error", chunks_created=0,
                error=f"File too large ({size_mb:.1f} MB). Max: {settings.max_upload_size_mb} MB",
            ))
            continue

        # ── Save original file permanently ────────────────────────────────────
        save_original(content, doc_id, file_name, settings.uploads_original_dir)

        tmp_path = os.path.join(settings.upload_dir, f"{doc_id}_{file_name}")
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)

            docs = load_file(tmp_path, file_name)
            docs_to_index[file_name] = docs
            logger.info(f"Loaded '{file_name}': {len(docs)} doc(s)")

            # ── Generate and save markdown ────────────────────────────────────
            md_content = generate_markdown(docs, file_name)
            save_markdown(md_content, doc_id, settings.markdown_files_dir)

        except Exception as e:
            results.append(FileUploadResult(
                file_name=file_name, doc_id=doc_id,
                file_type=Path(file_name).suffix.lstrip("."),
                status="error", chunks_created=0, error=str(e),
            ))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Phase 2: embed & index ────────────────────────────────────────────────
    if docs_to_index:
        try:
            index_results = index_documents(
                docs_by_file    = docs_to_index,
                client          = client,
                embeddings      = embeddings,
                collection_name = settings.qdrant_collection,
            )
            for file_name, info in index_results.items():
                results.append(FileUploadResult(
                    file_name     = file_name,
                    doc_id        = info["doc_id"],
                    file_type     = Path(file_name).suffix.lstrip("."),
                    status        = "indexed",
                    chunks_created= info["chunks_created"],
                ))
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return UploadResponse(
        total_files = len(files),
        successful  = sum(1 for r in results if r.status == "indexed"),
        failed      = sum(1 for r in results if r.status == "error"),
        results     = results,
    )


# =============================================================================
# QUERY HANDLER  (standard — full answer at once)
# =============================================================================

def query(
    request   : QueryRequest,
    client    : QdrantClient = Depends(get_client),
    embeddings               = Depends(get_emb),
    llm       : ChatOpenAI   = Depends(get_llm),
) -> QueryResponse:
    logger.info(f"Query: {request.question} (top_k={request.top_k})")
    try:
        result = query_with_references(
            question        = request.question,
            top_k           = request.top_k,
            client          = client,
            embeddings      = embeddings,
            llm             = llm,
            collection_name = settings.qdrant_collection,
        )
        return QueryResponse(
            question           = request.question,
            answer             = result["answer"],
            references         = result["references"],
            embedding_provider = settings.embedding_provider,
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STREAMING QUERY HANDLER
# =============================================================================

async def _stream_generator(
    question       : str,
    top_k          : int,
    client         : QdrantClient,
    embeddings,
    llm            : ChatOpenAI,
    collection_name: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields Server-Sent Events (SSE).

    Event types:
      data: {"type": "token",     "content": "<chunk>"}   ← one per LLM token
      data: {"type": "references","content": [...]}        ← after answer is done
      data: {"type": "done"}                               ← stream finished
    """
    from app.rag_chain import query_with_references_context  # helper (see rag_chain.py)

    try:
        # 1. Retrieve context chunks (synchronous — fast)
        context, references = query_with_references_context(
            question        = question,
            top_k           = top_k,
            client          = client,
            embeddings      = embeddings,
            collection_name = collection_name,
        )

        # 2. Build prompt
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        # 3. Stream LLM tokens
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # 4. Send references after answer completes
        refs_data = [
            {
                "file_name"      : r.file_name,
                "file_type"      : r.file_type,
                "page_number"    : r.page_number,
                "chunk_index"    : r.chunk_index,
                "chunk_text"     : r.chunk_text,
                "relevance_score": r.relevance_score,
            }
            for r in references
        ]
        yield f"data: {json.dumps({'type': 'references', 'content': refs_data})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


async def query_stream(
    request   : QueryRequest,
    client    : QdrantClient = Depends(get_client),
    embeddings               = Depends(get_emb),
    llm       : ChatOpenAI   = Depends(get_llm),
) -> StreamingResponse:
    """
    Stream the answer token-by-token using Server-Sent Events.
    Connect with EventSource in the browser or curl --no-buffer.
    """
    logger.info(f"Stream query: {request.question} (top_k={request.top_k})")
    return StreamingResponse(
        _stream_generator(
            question        = request.question,
            top_k           = request.top_k,
            client          = client,
            embeddings      = embeddings,
            llm             = llm,
            collection_name = settings.qdrant_collection,
        ),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control"  : "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering if behind proxy
        },
    )


# =============================================================================
# GET MARKDOWN FILE
# =============================================================================

def get_file_markdown(doc_id: str) -> Response:
    """Return the markdown version of an indexed document."""
    md_path = os.path.join(settings.markdown_files_dir, f"{doc_id}.md")
    if not os.path.exists(md_path):
        raise HTTPException(
            status_code=404,
            detail=f"Markdown file not found for doc_id '{doc_id}'. "
                   "Was this file uploaded after markdown support was added?"
        )
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="text/markdown; charset=utf-8")


# =============================================================================
# GET ORIGINAL FILE
# =============================================================================

def get_original_file(doc_id: str) -> FileResponse:
    """Return the original uploaded file as a download."""
    original_dir = settings.uploads_original_dir
    if not os.path.isdir(original_dir):
        raise HTTPException(status_code=404, detail="Original files directory not found.")

    # Files are stored as  {doc_id}_{original_filename}
    matches = [f for f in os.listdir(original_dir) if f.startswith(f"{doc_id}_")]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"Original file not found for doc_id '{doc_id}'."
        )

    file_path     = os.path.join(original_dir, matches[0])
    original_name = matches[0][len(doc_id) + 1:]   # strip "{doc_id}_" prefix
    return FileResponse(
        path         = file_path,
        filename     = original_name,
        media_type   = "application/octet-stream",
    )


# =============================================================================
# LIST DOCUMENTS
# =============================================================================

def list_documents(client: QdrantClient = Depends(get_client)) -> DocumentListResponse:
    """List all files currently indexed in Qdrant (one row per file)."""
    try:
        docs = list_indexed_documents(client, settings.qdrant_collection)
        return DocumentListResponse(
            documents = [DocumentInfo(**d) for d in docs],
            total     = len(docs),
        )
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DELETE DOCUMENT
# =============================================================================

def delete_document(
    doc_id: str,
    client: QdrantClient = Depends(get_client),
) -> DeleteResponse:
    """Remove a file and ALL its chunks from Qdrant, plus local markdown/original files."""
    try:
        all_docs = list_indexed_documents(client, settings.qdrant_collection)
        doc_info = next((d for d in all_docs if d["doc_id"] == doc_id), None)

        n_deleted = delete_indexed_document(doc_id, client, settings.qdrant_collection)
        if n_deleted == 0:
            raise HTTPException(status_code=404, detail=f"No document found with doc_id: {doc_id}")

        file_name = doc_info["file_name"] if doc_info else doc_id

        # Also remove local files if they exist
        md_path = os.path.join(settings.markdown_files_dir, f"{doc_id}.md")
        if os.path.exists(md_path):
            os.unlink(md_path)

        original_dir = settings.uploads_original_dir
        if os.path.isdir(original_dir):
            for f in os.listdir(original_dir):
                if f.startswith(f"{doc_id}_"):
                    os.unlink(os.path.join(original_dir, f))

        logger.info(f"Deleted '{file_name}': {n_deleted} chunks removed")
        return DeleteResponse(
            status        = "deleted",
            doc_id        = doc_id,
            file_name     = file_name,
            chunks_deleted= n_deleted,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health() -> dict:
    return {
        "status"            : "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_model"         : settings.llm_model,
        "collection"        : settings.qdrant_collection,
    }
