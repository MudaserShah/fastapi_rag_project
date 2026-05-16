import logging
import os
from pathlib import Path
from typing import Annotated, List

from fastapi import Depends, File, HTTPException, UploadFile
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.config import settings
from app.file_loader import is_supported, load_file
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

# ✅ FIX 1: Removed wrong imports — 'from grpc import StatusCode' and
#            'from traitlets.utils import descriptions' were unused and not installed.

logger = logging.getLogger(__name__)

# ==========================================================================
# app_state is populated by the lifespan in main.py at startup.
# The three get_* functions below are dependency functions — FastAPI calls
# them automatically via Depends() to inject the shared object into each handler.
# ==========================================================================

app_state: dict = {}  # Holds embeddings, qdrant_client, llm — built once at startup


def get_client() -> QdrantClient:
    return app_state["qdrant_client"]  # The Qdrant connection


def get_emb():
    return app_state["embeddings"]     # The embedding model


def get_llm() -> ChatOpenAI:
    return app_state["llm"]            # The OpenAI LLM


# ==========================================================================
# UPLOAD HANDLER
# ==========================================================================

async def upload_files_handler(
    files      : Annotated[List[UploadFile], File(description="One or more files: PDF, DOCX, MD, CSV, TXT")],
    client     : QdrantClient = Depends(get_client),
    embeddings                = Depends(get_emb),
) -> UploadResponse:
    """
    Upload one or many files.
    Each file is loaded, split into chunks, embedded, and stored in Qdrant.
    Returns a per-file status — one failure does not cancel the rest.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: List[FileUploadResult] = []  # Collects one result per file

    # ✅ FIX 2: Was 'doc_to_index' (typo) — must be 'docs_to_index' everywhere
    docs_to_index: dict = {}              # {file_name: List[Document]}

    # ---- Phase 1: Load each file from the uploaded bytes ----------------
    for upload in files:
        file_name = upload.filename or "unknown"

        # Reject unsupported types before doing any disk I/O
        if not is_supported(file_name):
            results.append(FileUploadResult(
                file_name     = file_name,
                doc_id        = compute_doc_id(file_name),
                file_type     = Path(file_name).suffix.lstrip("."),
                status        = "error",
                chunks_created= 0,
                error         = f"Unsupported type '{Path(file_name).suffix}'. Accepted: PDF, DOCX, MD, TXT",
            ))
            continue

        # Read the uploaded bytes into memory
        content = await upload.read()

        # Reject files that exceed the configured size limit
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            results.append(FileUploadResult(
                file_name     = file_name,
                doc_id        = compute_doc_id(file_name),
                file_type     = Path(file_name).suffix.lstrip("."),
                status        = "error",
                chunks_created= 0,
                error         = f"File too large ({size_mb:.1f} MB). Max allowed: {settings.max_upload_size_mb} MB",
            ))
            continue

        # LangChain loaders need a real file on disk — save bytes to a temp path
        tmp_path = os.path.join(
            settings.upload_dir,
            f"{compute_doc_id(file_name)}_{file_name}"
        )

        # ✅ FIX 3: Was 'finally' indented INSIDE the 'except' block — invalid Python.
        #           Moved to the correct level (same as try/except) so temp file is
        #           always deleted whether or not an exception occurred.
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)  # Write uploaded bytes to disk

            docs = load_file(tmp_path, file_name)  # Load into LangChain Documents
            docs_to_index[file_name] = docs
            logger.info(f"Loaded '{file_name}': {len(docs)} doc(s)")

        except Exception as e:
            results.append(FileUploadResult(
                file_name     = file_name,
                doc_id        = compute_doc_id(file_name),
                file_type     = Path(file_name).suffix.lstrip("."),
                status        = "error",
                chunks_created= 0,
                error         = str(e),
            ))

        finally:
            # Always delete the temp file — Documents are already in memory
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ---- Phase 2: Embed and store all successfully loaded files ----------------
    if docs_to_index:
        try:
            index_results = index_documents(
                docs_by_file    = docs_to_index,
                client          = client,
                embeddings      = embeddings,
                collection_name = settings.qdrant_collection,
            )
            # ✅ FIX 4: Was missing a comma after file_type= line — caused SyntaxError
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


# ==========================================================================
# QUERY HANDLER
# ==========================================================================

# ✅ FIX 5: The 'try' block was at module level (wrong indentation) — it's now
#           correctly indented inside the query() function body.
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


# ==========================================================================
# LIST DOCUMENTS HANDLER
# ==========================================================================

def list_documents(
    client: QdrantClient = Depends(get_client),
) -> DocumentListResponse:
    """List all files currently indexed in Qdrant (one row per file, not per chunk)."""
    try:
        docs = list_indexed_documents(client, settings.qdrant_collection)
        return DocumentListResponse(
            documents = [DocumentInfo(**d) for d in docs],
            total     = len(docs),
        )
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================================
# DELETE DOCUMENT HANDLER
# ==========================================================================

def delete_document(
    doc_id: str,
    client: QdrantClient = Depends(get_client),
) -> DeleteResponse:
    """Remove a file and ALL its chunks from Qdrant by doc_id."""
    try:
        # Look up the file name before deleting so we can include it in the response
        all_docs = list_indexed_documents(client, settings.qdrant_collection)
        doc_info = next((d for d in all_docs if d["doc_id"] == doc_id), None)

        n_deleted = delete_indexed_document(doc_id, client, settings.qdrant_collection)

        if n_deleted == 0:
            # ✅ FIX 6: Was 'data={...}' — HTTPException uses 'detail=', not 'data='
            raise HTTPException(
                status_code=404,
                detail=f"No document found with doc_id: {doc_id}",
            )

        file_name = doc_info["file_name"] if doc_info else doc_id
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


# ==========================================================================
# HEALTH CHECK HANDLER
# ==========================================================================

def health() -> dict:
    """Quick check that the server is running and shows the active configuration."""
    return {
        "status"            : "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_model"         : settings.llm_model,
        "collection"        : settings.qdrant_collection,
    }
