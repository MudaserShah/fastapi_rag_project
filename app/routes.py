from typing import Annotated, List

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.api import (
    get_client, get_emb, get_llm,
    upload_files_handler,
    query, query_stream,
    get_file_markdown, get_original_file,
    list_documents, delete_document, health,
)
from app.models import (
    DeleteResponse, DocumentListResponse,
    QueryRequest, QueryResponse, UploadResponse,
)

router = APIRouter()


# ─── Documents ────────────────────────────────────────────────────────────────

@router.post(
    "/upload", 
    response_model=UploadResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "One or more files: PDF, DOCX, MD, CSV, TXT"
                            }
                        },
                        "required": ["files"]
                    }
                }
            }
        }
    }
)
async def upload_files(
    files      : Annotated[List[UploadFile], File(description="PDF, DOCX, MD, CSV, TXT")],
    client     : QdrantClient = Depends(get_client),
    embeddings                = Depends(get_emb),
) -> UploadResponse:
    return await upload_files_handler(files, client, embeddings)


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents_route(client: QdrantClient = Depends(get_client)):
    return list_documents(client)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
def delete_document_route(doc_id: str, client: QdrantClient = Depends(get_client)):
    return delete_document(doc_id, client)


# ─── RAG ──────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_route(
    request   : QueryRequest,
    client    : QdrantClient = Depends(get_client),
    embeddings               = Depends(get_emb),
    llm       : ChatOpenAI   = Depends(get_llm),
):
    return query(request, client, embeddings, llm)


@router.post(
    "/query/stream",
    tags=["RAG"],
    summary="Query Stream Endpoint",
    description=(
        "Stream the answer token-by-token via Server-Sent Events (SSE).\n\n"
        "Each event is a JSON object:\n"
        "- `{type: 'token', content: '...'}` — one LLM token\n"
        "- `{type: 'references', content: [...]}` — source chunks after answer\n"
        "- `{type: 'done'}` — stream finished\n"
        "- `{type: 'error', content: '...'}` — if something goes wrong"
    ),
    response_class=StreamingResponse,
)
async def query_stream_route(
    request   : QueryRequest,
    client    : QdrantClient = Depends(get_client),
    embeddings               = Depends(get_emb),
    llm       : ChatOpenAI   = Depends(get_llm),
):
    return await query_stream(request, client, embeddings, llm)


# ─── Files ────────────────────────────────────────────────────────────────────

@router.get(
    "/files/{doc_id}/markdown",
    tags=["Files"],
    summary="Get File Markdown",
    response_class=Response,
    responses={200: {"content": {"text/markdown": {}}}},
)
def get_markdown_route(doc_id: str):
    """Return the Markdown version of an indexed document."""
    return get_file_markdown(doc_id)


@router.get(
    "/files/{doc_id}/original",
    tags=["Files"],
    summary="Get Original File",
)
def get_original_route(doc_id: str):
    """Download the original uploaded file."""
    return get_original_file(doc_id)


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["Health"])
def health_route() -> dict:
    return health()
