from typing import List
from fastapi import APIRouter, Depends, File, UploadFile
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.api import (
    delete_document,
    get_client,
    get_emb,
    get_llm,
    health,
    list_documents,
    query,
    upload_files_handler,
)
from app.models import (
    DeleteResponse,
    DocumentListResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# /upload  POST
# 🛠️ Safe schema override using openapi_extra to force Swagger file picker
# ─────────────────────────────────────────────────────────────────────────────
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
    files: List[UploadFile] = File(...),
    client: QdrantClient = Depends(get_client),
    embeddings = Depends(get_emb),
) -> UploadResponse:
    return await upload_files_handler(files, client, embeddings)


# /query  POST
@router.post("/query", response_model=QueryResponse)
def query_route(
    request: QueryRequest,
    client: QdrantClient = Depends(get_client),
    embeddings = Depends(get_emb),
    llm: ChatOpenAI = Depends(get_llm),
) -> QueryResponse:
    return query(request, client, embeddings, llm)


# /documents  GET
@router.get("/documents", response_model=DocumentListResponse)
def list_documents_route(
    client: QdrantClient = Depends(get_client),
) -> DocumentListResponse:
    return list_documents(client)


# /documents/{doc_id}  DELETE
@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document_route(
    doc_id: str,
    client: QdrantClient = Depends(get_client),
) -> DeleteResponse:
    return delete_document(doc_id, client)


# /health  GET
@router.get("/health")
def health_route() -> dict:
    return health()
