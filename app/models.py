# app/models.py

from typing import List, Optional
from pydantic import BaseModel, Field


# =========================================================
# Reference Schema
# =========================================================

class Reference(BaseModel):
    """One chunk that was retrieved and used to build the answer."""
    file_name: str          # e.g. "annual_report.pdf"
    file_type: str          # e.g. "pdf", "docx", "txt"
    page_number: int        # Which page/row the chunk came from (1-indexed)
    chunk_index: int        # Position of this chunk within the document
    chunk_text: str         # The exact text that was retrieved
    relevance_score: float  # Cosine similarity score between 0.0 and 1.0


# =========================================================
# Query Request
# =========================================================

class QueryRequest(BaseModel):
    """Request body for asking questions."""
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


# =========================================================
# Query Response
# =========================================================

class QueryResponse(BaseModel):
    """Final RAG answer response."""
    question: str
    answer: str
    references: List[Reference]
    embedding_provider: str


# =========================================================
# Single File Upload Result
# =========================================================

# ✅ FIX: Was {success: bool, chunks_indexed: int}
# api.py uses:  doc_id, file_type, status, chunks_created, error
class FileUploadResult(BaseModel):
    """Result for one uploaded file."""
    file_name: str
    doc_id: str
    file_type: str
    status: str                   # "indexed" or "error"
    chunks_created: int
    error: Optional[str] = None


# =========================================================
# Upload Response
# =========================================================

# ✅ FIX: Was {successful_files, failed_files} — api.py uses {successful, failed}
class UploadResponse(BaseModel):
    """Response after uploading files."""
    total_files: int
    successful: int
    failed: int
    results: List[FileUploadResult]


# =========================================================
# Document Info
# =========================================================

class DocumentInfo(BaseModel):
    """Indexed document metadata."""
    doc_id: str
    file_name: str
    file_type: str
    chunk_count: int


# =========================================================
# Document List Response
# =========================================================

# ✅ FIX: Was {total_documents} — api.py uses {total}
class DocumentListResponse(BaseModel):
    """List all indexed documents."""
    documents: List[DocumentInfo]
    total: int


# =========================================================
# Delete Response
# =========================================================

# ✅ FIX: Was {success: bool, message: str}
# api.py uses: {status, doc_id, file_name, chunks_deleted}
class DeleteResponse(BaseModel):
    """Response after deleting a document."""
    status: str
    doc_id: str
    file_name: str
    chunks_deleted: int


# =========================================================
# Health Response
# =========================================================

class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    qdrant_connected: bool
    embedding_provider: str
    llm_model: str


# =========================================================
# Error Response
# =========================================================

class ErrorResponse(BaseModel):
    """Standard API error response."""
    detail: str
