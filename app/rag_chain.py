# app/rag_chain.py

import uuid
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,

)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

from app.config import settings
from app.file_loader import load_file
from app.models import Reference

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 500   # Max characters per chunk
CHUNK_OVERLAP = 50    # Characters shared between consecutive chunks


# ==================================================================================
# EMBEDDINGS
# ==================================================================================

def get_embeddings(provider: str):
    """
    Return the embedding model.
    "openai"       -> text-embedding-3-small (1536 dims, paid API)
    "huggingface"  -> all-MiniLM-L6-v2       (384 dims,  free, CPU)
    """
    if provider == "openai":
        # ✅ FIX 1: Was models= (wrong kwarg) — correct is model=
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ==================================================================================
# QDRANT HELPERS
# ==================================================================================

def get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    """Create and return a Qdrant client connected to the cloud cluster."""
    return QdrantClient(url=url, api_key=api_key, timeout=120)


def compute_doc_id(file_name: str) -> str:
    """
    Produce a short, stable ID for a file by hashing its name.
    Same filename -> same ID every time.
    Every chunk of a file stores this ID so we can delete them all with one filter.
    """
    return hashlib.sha256(file_name.encode()).hexdigest()[:16]


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Return True if the named collection already exists in Qdrant."""
    existing = [c.name for c in client.get_collections().collections]
    return collection_name in existing


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create the Qdrant collection if it does not exist yet.
    Also creates a keyword payload index on 'doc_id' — required for DELETE filtering.
    If the collection already exists with a different vector dimension, raise a clear error.
    """
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{collection_name}' ({vector_size} dims)")

        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema="keyword",
        )
        return  # New collection — no dimension check needed

    # Collection already exists — ensure doc_id index is present.
    # Old collections may be missing it; Qdrant ignores this call if index exists.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema="keyword",
    )

    # Make sure the dimension matches our embedding model
    info = client.get_collection(collection_name)
    vectors_cfg = info.config.params.vectors

    # Newer qdrant-client returns a dict {"": VectorParams}; older returns VectorParams directly
    if isinstance(vectors_cfg, dict):
        existing_size = list(vectors_cfg.values())[0].size
    else:
        existing_size = vectors_cfg.size

    if existing_size != vector_size:
        raise ValueError(
            f"Collection '{collection_name}' was built with {existing_size}-dim vectors "
            f"but the current embedding model produces {vector_size}-dim vectors. "
            f"Change EMBEDDING_PROVIDER in .env to match, or delete the collection "
            f"in Qdrant Cloud and restart the server."
        )


# ==================================================================================
# INDEX DOCUMENTS
# ==================================================================================

# ✅ FIX 2: Completely rewritten to accept docs_by_file dict — old signature
#           (file_path, file_name) did not match how api.py calls this function.
def index_documents(
    docs_by_file: Dict,      # {file_name: List[Document]}
    client: QdrantClient,
    embeddings,
    collection_name: str,
) -> Dict:                   # {file_name: {"doc_id": str, "chunks_created": int}}
    """
    For each file in docs_by_file:
      1. Split LangChain Documents into smaller chunks
      2. Embed all chunks
      3. Upsert into Qdrant with full metadata in payload
    Returns a dict so api.py can build a FileUploadResult per file.
    """
    # Determine vector size from a quick test embedding
    sample_vector = embeddings.embed_query("test")
    vector_size = len(sample_vector)
    ensure_collection(client, collection_name, vector_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    results = {}

    for file_name, docs in docs_by_file.items():
        doc_id = compute_doc_id(file_name)

        # Split into smaller chunks (preserves metadata from each source Document)
        chunks = splitter.split_documents(docs)

        if not chunks:
            logger.warning(f"No chunks produced for '{file_name}'")
            results[file_name] = {"doc_id": doc_id, "chunks_created": 0}
            continue

        texts   = [chunk.page_content for chunk in chunks]
        vectors = embeddings.embed_documents(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text":        chunk.page_content,
                        "doc_id":      doc_id,
                        "file_name":   file_name,
                        "file_type":   chunk.metadata.get(
                                           "file_type",
                                           Path(file_name).suffix.lstrip(".")
                                       ),
                        "page_number": chunk.metadata.get("page_number", 1),
                        "chunk_index": i,
                        "created_at":  datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

        # ✅ Batch upsert — send 100 points at a time to avoid WriteTimeout
        BATCH_SIZE = 100
        for batch_start in range(0, len(points), BATCH_SIZE):
            batch = points[batch_start : batch_start + BATCH_SIZE]
            client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"  Upserted batch {batch_start // BATCH_SIZE + 1} "
                        f"({len(batch)} chunks) for '{file_name}'")

        results[file_name] = {"doc_id": doc_id, "chunks_created": len(points)}
        logger.info(f"Indexed '{file_name}': {len(points)} chunks total")

    return results


# ==================================================================================
# QUERY WITH REFERENCES
# ==================================================================================

# ✅ FIX 3: Added llm parameter, renamed query→question, added LLM answer generation.
#           Old function returned only context text with no actual LLM answer.
# ✅ FIX 4: Reference objects now use correct field names from models.py
#           (chunk_text, relevance_score, chunk_index, file_type, page_number)
def query_with_references(
    question: str,
    top_k: int,
    client: QdrantClient,
    embeddings,
    llm: ChatOpenAI,
    collection_name: str,
) -> Dict:
    """
    1. Embed the question
    2. Search Qdrant for top_k most similar chunks
    3. Feed chunks as context to the LLM
    4. Return answer + reference list
    """
    query_vector = embeddings.embed_query(question)

    # ✅ client.search() removed in qdrant-client v1.7+ — use query_points() instead
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    references: List[Reference] = []
    context_texts: List[str]    = []

    for r in search_result:
        payload = r.payload
        text    = payload.get("text", "")
        context_texts.append(text)

        references.append(
            Reference(
                file_name      = payload.get("file_name", ""),
                file_type      = payload.get("file_type", ""),
                page_number    = payload.get("page_number", 1),
                chunk_index    = payload.get("chunk_index", 0),
                chunk_text     = text,
                relevance_score= r.score,
            )
        )

    # Build context string and ask the LLM
    context = "\n\n".join(context_texts)
    prompt  = (
        f"Answer the question using ONLY the context below. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    llm_response = llm.invoke([HumanMessage(content=prompt)])
    answer       = llm_response.content

    return {
        "answer":     answer,
        "references": references,
    }


# ==================================================================================
# QUERY CONTEXT HELPER  (used by streaming endpoint)
# ==================================================================================

def query_with_references_context(
    question: str,
    top_k: int,
    client: QdrantClient,
    embeddings,
    collection_name: str,
):
    """
    Retrieve top_k chunks for a question and return (context_str, references_list).
    Used by the streaming handler so it can stream the LLM call separately.
    """
    query_vector = embeddings.embed_query(question)

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    references: List[Reference] = []
    context_texts: List[str]    = []

    for r in search_result:
        payload = r.payload
        text    = payload.get("text", "")
        context_texts.append(text)
        references.append(
            Reference(
                file_name      = payload.get("file_name", ""),
                file_type      = payload.get("file_type", ""),
                page_number    = payload.get("page_number", 1),
                chunk_index    = payload.get("chunk_index", 0),
                chunk_text     = text,
                relevance_score= r.score,
            )
        )

    return "".join(context_texts), references


# ==================================================================================
# DELETE DOCUMENT
# ==================================================================================

# ✅ FIX 5: Old function returned a dict — api.py expects an int (chunk count).
#           Now scrolls first to count matching chunks, then deletes them.
def delete_indexed_document(
    doc_id: str,
    client: QdrantClient,
    collection_name: str,
) -> int:
    """Delete all chunks belonging to doc_id. Returns the number of deleted chunks."""
    delete_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )

    # Count before deleting so we can return the number
    scroll_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=delete_filter,
        limit=10_000,
        with_payload=False,
        with_vectors=False,
    )
    count = len(scroll_result[0])

    if count == 0:
        return 0

    client.delete(
        collection_name=collection_name,
        points_selector=delete_filter,
    )

    logger.info(f"Deleted {count} chunks for doc_id='{doc_id}'")
    return count


# ==================================================================================
# LIST DOCUMENTS
# ==================================================================================

def list_indexed_documents(
    client: QdrantClient,
    collection_name: str,
) -> List[Dict]:
    """
    Scroll through all points and return one summary dict per unique file
    (not one per chunk). Used by api.py to build DocumentInfo objects.
    """
    scroll_result = client.scroll(
        collection_name=collection_name,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )

    docs: Dict[str, Dict] = {}

    for point in scroll_result[0]:
        payload = point.payload
        doc_id  = payload.get("doc_id")

        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id":      doc_id,
                "file_name":   payload.get("file_name", ""),
                "file_type":   payload.get("file_type", ""),
                "chunk_count": 0,
            }

        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())
