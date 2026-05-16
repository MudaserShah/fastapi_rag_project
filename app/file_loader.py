# A LangChain Document has two main fields:
# doc.page_content -> the text of this chunk
# doc.metadata     -> a dict with file_name, file_type, page_number, etc.
#
# api.py calls: docs = load_file(path, file_name)
# It does not need to know which loader to use — that decision is made here.

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

# ✅ FIX: Was logging.getlogger (lowercase g) — correct is logging.getLogger
logger = logging.getLogger(__name__)

# Set of file extensions we accept
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".csv", ".txt"}


def is_supported(file_name: str) -> bool:
    """Return True if we have a loader for this file extension."""
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


def load_file(file_path: str, file_name: str) -> List[Document]:
    """
    Load a file from disk and return a list of LangChain Documents.
    Every returned Document has file_name, file_type, and page_number in its metadata.
    """
    ext = Path(file_name).suffix.lower()  # e.g. ".pdf"

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info(f"Loading '{file_name}' (type: {ext})")

    docs = _load_by_extension(file_path, ext)

    for doc in docs:
        doc.metadata["file_name"] = file_name          # e.g. "report.pdf"
        doc.metadata["file_type"] = ext.lstrip(".")    # e.g. "pdf"

    logger.info(f" -> {len(docs)} document(s) loaded")
    return docs


def _load_by_extension(file_path: str, ext: str) -> List[Document]:
    """Pick the correct LangChain loader and return the raw documents."""

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)   # One document per page
        docs = loader.load()
        for doc in docs:
            # PyPDFLoader stores page as 0-indexed; convert to 1-indexed for humans
            raw_page = doc.metadata.get("page", 0)
            doc.metadata["page_number"] = raw_page + 1
        return docs

    if ext == ".docx":
        loader = Docx2txtLoader(file_path)  # One document for the whole file
        docs = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1  # DOCX has no page info
        return docs

    if ext in (".md", ".markdown"):
        try:
            loader = UnstructuredMarkdownLoader(file_path)  # Respects headings
            docs = loader.load()
        except (ImportError, ModuleNotFoundError):
            loader = TextLoader(file_path, encoding="utf-8")  # Fallback
            docs = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    if ext == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")  # One Document per row
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["page_number"] = i + 1  # Row number as page
        return docs

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")  # One Document for whole file
        docs = loader.load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    raise ValueError(f"No loader implemented for extension: {ext}")
