# app/markdown_generator.py
#
# Converts a list of LangChain Documents (already loaded by file_loader.py)
# into a clean Markdown string, then saves it to disk.
#
# Called during upload so every indexed file has a .md copy in markdown_files/

import os
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def generate_markdown(docs: List[Document], file_name: str) -> str:
    """
    Turn a list of LangChain Documents into a Markdown string.

    Layout:
    - H1 heading = the original file name
    - If multiple docs (e.g. PDF pages), each gets an H2 heading with page number
    - Single-doc files (TXT, DOCX, CSV) get their content without a page heading
    """
    lines: List[str] = []

    # ── Top-level heading ──────────────────────────────────────────────────────
    lines.append(f"# {file_name}\n")

    for doc in docs:
        page_num  = doc.metadata.get("page_number", 1)
        file_type = doc.metadata.get("file_type", "")
        content   = doc.page_content.strip()

        if not content:
            continue

        # ── Page / row heading (only when there are multiple docs) ─────────────
        if len(docs) > 1:
            if file_type == "csv":
                lines.append(f"\n## Row {page_num}\n")
            else:
                lines.append(f"\n## Page {page_num}\n")

        lines.append(content)
        lines.append("")   # blank line between sections

    return "\n".join(lines)


def save_markdown(content: str, doc_id: str, markdown_dir: str) -> str:
    """
    Write markdown content to  <markdown_dir>/<doc_id>.md
    Returns the full path of the saved file.
    """
    os.makedirs(markdown_dir, exist_ok=True)
    path = os.path.join(markdown_dir, f"{doc_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Markdown saved → {path}")
    return path


def save_original(content: bytes, doc_id: str, file_name: str, original_dir: str) -> str:
    """
    Persist the raw uploaded bytes to  <original_dir>/<doc_id>_<file_name>
    Returns the full path of the saved file.
    """
    os.makedirs(original_dir, exist_ok=True)
    path = os.path.join(original_dir, f"{doc_id}_{file_name}")
    with open(path, "wb") as f:
        f.write(content)
    logger.info(f"Original saved → {path}")
    return path
