"""
src/data_ingestion/ingestion_service.py
=   ======================================
Matches the existing IngestionService pattern exactly.

- Uses Docling DocumentConverter (same as original)
- Uses utils.logger_exceptions.get_logger (same as original)
- Keeps SUPPORTED_EXTENSIONS and VALID_DEPARTMENTS identical
- ingest_file()       → returns raw dict with Docling document object
- ingest_directory()  → returns list of raw dicts
- to_langchain_docs() → converts raw dicts → list[LangChain Document]
- load_file()         → ingest + convert in one call
- load_directory()    → ingest + convert in one call
"""

from __future__ import annotations

from pathlib import Path


from typing import Any

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants  (identical to original)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".md",
    ".txt",
    ".html",
    ".htm",
}

VALID_DEPARTMENTS: set[str] = {
    "finance",
    "hr_data",
    "marketing",
    "engineering",
    "general",
    "c_level",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class IngestionService:
    """
    Drop-in replacement for IngestionService.

    Keeps the same method signatures and return shapes as the
    original IngestionService so existing code continues to work.

    New methods:
        to_langchain_docs(raw_docs) -> list[Document]
        load_file(path, dept)       -> list[Document]   (convenience)
        load_directory(root_dir)    -> list[Document]   (convenience)
    """

    def __init__(self) -> None:
        self.converter = DocumentConverter()
        logger.info("IngestionService Initialized")

    # ------------------------------------------------------------------
    # Original interface  (unchanged from your IngestionService)
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: Path,
        department: str,
    ) -> dict[str, Any]:
        """
        Parse a single file with Docling.

        Returns:
            {
                "department" : str,
                "filename"   : str,
                "file_path"  : str,
                "file_type"  : str,    e.g. ".pdf"
                "document"   : Docling Document object,
            }
        """
        try:
            logger.info(f"Processing file: {file_path.name}")

            result = self.converter.convert(str(file_path))

            logger.info(f"Successfully parsed: {file_path.name}")

            return {
                "department": department,
                "filename":   file_path.name,
                "file_path":  str(file_path),
                "file_type":  file_path.suffix.lower(),
                "document":   result.document,
            }

        except Exception as e:
            logger.exception(f"Failed parsing file: {file_path}")
            raise e

    def ingest_directory(
        self,
        root_dir: Path,
    ) -> list[dict[str, Any]]:
        """
        Walk root_dir/<department>/<files> and ingest every
        supported file.  Identical logic to your original.

        Returns:
            list of raw dicts (same shape as ingest_file output)
        """
        documents: list[dict[str, Any]] = []

        logger.info(f"Starting ingestion from {root_dir}")

        if not root_dir.exists():
            logger.error(f"Directory not found: {root_dir}")
            return documents

        for department_dir in root_dir.iterdir():

            if not department_dir.is_dir():
                continue

            department = department_dir.name.lower()

            if department not in VALID_DEPARTMENTS:
                logger.warning(f"Invalid department folder: {department}")
                continue

            logger.info(f"Processing department: {department}")

            for file_path in department_dir.rglob("*"):

                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    document = self.ingest_file(
                        file_path  = file_path,
                        department = department,
                    )
                    documents.append(document)

                except Exception:
                    logger.exception(f"Skipping file: {file_path}")
                    continue

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    # ------------------------------------------------------------------
    # New: convert raw Docling dicts -> LangChain Documents
    # ------------------------------------------------------------------

    def to_langchain_docs(
        self,
        raw_docs: list[dict[str, Any]],
    ) -> list[Document]:
        """
        Convert raw dicts from ingest_file / ingest_directory
        into LangChain Document objects.

        Calls document.export_to_markdown() on the Docling object
        to get clean text, then wraps in a LangChain Document
        with metadata for downstream filtering.

        Metadata on every Document:
            source      : absolute file path
            filename    : file name only
            department  : department slug
            file_type   : .pdf / .docx / etc.

        Args:
            raw_docs : list[dict] from ingest_file / ingest_directory

        Returns:
            list[Document]
        """
        langchain_docs: list[Document] = []

        for raw in raw_docs:
            try:
                # Export Docling document object to plain markdown text
                text = raw["document"].export_to_markdown()

                if not text.strip():
                    logger.warning(
                        f"Empty content after parsing: {raw['filename']}"
                    )
                    continue

                metadata: dict[str, Any] = {
                    "source":     raw["file_path"],
                    "filename":   raw["filename"],
                    "department": raw["department"],
                    "file_type":  raw["file_type"],
                }

                langchain_docs.append(
                    Document(
                        page_content = text,
                        metadata     = metadata,
                    )
                )

                logger.info(
                    f"Converted: {raw['filename']} "
                    f"({len(text):,} chars)"
                )

            except Exception:
                logger.exception(
                    f"Failed to convert: {raw.get('filename', '?')}"
                )
                continue

        logger.info(
            f"LangChain docs ready: "
            f"{len(langchain_docs)} / {len(raw_docs)}"
        )
        return langchain_docs

    # ------------------------------------------------------------------
    # Convenience wrappers  (load + convert in one step)
    # ------------------------------------------------------------------

    def load_file(
        self,
        file_path:  Path,
        department: str,
    ) -> list[Document]:
        """
        Ingest one file and return LangChain Documents immediately.
        Convenience wrapper: ingest_file -> to_langchain_docs.
        """
        raw = self.ingest_file(file_path, department)
        return self.to_langchain_docs([raw])

    def load_directory(
        self,
        root_dir: Path,
    ) -> list[Document]:
        """
        Ingest all files in a directory and return LangChain Documents.
        Convenience wrapper: ingest_directory -> to_langchain_docs.
        """
        raw_docs = self.ingest_directory(root_dir)
        return self.to_langchain_docs(raw_docs)