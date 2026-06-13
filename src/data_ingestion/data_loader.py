"""
src/data_ingestion/data_loader.py
===================================
Loads department documents using Docling.
Folder name is mapped to standard department name via FOLDER_TO_DEPT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
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

# ── Folder name → standard department name ────────────────────────────────
# Allows flexible folder naming without breaking department routing
FOLDER_TO_DEPT: dict[str, str] = {
    "engineering": "engineering",
    "finance":     "finance",
    "marketing":   "marketing",
    "general":     "general",
    "hr":          "hr",          # if folder renamed to hr
    "hr_data":     "hr",          # ← hr_data folder → hr department
    "c_level":     "c_level",
}

# Valid department names after mapping
VALID_DEPARTMENTS: set[str] = set(FOLDER_TO_DEPT.values())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class IngestionService:

    def __init__(self) -> None:
        self.converter = DocumentConverter()
        logger.info("IngestionService Initialized")

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        file_path:  Path,
        department: str,
    ) -> dict[str, Any]:
        """
        Parse a single file with Docling.

        Returns:
            {
                "department" : str,
                "filename"   : str,
                "file_path"  : str,
                "file_type"  : str,
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
            logger.exception(f"Failed parsing: {file_path.name}")
            raise e

    def ingest_directory(
        self,
        root_dir: Path,
    ) -> list[dict[str, Any]]:
        """
        Walk root_dir/<folder>/<files> and ingest every supported file.
        Folder name is mapped to department name via FOLDER_TO_DEPT.
        """
        documents: list[dict[str, Any]] = []

        logger.info(f"Starting ingestion from {root_dir}")

        if not root_dir.exists():
            logger.error(f"Directory not found: {root_dir}")
            return documents

        for dept_folder in sorted(root_dir.iterdir()):

            if not dept_folder.is_dir():
                continue

            folder_name = dept_folder.name.lower()

            # ── Map folder name → standard department name ─────────
            department = FOLDER_TO_DEPT.get(folder_name)

            if department is None:
                logger.warning(
                    f"Unknown folder '{folder_name}' — skipping"
                )
                continue

            logger.info(
                f"Processing department: {department} "
                f"(folder='{folder_name}')"
            )

            for file_path in sorted(dept_folder.rglob("*")):

                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    doc = self.ingest_file(
                        file_path=file_path,
                        department=department,   # ← "hr" not "hr_data"
                    )
                    documents.append(doc)

                except Exception:
                    logger.exception(
                        f"Skipping: {file_path.name}"
                    )
                    continue

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    # ------------------------------------------------------------------
    # Convert Docling docs → LangChain Documents
    # ------------------------------------------------------------------

    def to_langchain_docs(
        self,
        raw_docs: list[dict[str, Any]],
    ) -> list[Document]:
        """
        Convert raw dicts → LangChain Documents.
        Calls document.export_to_markdown() for clean text.
        """
        langchain_docs: list[Document] = []

        for raw in raw_docs:
            try:
                text = raw["document"].export_to_markdown()

                if not text.strip():
                    logger.warning(
                        f"Empty content: {raw['filename']}"
                    )
                    continue

                langchain_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source":     raw["file_path"],
                            "filename":   raw["filename"],
                            "department": raw["department"],
                            "file_type":  raw["file_type"],
                        },
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
            f"{len(langchain_docs)}/{len(raw_docs)}"
        )
        return langchain_docs

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def load_file(
        self,
        file_path:  Path,
        department: str,
    ) -> list[Document]:
        raw = self.ingest_file(file_path, department)
        return self.to_langchain_docs([raw])

    def load_directory(
        self,
        root_dir: Path,
    ) -> list[Document]:
        raw_docs = self.ingest_directory(root_dir)
        return self.to_langchain_docs(raw_docs)