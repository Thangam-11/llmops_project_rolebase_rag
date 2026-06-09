from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {
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

VALID_DEPARTMENTS = {
    "finance",
    "hr_data",
    "marketing",
    "engineering",
    "general",
    "c_level",
}


class IngestionService:

    def __init__(self):
        self.converter = DocumentConverter()

        logger.info(
            "Ingestion Service Initialized"
        )

    def ingest_file(
        self,
        file_path: Path,
        department: str,
    ) -> dict[str, Any]:

        try:

            logger.info(
                f"Processing file: {file_path.name}"
            )

            result = self.converter.convert(
                str(file_path)
            )

            logger.info(
                f"Successfully parsed: {file_path.name}"
            )

            return {
                "department": department,
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "document": result.document,
            }

        except Exception as e:

            logger.exception(
                f"Failed parsing file: {file_path}"
            )

            raise e

    def ingest_directory(
        self,
        root_dir: Path,
    ) -> list[dict[str, Any]]:

        documents = []

        logger.info(
            f"Starting ingestion from {root_dir}"
        )

        if not root_dir.exists():

            logger.error(
                f"Directory not found: {root_dir}"
            )

            return documents

        for department_dir in root_dir.iterdir():

            if not department_dir.is_dir():
                continue

            department = department_dir.name.lower()

            if department not in VALID_DEPARTMENTS:

                logger.warning(
                    f"Invalid department folder: {department}"
                )

                continue

            logger.info(
                f"Processing department: {department}"
            )

            for file_path in department_dir.rglob("*"):

                if not file_path.is_file():
                    continue

                if (
                    file_path.suffix.lower()
                    not in SUPPORTED_EXTENSIONS
                ):
                    continue

                try:

                    document = self.ingest_file(
                        file_path=file_path,
                        department=department,
                    )

                    documents.append(document)

                except Exception:

                    logger.exception(
                        f"Skipping file: {file_path}"
                    )

                    continue

        logger.info(
            f"Total documents loaded: {len(documents)}"
        )

        return documents