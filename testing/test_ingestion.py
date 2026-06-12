from pathlib import Path

from config.settings import get_settings
from utils.logger_exceptions import get_logger

from src.data_ingestion.data_loader import IngestionService
from src.data_ingestion.chunker_service import ChunkingService
from src.ingestion_pipeline.ingest_to_qdrant import (
    to_langchain_docs,
)

logger = get_logger(__name__)
settings = get_settings()


def test_ingestion():

    logger.info(
        "Starting ingestion test..."
    )

    root = Path("data")

    print("\n" + "=" * 60)
    print("INGESTION TEST")
    print("=" * 60)

    # ----------------------------------------
    # Load files
    # ----------------------------------------

    loader = IngestionService()

    raw_docs = loader.ingest_directory(root)

    print(
        f"\nFiles Loaded: {len(raw_docs)}"
    )

    if not raw_docs:
        print(
            "❌ No files found"
        )
        return

    # ----------------------------------------
    # Show loaded files
    # ----------------------------------------

    print("\nLoaded Files")

    for idx, doc in enumerate(raw_docs, start=1):

        print("-" * 60)

        print(
            f"{idx}. "
            f"{doc['department']} | "
            f"{doc['filename']} | "
            f"{doc['file_type']}"
        )

    # ----------------------------------------
    # Chunk documents
    # ----------------------------------------

    chunker = ChunkingService()

    lc_docs = to_langchain_docs(
        raw_docs,
        chunker,
    )

    print(
        f"\nTotal Chunks: {len(lc_docs)}"
    )

    if not lc_docs:
        print(
            "❌ No chunks created"
        )
        return

    # ----------------------------------------
    # Show sample chunks
    # ----------------------------------------

    print("\nSample Chunks")

    for idx, doc in enumerate(
        lc_docs[:5],
        start=1,
    ):

        print("-" * 60)

        print(
            f"Chunk #{idx}"
        )

        print(
            f"Department: "
            f"{doc.metadata.get('department')}"
        )

        print(
            f"Filename: "
            f"{doc.metadata.get('filename')}"
        )

        print(
            f"Chunk Index: "
            f"{doc.metadata.get('chunk_index')}"
        )

        print(
            f"Chunk Type: "
            f"{doc.metadata.get('chunk_type')}"
        )

        print(
            f"Page: "
            f"{doc.metadata.get('page')}"
        )

        print(
            "\nContent:\n"
        )

        print(
            doc.page_content[:500]
        )

    # ----------------------------------------
    # Department summary
    # ----------------------------------------

    dept_counts = {}

    for doc in lc_docs:

        dept = doc.metadata.get(
            "department",
            "unknown",
        )

        dept_counts[dept] = (
            dept_counts.get(dept, 0)
            + 1
        )

    print("\n" + "=" * 60)
    print("DEPARTMENT SUMMARY")
    print("=" * 60)

    for dept, count in sorted(
        dept_counts.items()
    ):
        print(
            f"{dept:<15} {count}"
        )

    print("\n" + "=" * 60)
    print("INGESTION TEST PASSED")
    print("=" * 60)

    logger.info(
        "Ingestion test completed ✓"
    )


if __name__ == "__main__":
    test_ingestion()