from pathlib import Path

from src.data_ingestion.data_loader import (
    IngestionService,
)

from src.data_ingestion.chunker_service import (
    ChunkingService,
)

# Initialize Services
ingestion = IngestionService()
chunker = ChunkingService()

# Load Documents
documents = ingestion.ingest_directory(
    Path("data")
)

print(f"\nTotal Documents: {len(documents)}\n")

# Process Each Document
for doc in documents:

    print("\n" + "=" * 80)

    print(
        f"Department : {doc['department']}"
    )

    print(
        f"File       : {doc['filename']}"
    )

    print(
        f"Type       : {doc['file_type']}"
    )

    chunks = chunker.chunk_document(
        document=doc["document"],
        file_type=doc["file_type"],
    )

    print(
        f"Chunks Generated : {len(chunks)}"
    )

    if chunks:

        print("\nFirst Chunk Preview:\n")

        print(
            chunks[0]["text"][:500]
        )

        print("\nChunk Metadata:\n")

        print(
            {
                "chunk_id": chunks[0].get(
                    "chunk_id"
                ),
                "chunk_type": chunks[0].get(
                    "chunk_type"
                ),
                "headings": chunks[0].get(
                    "headings"
                ),
                "page": chunks[0].get(
                    "page"
                ),
            }
        )

print("\n" + "=" * 80)
print("Chunking Test Completed Successfully")