from pathlib import Path

from src.data_ingestion.data_loader import (
    IngestionService,
)

from src.data_ingestion.chunker_service import (
    ChunkingService,
)

from src.embedding_layer.embedding_service import (
    EmbeddingService,
)

from src.vectordb.qdrant_service import (
    QdrantService,
)

from utils.logger_exceptions import (
    get_logger,
)

logger = get_logger(__name__)


def main():

    logger.info(
        "Starting ingestion pipeline"
    )

    ingestion = IngestionService()

    chunker = ChunkingService()

    embedder = EmbeddingService()

    qdrant = QdrantService()

    qdrant.create_collection()

    qdrant.ensure_payload_index()

    documents = ingestion.ingest_directory(
        Path("data")
    )

    logger.info(
        f"Loaded {len(documents)} documents"
    )

    total_chunks = 0

    for doc in documents:

        department = doc["department"]

        filename = doc["filename"]

        file_type = doc["file_type"]

        logger.info(
            f"Processing {filename}",
            extra={
                "department": department
            },
        )

        chunks = chunker.chunk_document(
            document=doc["document"],
            file_type=file_type,
        )

        if not chunks:

            logger.warning(
                f"No chunks generated for {filename}"
            )

            continue

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            embedder.embed_texts(
                texts
            )
        )

        qdrant.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            department=department,
            filename=filename,
        )

        total_chunks += len(chunks)

        logger.info(
            f"Inserted {len(chunks)} chunks",
            extra={
                "file": filename,
                "department": department,
            },
        )

    logger.info(
        f"Pipeline completed. Total chunks={total_chunks}"
    )

    print("\n" + "=" * 60)

    print(
        f"Documents Processed : {len(documents)}"
    )

    print(
        f"Chunks Inserted     : {total_chunks}"
    )

    print("=" * 60)

    print(
        "Data successfully stored in Qdrant ✓"
    )


if __name__ == "__main__":
    main()