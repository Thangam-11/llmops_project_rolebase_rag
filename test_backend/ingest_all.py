from pathlib import Path

from src.data_ingestion.data_loader import IngestionService
from src.data_ingestion.chunker_service import ChunkingService
from src.embedding_layer.embedding_service import EmbeddingService
from src.vectordb.qdrant_store import QdrantService


def main():

    ingestion = IngestionService()
    chunker   = ChunkingService()
    embedder  = EmbeddingService()
    qdrant    = QdrantService()

    data_dir = Path("data")

    print("=" * 50)
    print(f"Loading documents from: {data_dir}")
    print("=" * 50)

    docs = ingestion.ingest_directory(data_dir)

    print(f"\nTotal documents loaded: {len(docs)}\n")

    total_chunks = 0

    for doc in docs:

        department = doc["department"]
        filename   = doc["filename"]
        file_type  = doc["file_type"]

        print(f"Processing : {filename}")
        print(f"Department : {department}")

        # Chunk
        chunks = chunker.chunk_document(
            document=doc["document"],
            file_type=file_type,
        )

        if not chunks:
            print(f"  skipped — no chunks\n")
            continue

        # Embed
        texts      = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)

        # Insert
        qdrant.insert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            department=department,
            filename=filename,
        )

        total_chunks += len(chunks)

        print(f"  chunks   : {len(chunks)}")
        print(f"  status   : inserted ✓\n")

    print("=" * 50)
    print(f"Total chunks indexed : {total_chunks}")
    print("Ingestion complete ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()