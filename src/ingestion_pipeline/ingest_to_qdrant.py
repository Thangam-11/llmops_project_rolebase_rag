"""
Ingest all department documents into Qdrant.

Run:
    python -m src.ingestion_pipeline.ingest_to_qdrant
"""

from pathlib import Path
from langchain_core.documents import Document

from src.data_ingestion.data_loader import IngestionService
from src.data_ingestion.chunker_service import ChunkingService
from src.embedding_layer.embedding_service import EmbeddingService
from src.vectordb.qdrant_service import QdrantStore
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Convert Docling raw docs → LangChain Documents
# ---------------------------------------------------------------------------

def to_langchain_docs(
    raw_docs: list[dict],
    chunker:  ChunkingService,
) -> list[Document]:

    lc_docs: list[Document] = []

    for raw in raw_docs:

        department = raw["department"]
        filename   = raw["filename"]
        file_type  = raw["file_type"]

        chunks = chunker.chunk_document(
            document=raw["document"],
            file_type=file_type,
        )

        if not chunks:
            logger.warning(
                f"No chunks from {filename} — skipping"
            )
            continue

        for chunk in chunks:
            lc_docs.append(
                Document(
                    page_content=chunk["text"],
                    metadata={
                        "department":  department,
                        "filename":    filename,
                        "chunk_index": chunk["chunk_id"],
                        "chunk_type":  chunk.get("chunk_type", "text"),
                        "page":        chunk.get("page"),
                        "headings":    chunk.get("headings", []),
                    },
                )
            )

    logger.info(
        f"Converted {len(raw_docs)} files "
        f"→ {len(lc_docs)} chunks"
    )

    return lc_docs


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def run_ingestion(data_dir: str = "data") -> dict:

    root = Path(data_dir)

    print("=" * 60)
    print(f"Starting ingestion from: {root.resolve()}")
    print("=" * 60)

    # ── Init services ──────────────────────────────────────────────────
    loader   = IngestionService()
    chunker  = ChunkingService()
    embedder = EmbeddingService(
        model_name=settings.embedding_model,
    )
    store = QdrantStore(
        embedding_service=embedder,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )

    # ── Step 1 — Load all files ────────────────────────────────────────
    raw_docs = loader.ingest_directory(root)

    if not raw_docs:
        print("❌ No files found — check your data/ folder")
        return {"files": 0, "chunks": 0}

    print(f"\nFiles loaded: {len(raw_docs)}\n")

    # ── Step 2 — Chunk + convert ───────────────────────────────────────
    lc_docs = to_langchain_docs(raw_docs, chunker)

    if not lc_docs:
        print("❌ No chunks produced")
        return {"files": len(raw_docs), "chunks": 0}

    # ── Step 3 — Embed + upsert into Qdrant ───────────────────────────
    print(f"Inserting {len(lc_docs)} chunks into Qdrant...")
    store.add_documents(lc_docs)

    # ── Summary ────────────────────────────────────────────────────────
    dept_counts: dict[str, int] = {}
    file_counts: dict[str, list] = {}

    for doc in lc_docs:
        dept = doc.metadata.get("department", "unknown")
        file = doc.metadata.get("filename", "unknown")

        dept_counts[dept] = dept_counts.get(dept, 0) + 1

        if dept not in file_counts:
            file_counts[dept] = []
        if file not in file_counts[dept]:
            file_counts[dept].append(file)

    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"{'Department':<15} {'Chunks':>8}  Files")
    print("-" * 60)

    for dept in sorted(dept_counts.keys()):
        files  = ", ".join(file_counts[dept])
        count  = dept_counts[dept]
        print(f"{dept:<15} {count:>8}  {files}")

    print("-" * 60)
    print(f"{'TOTAL':<15} {len(lc_docs):>8}  {len(raw_docs)} files")
    print("=" * 60)

    # Collection info
    info = store.collection_info()
    print(f"\nQdrant collection : {info['name']}")
    print(f"Total vectors     : {info['vector_count']}")
    print(f"Status            : {info['status']}")
    print("\n✅ Ingestion complete")

    return {
        "files":       len(raw_docs),
        "chunks":      len(lc_docs),
        "departments": dept_counts,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion("data")