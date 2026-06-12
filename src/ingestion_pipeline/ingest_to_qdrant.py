"""
src/rag_pipeline/ingestion_pipeline.py
=======================================
Correct ingestion pipeline using YOUR existing services:

    IngestionService    (Docling loader    — returns raw dicts)
         |
    ChunkingService     (HybridChunker     — returns list[dict])
         |
    chunks_to_langchain (attach metadata   — returns list[Document])
         |
    QdrantStore         (embed + upsert    — stores with full payload)

Run:
    python -m src.rag_pipeline.ingestion_pipeline
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.data_ingestion.data_loader      import IngestionService
from src.data_ingestion.chunker_service  import ChunkingService
from src.embedding_layer.embedding_service     import EmbeddingService
from src.vectordb.qdrant_store           import QdrantStore
from utils.logger_exceptions             import get_logger
from config.settings                     import get_settings

logger   = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Helper: convert ChunkingService dicts → LangChain Documents
# ---------------------------------------------------------------------------

def chunks_to_langchain(
    chunks:     list[dict],
    department: str,
    filename:   str,
) -> list[Document]:
    """
    Convert raw chunk dicts from ChunkingService into LangChain Documents.
    This attaches department + filename into metadata so Qdrant
    stores them as searchable payload fields.

    Input chunk dict shape:
        {
            "chunk_id"  : int,
            "text"      : str,
            "headings"  : list[str],
            "page"      : int | None,
            "chunk_type": "text" | "table",
        }
    """
    docs = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content = chunk["text"],
                metadata     = {
                    "department": department,
                    "filename":   filename,
                    "chunk_id":   chunk["chunk_id"],
                    "chunk_type": chunk["chunk_type"],
                    "headings":   chunk["headings"],
                    "page":       chunk["page"],
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    End-to-end ingestion pipeline using your existing services.

        loader  : IngestionService  — Docling, returns raw dicts
        chunker : ChunkingService   — HybridChunker, returns list[dict]
        store   : QdrantStore       — LangChain Qdrant wrapper
    """

    def __init__(self) -> None:
        self.loader  = IngestionService()
        self.chunker = ChunkingService()
        self.embedder = EmbeddingService(
            model_name = settings.embedding_model,
        )
        self.store = QdrantStore(
            embedding_service = self.embedder,
            url               = settings.qdrant_url,
            api_key           = settings.qdrant_api_key,
            collection_name   = settings.collection_name,
        )
        logger.info("IngestionPipeline initialised ✓")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data_dir: str | Path = "data") -> dict:
        """
        Ingest an entire data directory.

        Steps:
            1. ingest_directory()      raw Docling dicts
            2. chunk_document()        HybridChunker per file
            3. chunks_to_langchain()   attach department + filename metadata
            4. store.add_documents()   embed + upsert into Qdrant

        Returns:
            {"files": int, "chunks": int}
        """
        root = Path(data_dir)
        logger.info(f"=== Ingestion pipeline START | dir={root} ===")

        # Step 1: load all files via Docling
        raw_docs = self.loader.ingest_directory(root)

        if not raw_docs:
            logger.warning("No files loaded — pipeline stopped")
            return {"files": 0, "chunks": 0}

        total_chunks = 0
        all_lc_docs: list[Document] = []

        # Step 2+3: chunk each file and convert to LangChain Documents
        for raw in raw_docs:
            department = raw["department"]
            filename   = raw["filename"]
            file_type  = raw["file_type"]

            logger.info(f"Chunking: {filename} | dept={department}")

            chunks = self.chunker.chunk_document(
                document  = raw["document"],
                file_type = file_type,
            )

            if not chunks:
                logger.warning(f"No chunks for {filename} — skipping")
                continue

            # Attach department + filename into each chunk's metadata
            lc_docs = chunks_to_langchain(
                chunks     = chunks,
                department = department,
                filename   = filename,
            )

            all_lc_docs.extend(lc_docs)
            total_chunks += len(lc_docs)

            logger.info(
                f"  {filename} → {len(lc_docs)} chunks"
            )

        if not all_lc_docs:
            logger.warning("No chunks produced — pipeline stopped")
            return {"files": len(raw_docs), "chunks": 0}

        # Step 4: embed + upsert into Qdrant
        self.store.add_documents(all_lc_docs)

        stats = {"files": len(raw_docs), "chunks": total_chunks}
        logger.info(f"=== Ingestion pipeline DONE | {stats} ===")
        self._print_summary(stats)
        return stats

    def run_file(
        self,
        file_path:  str | Path,
        department: str,
    ) -> dict:
        """
        Ingest a single file.

        Returns:
            {"files": 1, "chunks": int}
        """
        path = Path(file_path)
        logger.info(f"=== Single-file ingestion | {path.name} ===")

        raw    = self.loader.ingest_file(path, department)
        chunks = self.chunker.chunk_document(
            document  = raw["document"],
            file_type = raw["file_type"],
        )

        if not chunks:
            logger.warning(f"No chunks produced for {path.name}")
            return {"files": 1, "chunks": 0}

        lc_docs = chunks_to_langchain(
            chunks     = chunks,
            department = department,
            filename   = path.name,
        )

        self.store.add_documents(lc_docs,batch_size=8)

        stats = {"files": 1, "chunks": len(lc_docs)}
        logger.info(f"=== Single-file ingestion DONE | {stats} ===")
        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(stats: dict) -> None:
        print()
        print("=" * 50)
        print(f"  Files processed : {stats['files']}")
        print(f"  Chunks inserted : {stats['chunks']}")
        print("=" * 50)
        print("  Data ingested into Qdrant ✓")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run("data")