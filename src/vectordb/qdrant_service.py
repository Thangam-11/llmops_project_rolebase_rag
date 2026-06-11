"""
src/vectordb/qdrant_store.py
==============================
LangChain-native Qdrant vector store wrapper.

Responsibilities:
  - Create / connect to a Qdrant collection
  - Ensure a payload index exists on the 'department' field
  - Ingest LangChain Documents with metadata
  - Similarity search filtered by department

Every point stored in Qdrant has this payload:
    {
        "metadata": {
            "department"  : str,
            
            "filename"    : str,
            "chunk_index" : int,
            "chunk_type"  : "text" | "table",
            "page"        : int | None,
        },
        "page_content": str,
    }
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from src.embedding_layer.embedding_service import EmbeddingService
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class QdrantStore:
    """
    LangChain QdrantVectorStore with department-aware filtered search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        url:             str = None,
        api_key:         str = None,
        collection_name: str = None,
    ) -> None:

        self._embedding_service = embedding_service
        self._collection_name   = (
            collection_name or settings.collection_name
        )
        self._url     = url     or settings.qdrant_url
        self._api_key = api_key or settings.qdrant_api_key or None

        # Raw Qdrant client — for admin ops
        self._client = QdrantClient(
            url     = self._url,
            api_key = self._api_key,
        )

        # LangChain vectorstore — for add/search
        self._store = self._init_store()

        logger.info(
            f"QdrantStore ready | "
            f"collection='{self._collection_name}'"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_store(self) -> QdrantVectorStore:
        """Create collection if needed, ensure index, return store."""
        self._ensure_collection()
        self._ensure_department_index()

        return QdrantVectorStore(
            client          = self._client,
            collection_name = self._collection_name,
            embedding       = self._embedding_service.as_langchain(),
        )

    def _ensure_collection(self) -> None:

        existing = {
            c.name
            for c in self._client.get_collections().collections
        }

        if self._collection_name in existing:
            logger.info(
                f"Collection '{self._collection_name}' exists ✓"
            )
            return

        self._client.create_collection(
            collection_name = self._collection_name,
            vectors_config  = VectorParams(
                size     = self._embedding_service.dimension(),
                distance = Distance.COSINE,
            ),
        )
        logger.info(
            f"Collection '{self._collection_name}' created ✓"
        )

    def _ensure_department_index(self) -> None:
        """
        Qdrant requires a payload index on filtered fields.
        Check both 'department' and 'metadata.department'
        since LangChain nests metadata.
        """
        info    = self._client.get_collection(self._collection_name)
        indexes = info.payload_schema or {}

        # LangChain Qdrant stores fields under metadata.*
        index_key = "metadata.department"

        if index_key in indexes or "department" in indexes:
            logger.info("department index exists ✓")
            return

        # Create index on metadata.department
        self._client.create_payload_index(
            collection_name = self._collection_name,
            field_name      = index_key,
            field_schema    = PayloadSchemaType.KEYWORD,
        )
        logger.info(
            f"Payload index on '{index_key}' created ✓"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """
        Embed and upsert LangChain Documents into Qdrant.
        Returns list of inserted point IDs.
        """
        if not documents:
            logger.warning(
                "add_documents called with empty list — skipping"
            )
            return []

        logger.info(
            f"Inserting {len(documents)} documents..."
        )

        ids = self._store.add_documents(documents)

        logger.info(
            f"Inserted {len(ids)} documents ✓"
        )

        return ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query:      str,
        department: str,
        k:          int = 5,
    ) -> list[Document]:
        """
        Similarity search filtered to one department.

        Returns Documents sorted by relevance (highest score first).
        Score is attached to doc.metadata['score'].
        """
        dept_filter = Filter(
            must=[
                FieldCondition(
                    key   = "metadata.department",
                    match = MatchValue(value=department),
                )
            ]
        )

        raw = self._store.similarity_search_with_score(
            query  = query,
            k      = k,
            filter = dept_filter,
        )

        docs: list[Document] = []
        for doc, score in raw:
            doc.metadata["score"] = round(float(score), 4)
            docs.append(doc)

        # ✅ Higher score = more similar — sort descending
        docs.sort(
            key=lambda d: d.metadata.get("score", 0.0),
            reverse=True,
        )

        logger.info(
            f"search | dept='{department}' | "
            f"hits={len(docs)}"
        )

        return docs

    def search_multi_department(
        self,
        query:      str,
        departments: list[str],
        k_per_dept: int = 5,
        top_k:      int = 5,
    ) -> list[Document]:
        """
        Search across multiple departments — used for C-Level users.
        Merges and re-ranks by score, returns top_k.
        """
        all_docs: list[Document] = []

        for dept in departments:
            try:
                docs = self.search(
                    query=query,
                    department=dept,
                    k=k_per_dept,
                )
                all_docs.extend(docs)

            except Exception as e:
                logger.warning(
                    f"Search failed for dept '{dept}': {e}"
                )

        # ✅ Sort descending — highest similarity first
        all_docs.sort(
            key=lambda d: d.metadata.get("score", 0.0),
            reverse=True,
        )

        top = all_docs[:top_k]

        logger.info(
            f"search_multi | depts={departments} | "
            f"total={len(all_docs)} | returning={len(top)}"
        )

        return top

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def collection_info(self) -> dict:
        info = self._client.get_collection(self._collection_name)
        return {
            "name":         self._collection_name,
            "vector_count": info.points_count,
            "status":       str(info.status),
        }

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection_name)
        logger.info(
            f"Collection '{self._collection_name}' deleted ✓"
        )