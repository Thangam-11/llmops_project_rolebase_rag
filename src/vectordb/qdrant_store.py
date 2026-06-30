from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

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

from src.embedding_layer.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class QdrantStore:

    def __init__(
        self,
        embedding_service: EmbeddingService = None,  # ← optional, uses singleton
        url:             str = None,                  # ← None = read from settings
        api_key:         str = None,                  # ← None = read from settings
        collection_name: str = None,                  # ← None = read from settings
        max_workers:     int = 8,                      # ← thread pool size for parallel search
    ) -> None:

        # ── Use singleton if not provided ──────────────────────────────
        self._embedding_service = (
            embedding_service or get_embedding_service()
        )

        # ── Read from settings if not explicitly passed ────────────────
        self._collection_name = (
            collection_name or settings.collection_name
        )
        self._url = (
            url or settings.qdrant_url
        )
        self._api_key = (
            api_key or settings.qdrant_api_key or None
        )

        # ── Raw Qdrant client — for admin ops ──────────────────────────
        self._client = QdrantClient(
            url     = self._url,
            api_key = self._api_key,
            timeout = 120,
        )

        # ── Thread pool — used to parallelize multi-department search ──
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # ── LangChain vectorstore — for add/search ─────────────────────
        self._store = self._init_store()

        logger.info(
            f"QdrantStore ready | "
            f"collection='{self._collection_name}' | "
            f"url='{self._url}'"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_store(self) -> QdrantVectorStore:
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
                f"Collection '{self._collection_name}' already exists"
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

        info    = self._client.get_collection(self._collection_name)
        indexes = info.payload_schema or {}

        index_key = "metadata.department"

        if index_key in indexes or "department" in indexes:
            logger.info("department index already exists ✓")
            return

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

        if not documents:
            logger.warning("add_documents — empty list, skipping")
            return []

        logger.info(f"Inserting {len(documents)} documents...")
        ids = self._store.add_documents(documents)
        logger.info(f"Inserted {len(ids)} documents ✓")

        return ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query:      str,
        department: str,
        k:          int = 5,
    ) -> list[Document]:

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

        docs.sort(
            key=lambda d: d.metadata.get("score", 0.0),
            reverse=True,
        )

        logger.info(
            f"search | dept='{department}' | "
            f"query='{query[:50]}' | "
            f"hits={len(docs)}"
        )

        return docs

    def search_multi_department(
        self,
        query:       str,
        departments: list[str],
        k_per_dept:  int = 5,
        top_k:       int = 5,
    ) -> list[Document]:
        """Searches all given departments concurrently (thread pool) instead
        of sequentially, since each Qdrant call is a blocking network call."""

        all_docs: list[Document] = []

        futures = {
            self._executor.submit(self.search, query, dept, k_per_dept): dept
            for dept in departments
        }

        for future in as_completed(futures):
            dept = futures[future]
            try:
                all_docs.extend(future.result())
            except Exception as e:
                logger.warning(f"Search failed for dept '{dept}': {e}")

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

    def shutdown(self) -> None:
        """Call on app shutdown to cleanly release thread pool workers."""
        self._executor.shutdown(wait=False)