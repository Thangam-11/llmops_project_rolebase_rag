from src.embedding_layer.embedding_service import EmbeddingService
from src.vectordb.qdrant_service import QdrantService
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

C_LEVEL_COLLECTIONS = [
    "engineering",
    "hr",
    "finance",
    "marketing",
    "general",
]


class RetrieverService:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service    = QdrantService()
        logger.info("RetrieverService initialized ✓")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        department: str,
        limit: int = 5,
    ) -> list[dict]:

        try:
            query_embedding = (
                self.embedding_service.embed_text(question)
            )

            if department == "c_level":
                return self._retrieve_all(
                    query_embedding, limit
                )

            return self._retrieve_one(
                query_embedding, department, limit
            )

        except Exception:
            logger.exception("Retrieval failed")
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve_one(
        self,
        query_embedding: list[float],
        department: str,
        limit: int,
    ) -> list[dict]:

        results = self.qdrant_service.search(
            query_embedding=query_embedding,
            department=department,
            limit=limit,
        )

        chunks = self._format(results)

        logger.info(
            f"Retrieved {len(chunks)} chunks "
            f"from '{department}'"
        )

        return chunks

    def _retrieve_all(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict]:

        all_chunks = []

        for collection in C_LEVEL_COLLECTIONS:
            try:
                results = self.qdrant_service.search(
                    query_embedding=query_embedding,
                    department=collection,
                    limit=limit,
                )
                all_chunks.extend(self._format(results))

            except Exception as e:
                logger.warning(
                    f"Collection '{collection}' failed: {e}"
                )

        all_chunks.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        top = all_chunks[:limit]

        logger.info(
            f"C-Level: {len(top)} chunks "
            f"from {len(C_LEVEL_COLLECTIONS)} collections"
        )

        return top

    def _format(self, results) -> list[dict]:

        chunks = []

        for r in results:
            chunks.append({
                "score":      round(r.score, 4),
                "department": r.payload.get("department", ""),
                "filename":   r.payload.get("filename", ""),
                "chunk_id":   r.payload.get("chunk_id", ""),
                "chunk_text": r.payload.get("chunk_text", ""),
                "chunk_type": r.payload.get("chunk_type", "text"),
                "headings":   r.payload.get("headings", []),
                "page":       r.payload.get("page"),
            })

        return chunks