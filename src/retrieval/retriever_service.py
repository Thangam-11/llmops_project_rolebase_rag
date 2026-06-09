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


class RetrieverService:

    def __init__(self):

        self.embedding_service = (
            EmbeddingService()
        )

        self.qdrant_service = (
            QdrantService()
        )

        logger.info(
            "Retriever Service Initialized"
        )

    def retrieve(
        self,
        question: str,
        department: str,
        limit: int = 5,
    ) -> list[dict]:

        try:

            query_embedding = (
                self.embedding_service
                .embed_text(question)
            )

            results = (
                self.qdrant_service.search(
                    query_embedding=query_embedding,
                    department=department,
                    limit=limit,
                )
            )

            retrieved_chunks = []

            for result in results:

                retrieved_chunks.append(
                    {
                        "score": result.score,
                        "department":
                            result.payload.get(
                                "department"
                            ),
                        "filename":
                            result.payload.get(
                                "filename"
                            ),
                        "chunk_id":
                            result.payload.get(
                                "chunk_id"
                            ),
                        "chunk_text":
                            result.payload.get(
                                "chunk_text"
                            ),
                    }
                )

            logger.info(
                f"Retrieved {len(retrieved_chunks)} chunks"
            )

            return retrieved_chunks

        except Exception:

            logger.exception(
                "Retrieval failed"
            )

            raise