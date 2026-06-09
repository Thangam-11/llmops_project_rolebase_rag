from sentence_transformers import (
    SentenceTransformer,
)

from utils.logger_exceptions import (
    get_logger,
)

logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self):

        logger.info(
            "Loading embedding model..."
        )

        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        logger.info(
            "Embedding model loaded"
        )

    def embed_text(
        self,
        text: str,
    ) -> list[float]:

        try:

            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
            )

            return embedding.tolist()

        except Exception:

            logger.exception(
                "Embedding generation failed"
            )

            raise

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        try:

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )

            return embeddings.tolist()

        except Exception:

            logger.exception(
                "Batch embedding failed"
            )

            raise

    def embedding_dimension(
        self,
    ) -> int:

        return (
            self.model
            .get_sentence_embedding_dimension()
        )