"""
Embedding Service
==================
BAAI/bge-base-en-v1.5 (768-dim)

Provides two interfaces:
  1. Direct methods  — embed_text(), embed_texts()
  2. LangChain API   — embed_query(), embed_documents(), as_langchain()
"""

from __future__ import annotations

from functools import cached_property

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME    = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE    = 32

QUERY_PREFIX  = (
    "Represent this sentence for "
    "searching relevant passages: "
)


class EmbeddingService:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device:     str = "cpu",
        batch_size: int = BATCH_SIZE,
    ) -> None:

        self._model_name = model_name
        self._device     = device
        self._batch_size = batch_size

        logger.info(
            f"Loading embedding model: "
            f"{model_name} on {device}"
        )

    # ------------------------------------------------------------------
    # Cached LangChain embeddings — loaded once
    # ------------------------------------------------------------------

    @cached_property
    def _embeddings(self) -> HuggingFaceEmbeddings:

        emb = HuggingFaceEmbeddings(
            model_name=self._model_name,
            model_kwargs={
                "device": self._device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
                # ✅ removed show_progress_bar here
                # LangChain passes it internally — duplicate causes error
            },
        )

        logger.info(
            f"Embedding model loaded ✓ "
            f"dim={EMBEDDING_DIM}"
        )

        return emb

    # ------------------------------------------------------------------
    # Original interface — your existing code stays working
    # ------------------------------------------------------------------

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """Single text embedding with BGE query prefix."""
        try:
            return self._embeddings.embed_query(
                QUERY_PREFIX + text
            )
        except Exception:
            logger.exception("embed_text failed")
            raise

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Batch embed list of texts — no prefix applied."""
        try:
            if not texts:
                return []
            return self._embeddings.embed_documents(texts)
        except Exception:
            logger.exception("embed_texts failed")
            raise

    # ------------------------------------------------------------------
    # LangChain interface — used by vectorstore and RAG chain
    # ------------------------------------------------------------------

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """LangChain-style query embedding with BGE prefix."""
        try:
            return self._embeddings.embed_query(
                QUERY_PREFIX + text
            )
        except Exception:
            logger.exception("embed_query failed")
            raise

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """LangChain-style document embedding — no prefix."""
        try:
            if not texts:
                return []
            return self._embeddings.embed_documents(texts)
        except Exception:
            logger.exception("embed_documents failed")
            raise

    def as_langchain(self) -> HuggingFaceEmbeddings:
        """
        Return raw LangChain embeddings object.
        Pass to QdrantVectorStore or any other vectorstore.
        """
        return self._embeddings

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    @staticmethod
    def dimension() -> int:
        return EMBEDDING_DIM
    
# Add this at the very bottom of embedding_service.py

from functools import lru_cache  # noqa: E402

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Global singleton — model loads once, reused everywhere.
    Prevents reloading 768-dim model on every search call.
    """
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model,
    )