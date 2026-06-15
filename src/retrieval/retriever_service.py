"""
src/retrieval/retriever.py
===========================
Department-aware retrieval layer built on QdrantStore.
...
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.vectordb.qdrant_store import QdrantStore
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

C_LEVEL_DEPARTMENTS: list[str] = [
    "finance",
    "hr",
    "marketing",
    "engineering",
    "general",
]

DEPT_COLLECTIONS: dict[str, list[str]] = {
    "engineering": ["engineering", "general"],
    "hr":          ["hr",          "general"],
    "finance":     ["finance",     "general"],
    "marketing":   ["marketing",   "general"],
    "general":     ["general"],
    "c_level":     C_LEVEL_DEPARTMENTS,
}


class RetrieverService:
    """
    Thin routing layer over QdrantStore.
    Args:
        store      : QdrantStore instance
        default_k  : number of chunks to return per query
    """

    def __init__(
        self,
        store: QdrantStore,
        default_k: int = 5,
    ) -> None:
        self._store = store
        self._default_k = default_k
        logger.info("RetrieverService ready ✓")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        department: str,
        k: int | None = None,
    ) -> list[Document]:
        k = k or self._default_k

        if department == "c_level":
            return self._retrieve_c_level(question, k)

        collections = DEPT_COLLECTIONS.get(department, [department])

        if len(collections) == 1:
            return self._retrieve_single(
                question=question,
                department=collections[0],
                k=k,
            )

        return self._retrieve_multiple(
            question=question,
            departments=collections,
            k=k,
        )

    # ------------------------------------------------------------------
    # Internal Retrieval Methods
    # ------------------------------------------------------------------

    def _retrieve_single(
        self,
        question: str,
        department: str,
        k: int,
    ) -> list[Document]:
        docs = self._store.search(
            query=question,
            department=department,
            k=k,
        )
        logger.info(f"Single retrieval | dept='{department}' | hits={len(docs)}")
        return docs

    def _retrieve_multiple(
        self,
        question: str,
        departments: list[str],
        k: int,
    ) -> list[Document]:
        docs = self._store.search_multi_department(
            query=question,
            departments=departments,
            k_per_dept=k,
            top_k=k,
        )
        logger.info(f"Multi-department retrieval | depts={departments} | hits={len(docs)}")
        return docs

    def _retrieve_c_level(
        self,
        question: str,
        k: int,
    ) -> list[Document]:
        docs = self._store.search_multi_department(
            query=question,
            departments=C_LEVEL_DEPARTMENTS,
            k_per_dept=k,
            top_k=k,
        )
        logger.info(f"C-Level retrieval | depts={C_LEVEL_DEPARTMENTS} | hits={len(docs)}")
        return docs

    # ------------------------------------------------------------------
    # Context formatting (used by the RAG chain)
    # ------------------------------------------------------------------

    @staticmethod
    def format_context(
        docs: list[Document],
        max_chars_per_chunk: int = 800,
        max_total_chars: int = 6000,
    ) -> str:
        if not docs:
            return "No relevant context found."

        parts: list[str] = []
        total_chars: int = 0

        for i, doc in enumerate(docs, 1):
            m        = doc.metadata
            filename = m.get("filename",   "unknown")
            dept     = m.get("department", "unknown")
            score    = m.get("score",      0.0)
            page     = m.get("page",       None)
            headings = m.get("headings",   [])

            heading_str = f" | section: {' > '.join(headings)}" if headings else ""
            page_str    = f" | page {page}" if page else ""

            header = (
                f"[{i}] {filename}"
                f" | dept={dept}"
                f"{heading_str}"
                f"{page_str}"
                f" | score={score:.4f}"
            )

            content = doc.page_content
            if len(content) > max_chars_per_chunk:
                content = content[:max_chars_per_chunk] + "…"

            chunk = f"{header}\n{content}"

            if total_chars + len(chunk) > max_total_chars:
                logger.info(f"Context truncated at chunk {i} ({total_chars} chars)")
                break

            parts.append(chunk)
            total_chars += len(chunk)

        return "\n\n---\n\n".join(parts)