"""
src/retrieval/retriever.py
===========================
Department-aware retrieval layer built on QdrantStore.

Role-based logic:
    - Regular users → search only their own department
    - C-Level users → search ALL departments, merge + re-rank

Returns LangChain Documents with 'score' in metadata.

Usage:
    retriever = RetrieverService(qdrant_store)
    docs      = retriever.retrieve("What is our Q4 profit?", department="finance")
    docs      = retriever.retrieve("Summarise all KPIs",     department="c_level")
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.vectordb.qdrant_store import QdrantStore
from utils.logger_exceptions         import get_logger

logger = get_logger(__name__)

# All departments a C-Level user can query
C_LEVEL_DEPARTMENTS: list[str] = [
    "finance",
    "hr",
    "marketing",
    "engineering",
    "general",
]


class RetrieverService:
    """
    Thin routing layer over QdrantStore.

    Args:
        store      : QdrantStore instance
        default_k  : number of chunks to return per query
    """

    def __init__(
        self,
        store:     QdrantStore,
        default_k: int = 5,
    ) -> None:
        self._store     = store
        self._default_k = default_k
        logger.info("RetrieverService ready ✓")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question:   str,
        department: str,
        k:          int | None = None,
    ) -> list[Document]:
        """
        Retrieve the most relevant chunks for a question.

        Args:
            question   : user query
            department : user's department (from JWT payload)
            k          : override number of results

        Returns:
            list[Document] sorted by relevance
        """
        k = k or self._default_k

        if department == "c_level":
            return self._retrieve_c_level(question, k)

        return self._retrieve_single(question, department, k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrieve_single(
        self,
        question:   str,
        department: str,
        k:          int,
    ) -> list[Document]:
        docs = self._store.search(
            query      = question,
            department = department,
            k          = k,
        )
        logger.info(
            f"Single retrieval | dept='{department}' | hits={len(docs)}"
        )
        return docs

    def _retrieve_c_level(
        self,
        question: str,
        k:        int,
    ) -> list[Document]:
        docs = self._store.search_multi_department(
            query       = question,
            departments = C_LEVEL_DEPARTMENTS,
            k_per_dept  = k,
            top_k       = k,
        )
        logger.info(
            f"C-Level retrieval | depts={C_LEVEL_DEPARTMENTS} | hits={len(docs)}"
        )
        return docs

    # ------------------------------------------------------------------
    # Context formatting (used by the RAG chain)
    # ------------------------------------------------------------------

    @staticmethod
    def format_context(docs: list[Document]) -> str:
        """
        Convert retrieved Documents into a formatted context string
        suitable for injection into a prompt.

        Format:
            [1] filename.pdf | dept=finance | score=0.9241
            <chunk text>
            ---
        """
        if not docs:
            return "No relevant context found."

        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            m        = doc.metadata
            filename = m.get("filename",   "unknown")
            dept     = m.get("department", "unknown")
            score    = m.get("score",      0.0)
            page     = m.get("page",       None)
            headings = m.get("headings",   [])

            heading_str = (
                f" | section: {' > '.join(headings)}" if headings else ""
            )
            page_str = f" | page {page}" if page else ""

            header = (
                f"[{i}] {filename}"
                f" | dept={dept}"
                f"{heading_str}"
                f"{page_str}"
                f" | score={score}"
            )
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)