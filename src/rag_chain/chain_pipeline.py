"""
src/rag_pipeline/rag_chain.py
===============================
Complete LangChain LCEL RAG chain.

Flow:
    question + department
         │
         ▼
    RetrieverService.retrieve()      ← Qdrant filtered search
         │
         ▼
    RetrieverService.format_context()
         │
         ▼
    get_rag_prompt(department)       ← ChatPromptTemplate per dept
         │
         ▼
    ChatOpenAI via OpenRouter        ← LLM generation
         │
         ▼
    StrOutputParser()
         │
         ▼
    {"answer": str, "sources": list, "department": str}
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from src.prompts_layer.prompts import get_rag_prompt
from src.embedding_layer.embedding_service import EmbeddingService,get_embedding_service
from src.vectordb.qdrant_store import QdrantStore
from src.retrieval.retriever_service import RetrieverService
from src.llm_layer.llm_connecter import LLMConnector
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class RAGChain:
    """
    Full RAG pipeline as a single reusable object.
    All heavy objects are created once and reused across calls.
    """

    def __init__(self) -> None:

        # Embedding model
        self._embedder = get_embedding_service()

        # Qdrant vector store
        self._store = QdrantStore(
            embedding_service=self._embedder,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
        )

        # Retriever wraps the store
        self._retriever = RetrieverService(
            store=self._store,
            default_k=5,
        )

        # LLM
        self._llm    = LLMConnector().get_llm()
        self._parser = StrOutputParser()

        logger.info("RAGChain ready ✓")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(
        self,
        question:   str,
        department: str,
        k:          int = 5,
    ) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            {
                "answer":     str,
                "sources":    list[dict],
                "department": str,
            }
        """
        logger.info(
            f"RAGChain.invoke | "
            f"dept={department} | "
            f"q={question[:80]}"
        )

        # ── Step 1: Retrieve ───────────────────────────────────────────
        docs = self._retriever.retrieve(
            question=question,
            department=department,
            k=k,
        )

        if not docs:
            logger.warning(
                "No documents retrieved — returning fallback"
            )
            return {
                "answer": (
                    "I don't have that information "
                    "in the available documents."
                ),
                "sources":    [],
                "department": department,
            }

        # ── Step 2: Format context ─────────────────────────────────────
        context = RetrieverService.format_context(docs)

        # ── Step 3: Build LCEL chain ───────────────────────────────────
        # get_rag_prompt returns ChatPromptTemplate — pipeable directly
        prompt = get_rag_prompt(department)
        chain  = prompt | self._llm | self._parser

        # ── Step 4: Generate ───────────────────────────────────────────
        answer = chain.invoke({
            "context":  context,
            "question": question,
        })

        logger.info("RAGChain answer generated ✓")

        # Build clean sources list
        sources = [
            {
                "text":       doc.page_content[:300] + "…"
                              if len(doc.page_content) > 300
                              else doc.page_content,
                "filename":   doc.metadata.get("filename", ""),
                "department": doc.metadata.get("department", ""),
                "page":       doc.metadata.get("page"),
                "score":      doc.metadata.get("score", 0.0),
            }
            for doc in docs
        ]

        return {
            "answer":     answer,
            "sources":    sources,
            "department": department,
        }

    # ------------------------------------------------------------------
    # Streaming variant
    # ------------------------------------------------------------------

    def stream(
        self,
        question:   str,
        department: str,
        k:          int = 5,
    ):
        """
        Streaming version — yields answer tokens one by one.

        Usage:
            for token in chain.stream("question", "finance"):
                print(token, end="", flush=True)
        """
        docs = self._retriever.retrieve(
            question=question,
            department=department,
            k=k,
        )

        if not docs:
            yield "I don't have that information in the available documents."
            return

        context = RetrieverService.format_context(docs)
        prompt  = get_rag_prompt(department)
        chain   = prompt | self._llm | self._parser

        for token in chain.stream({
            "context":  context,
            "question": question,
        }):
            yield token