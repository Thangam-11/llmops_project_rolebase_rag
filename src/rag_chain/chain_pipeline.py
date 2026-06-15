"""
src/rag_pipeline/rag_chain.py
===============================
Complete LangChain LCEL RAG chain with PII guardrail and RAGAS evaluation.

Flow:
    question + department
         │
         ▼
    PIIGuardrail.check_input()       ← block if PII in question
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
    PIIGuardrail.scrub_output()      ← scrub PII from answer
         │
         ▼
    RagasEvaluator.evaluate_single() ← optional quality scoring
         │
         ▼
    {"answer": str, "sources": list, "department": str, "quality": dict}
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from src.prompts_layer.prompts           import get_rag_prompt
from src.embedding_layer.embedding_service import EmbeddingService, get_embedding_service
from src.vectordb.qdrant_store           import QdrantStore
from src.retrieval.retriever_service     import RetrieverService
from src.llm_layer.llm_connecter         import LLMConnector
from src.pil_guardrils.pil_guard         import PIIGuardrail, PIIGuardResult
from src.ragas_evaluation.rags_evaluator import RagasEvaluator

from config.settings          import get_settings
from utils.logger_exceptions  import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class RAGChain:
    """
    Full RAG pipeline with PII guardrail and optional RAGAS evaluation.
    All heavy objects are created once and reused across calls.
    """

    def __init__(self, enable_evaluation: bool = False) -> None:

        # Embedding model
        self._embedder = get_embedding_service()

        # Qdrant vector store
        self._store = QdrantStore(
            embedding_service = self._embedder,
            url               = settings.qdrant_url,
            api_key           = settings.qdrant_api_key,
            collection_name   = settings.collection_name,
        )

        # Retriever wraps the store
        self._retriever = RetrieverService(
            store     = self._store,
            default_k = 5,
        )

        # LLM
        self._llm    = LLMConnector().get_llm()
        self._parser = StrOutputParser()

        # PII guardrail — always on
        self._pii = PIIGuardrail()

        # RAGAS evaluator — optional (adds ~10s per call)
        self._evaluator      = RagasEvaluator() if enable_evaluation else None
        self._enable_eval    = enable_evaluation

        logger.info(
            f"RAGChain ready ✓ | "
            f"evaluation={'on' if enable_evaluation else 'off'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(
        self,
        question:   str,
        department: str,
        k:          int  = 5,
    ) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            {
                "answer":     str,
                "sources":    list[dict],
                "department": str,
                "blocked":    bool,           # True if PII blocked the request
                "quality":    dict | None,    # RAGAS scores (if evaluation on)
            }
        """
        logger.info(
            f"RAGChain.invoke | dept={department} | q={question[:80]}"
        )

        # ── Step 1: PII check on input ─────────────────────────────────
        pii_result = self._pii.check_input(question)
        if not pii_result.allowed:
            logger.warning(f"Request blocked by PII guardrail: {pii_result.pii_found}")
            return {
                "answer":     pii_result.reason,
                "sources":    [],
                "department": department,
                "blocked":    True,
                "pii_found":   pii_result.pii_found,
                "quality":    None,
            }

        # ── Step 2: Retrieve ───────────────────────────────────────────
        docs = self._retriever.retrieve(
            question   = question,
            department = department,
            k          = k,
        )

        if not docs:
            logger.warning("No documents retrieved — returning fallback")
            return {
                "answer":     "I don't have that information in the available documents.",
                "sources":    [],
                "department": department,
                "blocked":    False,
                "quality":    None,
            }

        # ── Step 3: Format context ─────────────────────────────────────
        context = RetrieverService.format_context(docs)

        # ── Step 4: Build LCEL chain and generate ─────────────────────
        prompt = get_rag_prompt(department)
        chain  = prompt | self._llm | self._parser

        answer = chain.invoke({
            "context":  context,
            "question": question,
        })

        # ── Step 5: Scrub PII from answer ─────────────────────────────
        answer = self._pii.scrub_output(answer)

        logger.info("RAGChain answer generated ✓")

        # ── Step 6: Build sources list ─────────────────────────────────
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

        # ── Step 7: RAGAS evaluation (optional) ───────────────────────
        quality = None
        if self._enable_eval:
            try:
                contexts    = [doc.page_content for doc in docs]
                ragas_result = self._evaluator.evaluate_single(
                    question = question,
                    answer   = answer,
                    contexts = contexts,
                )
                quality = {
                    "overall":           ragas_result.overall_score,
                    "pass":              ragas_result.overall_score >= 0.5,
                    "faithfulness":      ragas_result.faithfulness,
                    "answer_relevancy":  ragas_result.answer_relevancy,
                    "context_precision": ragas_result.context_precision,
                    "context_recall":    ragas_result.context_recall,
                }
                logger.info(
                    f"RAGAS | overall={ragas_result.overall_score} | "
                    f"pass={quality['pass']}"
                )
            except Exception as e:
                logger.exception(f"RAGAS evaluation failed (non-fatal): {e}")

        return {
            "answer":     answer,
            "sources":    sources,
            "department": department,
            "blocked":    False,
            "quality":    quality,
        }

    # ------------------------------------------------------------------
    # Streaming variant (PII guardrail only — no RAGAS during stream)
    # ------------------------------------------------------------------

    def stream(
        self,
        question:   str,
        department: str,
        k:          int = 5,
    ):
        """
        Streaming version — yields answer tokens one by one.
        PII guardrail runs on input; output is scrubbed after full generation.

        Usage:
            for token in chain.stream("question", "finance"):
                print(token, end="", flush=True)
        """
        # PII check on input
        pii_result = self._pii.check_input(question)
        if not pii_result.allowed:
            yield pii_result.reason
            return

        docs = self._retriever.retrieve(
            question   = question,
            department = department,
            k          = k,
        )

        if not docs:
            yield "I don't have that information in the available documents."
            return

        context = RetrieverService.format_context(docs)
        prompt  = get_rag_prompt(department)
        chain   = prompt | self._llm | self._parser

        # Collect full answer for PII scrubbing
        full_answer = ""
        for token in chain.stream({"context": context, "question": question}):
            full_answer += token

        # Scrub output then yield as one clean chunk
        clean = self._pii.scrub_output(full_answer)
        yield clean