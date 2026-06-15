"""
src/rag_chain/chain_pipeline.py
================================
Complete LangChain LCEL RAG chain with PII guardrail
and optional RAGAS evaluation.

Flow:
    question + department
         │
         ▼
    PIIGuardrail.scrub_input()       ← scrub PII from question
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

import time

from langchain_core.output_parsers import StrOutputParser

from src.prompts_layer.prompts              import get_rag_prompt
from src.embedding_layer.embedding_service  import get_embedding_service
from src.vectordb.qdrant_store              import QdrantStore
from src.retrieval.retriever_service        import RetrieverService
from src.llm_layer.llm_connecter            import LLMConnector
from src.pil_guardrils.pil_guard            import PIIGuardrail, PIIGuardResult
from src.ragas_evaluation.rags_evaluator    import RagasEvaluator

from config.settings         import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class RAGChain:
    """
    Full RAG pipeline with PII guardrail and optional RAGAS evaluation.
    All heavy objects are created once and reused across calls.
    """

    def __init__(
        self,
        enable_evaluation: bool = False,
    ) -> None:

        # Embedding model — singleton
        self._embedder = get_embedding_service()

        # Qdrant vector store
        self._store = QdrantStore(
            embedding_service=self._embedder,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
        )

        # Retriever
        self._retriever = RetrieverService(
            store=self._store,
            default_k=5,
        )

        # LLM + parser
        self._llm    = LLMConnector().get_llm()
        self._parser = StrOutputParser()

        # PII guardrail — always on
        self._pii = PIIGuardrail()

        # RAGAS evaluator — optional
        self._evaluator   = RagasEvaluator() if enable_evaluation else None
        self._enable_eval = enable_evaluation

        logger.info(
            f"RAGChain ready ✓ | "
            f"evaluation={'on' if enable_evaluation else 'off'}"
        )

    # ------------------------------------------------------------------
    # Invoke — full pipeline
    # ------------------------------------------------------------------

    def invoke(
        self,
        question:   str,
        department: str,
        k:          int = 5,
    ) -> dict:

        start = time.perf_counter()

        logger.info(
            f"RAGChain.invoke | "
            f"dept={department} | "
            f"q={question[:80]}"
        )

        # ── Step 1: Scrub PII from input ───────────────────────────────
        pii_result     = self._pii.scrub_input(question)
        clean_question = pii_result.clean_text

        if pii_result.was_scrubbed:
            logger.info(
                f"Input PII scrubbed: {pii_result.pii_found}"
            )

        # ── Step 2: Retrieve ───────────────────────────────────────────
        docs = self._retriever.retrieve(
            question=clean_question,
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
                "sources":      [],
                "department":   department,
                "latency_ms":   0,
                "was_blocked":  False,
                "pii_scrubbed": pii_result.was_scrubbed,
                "quality":      None,
            }

        # ── Step 3: Format context ─────────────────────────────────────
        context = RetrieverService.format_context(docs)

        # ── Step 4: Build LCEL chain + generate ───────────────────────
        prompt = get_rag_prompt(department)
        chain  = prompt | self._llm | self._parser

        answer = chain.invoke({
            "context":  context,
            "question": clean_question,
        })

        # ── Step 5: Scrub PII from output ─────────────────────────────
        # Force str() — fixes LangChain TextAccessor error
        answer = self._pii.scrub_output(str(answer))

        # ── Step 6: Build sources list ─────────────────────────────────
        sources = [
            {
                "chunk_text": (
                    doc.page_content[:300] + "…"
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "score":      doc.metadata.get("score", 0.0),
                "filename":   doc.metadata.get("filename", ""),
                "department": doc.metadata.get("department", ""),
                "page":       doc.metadata.get("page"),
            }
            for doc in docs
        ]

        latency_ms = round(
            (time.perf_counter() - start) * 1000
        )

        # ── Step 7: RAGAS evaluation (optional) ───────────────────────
        quality = None
        if self._enable_eval and self._evaluator:
            try:
                contexts     = [doc.page_content for doc in docs]
                ragas_result = self._evaluator.evaluate_single(
                    question=question,
                    answer=answer,
                    contexts=contexts,
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
                    f"RAGAS | "
                    f"overall={ragas_result.overall_score:.3f} | "
                    f"pass={quality['pass']}"
                )
            except Exception as e:
                logger.warning(
                    f"RAGAS evaluation failed (non-fatal): {e}"
                )

        logger.info(
            f"RAGChain answer generated ✓ | "
            f"{latency_ms}ms"
        )

        return {
            "answer":       answer,
            "sources":      sources,
            "department":   department,
            "latency_ms":   latency_ms,
            "was_blocked":  False,
            "pii_scrubbed": pii_result.was_scrubbed,
            "quality":      quality,
        }

    # ------------------------------------------------------------------
    # Stream — yields tokens one by one
    # ------------------------------------------------------------------

    def stream(
        self,
        question:   str,
        department: str,
        k:          int = 5,
    ):
        """
        Streaming version — yields answer tokens one by one.
        PII guardrail runs on input.
        Output is collected then scrubbed before final yield.

        Usage:
            for token in chain.stream("question", "finance"):
                print(token, end="", flush=True)
        """

        # ── Step 1: Scrub input ────────────────────────────────────────
        pii_result     = self._pii.scrub_input(question)  # ✅ scrub_input
        clean_question = pii_result.clean_text

        # ── Step 2: Retrieve ───────────────────────────────────────────
        docs = self._retriever.retrieve(
            question=clean_question,
            department=department,
            k=k,
        )

        if not docs:
            yield (
                "I don't have that information "
                "in the available documents."
            )
            return

        # ── Step 3: Build context + chain ──────────────────────────────
        context = RetrieverService.format_context(docs)
        prompt  = get_rag_prompt(department)
        chain   = prompt | self._llm | self._parser

        # ── Step 4: Collect full answer ────────────────────────────────
        # Collect before scrubbing — can't scrub mid-stream
        full_answer = ""
        for token in chain.stream({
            "context":  context,
            "question": clean_question,
        }):
            full_answer += str(token)

        # ── Step 5: Scrub output then yield ───────────────────────────
        clean = self._pii.scrub_output(full_answer)
        yield clean