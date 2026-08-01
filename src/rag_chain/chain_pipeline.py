"""
src/rag_chain/chain_pipeline.py
================================
Complete LangChain LCEL RAG chain with layered guardrails and
optional RAGAS evaluation.

Flow:
    question + department
         │
         ▼
    is_prompt_injection()            ← Layer 1: fast keyword pre-filter
         │
         ▼
    NemoGuardrailService.check()     ← Layer 2: intent rails (off-topic/jailbreak/sensitive)
         │
         ▼
    PIIGuardrail.scrub_input()       ← Layer 3: scrub PII from question
         │
         ▼
    RetrieverService.retrieve()      ← Layer 4: Qdrant filtered search
         │
         ▼
    filter_safe_docs()               ← Layer 5: drop poisoned/injected doc chunks
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
    PIIGuardrail.scrub_output()      ← Layer 6: scrub/block PII in answer
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

from src.prompts_layer.prompts import get_rag_prompt
from src.embedding_layer.embedding_service import get_embedding_service
from src.vectordb.qdrant_store import QdrantStore
from src.retrieval.retriever_service import RetrieverService
from src.llm_layer.llm_connecter import LLMConnector
from src.pil_guardrils.pil_guard import (
    PIIGuardrail,
    PIIGuardResult,
    is_prompt_injection,
    filter_safe_docs,
)
from src.ragas_evaluation.ragas_evaluator import RagasEvaluator
from src.nemo_guardrils.nemo_guardrail_service import NemoGuardrailService

from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGChain:
    def __init__(
        self,
        enable_evaluation: bool | None = None,
    ) -> None:
        self._pii_enabled = settings.pii_guardrail_enabled
        self._nemo_guard = NemoGuardrailService()

        self._enable_eval = (
            settings.ragas_enabled
            if enable_evaluation is None
            else enable_evaluation
        )
        self._evaluator: RagasEvaluator | None = None
        if self._enable_eval:
            self._evaluator = RagasEvaluator()

        self._embedder = get_embedding_service()

        self._store = QdrantStore(
            embedding_service=self._embedder,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
        )

        self._retriever = RetrieverService(
            store=self._store,
            default_k=5,
        )

        self._llm = LLMConnector().get_llm()
        self._parser = StrOutputParser()

        self._pii = PIIGuardrail() if self._pii_enabled else None

        logger.info(
            f"RAGChain ready | "
            f"evaluation={'on' if self._enable_eval else 'off'} | "
            f"pii_guardrail={'on' if self._pii_enabled else 'off'} | "
            f"nemo_guardrails=on"
        )

    # ------------------------------------------------------------------
    # Shared helper — builds a consistent blocked-response payload
    # ------------------------------------------------------------------

    @staticmethod
    def _blocked_response(
        message: str,
        department: str,
        start: float,
        pii_result: PIIGuardResult | None = None,
    ) -> dict:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return {
            "answer": message,
            "sources": [],
            "department": department,
            "latency_ms": latency_ms,
            "was_blocked": True,
            "pii_scrubbed": pii_result.was_scrubbed if pii_result else False,
            "pii_found": pii_result.pii_found if pii_result else [],
            "pii_count": pii_result.count if pii_result else 0,
            "output_pii_found": [],
            "quality": None,
        }

    def invoke(
        self,
        question: str,
        department: str,
        k: int = 5,
    ) -> dict:
        start = time.perf_counter()

        logger.info(
            f"RAGChain.invoke | "
            f"dept={department} | "
            f"q={question[:80]}"
        )

        # ── Layer 1: fast keyword pre-filter (no LLM cost) ──────────────
        if is_prompt_injection(question):
            logger.warning(f"Blocked (keyword prompt injection): {question[:80]}")
            return self._blocked_response(
                "I maintain consistent guidelines regardless of how I am prompted.",
                department,
                start,
            )

        # ── Layer 2: NeMo intent rails (off-topic/jailbreak/sensitive) ──
        is_blocked, bot_message = self._nemo_guard.check(question)
        if is_blocked:
            logger.warning(f"Blocked (NeMo rail): {question[:80]}")
            return self._blocked_response(
                bot_message or "i can only answer questions about FinSolve company information.",
                department,
                start
            )

        # ── Layer 3: Presidio input PII scrub ────────────────────────────
        pii_result = (
            self._pii.scrub_input(question)
            if self._pii
            else PIIGuardResult(clean_text=question)
        )
        clean_question = pii_result.clean_text

        if pii_result.was_scrubbed:
            logger.info(f"Input PII anonymized: {pii_result.pii_found}")

        # ── Layer 4: retrieval ────────────────────────────────────────────
        docs = self._retriever.retrieve(
            question=clean_question,
            department=department,
            k=k,
        )

        # ── Layer 5: sanitize retrieved docs (indirect prompt injection) ─
        docs = filter_safe_docs(docs)

        if not docs:
            latency_ms = round((time.perf_counter() - start) * 1000)
            return {
                "answer": "I don't have that information in the available documents.",
                "sources": [],
                "department": department,
                "latency_ms": latency_ms,
                "was_blocked": False,
                "pii_scrubbed": pii_result.was_scrubbed,
                "pii_found": pii_result.pii_found,
                "pii_count": pii_result.count,
                "output_pii_found": [],
                "quality": None,
            }

        context = RetrieverService.format_context(docs)
        prompt = get_rag_prompt(department)
        chain = prompt | self._llm | self._parser

        answer = chain.invoke(
            {
                "context": context,
                "question": clean_question,
            }
        )

        # ── Layer 6: Presidio output scrub (redact or hard-block) ────────
        output_pii_found: list[str] = []
        was_blocked = False

        if self._pii:
            output_pii_result = self._pii.scrub_output(str(answer))
            answer = output_pii_result.clean_text
            output_pii_found = output_pii_result.pii_found
            was_blocked = output_pii_result.was_blocked
        else:
            answer = str(answer)

        if was_blocked:
            logger.warning(f"Output blocked — sensitive PII detected: {output_pii_found}")

        sources = [
            {
                "chunk_text": (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
                "score": doc.metadata.get("score", 0.0),
                "filename": doc.metadata.get("filename", ""),
                "department": doc.metadata.get("department", ""),
                "page": doc.metadata.get("page"),
            }
            for doc in docs
        ] if not was_blocked else []

        latency_ms = round((time.perf_counter() - start) * 1000)

        quality = None
        if self._enable_eval and self._evaluator and not was_blocked:
            try:
                contexts = [doc.page_content for doc in docs]
                ragas_result = self._evaluator.evaluate_single(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                )

                quality = {
                    "overall": ragas_result.overall_score,
                    "pass": ragas_result.overall_score >= 0.5,
                    "faithfulness": ragas_result.faithfulness,
                    "answer_relevancy": ragas_result.answer_relevancy,
                    "context_precision": ragas_result.context_precision,
                    "context_recall": ragas_result.context_recall,
                }

                logger.info(
                    f"RAGAS | overall={ragas_result.overall_score:.3f} | "
                    f"pass={quality['pass']}"
                )

            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}")

        logger.info(f"RAGChain answer generated | {latency_ms}ms")

        return {
            "answer": answer,
            "sources": sources,
            "department": department,
            "latency_ms": latency_ms,
            "was_blocked": was_blocked,
            "pii_scrubbed": pii_result.was_scrubbed,
            "pii_found": pii_result.pii_found,
            "pii_count": pii_result.count,
            "output_pii_found": sorted(set(output_pii_found)),
            "quality": quality,
        }

    def stream(
        self,
        question: str,
        department: str,
        k: int = 5,
    ):
        # Layer 1 + 2 apply here too, for consistency with invoke()
        if is_prompt_injection(question):
            logger.warning(f"Blocked stream (keyword prompt injection): {question[:80]}")
            yield "I maintain consistent guidelines regardless of how I am prompted."
            return

        is_blocked, bot_message = self._nemo_guard.check(question)
        if is_blocked:
            logger.warning(f"Blocked stream (NeMo rail): {question[:80]}")
            yield bot_message
            return

        pii_result = (
            self._pii.scrub_input(question)
            if self._pii
            else PIIGuardResult(clean_text=question)
        )
        clean_question = pii_result.clean_text

        docs = self._retriever.retrieve(
            question=clean_question,
            department=department,
            k=k,
        )
        docs = filter_safe_docs(docs)

        if not docs:
            yield "I don't have that information in the available documents."
            return

        context = RetrieverService.format_context(docs)
        prompt = get_rag_prompt(department)
        chain = prompt | self._llm | self._parser

        full_answer = ""
        for token in chain.stream(
            {
                "context": context,
                "question": clean_question,
            }
        ):
            full_answer += str(token)

        if self._pii:
            output_pii_result = self._pii.scrub_output(full_answer)
            yield output_pii_result.clean_text
        else:
            yield full_answer