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

from src.prompts_layer.prompts import get_rag_prompt
from src.embedding_layer.embedding_service import get_embedding_service
from src.ragas_evaluation.rags_evaluator import RagasEvaluator
from src.vectordb.qdrant_store import QdrantStore
from src.retrieval.retriever_service import RetrieverService
from src.llm_layer.llm_connecter import LLMConnector
from src.pil_guardrils.pil_guard import PIIGuardrail, PIIGuardResult
#from src.ragas_evaluation.rags_evaluator import RagasEvaluator

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
        #self._evaluator = RagasEvaluator() if self._enable_eval else None

        logger.info(
            f"RAGChain ready | "
            f"evaluation={'on' if self._enable_eval else 'off'} | "
            f"pii_guardrail={'on' if self._pii_enabled else 'off'}"
        )

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

        pii_result = (
            self._pii.scrub_input(question)
            if self._pii
            else PIIGuardResult(clean_text=question)
        )
        clean_question = pii_result.clean_text

        if pii_result.was_scrubbed:
            logger.info(f"Input PII anonymized: {pii_result.pii_found}")

        docs = self._retriever.retrieve(
            question=clean_question,
            department=department,
            k=k,
        )

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

        output_pii_found = []
        if self._pii:
            output_pii_found = [
                item["type"]
                for item in self._pii.detect_pii(str(answer))
            ]
            answer = self._pii.scrub_output(str(answer))
        else:
            answer = str(answer)

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
        ]

        latency_ms = round((time.perf_counter() - start) * 1000)

        quality = None
        if self._enable_eval and self._evaluator:
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
            "was_blocked": False,
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

        clean = self._pii.scrub_output(full_answer) if self._pii else full_answer
        yield clean