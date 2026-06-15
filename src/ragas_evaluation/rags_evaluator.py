"""
src/ragas_evaluation/ragas_evaluator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI

from src.embedding_layer.embedding_service import EmbeddingService
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RagasResult:
    question:          str
    answer:            str
    faithfulness:      float = 0.0
    answer_relevancy:  float = 0.0
    context_precision: float = 0.0
    context_recall:    float = 0.0
    overall_score:     float = 0.0

    def report(self) -> str:
        lines = [
            "=" * 55,
            "  RAGAS EVALUATION",
            "=" * 55,
            f"  Question          : {self.question[:65]}",
            f"  Answer            : {self.answer[:65]}",
            "",
            f"  Faithfulness      : {self.faithfulness:.3f}  {'✅' if self.faithfulness   >= 0.5 else '⚠️'}",
            f"  Answer Relevancy  : {self.answer_relevancy:.3f}  {'✅' if self.answer_relevancy  >= 0.5 else '⚠️'}",
            f"  Context Precision : {self.context_precision:.3f}  {'✅' if self.context_precision >= 0.5 else '⚠️'}",
            f"  Context Recall    : {self.context_recall:.3f}  {'✅' if self.context_recall   >= 0.5 else '⚠️'}",
            "",
            f"  Overall Score     : {self.overall_score:.3f}  {'✅ PASS' if self.overall_score >= 0.5 else '❌ FAIL'}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RagasEvaluator:
    """
    RAGAS evaluator using OpenRouter LLM + embeddings.
    Compatible with ragas 0.2.15.
    """

    def __init__(self) -> None:
        # LLM judge (OpenRouter)
        self._llm = ChatOpenAI(
            model          = settings.llm_model,
            openai_api_key = settings.openrouter_api_key,
            base_url       = settings.openrouter_base_url,
            temperature    = 0,
        )

        # Embeddings for answer relevancy metric
        self._embeddings = EmbeddingService()._embeddings

        self._metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        logger.info("RagasEvaluator ready ✓")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        question:     str,
        answer:       str,
        contexts:     list[str],
        ground_truth: str = "",
    ) -> RagasResult:
        data = {
            "question":    [question],
            "answer":      [answer],
            "contexts":    [contexts],
            "ground_truth":[ground_truth or answer],
        }

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation...")

        scores = evaluate(
            dataset    = dataset,
            metrics    = self._metrics,
            llm        = self._llm,
            embeddings = self._embeddings,
        )

        row = scores.to_pandas().iloc[0]

        faith  = float(row.get("faithfulness",      0.0) or 0.0)
        rel    = float(row.get("answer_relevancy",  0.0) or 0.0)
        prec   = float(row.get("context_precision", 0.0) or 0.0)
        recall = float(row.get("context_recall",    0.0) or 0.0)

        overall = (faith * 0.35 + rel * 0.30 + prec * 0.20 + recall * 0.15)

        result = RagasResult(
            question          = question,
            answer            = answer,
            faithfulness      = round(faith,   3),
            answer_relevancy  = round(rel,     3),
            context_precision = round(prec,    3),
            context_recall    = round(recall,  3),
            overall_score     = round(overall, 3),
        )

        logger.info(
            f"RAGAS done | overall={result.overall_score:.3f} | "
            f"faith={faith:.3f} | rel={rel:.3f}"
        )

        return result

    def evaluate_batch(
        self,
        test_cases: list[dict],
    ) -> list[RagasResult]:
        results = []
        for i, tc in enumerate(test_cases, 1):
            logger.info(f"RAGAS batch: case {i}/{len(test_cases)}")
            try:
                result = self.evaluate_single(
                    question     = tc["question"],
                    answer       = tc["answer"],
                    contexts     = tc["contexts"],
                    ground_truth = tc.get("ground_truth", ""),
                )
                results.append(result)
            except Exception as e:
                logger.exception(f"RAGAS failed for case {i}: {e}")
        return results

    def summary(self, results: list[RagasResult]) -> dict:
        if not results:
            return {}
        n = len(results)
        return {
            "total_cases":      n,
            "avg_faithfulness": round(sum(r.faithfulness      for r in results) / n, 3),
            "avg_relevancy":    round(sum(r.answer_relevancy  for r in results) / n, 3),
            "avg_precision":    round(sum(r.context_precision for r in results) / n, 3),
            "avg_recall":       round(sum(r.context_recall    for r in results) / n, 3),
            "avg_overall":      round(sum(r.overall_score     for r in results) / n, 3),
            "pass_rate":        round(sum(1 for r in results if r.overall_score >= 0.5) / n, 3),
        }


if __name__ == "__main__":
    evaluator = RagasEvaluator()
    result = evaluator.evaluate_single(
        question     = "What is Q4 revenue?",
        answer       = "Q4 revenue was $2.6B",
        contexts     = ["Q4 2024 revenue was $2.6 billion..."],
        ground_truth = "Q4 revenue was $2.6 billion",
    )
    print(result.report())