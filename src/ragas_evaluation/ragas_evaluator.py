"""
src/ragas_evaluation/ragas_evaluator.py
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.evaluation import EvaluationResult
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config.settings import get_settings
from src.embedding_layer.embedding_service import get_embedding_service
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RagasResult:
    question: str
    answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0

    def report(self) -> str:
        status = "PASS" if self.overall_score >= 0.5 else "FAIL"

        lines = [
            "=" * 55,
            "RAGAS EVALUATION",
            "=" * 55,
            f"Question          : {self.question[:65]}",
            f"Answer            : {self.answer[:65]}",
            "",
            f"Faithfulness      : {self.faithfulness:.3f}",
            f"Answer Relevancy  : {self.answer_relevancy:.3f}",
            f"Context Precision : {self.context_precision:.3f}",
            f"Context Recall    : {self.context_recall:.3f}",
            "",
            f"Overall Score     : {self.overall_score:.3f} {status}",
            "=" * 55,
        ]
        return "\n".join(lines)


class RagasEvaluator:
    """
    RAGAS evaluator using OpenRouter-compatible ChatOpenAI and
    the same embedding model as the retrieval pipeline.
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0,
        )

        self._embeddings = get_embedding_service().as_langchain()

        self._metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        logger.info("RagasEvaluator ready")

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> RagasResult:
        if not contexts:
            logger.warning("RAGAS skipped because contexts are empty")
            return RagasResult(question=question, answer=answer)

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth or answer],
        }

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation")

        scores = evaluate(
            dataset=dataset,
            metrics=self._metrics,
            llm=self._llm,
            embeddings=self._embeddings,
        )

        if not isinstance(scores, EvaluationResult):
            raise TypeError(f"Expected EvaluationResult from ragas.evaluate, got {type(scores)}")

        row = scores.to_pandas().iloc[0]

        faithfulness_score = self._score(row, "faithfulness")
        relevancy_score = self._score(row, "answer_relevancy")
        precision_score = self._score(row, "context_precision")
        recall_score = self._score(row, "context_recall")

        overall_score = (
            faithfulness_score * 0.35
            + relevancy_score * 0.30
            + precision_score * 0.20
            + recall_score * 0.15
        )

        result = RagasResult(
            question=question,
            answer=answer,
            faithfulness=round(faithfulness_score, 3),
            answer_relevancy=round(relevancy_score, 3),
            context_precision=round(precision_score, 3),
            context_recall=round(recall_score, 3),
            overall_score=round(overall_score, 3),
        )

        logger.info(
            f"RAGAS done | overall={result.overall_score:.3f} | "
            f"faithfulness={result.faithfulness:.3f} | "
            f"relevancy={result.answer_relevancy:.3f}"
        )

        return result

    def evaluate_batch(
        self,
        test_cases: list[dict],
    ) -> list[RagasResult | None]:
        results: list[RagasResult | None] = []

        for index, test_case in enumerate(test_cases, 1):
            logger.info(f"RAGAS batch case {index}/{len(test_cases)}")

            try:
                result = self.evaluate_single(
                    question=test_case["question"],
                    answer=test_case["answer"],
                    contexts=test_case["contexts"],
                    ground_truth=test_case.get("ground_truth", ""),
                )
                results.append(result)

            except Exception as exc:
                logger.exception(f"RAGAS failed for case {index}: {exc}")
                results.append(None)

        return results

    def summary(self, results: list[RagasResult]) -> dict:
        if not results:
            return {}

        total = len(results)

        return {
            "total_cases": total,
            "avg_faithfulness": round(
                sum(item.faithfulness for item in results) / total,
                3,
            ),
            "avg_relevancy": round(
                sum(item.answer_relevancy for item in results) / total,
                3,
            ),
            "avg_precision": round(
                sum(item.context_precision for item in results) / total,
                3,
            ),
            "avg_recall": round(
                sum(item.context_recall for item in results) / total,
                3,
            ),
            "avg_overall": round(
                sum(item.overall_score for item in results) / total,
                3,
            ),
            "pass_rate": round(
                sum(1 for item in results if item.overall_score >= 0.5) / total,
                3,
            ),
        }

    @staticmethod
    def _score(row, key: str) -> float:
        value = row.get(key, 0.0)

        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


if __name__ == "__main__":
    evaluator = RagasEvaluator()
    result = evaluator.evaluate_single(
        question="What is Q4 revenue?",
        answer="Q4 revenue was $2.6B",
        contexts=["Q4 2024 revenue was $2.6 billion."],
        ground_truth="Q4 revenue was $2.6 billion",
    )
    print(result.report())