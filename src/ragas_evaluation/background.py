"""
src/ragas_evaluation/background.py
Fire-and-forget RAGAS scoring, run after the response is already sent.
"""
from __future__ import annotations

from sqlalchemy import update
from models.database import AsyncSessionLocal
from models.model import QueryLog
from src.ragas_evaluation.rags_evaluator import RagasEvaluator
from src.monitoring.metrices import RAGAS_SCORE, RAGAS_METRIC_SCORE
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

_evaluator: RagasEvaluator | None = None


def get_evaluator() -> RagasEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RagasEvaluator()
    return _evaluator


async def evaluate_and_store(
    query_log_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    department: str,
) -> None:
    if not contexts:
        logger.warning(f"Skipping background RAGAS for {query_log_id}: no contexts")
        return

    try:
        evaluator = get_evaluator()
        result = evaluator.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        async with AsyncSessionLocal() as db:
            await db.execute(
                update(QueryLog)
                .where(QueryLog.id == query_log_id)
                .values(
                    faithfulness=result.faithfulness,
                    answer_relevancy=result.answer_relevancy,
                    context_precision=result.context_precision,
                    context_recall=result.context_recall,
                    overall_score=result.overall_score,
                )
            )
            await db.commit()

        RAGAS_SCORE.labels(department=department).set(result.overall_score)
        for metric, value in (
            ("faithfulness", result.faithfulness),
            ("answer_relevancy", result.answer_relevancy),
            ("context_precision", result.context_precision),
            ("context_recall", result.context_recall),
        ):
            RAGAS_METRIC_SCORE.labels(department=department, metric=metric).set(value)

        logger.info(f"Background RAGAS stored | query_id={query_log_id} | overall={result.overall_score:.3f}")

    except Exception:
        logger.exception(f"Background RAGAS evaluation failed for query_id={query_log_id}")