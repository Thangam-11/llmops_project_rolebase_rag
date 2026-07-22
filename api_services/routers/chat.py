"""
Chat router with Prometheus metrics and LangSmith tracing.
"""
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db
from models.model import User, QueryLog
from auth.dependencies import get_current_user
from api_services.schemas.query import (
    QueryRequest,
    QueryResponse,
    QueryHistoryResponse,
    QueryHistoryItem,
)
from src.rag_chain.chain_pipeline import RAGChain
from src.monitoring.metrices import (
    RAG_REQUESTS,
    RAG_LATENCY,
    PII_BLOCKS,
    PII_DETECTIONS,
    RAGAS_SCORE,
    RAGAS_METRIC_SCORE,
    RETRIEVAL_DOCS,
    HTTP_REQUESTS,
    HTTP_LATENCY,
)
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chain = get_rag_chain()
    department = user.department.value
    start = time.perf_counter()
    try:
        result = await run_in_threadpool(
            chain.invoke,
            question=req.question,
            department=department,
            k=req.top_k,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        status_label = "blocked" if result.get("was_blocked") else "success"
        RAG_REQUESTS.labels(department=department, status=status_label).inc()
        RAG_LATENCY.labels(department=department).observe(latency_ms / 1000)
        HTTP_REQUESTS.labels(method="POST", endpoint="/chat/query", status=200).inc()
        HTTP_LATENCY.labels(endpoint="/chat/query").observe(latency_ms / 1000)

        if result.get("pii_found"):
            for pii_type in result["pii_found"]:
                PII_DETECTIONS.labels(
                    direction="input",
                    pii_type=pii_type,
                ).inc()

        if result.get("output_pii_found"):
            for pii_type in result["output_pii_found"]:
                PII_DETECTIONS.labels(
                    direction="output",
                    pii_type=pii_type,
                ).inc()

        if result.get("was_blocked") and result.get("pii_found"):
            for pii_type in result["pii_found"]:
                PII_BLOCKS.labels(pii_type=pii_type).inc()

        if result.get("quality"):
            RAGAS_SCORE.labels(department=department).set(
                result["quality"]["overall"]
            )
            for metric in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ):
                RAGAS_METRIC_SCORE.labels(
                    department=department,
                    metric=metric,
                ).set(result["quality"][metric])

        if result.get("sources"):
            RETRIEVAL_DOCS.labels(department=department).observe(
                len(result["sources"])
            )

        mapped_sources = [
            {
                "text": s.get("chunk_text", ""),
                "score": s.get("score", 0.0),
                "filename": s.get("filename"),
                "department": s.get("department"),
                "page": s.get("page"),
            }
            for s in result.get("sources", [])
        ]

        query_log = QueryLog(
            user_id=user.id,
            department=user.department,
            question=req.question,
            answer=result.get("answer"),
            sources=result.get("sources", []),
            latency_ms=latency_ms,
            was_blocked=result.get("was_blocked", False),
        )
        db.add(query_log)
        await db.commit()
        await db.refresh(query_log)  # get the generated id back

        # Schedule background RAGAS evaluation (only if not already scored
        # and not blocked, to avoid wasting eval calls on PII-blocked answers)
        if not result.get("was_blocked") and not result.get("quality"):
            try:
                from src.ragas_evaluation.background import evaluate_and_store
                background_tasks.add_task(
                    evaluate_and_store,
                    query_id=query_log.id,
                    question=req.question,
                    answer=result.get("answer"),
                    sources=result.get("sources", []),
                    department=department,
                )
            except Exception:
                logger.exception(
                    "Skipping background RAGAS evaluation: evaluate_and_store "
                    "unavailable (check ragas / langchain_community versions)"
                )

        return QueryResponse(
            query_id=str(query_log.id),
            answer=result["answer"],
            sources=mapped_sources,
            department=department,
            latency_ms=latency_ms,
            was_blocked=result.get("was_blocked", False),
            block_reason=result["answer"] if result.get("was_blocked") else None,
            pii_found=result.get("pii_found"),
            quality=result.get("quality"),
        )

    except Exception as e:
        await db.rollback()
        RAG_REQUESTS.labels(department=department, status="error").inc()
        HTTP_REQUESTS.labels(method="POST", endpoint="/chat/query", status=500).inc()
        logger.exception(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing failed",
        )


@router.get("/stream")
async def stream(
    question: str,
    user: User = Depends(get_current_user),
):
    chain = get_rag_chain()
    department = user.department.value

    HTTP_REQUESTS.labels(method="GET", endpoint="/chat/stream", status=200).inc()

    async def event_stream():
        for token in chain.stream(
            question=question,
            department=department,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.get("/history", response_model=QueryHistoryResponse)
async def history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    HTTP_REQUESTS.labels(method="GET", endpoint="/chat/history", status=200).inc()

    result = await db.execute(
        select(QueryLog)
        .where(QueryLog.user_id == user.id)
        .order_by(QueryLog.created_at.desc())
        .limit(20)
    )

    logs = result.scalars().all()

    items = [
        QueryHistoryItem(
            id=str(log.id),
            question=log.question,
            answer=log.answer,
            department=log.department.value,
            latency_ms=log.latency_ms,
            sources=log.sources,
            was_blocked=log.was_blocked,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]

    return QueryHistoryResponse(total=len(items), queries=items)
