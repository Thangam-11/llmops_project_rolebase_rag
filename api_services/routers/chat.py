"""
Chat router — with Prometheus metrics + LangSmith tracing
"""
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.database            import get_db
from models.model               import User, QueryLog
from auth.dependencies          import get_current_user
from api_services.schemas.query import (
    QueryRequest,
    QueryResponse,
    QueryHistoryResponse,
    QueryHistoryItem,
)
from src.rag_chain.chain_pipeline import RAGChain
from src.monitoring.metrices     import (
    RAG_REQUESTS,
    RAG_LATENCY,
    PII_BLOCKS,
    RAGAS_SCORE,
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
    req:  QueryRequest,
    user: User         = Depends(get_current_user),
    db:   AsyncSession = Depends(get_db),
):
    chain      = get_rag_chain()
    department = user.department.value
    start      = time.perf_counter()

    try:
        result     = chain.invoke(
            question   = req.question,
            department = department,
            k          = req.top_k,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        # ── Prometheus metrics ─────────────────────────────────────────
        status_label = "blocked" if result.get("blocked") else "success"
        RAG_REQUESTS.labels(department=department, status=status_label).inc()
        RAG_LATENCY.labels(department=department).observe(latency_ms / 1000)
        HTTP_REQUESTS.labels(method="POST", endpoint="/chat/query", status=200).inc()
        HTTP_LATENCY.labels(endpoint="/chat/query").observe(latency_ms / 1000)

        if result.get("blocked") and result.get("pii_found"):
            for pii_type in result["pii_found"]:
                PII_BLOCKS.labels(pii_type=pii_type).inc()

        if result.get("quality"):
            RAGAS_SCORE.labels(department=department).set(
                result["quality"]["overall"]
            )

        if result.get("sources"):
            RETRIEVAL_DOCS.labels(department=department).observe(
                len(result["sources"])
            )

        # ── Map sources chunk_text → text ──────────────────────────────
        mapped_sources = [
            {
                "text":       s.get("chunk_text", ""),
                "score":      s.get("score", 0.0),
                "filename":   s.get("filename"),
                "department": s.get("department"),
                "page":       s.get("page"),
            }
            for s in result.get("sources", [])
        ]

        # ── Log to PostgreSQL ──────────────────────────────────────────
        db.add(QueryLog(
            user_id     = user.id,
            department  = user.department,
            question    = req.question,
            answer      = result.get("answer"),
            sources     = result.get("sources", []),
            latency_ms  = latency_ms,
            was_blocked = result.get("blocked", False),
        ))
        await db.commit()

        # ── Build response ─────────────────────────────────────────────
        return QueryResponse(
            query_id     = str(uuid.uuid4()),
            answer       = result["answer"],
            sources      = mapped_sources,
            department   = department,
            latency_ms   = latency_ms,
            was_blocked  = result.get("blocked", False),
            block_reason = result["answer"] if result.get("blocked") else None,
            pii_found    = result.get("pii_found"),
            quality      = result.get("quality"),
        )

    except Exception as e:
        RAG_REQUESTS.labels(department=department, status="error").inc()
        HTTP_REQUESTS.labels(method="POST", endpoint="/chat/query", status=500).inc()
        logger.exception(f"Query failed: {e}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Query processing failed",
        )


@router.get("/stream")
async def stream(
    question: str,
    user:     User = Depends(get_current_user),
):
    chain      = get_rag_chain()
    department = user.department.value

    HTTP_REQUESTS.labels(method="GET", endpoint="/chat/stream", status=200).inc()

    async def event_stream():
        for token in chain.stream(
            question   = question,
            department = department,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.get("/history", response_model=QueryHistoryResponse)
async def history(
    user: User         = Depends(get_current_user),
    db:   AsyncSession = Depends(get_db),
):
    HTTP_REQUESTS.labels(method="GET", endpoint="/chat/history", status=200).inc()

    result = await db.execute (
        select(QueryLog)
        .where(QueryLog.user_id == user.id)
        .order_by(QueryLog.created_at.desc())
        .limit(20)
    )
    logs  = result.scalars().all()
    items = [
        QueryHistoryItem(
            id          = str(log.id),
            question    = log.question,
            answer      = log.answer,
            department  = log.department.value,
            latency_ms  = log.latency_ms,
            sources     = log.sources,
            was_blocked = log.was_blocked,
            created_at  = log.created_at.isoformat(),
        )
        for log in logs
    ]
    return QueryHistoryResponse(total=len(items), queries=items)