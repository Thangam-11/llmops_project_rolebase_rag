"""
Chat router — uses schemas from api_services/schema/
"""
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
from utils.logger_exceptions import get_logger

logger     = get_logger(__name__)
router     = APIRouter(prefix="/chat", tags=["Chat"])
_rag_chain = None


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
    chain  = get_rag_chain()
    result = chain.invoke(
        question=req.question,
        department=user.department.value,
        k=req.top_k,
    )

    # Log to PostgreSQL
    db.add(QueryLog(
        user_id=user.id,
        department=user.department,
        question=req.question,
        answer=result.get("answer"),
        sources=result.get("sources", []),
        latency_ms=result.get("latency_ms", 0),
        was_blocked=result.get("was_blocked", False),
    ))
    await db.flush()

    return result


@router.get("/stream")
async def stream(
    question: str,
    user:     User = Depends(get_current_user),
):
    chain = get_rag_chain()

    async def event_stream():
        for token in chain.stream(
            question=question,
            department=user.department.value,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.get(
    "/history",
    response_model=QueryHistoryResponse,
)
async def history(
    user: User         = Depends(get_current_user),
    db:   AsyncSession = Depends(get_db),
):
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

    return QueryHistoryResponse(
        total=len(items),
        queries=items,
    )