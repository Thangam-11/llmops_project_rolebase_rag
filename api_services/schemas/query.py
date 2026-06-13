"""
RAG query request/response schemas.
"""
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=4000,
        description="User question",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


class SourceItem(BaseModel):
    chunk_text: str
    score:      float
    filename:   str | None = None
    department: str | None = None
    page:       int | None = None


class QueryResponse(BaseModel):
    query_id:   str
    answer:     str
    sources:    list[SourceItem]
    department: str
    latency_ms: int
    was_blocked: bool
    block_reason: str | None = None


class QueryHistoryItem(BaseModel):
    id:          str
    question:    str
    answer:      str | None
    department:  str
    latency_ms:  int
    sources:     list | None
    was_blocked: bool
    created_at:  str


class QueryHistoryResponse(BaseModel):
    total:   int
    queries: list[QueryHistoryItem]