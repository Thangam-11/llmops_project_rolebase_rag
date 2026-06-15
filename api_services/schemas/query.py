"""
RAG query request/response schemas.
"""
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        min_length   = 3,
        max_length   = 4000,
        description  = "User question",
    )
    top_k: int = Field(
        default     = 5,
        ge          = 1,
        le          = 20,
        description = "Number of chunks to retrieve",
    )


# ── Nested ────────────────────────────────────────────────────────────────────

class SourceItem(BaseModel):
    text:       str             # ← was chunk_text, matches rag_chain.py output
    score:      float
    filename:   str | None = None
    department: str | None = None
    page:       int | None = None


class QualityScores(BaseModel):
    """RAGAS evaluation scores — only present when enable_evaluation=True."""
    overall:           float
    pass_:             bool  = Field(alias="pass")
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    context_recall:    float

    model_config = {"populate_by_name": True}


# ── Response ──────────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    query_id:     str
    answer:       str
    sources:      list[SourceItem]
    department:   str
    latency_ms:   int
    was_blocked:  bool
    block_reason: str | None        = None
    pii_found:    list[str] | None  = None   # ← PII types caught e.g. ["US_SSN"]
    quality:      QualityScores | None = None # ← RAGAS scores if evaluation on


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