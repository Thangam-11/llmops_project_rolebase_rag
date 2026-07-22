from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=5000
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int


class QueryLogResponse(BaseModel):
    question: str
    answer: str | None
    latency_ms: int
    faithfulness: float | None
    answer_relevancy: float | None

    model_config = {
        "from_attributes": True
    }