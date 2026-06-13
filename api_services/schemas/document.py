"""
Document ingestion schemas.
"""
from datetime import datetime
from pydantic import BaseModel
from models.model import Department


class DocumentResponse(BaseModel):
    id:           str
    filename:     str
    department:   str
    file_size:    int
    chunk_count:  int
    status:       str
    created_at:   str

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    files:       int
    chunks:      int
    department:  str
    status:      str = "success"


class CollectionInfoResponse(BaseModel):
    name:         str
    vector_count: int
    status:       str
    departments:  dict[str, int]