from uuid import UUID
from datetime import datetime

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    id: UUID
    filename: str
    original_name: str
    department: str
    chunk_count: int
    status: str

    model_config = {
        "from_attributes": True
    }


class DocumentListResponse(BaseModel):
    id: UUID
    filename: str
    department: str
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class DocumentDetailResponse(BaseModel):
    id: UUID
    filename: str
    original_name: str
    file_size: int
    mime_type: str
    department: str
    chunk_count: int
    status: str
    created_at: datetime

    model_config = {
        "from_attributes": True
    }