from uuid import UUID

from pydantic import (
    BaseModel,
    EmailStr,
)


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str | None = None
    department: str


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    username: str
    full_name: str | None
    department: str
    role: str

    model_config = {
        "from_attributes": True
    }