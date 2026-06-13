"""
User management schemas.
"""
from pydantic import BaseModel, EmailStr, Field, field_validator
from models.model import Department, UserRole


class UserCreate(BaseModel):
    email:      EmailStr
    username:   str        = Field(min_length=3, max_length=100)
    password:   str        = Field(min_length=8, max_length=100)
    full_name:  str | None = None
    department: Department
    role:       UserRole   = UserRole.viewer

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username may only contain letters, "
                "numbers, _ and -"
            )
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strong(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError(
                "Need at least one uppercase letter"
            )
        if not any(c.isdigit() for c in v):
            raise ValueError(
                "Need at least one digit"
            )
        return v


class UserResponse(BaseModel):
    id:          str
    email:       str
    username:    str
    full_name:   str | None
    department:  str
    role:        str
    is_active:   bool
    is_verified: bool
    created_at:  str

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    full_name:  str | None   = None
    email:      EmailStr | None = None


class UserRoleUpdate(BaseModel):
    """Admin only — change role or department."""
    role:       UserRole   | None = None
    department: Department | None = None


class UserListResponse(BaseModel):
    total: int
    users: list[UserResponse]