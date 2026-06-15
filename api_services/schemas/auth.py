"""
Auth request/response schemas.
"""
from pydantic import BaseModel, EmailStr, Field, field_validator
from models.model import Department, UserRole


# ── Request schemas ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:      EmailStr
    username:   str        = Field(min_length=3, max_length=100)
    password:   str        = Field(min_length=8, max_length=100)
    full_name:  str | None = None
    department: Department
    role:       UserRole   = UserRole.viewer
    model_config = {"from_attributes": True}  

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
                "Password needs at least one uppercase letter"
            )
        if not any(c.isdigit() for c in v):
            raise ValueError(
                "Password needs at least one digit"
            )
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


# ── Response schemas ───────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id:          str
    email:       str
    username:    str
    full_name:   str | None
    department:  str
    role:        str
    is_active:   bool
    is_verified: bool

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int
    user:          UserResponse


class AccessTokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int


class MeResponse(BaseModel):
    id:                   str
    email:                str
    username:             str
    full_name:            str | None
    department:           str
    role:                 str
    is_active:            bool
    last_login:           str | None
    allowed_collections:  list[str]