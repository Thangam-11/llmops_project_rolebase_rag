import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, Float, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from models.database import Base


class Department(str, enum.Enum):
    engineering = "engineering"
    hr          = "hr"
    finance     = "finance"
    marketing   = "marketing"
    general     = "general"    # employee level — general info only
    c_level     = "c_level"    # full access to all departments

class UserRole(str, enum.Enum):
    admin    = "admin"      # system admin
    manager  = "manager"    # dept manager — upload + query
    analyst  = "analyst"    # dept analyst — query + view logs
    viewer   = "viewer"     # employee — general info only


# Department → which Qdrant collections they can access
DEPARTMENT_ACCESS = {
    Department.engineering: ["engineering", "general"],
    Department.hr:          ["hr", "general"],
    Department.finance:     ["finance", "general"],
    Department.marketing:   ["marketing", "general"],
    Department.general:     ["general"],
    Department.c_level:     ["engineering", "hr", "finance", "marketing", "general"],
}


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str]    = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None]  = mapped_column(String(255), nullable=True)
    department: Mapped[Department] = mapped_column(SAEnum(Department), nullable=False)
    role: Mapped[UserRole]         = mapped_column(SAEnum(UserRole), nullable=False, default=UserRole.viewer)
    is_active: Mapped[bool]        = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool]      = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime]   = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    documents:      Mapped[list["Document"]]     = relationship(back_populates="owner",  cascade="all, delete-orphan")
    queries:        Mapped[list["QueryLog"]]     = relationship(back_populates="user",   cascade="all, delete-orphan")
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(back_populates="user",   cascade="all, delete-orphan")

    @property
    def allowed_collections(self) -> list[str]:
        """Collections this user can query based on department."""
        if self.role == UserRole.admin:
            return ["engineering", "hr", "finance", "marketing", "general"]
        return DEPARTMENT_ACCESS.get(self.department, ["general"])

    @property
    def primary_collection(self) -> str:
        return self.department.value

    @property
    def can_upload(self) -> bool:
        return self.role in (UserRole.admin, UserRole.manager)

    @property
    def can_view_logs(self) -> bool:
        return self.role in (UserRole.admin, UserRole.manager, UserRole.analyst)

    def __repr__(self):
        return f"<User {self.username} dept={self.department} role={self.role}>"


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID]   = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_revoked: Mapped[bool]     = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(50),  nullable=True)

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID]= mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    department: Mapped[Department] = mapped_column(SAEnum(Department), nullable=False, index=True)
    filename: Mapped[str]      = mapped_column(String(500), nullable=False)
    original_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int]     = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str]     = mapped_column(String(100), nullable=False)
    chunk_count: Mapped[int]   = mapped_column(Integer, default=0)
    status: Mapped[str]        = mapped_column(String(50), default="indexed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    owner:  Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    qdrant_point_id: Mapped[str]   = mapped_column(String(255), nullable=False, index=True)
    chunk_index: Mapped[int]       = mapped_column(Integer, nullable=False)
    content: Mapped[str]           = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime]   = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID]         = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    department: Mapped[Department]    = mapped_column(SAEnum(Department), nullable=False)
    question: Mapped[str]     = mapped_column(Text, nullable=False)
    answer: Mapped[str | None]= mapped_column(Text, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int]   = mapped_column(Integer, default=0)
    was_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    block_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    faithfulness: Mapped[float | None]      = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None]  = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="queries")