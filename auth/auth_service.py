"""
auth/auth_service.py
====================
Handles user registration, login, token refresh, and logout.
"""
import hashlib
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.jwt_service import JWTService
from auth.security import hash_password, verify_password
from models.model import User, RefreshToken
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


def _hash_token(raw_token: str) -> str:
    """SHA-256 hash of refresh token for safe DB storage."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


class AuthService:

    # ── Register ──────────────────────────────────────────────────────────────

    async def register(
        self,
        email:      str,
        username:   str,
        password:   str,
        department: str,
        role:       str,
        db:         AsyncSession,
        full_name:  str | None = None,
    ) -> User:
        result = await db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )

        result = await db.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already taken",
            )

        user = User(
            email           = email,
            username        = username,
            hashed_password = hash_password(password),
            full_name       = full_name,
            department      = department,
            role            = role,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info(f"User registered: {username} | dept={department} role={role}")
        return user

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(
        self,
        email:      str,
        password:   str,
        db:         AsyncSession,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> dict:
        result = await db.execute(select(User).where(User.email == email))
        user   = result.scalar_one_or_none()

        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled",
            )

        # Create tokens
        token_data    = {"user_id": str(user.id), "role": user.role.value}
        access_token  = JWTService.create_access_token(token_data)
        refresh_token = JWTService.create_refresh_token(token_data)

        # Store hashed refresh token in DB
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=settings.refresh_token_expire_days
        )
        db.add(RefreshToken(
            user_id    = user.id,
            token_hash = _hash_token(refresh_token),
            expires_at = expires_at,
            user_agent = user_agent,
            ip_address = ip_address,
        ))

        # Update last login
        user.last_login = datetime.now(timezone.utc)
        await db.commit()

        logger.info(f"User logged in: {user.username}")
        return {
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "token_type":    "bearer",
            "expires_in":    settings.access_token_expire_minutes * 60,
            "user": {
                "id":          str(user.id),
                "email":       user.email,
                "username":    user.username,
                "full_name":   user.full_name,
                "department":  user.department.value,
                "role":        user.role.value,
                "is_active":   user.is_active,
                "is_verified": user.is_verified,
            }
        }

    # ── Refresh ───────────────────────────────────────────────────────────────

    async def refresh(self, raw_token: str, db: AsyncSession) -> dict:
        try:
            payload = JWTService.decode_token(raw_token)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        token_hash = _hash_token(raw_token)
        result     = await db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.is_revoked == False,
            )
        )
        stored = result.scalar_one_or_none()

        if not stored or stored.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token expired or revoked",
            )

        token_data   = {"user_id": payload["user_id"], "role": payload["role"]}
        access_token = JWTService.create_access_token(token_data)
        logger.info(f"Token refreshed for user_id={payload['user_id']}")
        return {
            "access_token":  access_token,
            "refresh_token": raw_token,
            "token_type":    "bearer",
            "expires_in":    settings.access_token_expire_minutes * 60,
        }

    # ── Logout ────────────────────────────────────────────────────────────────

    async def logout(self, raw_token: str, db: AsyncSession) -> None:
        token_hash = _hash_token(raw_token)
        result     = await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        stored = result.scalar_one_or_none()
        if stored:
            stored.is_revoked = True
            await db.commit()
            logger.info(f"Refresh token revoked for user_id={stored.user_id}")