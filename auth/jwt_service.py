from datetime import datetime, timedelta, timezone
from jose import ExpiredSignatureError, JWTError, jwt
from config.settings import get_settings

settings = get_settings()


class TokenExpiredError(Exception):
    """Raised when the token has expired."""

class TokenInvalidError(Exception):
    """Raised when the token is invalid or malformed."""


class JWTService:

    @staticmethod
    def _build_payload(data: dict, expire: datetime) -> dict:
        return {
            **data,
            "iat": datetime.now(timezone.utc),
            "exp": expire,
        }

    @staticmethod
    def create_access_token(data: dict) -> str:
        """Create short-lived access token."""
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
        return jwt.encode(
            JWTService._build_payload(data, expire),
            settings.secret_key,
            algorithm=settings.algorithm,
        )

    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create long-lived refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.refresh_token_expire_days
        )
        return jwt.encode(
            JWTService._build_payload(data, expire),
            settings.secret_key,
            algorithm=settings.algorithm,
        )

    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode and validate JWT. Raises TokenExpiredError or TokenInvalidError on failure."""
        try:
            return jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm],
            )
        except ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except JWTError as e:
            raise TokenInvalidError(f"Invalid token: {e}") from e