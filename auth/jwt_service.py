from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt

from config.settings import get_settings

settings = get_settings()


def create_access_token(data: dict) -> str:
    """
    Create short-lived access token.
    """

    payload = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    payload.update({"exp": expire})

    encoded_jwt = jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm,
    )

    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create long-lived refresh token.
    """

    payload = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(days=7)

    payload.update({"exp": expire})

    encoded_jwt = jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm,
    )

    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT.
    """

    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )

        return payload

    except JWTError:
        return {}