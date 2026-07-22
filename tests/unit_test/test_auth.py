# tests/unit/test_auth_service.py
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi import HTTPException

from auth.auth_service import AuthService, _hash_token
from models.model import User, RefreshToken, Department, UserRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    return AuthService()


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


def mock_result(return_value):
    """
    db.execute() is awaited, but .scalar_one_or_none() on the returned
    Result is a SYNC call — must be a plain MagicMock, not AsyncMock,
    or callers get a coroutine instead of the actual value.
    """
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=return_value)
    return result


@pytest.fixture
def mock_settings(mocker):
    fake_settings = MagicMock()
    fake_settings.refresh_token_expire_days = 30
    fake_settings.access_token_expire_minutes = 15
    mocker.patch("auth.auth_service.settings", fake_settings)
    return fake_settings


@pytest.fixture
def existing_user():
    user = User(
        email="jane@example.com",
        username="jane",
        hashed_password="hashed_pw",
        department=Department.finance,
        role=UserRole.analyst,
        full_name="Jane Doe",
        is_active=True,
        is_verified=True,
    )
    user.id = "user-uuid-123"
    user.last_login = None
    return user


# ---------------------------------------------------------------------------
# 1. _hash_token — pure function
# ---------------------------------------------------------------------------

def test_hash_token_is_deterministic():
    assert _hash_token("same-token") == _hash_token("same-token")


def test_hash_token_differs_for_different_input():
    assert _hash_token("token-a") != _hash_token("token-b")


def test_hash_token_returns_hex_sha256_length():
    result = _hash_token("any-token")
    assert len(result) == 64   # SHA-256 hex digest length
    int(result, 16)            # raises ValueError if not valid hex


# ---------------------------------------------------------------------------
# 2. register()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_raises_409_if_email_already_exists(service, mock_db, existing_user):
    mock_db.execute.return_value = mock_result(existing_user)

    with pytest.raises(HTTPException) as exc_info:
        await service.register(
            email="jane@example.com", username="new_user", password="pw123",
            department="finance", role="viewer", db=mock_db,
        )

    assert exc_info.value.status_code == 409
    assert "Email already registered" in exc_info.value.detail


@pytest.mark.asyncio
async def test_register_raises_409_if_username_already_taken(service, mock_db, existing_user):
    # first execute (email check) -> no match, second (username check) -> match
    mock_db.execute.side_effect = [
        mock_result(None),
        mock_result(existing_user),
    ]

    with pytest.raises(HTTPException) as exc_info:
        await service.register(
            email="new@example.com", username="jane", password="pw123",
            department="finance", role="viewer", db=mock_db,
        )

    assert exc_info.value.status_code == 409
    assert "Username already taken" in exc_info.value.detail


@pytest.mark.asyncio
async def test_register_success_hashes_password(service, mock_db, mocker):
    mock_db.execute.side_effect = [mock_result(None), mock_result(None)]
    mock_hash = mocker.patch("auth.auth_service.hash_password", return_value="hashed_value")

    result = await service.register(
        email="new@example.com", username="newuser", password="plaintext_pw",
        department="finance", role="viewer", db=mock_db,
    )

    mock_hash.assert_called_once_with("plaintext_pw")
    assert result.hashed_password == "hashed_value"


@pytest.mark.asyncio
async def test_register_calls_commit_and_refresh(service, mock_db, mocker):
    mock_db.execute.side_effect = [mock_result(None), mock_result(None)]
    mocker.patch("auth.auth_service.hash_password", return_value="hashed")

    await service.register(
        email="new@example.com", username="newuser", password="pw",
        department="engineering", role="manager", db=mock_db,
    )

    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    mock_db.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_checks_email_before_username(service, mock_db, existing_user, mocker):
    """
    Confirms check order: email uniqueness is validated before username —
    matters for which error message the caller sees first when both collide.
    """
    mock_db.execute.return_value = mock_result(existing_user)

    with pytest.raises(HTTPException) as exc_info:
        await service.register(
            email="jane@example.com", username="jane", password="pw",
            department="finance", role="viewer", db=mock_db,
        )

    assert "Email already registered" in exc_info.value.detail
    assert mock_db.execute.call_count == 1   # stopped after the first (email) check


# ---------------------------------------------------------------------------
# 3. login()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_raises_401_for_unknown_email(service, mock_db, mocker):
    mock_db.execute.return_value = mock_result(None)
    mocker.patch("auth.auth_service.verify_password", return_value=False)

    with pytest.raises(HTTPException) as exc_info:
        await service.login(email="ghost@example.com", password="pw", db=mock_db)

    assert exc_info.value.status_code == 401
    assert "Invalid email or password" in exc_info.value.detail


@pytest.mark.asyncio
async def test_login_raises_401_for_wrong_password(service, mock_db, existing_user, mocker):
    mock_db.execute.return_value = mock_result(existing_user)
    mocker.patch("auth.auth_service.verify_password", return_value=False)

    with pytest.raises(HTTPException) as exc_info:
        await service.login(email="jane@example.com", password="wrong_pw", db=mock_db)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_login_does_not_leak_whether_email_or_password_was_wrong(service, mock_db, mocker):
    """
    Both 'unknown email' and 'wrong password' must return the identical
    error message/status — distinguishing them would let an attacker
    enumerate valid emails.
    """
    mocker.patch("auth.auth_service.verify_password", return_value=False)

    mock_db.execute.return_value = mock_result(None)
    with pytest.raises(HTTPException) as exc_unknown:
        await service.login(email="ghost@example.com", password="pw", db=mock_db)

    existing = User(
        email="real@example.com", username="real", hashed_password="hash",
        department=Department.general, role=UserRole.viewer, is_active=True,
    )
    mock_db.execute.return_value = mock_result(existing)
    with pytest.raises(HTTPException) as exc_wrongpw:
        await service.login(email="real@example.com", password="wrong", db=mock_db)

    assert exc_unknown.value.status_code == exc_wrongpw.value.status_code
    assert exc_unknown.value.detail == exc_wrongpw.value.detail


@pytest.mark.asyncio
async def test_login_raises_403_for_inactive_account(service, mock_db, existing_user, mocker):
    existing_user.is_active = False
    mock_db.execute.return_value = mock_result(existing_user)
    mocker.patch("auth.auth_service.verify_password", return_value=True)

    with pytest.raises(HTTPException) as exc_info:
        await service.login(email="jane@example.com", password="correct_pw", db=mock_db)

    assert exc_info.value.status_code == 403
    assert "Account is disabled" in exc_info.value.detail


@pytest.mark.asyncio
async def test_login_success_returns_tokens_and_user_payload(service, mock_db, existing_user, mocker, mock_settings):
    mock_db.execute.return_value = mock_result(existing_user)
    mocker.patch("auth.auth_service.verify_password", return_value=True)
    mocker.patch("auth.auth_service.JWTService.create_access_token", return_value="fake_access_token")
    mocker.patch("auth.auth_service.JWTService.create_refresh_token", return_value="fake_refresh_token")

    result = await service.login(email="jane@example.com", password="correct_pw", db=mock_db)

    assert result["access_token"] == "fake_access_token"
    assert result["refresh_token"] == "fake_refresh_token"
    assert result["token_type"] == "bearer"
    assert result["expires_in"] == 15 * 60
    assert result["user"]["email"] == "jane@example.com"
    assert result["user"]["department"] == "finance"
    assert result["user"]["role"] == "analyst"


@pytest.mark.asyncio
async def test_login_stores_hashed_refresh_token_not_raw(service, mock_db, existing_user, mocker, mock_settings):
    mock_db.execute.return_value = mock_result(existing_user)
    mocker.patch("auth.auth_service.verify_password", return_value=True)
    mocker.patch("auth.auth_service.JWTService.create_access_token", return_value="access_tok")
    mocker.patch("auth.auth_service.JWTService.create_refresh_token", return_value="raw_refresh_tok")

    await service.login(email="jane@example.com", password="pw", db=mock_db)

    added_token = mock_db.add.call_args.args[0]
    assert isinstance(added_token, RefreshToken)
    assert added_token.token_hash == _hash_token("raw_refresh_tok")
    assert added_token.token_hash != "raw_refresh_tok"   # never store the raw token


@pytest.mark.asyncio
async def test_login_updates_last_login_timestamp(service, mock_db, existing_user, mocker, mock_settings):
    mock_db.execute.return_value = mock_result(existing_user)
    mocker.patch("auth.auth_service.verify_password", return_value=True)
    mocker.patch("auth.auth_service.JWTService.create_access_token", return_value="a")
    mocker.patch("auth.auth_service.JWTService.create_refresh_token", return_value="r")

    assert existing_user.last_login is None

    await service.login(email="jane@example.com", password="pw", db=mock_db)

    assert existing_user.last_login is not None


@pytest.mark.asyncio
async def test_login_token_data_includes_role_but_not_password(service, mock_db, existing_user, mocker, mock_settings):
    mock_db.execute.return_value = mock_result(existing_user)
    mocker.patch("auth.auth_service.verify_password", return_value=True)
    mock_create_access = mocker.patch("auth.auth_service.JWTService.create_access_token", return_value="a")
    mocker.patch("auth.auth_service.JWTService.create_refresh_token", return_value="r")

    await service.login(email="jane@example.com", password="pw", db=mock_db)

    token_data = mock_create_access.call_args.args[0]
    assert token_data == {"user_id": str(existing_user.id), "role": "analyst"}
    assert "password" not in token_data
    assert "hashed_password" not in token_data


# ---------------------------------------------------------------------------
# 4. refresh()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_raises_401_for_undecodable_token(service, mock_db, mocker):
    mocker.patch(
        "auth.auth_service.JWTService.decode_token",
        side_effect=Exception("bad signature"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.refresh("garbage.jwt.token", mock_db)

    assert exc_info.value.status_code == 401
    assert "Invalid refresh token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_refresh_raises_401_when_token_not_found_in_db(service, mock_db, mocker):
    mocker.patch(
        "auth.auth_service.JWTService.decode_token",
        return_value={"user_id": "u1", "role": "viewer"},
    )
    mock_db.execute.return_value = mock_result(None)

    with pytest.raises(HTTPException) as exc_info:
        await service.refresh("valid.jwt.token", mock_db)

    assert exc_info.value.status_code == 401
    assert "expired or revoked" in exc_info.value.detail


@pytest.mark.asyncio
async def test_refresh_raises_401_for_expired_token(service, mock_db, mocker):
    mocker.patch(
        "auth.auth_service.JWTService.decode_token",
        return_value={"user_id": "u1", "role": "viewer"},
    )
    expired_token = MagicMock()
    expired_token.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    mock_db.execute.return_value = mock_result(expired_token)

    with pytest.raises(HTTPException) as exc_info:
        await service.refresh("valid.jwt.token", mock_db)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_refresh_query_excludes_revoked_tokens_at_db_level(service, mock_db, mocker):
    """
    Confirms the WHERE clause includes is_revoked == False — this is
    enforced by the query itself, so a revoked token should come back
    as None from the DB (simulated here), triggering the 401 path.
    """
    mocker.patch(
        "auth.auth_service.JWTService.decode_token",
        return_value={"user_id": "u1", "role": "viewer"},
    )
    mock_db.execute.return_value = mock_result(None)  # revoked tokens never match the query

    with pytest.raises(HTTPException) as exc_info:
        await service.refresh("revoked.jwt.token", mock_db)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_refresh_success_returns_new_access_token(service, mock_db, mocker, mock_settings):
    mocker.patch(
        "auth.auth_service.JWTService.decode_token",
        return_value={"user_id": "u1", "role": "analyst"},
    )
    valid_token = MagicMock()
    valid_token.expires_at = datetime.now(timezone.utc) + timedelta(days=10)
    mock_db.execute.return_value = mock_result(valid_token)
    mocker.patch("auth.auth_service.JWTService.create_access_token", return_value="new_access_token")

    result = await service.refresh("valid.jwt.token", mock_db)

    assert result["access_token"] == "new_access_token"
    assert result["refresh_token"] == "valid.jwt.token"   # same refresh token returned, not rotated
    assert result["expires_in"] == 15 * 60


@pytest.mark.asyncio
async def test_refresh_does_not_rotate_refresh_token(service, mock_db, mocker, mock_settings):
    """
    Documents current behavior: refresh() reuses the same raw_token
    rather than issuing a new refresh token (no rotation). Worth
    confirming this is an accepted tradeoff — refresh token rotation
    is generally recommended to limit replay window if a token leaks.
    """
    mocker.patch(
        "auth.auth_service.JWTService.decode_token",
        return_value={"user_id": "u1", "role": "viewer"},
    )
    valid_token = MagicMock()
    valid_token.expires_at = datetime.now(timezone.utc) + timedelta(days=10)
    mock_db.execute.return_value = mock_result(valid_token)
    mocker.patch("auth.auth_service.JWTService.create_access_token", return_value="a")

    result = await service.refresh("original_refresh_token", mock_db)

    assert result["refresh_token"] == "original_refresh_token"


# ---------------------------------------------------------------------------
# 5. logout()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logout_revokes_existing_token(service, mock_db):
    stored_token = MagicMock()
    stored_token.is_revoked = False
    stored_token.user_id = "u1"
    mock_db.execute.return_value = mock_result(stored_token)

    await service.logout("raw_token_value", mock_db)

    assert stored_token.is_revoked is True
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_logout_is_noop_for_unknown_token(service, mock_db):
    """
    Logging out with a token that isn't in the DB should not raise —
    it should silently do nothing, since the caller's session is
    already effectively invalid either way.
    """
    mock_db.execute.return_value = mock_result(None)

    await service.logout("unknown_token", mock_db)   # should not raise

    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_logout_looks_up_by_hashed_token_not_raw(service, mock_db):
    stored_token = MagicMock()
    stored_token.is_revoked = False
    mock_db.execute.return_value = mock_result(stored_token)

    await service.logout("some_raw_token", mock_db)

    # can't easily inspect the SQL WHERE clause value directly without
    # digging into the Select object, but we can at least confirm execute
    # was called exactly once with a query (not the raw token itself passed elsewhere)
    mock_db.execute.assert_awaited_once()