
import pytest
from fastapi import HTTPException
from sqlalchemy import select

from auth.auth_service import AuthService, _hash_token
from auth.jwt_service import JWTService
from models.model import User, RefreshToken

pytestmark = pytest.mark.integration


@pytest.fixture
def service():
    return AuthService()


async def register_test_user(service, db, **overrides):
    defaults = dict(
        email="loginuser@example.com",
        username="loginuser",
        password="secure_password_123",
        department="finance",
        role="analyst",
    )
    defaults.update(overrides)
    return await service.register(db=db, **defaults)


# ---------------------------------------------------------------------------
# 1. Full register -> login chain, real DB + real JWT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_after_register_returns_valid_real_jwts(service, test_db_session):
    await register_test_user(service, test_db_session)

    result = await service.login(
        email="loginuser@example.com",
        password="secure_password_123",
        db=test_db_session,
    )

    # confirm these are REAL, decodable JWTs, not placeholder strings
    access_payload = JWTService.decode_token(result["access_token"])
    refresh_payload = JWTService.decode_token(result["refresh_token"])

    assert access_payload["role"] == "analyst"
    assert refresh_payload["role"] == "analyst"
    assert access_payload["user_id"] == refresh_payload["user_id"]


@pytest.mark.asyncio
async def test_login_wrong_password_fails_against_real_hash(service, test_db_session):
    await register_test_user(service, test_db_session, email="wrongpw@example.com", username="wrongpw_user")

    with pytest.raises(HTTPException) as exc_info:
        await service.login(
            email="wrongpw@example.com",
            password="totally_wrong_password",
            db=test_db_session,
        )

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_email_fails(service, test_db_session):
    with pytest.raises(HTTPException) as exc_info:
        await service.login(
            email="ghost@example.com", password="anything", db=test_db_session,
        )

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# 2. Refresh token is really stored hashed in Postgres
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_stores_hashed_refresh_token_in_real_db(service, test_db_session):
    await register_test_user(service, test_db_session, email="hashcheck@example.com", username="hashcheck_user")

    result = await service.login(
        email="hashcheck@example.com", password="secure_password_123", db=test_db_session,
    )
    raw_refresh_token = result["refresh_token"]

    db_result = await test_db_session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == _hash_token(raw_refresh_token))
    )
    stored = db_result.scalar_one_or_none()

    assert stored is not None
    assert stored.token_hash != raw_refresh_token   # never the raw token
    assert stored.is_revoked is False


@pytest.mark.asyncio
async def test_login_updates_last_login_in_real_db(service, test_db_session):
    user = await register_test_user(service, test_db_session, email="lastlogin@example.com", username="lastlogin_user")

    result = await test_db_session.execute(select(User).where(User.id == user.id))
    before_login = result.scalar_one()
    assert before_login.last_login is None

    await service.login(email="lastlogin@example.com", password="secure_password_123", db=test_db_session)

    result = await test_db_session.execute(select(User).where(User.id == user.id))
    after_login = result.scalar_one()
    assert after_login.last_login is not None


# ---------------------------------------------------------------------------
# 3. Full chain: register -> login -> refresh -> logout, all against real DB
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_auth_chain_register_login_refresh_logout(service, test_db_session):
    """
    The complete lifecycle in one real flow — proves each step's output
    is genuinely consumable by the next step against a real database,
    which is exactly what unit tests (each testing steps in isolation
    with mocked inputs) cannot guarantee.
    """
    # 1. Register
    await register_test_user(service, test_db_session, email="fullchain@example.com", username="fullchain_user")

    # 2. Login
    login_result = await service.login(
        email="fullchain@example.com", password="secure_password_123", db=test_db_session,
    )
    original_access_token = login_result["access_token"]
    refresh_token = login_result["refresh_token"]

    # 3. Refresh — exchange refresh token for a new access token
    refresh_result = await service.refresh(refresh_token, test_db_session)
    new_access_token = refresh_result["access_token"]

    assert new_access_token != original_access_token   # genuinely new token issued
    new_payload = JWTService.decode_token(new_access_token)
    assert new_payload["role"] == "analyst"

    # 4. Logout — revoke the refresh token
    await service.logout(refresh_token, test_db_session)

    db_result = await test_db_session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == _hash_token(refresh_token))
    )
    revoked_token = db_result.scalar_one()
    assert revoked_token.is_revoked is True

    # 5. Confirm refresh no longer works after logout
    with pytest.raises(HTTPException) as exc_info:
        await service.refresh(refresh_token, test_db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_refresh_with_revoked_token_fails(service, test_db_session):
    await register_test_user(service, test_db_session, email="revoketest@example.com", username="revoketest_user")

    login_result = await service.login(
        email="revoketest@example.com", password="secure_password_123", db=test_db_session,
    )
    refresh_token = login_result["refresh_token"]

    await service.logout(refresh_token, test_db_session)

    with pytest.raises(HTTPException) as exc_info:
        await service.refresh(refresh_token, test_db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_refresh_with_garbage_token_fails(service, test_db_session):
    with pytest.raises(HTTPException) as exc_info:
        await service.refresh("not.a.real.jwt.token", test_db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_logout_nonexistent_token_does_not_raise(service, test_db_session):
    # should be a silent no-op, not an error, per your AuthService.logout() design
    await service.logout("a_token_that_was_never_issued", test_db_session)


# ---------------------------------------------------------------------------
# 4. Two different users get independently valid tokens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_users_login_independently_with_correct_role_claims(service, test_db_session):
    """
    Confirms token claims are correctly scoped per-user — a bug where
    role/department gets cross-contaminated between concurrent logins
    would be a serious access-control failure.
    """
    await register_test_user(
        service, test_db_session, email="userA@example.com", username="userA",
        department="finance", role="viewer",
    )
    await register_test_user(
        service, test_db_session, email="userB@example.com", username="userB",
        department="engineering", role="admin",
    )

    login_a = await service.login(email="userA@example.com", password="secure_password_123", db=test_db_session)
    login_b = await service.login(email="userB@example.com", password="secure_password_123", db=test_db_session)

    payload_a = JWTService.decode_token(login_a["access_token"])
    payload_b = JWTService.decode_token(login_b["access_token"])

    assert payload_a["role"] == "viewer"
    assert payload_b["role"] == "admin"
    assert payload_a["user_id"] != payload_b["user_id"]