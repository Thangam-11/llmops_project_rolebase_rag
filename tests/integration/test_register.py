import pytest
from fastapi import HTTPException
from sqlalchemy import select

from auth.auth_service import AuthService
from auth.security import verify_password
from models.model import User

pytestmark = pytest.mark.integration


@pytest.fixture
def service():
    return AuthService()


# ---------------------------------------------------------------------------
# 1. Successful registration — real hash, real DB row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_creates_real_user_in_database(service, test_db_session):
    user = await service.register(
        email="alice@example.com",
        username="alice",
        password="correct_horse_battery_staple",
        department="finance",
        role="analyst",
        db=test_db_session,
    )

    result = await test_db_session.execute(select(User).where(User.email == "alice@example.com"))
    found = result.scalar_one_or_none()

    assert found is not None
    assert found.username == "alice"
    assert found.id == user.id


@pytest.mark.asyncio
async def test_register_password_is_really_hashed_and_verifiable(service, test_db_session):
    """
    Confirms the REAL bcrypt hash+verify round-trip works end to end —
    a unit test mocking hash_password can't catch a broken bcrypt
    config, wrong hash rounds, or a hashing/verification mismatch.
    """
    plaintext = "my_actual_password_123"

    user = await service.register(
        email="bob@example.com",
        username="bob",
        password=plaintext,
        department="engineering",
        role="viewer",
        db=test_db_session,
    )

    assert user.hashed_password != plaintext
    assert verify_password(plaintext, user.hashed_password) is True
    assert verify_password("wrong_password", user.hashed_password) is False


# ---------------------------------------------------------------------------
# 2. Real DB-level uniqueness enforcement (the actual backstop)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_duplicate_email_raises_409_via_real_query(service, test_db_session):
    await service.register(
        email="dup@example.com", username="first_user", password="pw123456",
        department="hr", role="viewer", db=test_db_session,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.register(
            email="dup@example.com", username="second_user", password="pw123456",
            department="hr", role="viewer", db=test_db_session,
        )

    assert exc_info.value.status_code == 409
    assert "Email already registered" in exc_info.value.detail


@pytest.mark.asyncio
async def test_register_duplicate_username_raises_409_via_real_query(service, test_db_session):
    await service.register(
        email="user1@example.com", username="dupname", password="pw123456",
        department="marketing", role="viewer", db=test_db_session,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.register(
            email="user2@example.com", username="dupname", password="pw123456",
            department="marketing", role="viewer", db=test_db_session,
        )

    assert exc_info.value.status_code == 409
    assert "Username already taken" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 3. Enum handling with real values from a request-like string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("department,role", [
    ("finance", "viewer"),
    ("hr", "analyst"),
    ("engineering", "manager"),
    ("marketing", "viewer"),
    ("general", "viewer"),
    ("c_level", "admin"),
])
@pytest.mark.asyncio
async def test_register_persists_every_department_and_role_combination(
    service, test_db_session, department, role
):
    user = await service.register(
        email=f"{department}_{role}@example.com",
        username=f"{department}_{role}_user",
        password="pw123456",
        department=department,
        role=role,
        db=test_db_session,
    )

    result = await test_db_session.execute(select(User).where(User.id == user.id))
    found = result.scalar_one()

    assert found.department.value == department
    assert found.role.value == role


@pytest.mark.asyncio
async def test_register_with_invalid_department_string_raises(service, test_db_session):
    """
    Confirms an invalid department string is rejected — either by
    application-level validation before this point, or by the real
    Postgres ENUM column. Either way, garbage must not silently persist.
    """
    with pytest.raises((HTTPException, ValueError, Exception)):
        await service.register(
            email="bad_dept@example.com", username="bad_dept_user", password="pw123456",
            department="not_a_real_department", role="viewer", db=test_db_session,
        )