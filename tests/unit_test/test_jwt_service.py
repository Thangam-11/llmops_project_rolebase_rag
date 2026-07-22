# tests/unit/test_jwt_service.py
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from jose import jwt as real_jwt

from auth.jwt_service import (
    JWTService,
    TokenExpiredError,
    TokenInvalidError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_settings(mocker):
    """
    Patch the module-level `settings` singleton with known, stable test
    values so expiry math and secret/algorithm checks are deterministic.
    """
    fake_settings = MagicMock()
    fake_settings.secret_key = "test-secret-key-for-unit-tests-only"
    fake_settings.algorithm = "HS256"
    fake_settings.access_token_expire_minutes = 15
    fake_settings.refresh_token_expire_days = 30
    mocker.patch("auth.jwt_service.settings", fake_settings)
    return fake_settings


# ---------------------------------------------------------------------------
# 1. Round-trip correctness
# ---------------------------------------------------------------------------

def test_access_token_round_trips_data():
    token = JWTService.create_access_token({"user_id": "u123", "role": "analyst"})

    decoded = JWTService.decode_token(token)

    assert decoded["user_id"] == "u123"
    assert decoded["role"] == "analyst"


def test_refresh_token_round_trips_data():
    token = JWTService.create_refresh_token({"user_id": "u456", "role": "admin"})

    decoded = JWTService.decode_token(token)

    assert decoded["user_id"] == "u456"
    assert decoded["role"] == "admin"


def test_access_and_refresh_tokens_for_same_data_are_different_strings():
    data = {"user_id": "u1", "role": "viewer"}
    access = JWTService.create_access_token(data)
    refresh = JWTService.create_refresh_token(data)

    assert access != refresh   # different expiry -> different payload -> different signature


# ---------------------------------------------------------------------------
# 2. Payload structure — iat/exp always present
# ---------------------------------------------------------------------------

def test_access_token_payload_includes_iat_and_exp():
    token = JWTService.create_access_token({"user_id": "u1"})

    decoded = JWTService.decode_token(token)

    assert "iat" in decoded
    assert "exp" in decoded


def test_access_token_data_key_named_exp_gets_overwritten_by_real_expiry():
    """
    Documents current behavior: _build_payload does {**data, iat:, exp:},
    so if `data` itself contains an "exp" key, the real expiry always
    wins (spread happens first, explicit keys after override it).
    This is the SAFE behavior — a caller can't smuggle a fake expiry
    through the data dict. Locking this in explicitly.
    """
    fake_past_exp = 0  # if this survived, the token would be "already expired"
    token = JWTService.create_access_token({"user_id": "u1", "exp": fake_past_exp})

    decoded = JWTService.decode_token(token)  # must NOT raise TokenExpiredError

    assert decoded["exp"] != fake_past_exp
    assert decoded["user_id"] == "u1"


# ---------------------------------------------------------------------------
# 3. Expiry math — access vs refresh token lifetimes
# ---------------------------------------------------------------------------
def test_access_token_expiry_matches_configured_minutes(mock_settings):
    before = datetime.now(timezone.utc)
    token = JWTService.create_access_token({"user_id": "u1"})

    decoded = JWTService.decode_token(token)
    exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)

    expected_seconds = mock_settings.access_token_expire_minutes * 60
    actual_seconds = (exp - before).total_seconds()

    # allow a couple seconds of slack for JWT's integer-timestamp truncation
    # and test execution time
    assert abs(actual_seconds - expected_seconds) < 3

def test_refresh_token_expiry_matches_configured_days(mock_settings):
    before = datetime.now(timezone.utc)
    token = JWTService.create_refresh_token({"user_id": "u1"})

    decoded = JWTService.decode_token(token)
    exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)

    expected = before + timedelta(days=mock_settings.refresh_token_expire_days)
    assert abs((exp - expected).total_seconds()) < 5   # small tolerance for test execution time


def test_refresh_token_expiry_is_much_later_than_access_token():
    access = JWTService.decode_token(JWTService.create_access_token({"user_id": "u1"}))
    refresh = JWTService.decode_token(JWTService.create_refresh_token({"user_id": "u1"}))

    assert refresh["exp"] > access["exp"]


# ---------------------------------------------------------------------------
# 4. Expired token handling
# ---------------------------------------------------------------------------

def test_decode_expired_token_raises_token_expired_error(mock_settings):
    """
    Manually craft an already-expired token using the real jose.encode,
    rather than mocking, so we exercise the real expiry check inside
    jose's jwt.decode.
    """
    past_expiry = datetime.now(timezone.utc) - timedelta(minutes=5)
    payload = {"user_id": "u1", "iat": datetime.now(timezone.utc) - timedelta(minutes=20), "exp": past_expiry}

    expired_token = real_jwt.encode(
        payload,
        mock_settings.secret_key,
        algorithm=mock_settings.algorithm,
    )

    with pytest.raises(TokenExpiredError, match="Token has expired"):
        JWTService.decode_token(expired_token)


# ---------------------------------------------------------------------------
# 5. Invalid / tampered token handling
# ---------------------------------------------------------------------------

def test_decode_garbage_string_raises_token_invalid_error():
    with pytest.raises(TokenInvalidError):
        JWTService.decode_token("not.a.valid.jwt.token.at.all")


def test_decode_empty_string_raises_token_invalid_error():
    with pytest.raises(TokenInvalidError):
        JWTService.decode_token("")


def test_decode_token_signed_with_wrong_secret_raises_token_invalid_error(mock_settings):
    """
    Critical security test: a token signed with a DIFFERENT secret key
    must be rejected — this is what actually prevents token forgery.
    """
    wrong_secret_token = real_jwt.encode(
        {"user_id": "attacker", "exp": datetime.now(timezone.utc) + timedelta(minutes=10)},
        "a-completely-different-secret-key",
        algorithm=mock_settings.algorithm,
    )

    with pytest.raises(TokenInvalidError):
        JWTService.decode_token(wrong_secret_token)


def test_decode_token_tampered_payload_raises_token_invalid_error(mock_settings):
    """
    Takes a legitimately signed token and flips a character in the
    payload segment — signature verification must catch this.
    """
    token = JWTService.create_access_token({"user_id": "u1", "role": "viewer"})
    header, payload, signature = token.split(".")

    tampered_payload = ("X" + payload[1:]) if payload[0] != "X" else ("Y" + payload[1:])
    tampered_token = f"{header}.{tampered_payload}.{signature}"

    with pytest.raises(TokenInvalidError):
        JWTService.decode_token(tampered_token)


def test_decode_token_wrong_algorithm_configured_raises_token_invalid_error(mock_settings):
    """
    If a token was signed with HS256 but decode_token is configured
    to expect a different algorithm, it must be rejected.
    """
    token = real_jwt.encode(
        {"user_id": "u1", "exp": datetime.now(timezone.utc) + timedelta(minutes=10)},
        mock_settings.secret_key,
        algorithm="HS256",
    )

    mock_settings.algorithm = "HS512"   # decode_token now expects a different algorithm

    with pytest.raises(TokenInvalidError):
        JWTService.decode_token(token)


# ---------------------------------------------------------------------------
# 6. _build_payload — internal helper
# ---------------------------------------------------------------------------

def test_build_payload_merges_data_with_iat_and_exp():
    expire = datetime.now(timezone.utc) + timedelta(minutes=10)
    payload = JWTService._build_payload({"user_id": "u1", "role": "viewer"}, expire)

    assert payload["user_id"] == "u1"
    assert payload["role"] == "viewer"
    assert payload["exp"] == expire
    assert "iat" in payload


def test_build_payload_does_not_mutate_original_data_dict():
    original_data = {"user_id": "u1"}
    expire = datetime.now(timezone.utc) + timedelta(minutes=10)

    JWTService._build_payload(original_data, expire)

    assert original_data == {"user_id": "u1"}   # unchanged — confirms {**data,...} doesn't mutate input