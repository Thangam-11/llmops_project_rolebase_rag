# tests/unit/test_db_session.py
from unittest.mock import MagicMock, AsyncMock

import pytest

from models.database import get_db
from config.settings import get_settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session():
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_session_local(mocker, mock_session):
    """
    AsyncSessionLocal() returns an object used as `async with ... as session`.
    Mock the async context manager protocol directly.
    """
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)  # False = don't suppress exceptions

    mock_factory = MagicMock(return_value=mock_ctx)
    mocker.patch("models.database.AsyncSessionLocal", mock_factory)
    return mock_factory


# ---------------------------------------------------------------------------
# 1. Success path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_db_yields_session(mock_session_local, mock_session):
    gen = get_db()
    session = await gen.__anext__()

    assert session is mock_session

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_get_db_commits_on_success(mock_session_local, mock_session):
    gen = get_db()
    await gen.__anext__()

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    mock_session.commit.assert_awaited_once()
    mock_session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_get_db_closes_session_on_success(mock_session_local, mock_session):
    gen = get_db()
    await gen.__anext__()

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

    mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. Failure path — rollback, re-raise, still close
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_db_rolls_back_on_exception(mock_session_local, mock_session):
    gen = get_db()
    await gen.__anext__()

    with pytest.raises(ValueError, match="boom"):
        await gen.athrow(ValueError("boom"))

    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_get_db_reraises_original_exception(mock_session_local, mock_session):
    """
    Confirms the except block re-raises rather than swallowing —
    a caller relying on this dependency must see the failure.
    """
    gen = get_db()
    await gen.__anext__()

    with pytest.raises(RuntimeError, match="db write failed"):
        await gen.athrow(RuntimeError("db write failed"))


@pytest.mark.asyncio
async def test_get_db_closes_session_even_on_exception(mock_session_local, mock_session):
    """
    The `finally: await session.close()` must run regardless of
    success or failure — this is the connection-leak guarantee.
    """
    gen = get_db()
    await gen.__anext__()

    with pytest.raises(ValueError):
        await gen.athrow(ValueError("boom"))

    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_db_does_not_call_commit_after_rollback(mock_session_local, mock_session):
    gen = get_db()
    await gen.__anext__()

    with pytest.raises(ValueError):
        await gen.athrow(ValueError("boom"))

    assert mock_session.commit.await_count == 0
    assert mock_session.rollback.await_count == 1


# ---------------------------------------------------------------------------
# 3. Engine construction config (import-time settings)
# ---------------------------------------------------------------------------
def test_engine_echo_true_in_development(mocker):
    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/db"
    mock_settings.environment = "development"
    mocker.patch("config.settings.get_settings", return_value=mock_settings)

    mock_create_engine = mocker.patch("sqlalchemy.ext.asyncio.create_async_engine")

    import importlib
    import models.database as db_module
    importlib.reload(db_module)

    call_kwargs = mock_create_engine.call_args.kwargs
    assert call_kwargs["echo"] is True


def test_engine_echo_false_in_production(mocker):
    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/db"
    mock_settings.environment = "production"
    mocker.patch("config.settings.get_settings", return_value=mock_settings)

    mock_create_engine = mocker.patch("sqlalchemy.ext.asyncio.create_async_engine")

    import importlib
    import models.database as db_module
    importlib.reload(db_module)

    call_kwargs = mock_create_engine.call_args.kwargs
    assert call_kwargs["echo"] is False


def test_engine_echo_is_case_insensitive(mocker):
    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/db"
    mock_settings.environment = "DEVELOPMENT"
    mocker.patch("config.settings.get_settings", return_value=mock_settings)

    mock_create_engine = mocker.patch("sqlalchemy.ext.asyncio.create_async_engine")

    import importlib
    import models.database as db_module
    importlib.reload(db_module)

    call_kwargs = mock_create_engine.call_args.kwargs
    assert call_kwargs["echo"] is True


def test_engine_pool_settings_are_passed_through(mocker):
    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/db"
    mock_settings.environment = "production"
    mocker.patch("config.settings.get_settings", return_value=mock_settings)

    mock_create_engine = mocker.patch("sqlalchemy.ext.asyncio.create_async_engine")

    import importlib
    import models.database as db_module
    importlib.reload(db_module)

    call_kwargs = mock_create_engine.call_args.kwargs
    assert call_kwargs["pool_size"] == 10
    assert call_kwargs["max_overflow"] == 20
    assert call_kwargs["pool_pre_ping"] is True