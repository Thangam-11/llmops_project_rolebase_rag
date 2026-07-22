# tests/unit/test_llm_connector.py
from unittest.mock import MagicMock

import pytest

from src.llm_layer.llm_connecter import LLMConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(mocker):
    """
    Patch the module-level `settings` object directly — it's already been
    resolved at import time via `settings = get_settings()`, so patching
    get_settings() itself would have no effect here.
    """
    fake_settings = MagicMock()
    fake_settings.llm_model = "meta-llama/llama-3.3-70b-instruct"
    fake_settings.openrouter_api_key = "test-api-key"
    fake_settings.openrouter_base_url = "https://openrouter.ai/api/v1"

    mocker.patch("src.llm_layer.llm_connecter.settings", fake_settings)
    return fake_settings


@pytest.fixture
def mock_chat_openai(mocker):
    mock_cls = mocker.patch("src.llm_layer.llm_connecter.ChatOpenAI")
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


# ---------------------------------------------------------------------------
# 1. __init__ defaults and overrides
# ---------------------------------------------------------------------------

def test_init_defaults_model_from_settings(mock_settings):
    connector = LLMConnector()

    assert connector._model == "meta-llama/llama-3.3-70b-instruct"
    assert connector._temperature == 0.0
    assert connector._max_tokens == 1000


def test_init_explicit_model_overrides_settings(mock_settings):
    connector = LLMConnector(model="openai/gpt-4o")

    assert connector._model == "openai/gpt-4o"


def test_init_explicit_temperature_and_max_tokens(mock_settings):
    connector = LLMConnector(temperature=0.7, max_tokens=2000)

    assert connector._temperature == 0.7
    assert connector._max_tokens == 2000


def test_init_empty_string_model_falls_back_to_settings(mock_settings):
    # explicit empty string is falsy -> `model or settings.llm_model` kicks in
    connector = LLMConnector(model="")

    assert connector._model == "meta-llama/llama-3.3-70b-instruct"


# ---------------------------------------------------------------------------
# 2. get_llm() — construction params
# ---------------------------------------------------------------------------

def test_get_llm_passes_correct_model_and_credentials(mock_settings, mock_chat_openai):
    mock_cls, _ = mock_chat_openai
    connector = LLMConnector(model="anthropic/claude-3.5-sonnet", temperature=0.2, max_tokens=500)

    connector.get_llm()

    mock_cls.assert_called_once_with(
        model="anthropic/claude-3.5-sonnet",
        openai_api_key="test-api-key",
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=500,
        default_headers={
            "HTTP-Referer": "https://finsolve.internal",
            "X-Title": "FinSolve RAG",
        },
    )


def test_get_llm_returns_chat_openai_instance(mock_settings, mock_chat_openai):
    _, mock_instance = mock_chat_openai
    connector = LLMConnector()

    result = connector.get_llm()

    assert result is mock_instance


def test_get_llm_uses_default_model_when_none_specified(mock_settings, mock_chat_openai):
    mock_cls, _ = mock_chat_openai
    connector = LLMConnector()

    connector.get_llm()

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model"] == "meta-llama/llama-3.3-70b-instruct"


# ---------------------------------------------------------------------------
# 3. health_check() — success path
# ---------------------------------------------------------------------------

def test_health_check_returns_true_on_success(mock_settings, mock_chat_openai):
    _, mock_instance = mock_chat_openai
    mock_instance.invoke.return_value = "pong"

    connector = LLMConnector()
    result = connector.health_check()

    assert result is True
    mock_instance.invoke.assert_called_once_with("ping")


def test_health_check_uses_fixed_max_tokens_5_regardless_of_instance_setting(mock_settings, mock_chat_openai):
    mock_cls, _ = mock_chat_openai
    connector = LLMConnector(max_tokens=2000)   # instance-level setting

    connector.health_check()

    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["max_tokens"] == 5   # health_check always overrides to 5


def test_health_check_does_not_pass_temperature(mock_settings, mock_chat_openai):
    """
    health_check builds its own ChatOpenAI without a `temperature` kwarg —
    confirms this ping call is deliberately minimal/separate from get_llm().
    """
    mock_cls, _ = mock_chat_openai
    connector = LLMConnector(temperature=0.9)

    connector.health_check()

    call_kwargs = mock_cls.call_args.kwargs
    assert "temperature" not in call_kwargs


# ---------------------------------------------------------------------------
# 4. health_check() — failure path (swallows exceptions, unlike other services)
# ---------------------------------------------------------------------------

def test_health_check_returns_false_on_invoke_failure(mock_settings, mock_chat_openai):
    _, mock_instance = mock_chat_openai
    mock_instance.invoke.side_effect = ConnectionError("network down")

    connector = LLMConnector()
    result = connector.health_check()

    assert result is False


def test_health_check_returns_false_on_construction_failure(mock_settings, mocker):
    mocker.patch(
        "src.llm_layer.llm_connecter.ChatOpenAI",
        side_effect=ValueError("bad api key"),
    )

    connector = LLMConnector()
    result = connector.health_check()

    assert result is False


def test_health_check_never_raises(mock_settings, mock_chat_openai):
    """
    Explicitly confirms health_check swallows ANY exception type —
    this must never bubble up, since callers likely use it in a
    liveness/readiness endpoint that shouldn't 500.
    """
    _, mock_instance = mock_chat_openai
    mock_instance.invoke.side_effect = Exception("anything")

    connector = LLMConnector()
    try:
        result = connector.health_check()
    except Exception:
        pytest.fail("health_check() must never raise")

    assert result is False