# tests/unit/test_embedding_service.py
from unittest.mock import MagicMock

import pytest

from src.embedding_layer.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    QUERY_PREFIX,
    EMBEDDING_DIM,
    MODEL_NAME,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hf_embeddings(mocker):
    """
    Patch HuggingFaceEmbeddings where it's imported into embedding_service,
    so no real model download happens.
    """
    mock_cls = mocker.patch("src.embedding_layer.embedding_service.HuggingFaceEmbeddings")
    mock_instance = MagicMock()
    mock_instance.embed_query.return_value = [0.1] * EMBEDDING_DIM
    mock_instance.embed_documents.return_value = [[0.1] * EMBEDDING_DIM]
    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


@pytest.fixture
def service(mock_hf_embeddings):
    return EmbeddingService(model_name="test-model", device="cpu")


# ---------------------------------------------------------------------------
# 1. Prefix behavior — the most important thing in this file
# ---------------------------------------------------------------------------

def test_embed_text_prepends_query_prefix(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    service.embed_text("what is X?")

    mock_instance.embed_query.assert_called_once_with(
        QUERY_PREFIX + "what is X?"
    )


def test_embed_query_prepends_query_prefix(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    service.embed_query("what is X?")

    mock_instance.embed_query.assert_called_once_with(
        QUERY_PREFIX + "what is X?"
    )


def test_embed_texts_does_not_prepend_prefix(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    service.embed_texts(["doc one", "doc two"])

    mock_instance.embed_documents.assert_called_once_with(
        ["doc one", "doc two"]
    )


def test_embed_documents_does_not_prepend_prefix(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    service.embed_documents(["doc one", "doc two"])

    mock_instance.embed_documents.assert_called_once_with(
        ["doc one", "doc two"]
    )


# ---------------------------------------------------------------------------
# 2. Empty-list short circuit
# ---------------------------------------------------------------------------

def test_embed_texts_empty_list_returns_empty_without_calling_model(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    result = service.embed_texts([])

    assert result == []
    mock_instance.embed_documents.assert_not_called()


def test_embed_documents_empty_list_returns_empty_without_calling_model(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    result = service.embed_documents([])

    assert result == []
    mock_instance.embed_documents.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Exception handling — logs and re-raises, doesn't swallow
# ---------------------------------------------------------------------------

def test_embed_text_reraises_on_failure(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings
    mock_instance.embed_query.side_effect = RuntimeError("model error")

    with pytest.raises(RuntimeError, match="model error"):
        service.embed_text("query")


def test_embed_texts_reraises_on_failure(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings
    mock_instance.embed_documents.side_effect = RuntimeError("model error")

    with pytest.raises(RuntimeError, match="model error"):
        service.embed_texts(["doc"])


def test_embed_query_reraises_on_failure(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings
    mock_instance.embed_query.side_effect = RuntimeError("model error")

    with pytest.raises(RuntimeError, match="model error"):
        service.embed_query("query")


def test_embed_documents_reraises_on_failure(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings
    mock_instance.embed_documents.side_effect = RuntimeError("model error")

    with pytest.raises(RuntimeError, match="model error"):
        service.embed_documents(["doc"])


# ---------------------------------------------------------------------------
# 4. cached_property — model instantiated only once
# ---------------------------------------------------------------------------

def test_embeddings_model_instantiated_only_once(service, mock_hf_embeddings):
    mock_cls, _ = mock_hf_embeddings

    service.embed_text("a")
    service.embed_texts(["b"])
    service.embed_query("c")
    service.embed_documents(["d"])

    mock_cls.assert_called_once()   # constructor called exactly once despite 4 calls


def test_embeddings_model_constructed_with_correct_kwargs(mock_hf_embeddings):
    mock_cls, _ = mock_hf_embeddings

    svc = EmbeddingService(model_name="custom-model", device="cuda")
    svc.embed_text("trigger lazy load")

    mock_cls.assert_called_once_with(
        model_name="custom-model",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# 5. as_langchain
# ---------------------------------------------------------------------------

def test_as_langchain_returns_same_underlying_instance(service, mock_hf_embeddings):
    _, mock_instance = mock_hf_embeddings

    service.embed_text("trigger lazy load")   # force cached_property to resolve
    result = service.as_langchain()

    assert result is mock_instance


# ---------------------------------------------------------------------------
# 6. Dimension utilities
# ---------------------------------------------------------------------------

def test_embedding_dimension_returns_768(service):
    assert service.embedding_dimension() == 768


def test_dimension_static_method_returns_768():
    assert EmbeddingService.dimension() == 768


def test_dimension_accessible_without_instance():
    # static method — should not require constructing the service (no model load)
    assert EmbeddingService.dimension() == EMBEDDING_DIM


# ---------------------------------------------------------------------------
# 7. get_embedding_service — singleton behavior
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_singleton_cache():
    """lru_cache persists across tests unless cleared — reset before and after."""
    get_embedding_service.cache_clear()
    yield
    get_embedding_service.cache_clear()


def test_get_embedding_service_returns_same_instance(mocker, mock_hf_embeddings):
    mock_settings = MagicMock()
    mock_settings.embedding_model = "test-model"
    mocker.patch(
        "src.embedding_layer.embedding_service.get_settings",
        return_value=mock_settings,
    )

    first = get_embedding_service()
    second = get_embedding_service()

    assert first is second


def test_get_embedding_service_uses_settings_embedding_model(mocker, mock_hf_embeddings):
    mock_settings = MagicMock()
    mock_settings.embedding_model = "BAAI/bge-large-en-v1.5"
    mocker.patch(
        "src.embedding_layer.embedding_service.get_settings",
        return_value=mock_settings,
    )

    svc = get_embedding_service()

    assert svc._model_name == "BAAI/bge-large-en-v1.5"


def test_get_embedding_service_calls_get_settings_only_once_across_calls(mocker, mock_hf_embeddings):
    mock_settings = MagicMock()
    mock_settings.embedding_model = "test-model"
    mock_get_settings = mocker.patch(
        "src.embedding_layer.embedding_service.get_settings",
        return_value=mock_settings,
    )

    get_embedding_service()
    get_embedding_service()

    mock_get_settings.assert_called_once()   # lru_cache short-circuits subsequent calls