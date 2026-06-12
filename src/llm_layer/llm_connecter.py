"""
src/llm/llm_connector.py
==========================
LangChain ChatOpenAI pointed at OpenRouter.

OpenRouter is OpenAI API-compatible, so ChatOpenAI works directly
by overriding the base_url and api_key.

Supports any model available on OpenRouter:
    meta-llama/llama-3.3-70b-instruct   (default)
    openai/gpt-4o
    anthropic/claude-3.5-sonnet
    google/gemini-2.0-flash
    mistralai/mistral-large
    ...

Usage:
    llm    = LLMConnector().get_llm()
    result = llm.invoke("Hello")

    # Health check
    ok = LLMConnector().health_check()
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()


class LLMConnector:
    """
    Factory for a LangChain ChatOpenAI instance that targets OpenRouter.

    Args:
        model       : model string (defaults to settings.llm_model)
        temperature : generation temperature (0 = deterministic)
        max_tokens  : max output tokens
    """

    def __init__(
        self,
        model:       str   = "",
        temperature: float = 0.0,
        max_tokens:  int   = 1000,
    ) -> None:
        self._model       = model or settings.llm_model
        self._temperature = temperature
        self._max_tokens  = max_tokens
        logger.info(
            f"LLMConnector | model={self._model} "
            f"| temp={temperature} | max_tokens={max_tokens}"
        )

    def get_llm(self) -> ChatOpenAI:
        """
        Build and return a ChatOpenAI instance targeting OpenRouter.
        The object is lightweight — create one per request or cache it.
        """
        return ChatOpenAI(
            model            = self._model,
            openai_api_key   = settings.openrouter_api_key,
            openai_api_base  = settings.openrouter_base_url,
            temperature      = self._temperature,
            max_tokens       = self._max_tokens,
            # OpenRouter recommended headers
            default_headers  = {
                "HTTP-Referer": "https://finsolve.internal",
                "X-Title":      "FinSolve RAG",
            },
        )

    def health_check(self) -> bool:
        """Ping the LLM with a minimal request. Returns True if reachable."""
        try:
            llm = ChatOpenAI(
                model           = self._model,
                openai_api_key  = settings.openrouter_api_key,
                openai_api_base = settings.openrouter_base_url,
                max_tokens      = 5,
            )
            llm.invoke("ping")
            logger.info("LLM health check ✓")
            return True
        except Exception:
            logger.exception("LLM health check failed")
            return False