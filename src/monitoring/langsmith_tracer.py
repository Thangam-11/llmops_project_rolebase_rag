"""
src/monitoring/langsmith_tracer.py
====================================
LangSmith tracing setup.
Enables full LangChain call tracing — every LLM call,
retrieval step, prompt, token count, and latency is
logged to LangSmith automatically once env vars are set.
"""

import os

from utils.logger_exceptions import get_logger
from config.settings import get_settings

logger   = get_logger(__name__)
settings = get_settings()


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing via environment variables.
    LangChain picks these up automatically — no code changes
    needed in the chain itself.

    Returns True if tracing is enabled.
    """
    # Check if enabled and key exists
    if not settings.langsmith_enabled:
        logger.info("LangSmith tracing disabled (LANGSMITH_ENABLED=false)")
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LangSmith enabled but LANGSMITH_API_KEY is empty — "
            "tracing will not work"
        )
        return False

    # Set environment variables — LangChain reads these automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]    = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

    logger.info(
        f"LangSmith tracing enabled ✓ | "
        f"project='{settings.langsmith_project}'"
    )
    return True


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently active."""
    return (
        os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        and bool(os.environ.get("LANGCHAIN_API_KEY"))
    )