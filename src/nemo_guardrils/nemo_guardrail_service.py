# src/pil_guardrils/nemo_guardrail_service.py
from __future__ import annotations

import os
from nemoguardrails import LLMRails, RailsConfig

from src.nemo_guardrils.nemo_rails_config import (   # ← fixed: pil_guardrils, not nemo_guardrils
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS,
)
from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
settings = get_settings()


class NemoGuardrailService:
    def __init__(self) -> None:
        # OpenRouter is OpenAI-compatible, so this reuses your existing key.
        os.environ.setdefault("OPENAI_API_KEY", settings.openrouter_api_key)

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT,
        )
        self._rails = LLMRails(config)
        logger.info("NeMo Guardrails ready ✓")

    def check(self, question: str) -> tuple[bool, str | None]:
        """
        Returns (blocked, bot_message). If a rail fired, bot_message is
        the canned refusal text — return it directly instead of running
        your normal RAG chain.
        """
        response = self._rails.generate(messages=[{"role": "user", "content": question}])
        text = response.get("content", "") if isinstance(response, dict) else str(response)

        for indicator in RAIL_INDICATORS:
            if indicator in text:
                return True, text

        return False, None