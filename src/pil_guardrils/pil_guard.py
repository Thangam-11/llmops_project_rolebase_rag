"""
src/pil_guardrils/pil_guard.py
================================
PII detection and scrubbing using Microsoft Presidio.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class PIIGuardResult:
    """
    Structured result from PIIGuardrail.scrub_input().
    Importable by chain_pipeline.py.
    """
    clean_text:   str
    pii_found:    list[str] = field(default_factory=list)
    was_scrubbed: bool      = False

    def __bool__(self) -> bool:
        return self.was_scrubbed


# ── Entity lists ───────────────────────────────────────────────────────────

INPUT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",
]

OUTPUT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
]


# ── Guard class ────────────────────────────────────────────────────────────

class PIIGuardrail:

    def __init__(self) -> None:
        self._analyzer   = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        logger.info("PIIGuardrail (Presidio) ready ✓")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _to_str(self, text) -> str:
        """
        Force convert to plain str.
        Fixes LangChain TextAccessor → str issue.
        """
        if isinstance(text, str):
            return text
        return str(text)

    def analyze(
        self,
        text,
        entities: list[str] | None = None,
    ) -> list:
        """Detect PII entities. Always converts to str first."""
        clean_text = self._to_str(text)

        if not clean_text.strip():
            return []

        return self._analyzer.analyze(
            text=clean_text,
            language="en",
            entities=entities,
        )

    # ------------------------------------------------------------------
    # Input scrubbing — returns PIIGuardResult
    # ------------------------------------------------------------------

    def scrub_input(self, query) -> PIIGuardResult:
        """
        Scrub PII from user input.
        Returns PIIGuardResult with clean_text and pii_found list.
        """
        clean_text = self._to_str(query)
        results    = self.analyze(
            clean_text,
            entities=INPUT_ENTITIES,
        )

        if not results:
            return PIIGuardResult(
                clean_text=clean_text,
                pii_found=[],
                was_scrubbed=False,
            )

        anonymized = self._anonymizer.anonymize(
            text=clean_text,
            analyzer_results=results,
        )

        pii_types = [r.entity_type for r in results]
        logger.info(f"Input PII scrubbed: {pii_types}")

        return PIIGuardResult(
            clean_text=anonymized.text,
            pii_found=pii_types,
            was_scrubbed=True,
        )

    # ------------------------------------------------------------------
    # Output scrubbing — returns plain str
    # ------------------------------------------------------------------

    def scrub_output(self, answer) -> str:
        """
        Scrub PII from LLM answer.
        Converts TextAccessor → str before Presidio call.
        """
        clean_answer = self._to_str(answer)

        if not clean_answer.strip():
            return clean_answer

        results = self.analyze(
            clean_answer,
            entities=OUTPUT_ENTITIES,
        )

        if not results:
            return clean_answer

        anonymized = self._anonymizer.anonymize(
            text=clean_answer,
            analyzer_results=results,
        )

        pii_types = [r.entity_type for r in results]
        logger.info(f"Output PII scrubbed: {pii_types}")

        return anonymized.text

    # ------------------------------------------------------------------
    # Detection only
    # ------------------------------------------------------------------

    def detect_pii(self, text) -> list[dict]:
        """Detect PII without anonymizing."""
        clean_text = self._to_str(text)
        results    = self.analyze(clean_text)

        return [
            {
                "type":  r.entity_type,
                "score": round(r.score, 3),
                "start": r.start,
                "end":   r.end,
            }
            for r in results
        ]