from __future__ import annotations

from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


@dataclass
class PIIGuardResult:
    clean_text: str
    pii_found: list[str] = field(default_factory=list)
    was_scrubbed: bool = False
    count: int = 0

    def __bool__(self) -> bool:
        return self.was_scrubbed


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


class PIIGuardrail:
    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
            "IBAN_CODE": OperatorConfig("replace", {"new_value": "[IBAN]"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[IP_ADDRESS]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
            "NRP": OperatorConfig("replace", {"new_value": "[NRP]"}),
        }
        logger.info("PIIGuardrail with Presidio anonymizer ready")

    def _to_str(self, text) -> str:
        if isinstance(text, str):
            return text
        return str(text)

    def analyze(self, text, entities: list[str] | None = None) -> list:
        clean_text = self._to_str(text)

        if not clean_text.strip():
            return []

        return self._analyzer.analyze(
            text=clean_text,
            language="en",
            entities=entities,
        )

    def scrub_input(self, query) -> PIIGuardResult:
        clean_text = self._to_str(query)
        results = self.analyze(clean_text, entities=INPUT_ENTITIES)

        if not results:
            return PIIGuardResult(clean_text=clean_text)

        anonymized = self._anonymizer.anonymize(
            text=clean_text,
            analyzer_results=results,
            operators=self._operators,
        )

        pii_types = sorted({r.entity_type for r in results})
        logger.info(f"Input PII anonymized: {pii_types}")

        return PIIGuardResult(
            clean_text=anonymized.text,
            pii_found=pii_types,
            was_scrubbed=True,
            count=len(results),
        )

    def scrub_output(self, answer) -> str:
        clean_answer = self._to_str(answer)

        if not clean_answer.strip():
            return clean_answer

        results = self.analyze(clean_answer, entities=OUTPUT_ENTITIES)

        if not results:
            return clean_answer

        anonymized = self._anonymizer.anonymize(
            text=clean_answer,
            analyzer_results=results,
            operators=self._operators,
        )

        pii_types = sorted({r.entity_type for r in results})
        logger.info(f"Output PII anonymized: {pii_types}")

        return anonymized.text

    def detect_pii(self, text) -> list[dict]:
        clean_text = self._to_str(text)
        results = self.analyze(clean_text)

        return [
            {
                "type": r.entity_type,
                "score": round(r.score, 3),
                "start": r.start,
                "end": r.end,
            }
            for r in results
        ]