"""
src/pil_guardrils/pil_guard.py
================================
PII detection/scrubbing (Presidio) + prompt-injection keyword filter +
retrieved-document sanitization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


@dataclass
class PIIGuardResult:
    clean_text: str
    pii_found: list[str] = field(default_factory=list)
    was_scrubbed: bool = False
    was_blocked: bool = False
    count: int = 0

    def __bool__(self) -> bool:
        return self.was_scrubbed


INPUT_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "IBAN_CODE", "IP_ADDRESS", "LOCATION", "NRP",
    "EMPLOYEE_ID", "PROJECT_CODE",
]

OUTPUT_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "IBAN_CODE", "EMPLOYEE_ID", "PROJECT_CODE", "SALARY",
]

BLOCK_ON_OUTPUT = {"CREDIT_CARD", "IBAN_CODE", "SALARY"}

ENTITY_THRESHOLDS: dict[str, float] = {
    "CREDIT_CARD": 0.3,
    "IBAN_CODE": 0.3,
    "EMAIL_ADDRESS": 0.5,
    "PHONE_NUMBER": 0.5,
    "IP_ADDRESS": 0.5,
    "EMPLOYEE_ID": 0.4,
    "PROJECT_CODE": 0.4,
    "SALARY": 0.4,
    "PERSON": 0.7,
    "LOCATION": 0.75,
    "NRP": 0.75,
}
DEFAULT_THRESHOLD = 0.6


def _build_custom_recognizers() -> list[PatternRecognizer]:
    employee_id = PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[Pattern(name="employee_id", regex=r"\bEMP[- ]?\d{4,6}\b", score=0.85)],
        context=["employee", "staff", "id"],
    )
    project_code = PatternRecognizer(
        supported_entity="PROJECT_CODE",
        patterns=[Pattern(name="project_code", regex=r"\bPROJ-[A-Z0-9]{4,8}\b", score=0.85)],
        context=["project", "codename"],
    )
    salary = PatternRecognizer(
        supported_entity="SALARY",
        patterns=[
            Pattern(name="salary_currency", regex=r"[\$₹€£]\s?\d[\d,]*(\.\d{1,2})?", score=0.4),
            Pattern(name="salary_lpa", regex=r"\b\d+(\.\d+)?\s?(LPA|lakhs?|lacs?)\b", score=0.6),
        ],
        context=["salary", "pay", "compensation", "wage", "ctc", "package", "income", "earns", "paid"],
    )
    return [employee_id, project_code, salary]


class PIIGuardrail:
    def __init__(self) -> None:
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()

        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        for recognizer in _build_custom_recognizers():
            self._analyzer.registry.add_recognizer(recognizer)

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
            "EMPLOYEE_ID": OperatorConfig("replace", {"new_value": "[EMPLOYEE_ID]"}),
            "PROJECT_CODE": OperatorConfig("replace", {"new_value": "[PROJECT_CODE]"}),
        }
        logger.info("PIIGuardrail with Presidio anonymizer ready")

    def _to_str(self, text) -> str:
        return text if isinstance(text, str) else str(text)

    def analyze(self, text, entities: list[str] | None = None) -> list:
        clean_text = self._to_str(text)
        if not clean_text.strip():
            return []

        results = self._analyzer.analyze(
            text=clean_text,
            language="en",
            entities=entities,
        )
        return [
            r for r in results
            if r.score >= ENTITY_THRESHOLDS.get(r.entity_type, DEFAULT_THRESHOLD)
        ]

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

    def scrub_output(self, answer) -> PIIGuardResult:
        clean_answer = self._to_str(answer)

        if not clean_answer.strip():
            return PIIGuardResult(clean_text=clean_answer)

        results = self.analyze(clean_answer, entities=OUTPUT_ENTITIES)

        if not results:
            return PIIGuardResult(clean_text=clean_answer)

        pii_types = sorted({r.entity_type for r in results})

        if any(t in BLOCK_ON_OUTPUT for t in pii_types):
            logger.warning(f"Output blocked — sensitive PII detected: {pii_types}")
            return PIIGuardResult(
                clean_text="This response contained sensitive information and was blocked.",
                pii_found=pii_types,
                was_scrubbed=True,
                was_blocked=True,
                count=len(results),
            )

        anonymized = self._anonymizer.anonymize(
            text=clean_answer,
            analyzer_results=results,
            operators=self._operators,
        )
        logger.info(f"Output PII anonymized: {pii_types}")

        return PIIGuardResult(
            clean_text=anonymized.text,
            pii_found=pii_types,
            was_scrubbed=True,
            count=len(results),
        )

    def detect_pii(self, text) -> list[dict]:
        clean_text = self._to_str(text)
        results = self.analyze(clean_text)
        return [
            {"type": r.entity_type, "score": round(r.score, 3), "start": r.start, "end": r.end}
            for r in results
        ]


# ── Fast keyword pre-filter (Layer 1, no LLM cost) ────────────────────────
PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "forget previous instructions",
    "forget everything above",
    "ignore system prompt",
    "developer mode",
    "reveal system prompt",
    "bypass safety",
    "disable guardrails",
    "pretend you are",
    "act as dan",
    "print your prompt",
    "repeat your instructions",
]


def is_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in PROMPT_INJECTION_PATTERNS)


# ── Retrieved-document sanitization (Layer 5) ──────────────────────────────
BAD_DOC_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "developer instructions",
    "ignore the above",
    "disregard the context",
]


def filter_safe_docs(docs: list) -> list:
    safe_docs = []
    for doc in docs:
        text = doc.page_content.lower()
        if any(pattern in text for pattern in BAD_DOC_PATTERNS):
            logger.warning(f"Dropped suspicious doc chunk: {doc.metadata.get('filename', 'unknown')}")
            continue
        safe_docs.append(doc)
    return safe_docs