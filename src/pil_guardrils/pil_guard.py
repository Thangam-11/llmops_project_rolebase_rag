from __future__ import annotations

from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_ENTITIES = [
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
]

OUTPUT_ENTITIES = [
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "PERSON",
    "LOCATION",
]

SCORE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Custom SSN recognizer (Presidio built-in is unreliable)
# ---------------------------------------------------------------------------

def _build_ssn_recognizer() -> PatternRecognizer:
    patterns = [
        Pattern(
            name  = "SSN_dashes",
            regex = r"\b\d{3}-\d{2}-\d{4}\b",
            score = 0.85,
        ),
        Pattern(
            name  = "SSN_spaces",
            regex = r"\b\d{3} \d{2} \d{4}\b",
            score = 0.85,
        ),
        Pattern(
            name  = "SSN_plain",
            regex = r"\b\d{9}\b",
            score = 0.5,   # lower confidence — 9 digits alone is ambiguous
        ),
    ]
    return PatternRecognizer(
        supported_entity = "US_SSN",
        patterns         = patterns,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PIIGuardResult:
    allowed:   bool
    reason:    str  = ""
    pii_found: list = None

    def __post_init__(self):
        if self.pii_found is None:
            self.pii_found = []


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------

class PIIGuardrail:

    def __init__(self) -> None:
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        nlp_engine = provider.create_engine()

        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        # Register custom SSN recognizer
        self._analyzer.registry.add_recognizer(_build_ssn_recognizer())

        self._anonymizer = AnonymizerEngine()
        logger.info("PIIGuardrail (Presidio) ready ✓")

    def analyze(self, text: str, entities: list[str] | None = None, language: str = "en") -> list:
        return self._analyzer.analyze(
            text            = text,
            entities        = entities,
            language        = language,
            score_threshold = SCORE_THRESHOLD,
        )

    def anonymize(self, text: str, entities: list[str] | None = None) -> str:
        results = self.analyze(text, entities=entities)
        if not results:
            return text
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text

    def check_input(self, question: str) -> PIIGuardResult:
        results = self.analyze(question, entities=INPUT_ENTITIES)
        if not results:
            return PIIGuardResult(allowed=True)

        found_types = list({r.entity_type for r in results})
        friendly    = ", ".join(t.replace("_", " ").title() for t in found_types)

        logger.warning(f"PII in user question: {found_types} | q={question[:60]}")

        return PIIGuardResult(
            allowed   = False,
            reason    = (
                f"Your question appears to contain sensitive information "
                f"({friendly}). Please rephrase without personal data."
            ),
            pii_found = found_types,
        )

    def scrub_output(self, answer: str) -> str:
        results = self.analyze(answer, entities=OUTPUT_ENTITIES)
        if not results:
            return answer

        found_types = list({r.entity_type for r in results})
        logger.warning(f"PII scrubbed from LLM output: {found_types}")

        return self._anonymizer.anonymize(text=answer, analyzer_results=results).text


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    guard = PIIGuardrail()

    r = guard.check_input("My SSN is 123-45-6789 and card is 4111-1111-1111-1111")
    print(r.allowed, r.reason)

    print(guard.scrub_output("Email John at john@example.com, SSN 123-45-6789"))

    r2 = guard.check_input("What is the Q4 revenue?")
    print(r2.allowed)