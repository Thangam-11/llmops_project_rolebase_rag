from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
})
nlp_engine = provider.create_engine()
a = AnalyzerEngine(nlp_engine=nlp_engine)

# Test different SSN formats
tests = [
    "My SSN is 123-45-6789",
    "Social Security Number: 123-45-6789",
    "ssn: 123456789",
    "My social security is 123-45-6789",
    "My card is 4111-1111-1111-1111",
    "Call me at john@example.com",
]

for t in tests:
    results = a.analyze(t, language="en", score_threshold=0.3)
    print(f"\nText: {t}")
    for r in results:
        print(f"  → {r.entity_type} | score={r.score:.2f} | '{t[r.start:r.end]}'")