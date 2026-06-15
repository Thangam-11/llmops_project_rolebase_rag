"""
tests/test_metrics_simple.py
"""
from src.monitoring.metrices import (
    RAG_REQUESTS,
    RAG_LATENCY,
    PII_BLOCKS,
    RAGAS_SCORE,
    RETRIEVAL_DOCS,
    HTTP_REQUESTS,
    HTTP_LATENCY,
    AUTH_LOGINS,
    AUTH_REGISTRATIONS,
    start_metrics_server,
)
from unittest.mock import patch

def test_rag_requests():
    RAG_REQUESTS.labels(department="hr", status="success").inc()
    RAG_REQUESTS.labels(department="finance", status="blocked").inc()
    RAG_REQUESTS.labels(department="marketing", status="no_docs").inc()
    RAG_REQUESTS.labels(department="engineering", status="error").inc()
    print("✓ RAG_REQUESTS OK")

def test_rag_latency():
    RAG_LATENCY.labels(department="hr").observe(1.5)
    RAG_LATENCY.labels(department="finance").observe(3.0)
    print("✓ RAG_LATENCY OK")

def test_pii_blocks():
    PII_BLOCKS.labels(pii_type="EMAIL_ADDRESS").inc()
    PII_BLOCKS.labels(pii_type="PHONE_NUMBER").inc()
    print("✓ PII_BLOCKS OK")

def test_ragas_score():
    RAGAS_SCORE.labels(department="finance").set(0.87)
    value = RAGAS_SCORE.labels(department="finance")._value.get()
    assert value == 0.87, f"Expected 0.87, got {value}"
    print("✓ RAGAS_SCORE OK")

def test_retrieval_docs():
    RETRIEVAL_DOCS.labels(department="hr").observe(5)
    RETRIEVAL_DOCS.labels(department="finance").observe(0)
    print("✓ RETRIEVAL_DOCS OK")

def test_http_requests():
    HTTP_REQUESTS.labels(method="GET", endpoint="/query", status="200").inc()
    HTTP_REQUESTS.labels(method="POST", endpoint="/auth/login", status="401").inc()
    print("✓ HTTP_REQUESTS OK")

def test_http_latency():
    HTTP_LATENCY.labels(endpoint="/query").observe(0.3)
    HTTP_LATENCY.labels(endpoint="/auth/login").observe(0.05)
    print("✓ HTTP_LATENCY OK")

def test_auth_logins():
    AUTH_LOGINS.labels(status="success").inc()
    AUTH_LOGINS.labels(status="failed").inc()
    print("✓ AUTH_LOGINS OK")

def test_auth_registrations():
    before = AUTH_REGISTRATIONS._value.get()
    AUTH_REGISTRATIONS.inc()
    after = AUTH_REGISTRATIONS._value.get()
    assert after == before + 1.0, f"Expected {before + 1.0}, got {after}"
    print("✓ AUTH_REGISTRATIONS OK")

def test_metrics_server():
    with patch("src.monitoring.metrics.start_http_server") as mock_start:
        start_metrics_server(port=9090)
        assert mock_start.called
        assert mock_start.call_args[0][0] == 9090
    print("✓ start_metrics_server OK")

if __name__ == "__main__":
    tests = [
        test_rag_requests,
        test_rag_latency,
        test_pii_blocks,
        test_ragas_score,
        test_retrieval_docs,
        test_http_requests,
        test_http_latency,
        test_auth_logins,
        test_auth_registrations,
        test_metrics_server,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")