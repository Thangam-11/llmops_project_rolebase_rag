"""
src/monitoring/metrics.py
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

# ── RAG pipeline ─────────────────────────────────────────────────────────────

RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total RAG queries",
    ["department", "status"],       # status: success | blocked | no_docs | error
)

RAG_LATENCY = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG query latency",
    ["department"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

PII_BLOCKS = Counter(
    "rag_pii_blocks_total",
    "Requests blocked by PII guardrail",
    ["pii_type"],
)
PII_DETECTIONS = Counter(
    "rag_pii_detections_total",
    "PII entities detected and anonymized by Presidio",
    ["direction", "pii_type"],
)


RAGAS_SCORE = Gauge(
    "rag_ragas_score",
    "Latest RAGAS overall score",
    ["department"],
)


RAGAS_METRIC_SCORE = Gauge(
    "rag_ragas_metric_score",
    "Latest RAGAS metric score by department and metric",
    ["department", "metric"],
)


RETRIEVAL_DOCS = Histogram(
    "rag_retrieval_docs_count",
    "Number of docs retrieved per query",
    ["department"],
    buckets=[1, 2, 3, 5, 8, 10],
)

# ── HTTP layer ────────────────────────────────────────────────────────────────

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

AUTH_LOGINS = Counter(
    "auth_logins_total",
    "Total login attempts",
    ["status"],                     # status: success | failed
)

AUTH_REGISTRATIONS = Counter(
    "auth_registrations_total",
    "Total user registrations",
)


# ── Server ────────────────────────────────────────────────────────────────────

def start_metrics_server(port: int = 9090) -> None:
    start_http_server(port)
    logger.info(f"Prometheus metrics server started ✓ on :{port}/metrics")