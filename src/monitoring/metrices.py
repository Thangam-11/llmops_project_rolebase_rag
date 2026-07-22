"""
src/monitoring/metrics.py
=========================
Prometheus metrics for the Role-Based RAG API.

Metrics groups:
  - System      : CPU / memory / disk
  - LLM         : requests, latency, token counts, cost
  - RAG pipeline : requests, latency, retrieval, RAGAS scores
  - HTTP layer  : request counts + latency per endpoint
  - Auth        : login / registration counts
  - PII         : detections + blocks
"""

from __future__ import annotations

import platform
import threading
import time
from contextlib import contextmanager

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)


# ── System ────────────────────────────────────────────────────────────────────

SYSTEM_CPU = Gauge(
    "system_cpu_usage_percent",
    "CPU usage percentage",
)

SYSTEM_MEMORY = Gauge(
    "system_memory_usage_percent",
    "Memory usage percentage",
)

SYSTEM_DISK = Gauge(
    "system_disk_usage_percent",
    "Disk usage percentage",
)

# ── LLM ──────────────────────────────────────────────────────────────────────

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests",
)

LLM_LATENCY = Histogram(
    "llm_response_time_seconds",
    "LLM response latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

LLM_PROMPT_TOKENS = Counter(
    "llm_prompt_tokens_total",
    "Total prompt tokens sent to the LLM",
)

LLM_COMPLETION_TOKENS = Counter(
    "llm_completion_tokens_total",
    "Total completion tokens received from the LLM",
)

LLM_TOTAL_TOKENS = Counter(
    "llm_total_tokens_total",
    "Total tokens (prompt + completion)",
)

LLM_COST = Counter(
    "llm_cost_usd_total",
    "Estimated LLM cost in USD",
)

# ── RAG pipeline ──────────────────────────────────────────────────────────────

RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total RAG queries",
    ["department", "status"],       # status: success | blocked | no_docs | error
)

RAG_LATENCY = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG query latency in seconds",
    ["department"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Qdrant retrieval latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)

PIPELINE_LATENCY = Histogram(
    "rag_pipeline_latency_seconds",
    "Full pipeline latency (excluding evaluation) in seconds",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

PROMPT_INJECTION = Counter(
    "prompt_injection_detected_total",
    "Prompt injection attacks detected",
)

PII_BLOCKS = Counter(
    "rag_pii_blocks_total",
    "Requests blocked by PII guardrail",
    ["pii_type"],
)

PII_DETECTIONS = Counter(
    "rag_pii_detections_total",
    "PII entities detected and anonymised by Presidio",
    ["direction", "pii_type"],     # direction: input | output
)

RAGAS_SCORE = Gauge(
    "rag_ragas_score",
    "Latest RAGAS overall score (0–1)",
    ["department"],
)

RAGAS_METRIC_SCORE = Gauge(
    "rag_ragas_metric_score",
    "Latest RAGAS metric score by department and metric",
    ["department", "metric"],
)

RETRIEVAL_DOCS = Histogram(
    "rag_retrieval_docs_count",
    "Number of documents retrieved per query",
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
    "HTTP request latency in seconds",
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


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextmanager
def track_latency(histogram: Histogram):
    """
    Context manager that records elapsed seconds into a Histogram.

    Usage:
        with track_latency(RETRIEVAL_LATENCY):
            docs = retriever.retrieve(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - t0)


@contextmanager
def track_latency_with_labels(histogram: Histogram, **labels):
    """
    Context manager for labelled Histograms.

    Usage:
        with track_latency_with_labels(RAG_LATENCY, department="finance"):
            result = chain.invoke(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(time.perf_counter() - t0)


# ── System metrics updater ────────────────────────────────────────────────────

def _disk_usage_percent() -> float:
    """Return disk usage for the root partition (cross-platform)."""
    root = "C:\\" if platform.system() == "Windows" else "/"
    try:
        return psutil.disk_usage(root).percent
    except Exception:
        return 0.0


def update_system_metrics() -> None:
    """Snapshot current system resource usage into Gauges."""
    SYSTEM_CPU.set(psutil.cpu_percent(interval=None))
    SYSTEM_MEMORY.set(psutil.virtual_memory().percent)
    SYSTEM_DISK.set(_disk_usage_percent())


def _system_metrics_loop(interval: int) -> None:
    """Background thread target — updates system gauges every `interval` seconds."""
    while True:
        try:
            update_system_metrics()
        except Exception as exc:
            logger.warning(f"System metrics update failed: {exc}")
        time.sleep(interval)


# ── Server ────────────────────────────────────────────────────────────────────

def start_metrics_server(
    port: int = 9090,
    system_metrics_interval: int = 15,
) -> None:
    """
    Start the Prometheus HTTP metrics server and launch the background
    system-metrics updater thread.

    Args:
        port                    : Port to expose /metrics on (default 9090).
        system_metrics_interval : How often (seconds) to refresh CPU/memory/disk
                                  gauges (default 15s).
    """
    start_http_server(port)
    logger.info(f"Prometheus metrics server started ✓ on :{port}/metrics")

    # Seed an immediate reading so Grafana doesn't start with empty gauges.
    update_system_metrics()

    thread = threading.Thread(
        target=_system_metrics_loop,
        args=(system_metrics_interval,),
        daemon=True,        # dies automatically when the main process exits
        name="system-metrics-updater",
    )
    thread.start()
    logger.info(
        f"System metrics updater started ✓ "
        f"(interval={system_metrics_interval}s)"
    )