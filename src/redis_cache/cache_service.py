"""
src/cache/redis_cache_service.py
==================================
Response-level cache for RAGChain.invoke() results.
Keyed on (question, department, top_k) so identical repeated questions
skip guardrails, retrieval, and the LLM call entirely.
"""

from __future__ import annotations

import hashlib
import json

import redis

from config.settings import get_settings
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _make_key(question: str, department: str, k: int) -> str:
    raw = f"{department}:{k}:{question.strip().lower()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"rag_cache:{digest}"


class RedisCacheService:
    def __init__(self) -> None:
        self._enabled = settings.redis_cache_enabled
        self._ttl = settings.redis_cache_ttl_seconds
        self._client: redis.Redis | None = None

        if not self._enabled:
            logger.info("Redis cache disabled via settings")
            return

        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self._client.ping()
            logger.info(f"Redis cache ready ✓ | url={settings.redis_url}")
        except Exception:
            logger.exception(
                "Redis connection failed — caching disabled for this session"
            )
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def get(
        self,
        question: str,
        department: str,
        k: int,
    ) -> dict | None:
        if self._client is None:
            return None

        try:
            key = _make_key(question, department, k)
            raw = self._client.get(key)

            if raw is None:
                return None

            logger.info(f"Cache HIT | key={key[:20]}...")
            return json.loads(raw)

        except Exception:
            logger.exception(
                "Redis GET failed — falling through to live query"
            )
            return None

    def set(
        self,
        question: str,
        department: str,
        k: int,
        result: dict,
    ) -> None:
        if self._client is None:
            return

        # Don't cache blocked responses
        if result.get("was_blocked"):
            return

        try:
            key = _make_key(question, department, k)

            # Remove request-specific fields
            cache_result = result.copy()
            cache_result.pop("latency_ms", None)
            cache_result.pop("cache_hit", None)

            self._client.setex(
                key,
                self._ttl,
                json.dumps(cache_result),
            )

            logger.info(
                f"Cache SET | key={key[:20]}... | ttl={self._ttl}s"
            )

        except Exception:
            logger.exception(
                "Redis SET failed — continuing without caching this result"
            )

    def invalidate_all(self) -> int:
        if self._client is None:
            return 0

        keys = list(self._client.scan_iter("rag_cache:*"))

        if keys:
            self._client.delete(*keys)

        logger.info(
            f"Cache invalidated | {len(keys)} keys removed"
        )

        return len(keys)