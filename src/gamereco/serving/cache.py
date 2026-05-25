"""Redis caching wrapper for serving recommendations.

We use Redis as a read-through cache in front of Postgres + the vector
search. Hit-rate on the steady-state traffic pattern (top decile of
users accounting for ~80% of requests) drives the 185 ms P95 latency
target.
"""

from __future__ import annotations

import json

import redis

from gamereco.common.config import redis_settings
from gamereco.common.schemas import RecommendationItem

KEY_PREFIX = "gamereco:recs:"


class RecommendationCache:
    def __init__(self, client: redis.Redis | None = None) -> None:
        cfg = redis_settings()
        self._client = client or redis.Redis(
            host=cfg.host, port=cfg.port, decode_responses=True
        )
        self._ttl = cfg.ttl_seconds

    @staticmethod
    def _key(user_id: str) -> str:
        return f"{KEY_PREFIX}{user_id}"

    def get(self, user_id: str) -> list[RecommendationItem] | None:
        raw = self._client.get(self._key(user_id))
        if raw is None:
            return None
        payload = json.loads(raw)
        return [RecommendationItem(**item) for item in payload]

    def set(self, user_id: str, items: list[RecommendationItem]) -> None:
        payload = json.dumps([item.model_dump() for item in items])
        self._client.set(self._key(user_id), payload, ex=self._ttl)

    def invalidate(self, user_id: str) -> None:
        self._client.delete(self._key(user_id))

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
