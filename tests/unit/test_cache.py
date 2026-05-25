"""Tests for the Redis cache wrapper."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from gamereco.common.schemas import RecommendationItem
from gamereco.serving.cache import KEY_PREFIX, RecommendationCache


@pytest.fixture
def fake_client() -> MagicMock:
    return MagicMock()


def _item() -> RecommendationItem:
    return RecommendationItem(steam_appid=440, name="TF2", header_image=None, score=0.9)


def test_cache_set_writes_json(fake_client: MagicMock) -> None:
    cache = RecommendationCache(client=fake_client)
    cache.set("u", [_item()])
    args, kwargs = fake_client.set.call_args
    assert args[0] == f"{KEY_PREFIX}u"
    payload = json.loads(args[1])
    assert payload[0]["steam_appid"] == 440
    assert "ex" in kwargs


def test_cache_get_returns_none_on_miss(fake_client: MagicMock) -> None:
    fake_client.get.return_value = None
    cache = RecommendationCache(client=fake_client)
    assert cache.get("u") is None


def test_cache_get_round_trip(fake_client: MagicMock) -> None:
    fake_client.get.return_value = json.dumps([_item().model_dump()])
    cache = RecommendationCache(client=fake_client)
    out = cache.get("u")
    assert out is not None
    assert out[0].steam_appid == 440


def test_cache_invalidate_calls_delete(fake_client: MagicMock) -> None:
    cache = RecommendationCache(client=fake_client)
    cache.invalidate("u")
    fake_client.delete.assert_called_once_with(f"{KEY_PREFIX}u")


def test_cache_ping_returns_bool(fake_client: MagicMock) -> None:
    fake_client.ping.return_value = True
    cache = RecommendationCache(client=fake_client)
    assert cache.ping() is True


def test_cache_ping_swallows_redis_error(fake_client: MagicMock) -> None:
    import redis

    fake_client.ping.side_effect = redis.RedisError("boom")
    cache = RecommendationCache(client=fake_client)
    assert cache.ping() is False
