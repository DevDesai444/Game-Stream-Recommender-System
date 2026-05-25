"""FastAPI service unit tests using TestClient and a fake store/cache."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from gamereco.common.schemas import RecommendationItem
from gamereco.serving import api as api_module


def _item(appid: int = 440) -> RecommendationItem:
    return RecommendationItem(steam_appid=appid, name=f"g-{appid}", header_image=None, score=0.5)


@pytest.fixture
def client() -> TestClient:
    store = MagicMock()
    cache = MagicMock()
    cache.ping.return_value = True
    api_module.app.state.store = store
    api_module.app.state.cache = cache
    return TestClient(api_module.app)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_endpoint_reports_redis_down(client: TestClient) -> None:
    api_module.app.state.cache.ping.return_value = False
    resp = client.get("/health")
    assert resp.json()["redis"] == "down"


def test_recommendations_returns_cached_payload(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = [_item()]
    resp = client.get("/recommendations/u1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_from"] == "cache"
    assert body["items"][0]["steam_appid"] == 440


def test_recommendations_falls_back_to_store(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = None
    api_module.app.state.store.fetch_user_recommendations.return_value = [_item(100)]
    resp = client.get("/recommendations/u1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_from"] == "postgres"
    api_module.app.state.cache.set.assert_called_once()


def test_recommendations_404_when_empty(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = None
    api_module.app.state.store.fetch_user_recommendations.return_value = []
    resp = client.get("/recommendations/unknown")
    assert resp.status_code == 404


def test_recommendations_respects_limit_query(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = [_item(1), _item(2), _item(3)]
    resp = client.get("/recommendations/u1?limit=2")
    body = resp.json()
    assert len(body["items"]) == 2


def test_similar_endpoint(client: TestClient) -> None:
    api_module.app.state.store.similar_games.return_value = [_item(7)]
    resp = client.get("/similar/440")
    assert resp.status_code == 200
    assert resp.json()["served_from"] == "pgvector"


def test_similar_endpoint_404(client: TestClient) -> None:
    api_module.app.state.store.similar_games.return_value = []
    resp = client.get("/similar/9999")
    assert resp.status_code == 404


def test_global_endpoint(client: TestClient) -> None:
    api_module.app.state.store.global_top.return_value = [_item(1)]
    resp = client.get("/global")
    body = resp.json()
    assert body["user_id"] == "__global__"
    assert len(body["items"]) == 1


def test_latency_field_is_set(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = [_item()]
    resp = client.get("/recommendations/u1")
    assert resp.json()["latency_ms"] >= 0
