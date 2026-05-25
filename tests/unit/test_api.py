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


def test_recommendations_falls_back_to_personal_store(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = None
    api_module.app.state.store.fetch_user_recommendations.return_value = [_item(100)]
    resp = client.get("/recommendations/u1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_from"] == "personal"
    api_module.app.state.cache.set.assert_called_once()


def test_recommendations_falls_through_to_cohort(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = None
    api_module.app.state.store.fetch_user_recommendations.return_value = []
    api_module.app.state.store.user_cohort.return_value = 4
    api_module.app.state.store.cohort_top.return_value = [_item(200)]
    resp = client.get("/recommendations/u_new")
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_from"] == "cohort"
    assert body["items"][0]["steam_appid"] == 200


def test_recommendations_falls_through_to_global(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = None
    api_module.app.state.store.fetch_user_recommendations.return_value = []
    api_module.app.state.store.user_cohort.return_value = None
    api_module.app.state.store.global_top.return_value = [_item(300)]
    resp = client.get("/recommendations/u_brand_new")
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_from"] == "global_fallback"


def test_recommendations_503_when_backend_empty(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = None
    api_module.app.state.store.fetch_user_recommendations.return_value = []
    api_module.app.state.store.user_cohort.return_value = None
    api_module.app.state.store.global_top.return_value = []
    resp = client.get("/recommendations/u")
    assert resp.status_code == 503


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


def test_onboard_blends_pgvector_neighbours(client: TestClient) -> None:
    api_module.app.state.store.similar_games.side_effect = [
        [_item(10), _item(20)],
        [_item(20), _item(30)],
    ]
    resp = client.post(
        "/onboard",
        json={"user_id": "u_new", "liked_steam_appids": [1, 2]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_from"] == "onboarding_pgvector"
    appids = [i["steam_appid"] for i in body["items"]]
    assert 10 in appids and 20 in appids and 30 in appids
    api_module.app.state.cache.set.assert_called()


def test_onboard_drops_seed_games_from_results(client: TestClient) -> None:
    api_module.app.state.store.similar_games.return_value = [_item(1), _item(99)]
    resp = client.post(
        "/onboard",
        json={"user_id": "u_new", "liked_steam_appids": [1]},
    )
    body = resp.json()
    assert all(i["steam_appid"] != 1 for i in body["items"])


def test_onboard_falls_back_to_global_when_no_neighbours(client: TestClient) -> None:
    api_module.app.state.store.similar_games.return_value = []
    api_module.app.state.store.global_top.return_value = [_item(500)]
    resp = client.post(
        "/onboard",
        json={"user_id": "u_new", "liked_steam_appids": [1]},
    )
    body = resp.json()
    assert body["served_from"] == "global_fallback"
    assert body["items"][0]["steam_appid"] == 500


def test_onboard_validates_payload(client: TestClient) -> None:
    resp = client.post(
        "/onboard",
        json={"user_id": "u", "liked_steam_appids": []},
    )
    assert resp.status_code == 422
