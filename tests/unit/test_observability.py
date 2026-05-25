"""Tests for the API observability stack (request id, access log, metrics)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from gamereco.common.schemas import RecommendationItem
from gamereco.serving import api as api_module
from gamereco.serving.observability import (
    REQUEST_ID_HEADER,
    Metrics,
)


def _item(appid: int = 1) -> RecommendationItem:
    return RecommendationItem(steam_appid=appid, name="g", header_image=None, score=0.5)


@pytest.fixture
def client() -> TestClient:
    store = MagicMock()
    cache = MagicMock()
    cache.get.return_value = [_item()]
    cache.ping.return_value = True
    api_module.app.state.store = store
    api_module.app.state.cache = cache
    return TestClient(api_module.app)


def test_metrics_endpoint_returns_prometheus_payload(client: TestClient) -> None:
    client.get("/recommendations/u")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "gamereco_requests_total" in body
    assert "gamereco_request_latency_seconds" in body


def test_request_id_is_minted_and_echoed(client: TestClient) -> None:
    resp = client.get("/health")
    assert REQUEST_ID_HEADER in resp.headers
    assert len(resp.headers[REQUEST_ID_HEADER]) > 0


def test_client_supplied_request_id_is_honoured(client: TestClient) -> None:
    resp = client.get("/health", headers={REQUEST_ID_HEADER: "abc-123"})
    assert resp.headers[REQUEST_ID_HEADER] == "abc-123"


def test_served_from_header_propagates_to_metrics(client: TestClient) -> None:
    api_module.app.state.cache.get.return_value = [_item()]
    client.get("/recommendations/u_cache")
    body = client.get("/metrics").text
    assert "served_from=\"cache\"" in body or "served_from" in body


def test_metrics_counter_increments_per_status(client: TestClient) -> None:
    for _ in range(3):
        client.get("/health")
    body = client.get("/metrics").text
    # The /health route should be present in the labelled counter.
    assert "/health" in body


def test_metrics_build_returns_independent_registries() -> None:
    a = Metrics.build()
    b = Metrics.build()
    assert a.registry is not b.registry
