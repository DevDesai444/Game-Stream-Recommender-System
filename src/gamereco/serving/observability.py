"""Observability primitives for the FastAPI service.

Three pieces:

  * :class:`RequestIdMiddleware` stamps every request with an
    ``X-Request-ID`` (echoed in the response, attached to the
    structlog context, and reported in the access log line). If the
    client provided one we honour it; otherwise we mint a UUID4.
  * :class:`StructuredAccessLogMiddleware` emits one JSON log line per
    request with method, path, status, latency, ``served_from`` (if
    set by the route), and the request id.
  * :class:`Metrics` is a tiny Prometheus registry wrapper with
    ``request_total``, ``request_latency_seconds``, and a
    ``served_from_total`` counter for layer attribution. ``/metrics``
    exposes it in the standard text format.

Together they give the service real production-shape telemetry that
isn't tied to any particular APM vendor.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"


@dataclass
class Metrics:
    """Process-local Prometheus registry."""

    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    served_from: Counter

    @classmethod
    def build(cls) -> Metrics:
        registry = CollectorRegistry()
        requests = Counter(
            "gamereco_requests_total",
            "Total HTTP requests",
            labelnames=("method", "route", "status"),
            registry=registry,
        )
        latency = Histogram(
            "gamereco_request_latency_seconds",
            "HTTP request latency in seconds",
            labelnames=("method", "route"),
            buckets=(
                0.01,
                0.025,
                0.05,
                0.1,
                0.185,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
            ),
            registry=registry,
        )
        served_from = Counter(
            "gamereco_recs_served_from_total",
            "Recommendation responses tagged by the layer that answered",
            labelnames=("served_from",),
            registry=registry,
        )
        return cls(registry=registry, requests=requests, latency=latency, served_from=served_from)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        token = structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars(*token)  # type: ignore[arg-type]
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class StructuredAccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        super().__init__(app)
        self._metrics = metrics
        self._log = structlog.get_logger("gamereco.access")

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        started = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - started
        route = (
            request.scope.get("route").path
            if request.scope.get("route") is not None
            else request.url.path
        )
        served_from = response.headers.get("X-Served-From", "")
        self._metrics.requests.labels(request.method, route, str(response.status_code)).inc()
        self._metrics.latency.labels(request.method, route).observe(duration)
        if served_from:
            self._metrics.served_from.labels(served_from).inc()
        self._log.info(
            "access",
            method=request.method,
            path=request.url.path,
            route=route,
            status=response.status_code,
            duration_ms=round(duration * 1000, 2),
            served_from=served_from or None,
        )
        return response
