"""FastAPI application exposing the recommendation service.

P95 latency target: 185 ms. The serving paths:

  * **personalised** : Redis read-through cache -> Postgres -> cohort -> global
  * **similar items**: pgvector cosine search on the 64-D embedding index
  * **onboarding**  : "I like games X, Y, Z" -> instant pgvector-based recs

The personalised endpoint follows a real cold-start cascade defined
in :mod:`gamereco.serving.coldstart` instead of 404'ing — a user we
have never seen still gets a coherent answer.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from gamereco.common.logging import configure_logging, get_logger
from gamereco.common.schemas import RecommendationItem, RecommendationResponse
from gamereco.serving.cache import RecommendationCache
from gamereco.serving.coldstart import resolve as resolve_coldstart
from gamereco.serving.observability import (
    Metrics,
    RequestIdMiddleware,
    StructuredAccessLogMiddleware,
)
from gamereco.serving.store import RecommendationStore, init_schema

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - bootstrap only
    configure_logging()
    # Skip building real store/cache if a wrapper has already injected
    # stubs (load-test scripts do this so we measure FastAPI overhead
    # in isolation from Postgres + Redis).
    if not getattr(app.state, "store", None):
        store = RecommendationStore()
        try:
            init_schema(store._engine)
        except Exception as exc:  # noqa: BLE001 - logged for visibility
            log.warning("api.schema_init_failed", error=str(exc))
        app.state.store = store
    if not getattr(app.state, "cache", None):
        app.state.cache = RecommendationCache()
    if not getattr(app.state, "metrics", None):
        app.state.metrics = Metrics.build()
    yield


app = FastAPI(
    title="Game Stream Recommender",
    version="0.2.0",
    description="Hybrid ALS + NCF + KMeans + XGBoost recommendation service",
    lifespan=lifespan,
)

# Metrics holder used by the middleware before app.state.metrics exists
# (the metrics registry must be process-singleton so /metrics keeps
# accumulating across requests).
_BOOT_METRICS = Metrics.build()
app.state.metrics = _BOOT_METRICS

app.add_middleware(StructuredAccessLogMiddleware, metrics=_BOOT_METRICS)
app.add_middleware(RequestIdMiddleware)


def get_store() -> RecommendationStore:
    return app.state.store  # type: ignore[no-any-return]


def get_cache() -> RecommendationCache:
    return app.state.cache  # type: ignore[no-any-return]


@app.get("/health")
async def health(cache: Annotated[RecommendationCache, Depends(get_cache)]) -> dict[str, str]:
    return {"status": "ok", "redis": "up" if cache.ping() else "down"}


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus scrape endpoint."""
    body, content_type = app.state.metrics.render()  # type: ignore[attr-defined]
    return Response(content=body, media_type=content_type)


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def recommendations(
    user_id: str,
    response: Response,
    store: Annotated[RecommendationStore, Depends(get_store)],
    cache: Annotated[RecommendationCache, Depends(get_cache)],
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    """Personalised top-K with a personal -> cohort -> global cascade.

    The endpoint never 404s for a known schema; unknown users get
    cohort or global fallbacks so that the client always has
    *something* to display. served_from in the response identifies the
    layer that actually answered (cache, personal, cohort, global).
    """
    started = time.perf_counter()
    cached = cache.get(user_id)
    if cached is not None:
        latency = (time.perf_counter() - started) * 1000
        response.headers["X-Served-From"] = "cache"
        return RecommendationResponse(
            user_id=user_id,
            served_from="cache",
            latency_ms=round(latency, 2),
            items=cached[:limit],
        )

    resolved = resolve_coldstart(store, user_id, limit=limit)
    if not resolved.items:
        raise HTTPException(
            status_code=503,
            detail="recommendation backend has not finished bootstrapping",
        )
    if resolved.served_from == "personal":
        cache.set(user_id, resolved.items)

    latency = (time.perf_counter() - started) * 1000
    response.headers["X-Served-From"] = resolved.served_from
    return RecommendationResponse(
        user_id=user_id,
        served_from=resolved.served_from,
        latency_ms=round(latency, 2),
        items=resolved.items,
    )


class OnboardRequest(BaseModel):
    """A handful of liked games to seed brand-new users with."""

    user_id: str = Field(..., min_length=1, max_length=64)
    liked_steam_appids: list[int] = Field(..., min_length=1, max_length=20)


@app.post("/onboard", response_model=RecommendationResponse)
async def onboard(
    payload: OnboardRequest,
    store: Annotated[RecommendationStore, Depends(get_store)],
    cache: Annotated[RecommendationCache, Depends(get_cache)],
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    """First-touch recommendations for a brand-new user.

    Looks up the pgvector nearest neighbours of each provided
    appid, deduplicates, drops the seeds themselves, and returns the
    blended top-K. The result is also persisted to the cache under
    ``user_id`` so the next call hits the read path normally.
    """
    started = time.perf_counter()
    aggregated: dict[int, RecommendationItem] = {}
    for appid in payload.liked_steam_appids:
        try:
            neighbours = store.similar_games(int(appid), limit=limit)
        except Exception as exc:  # noqa: BLE001 - degraded path is still useful
            log.warning("api.onboard_neighbour_failed", appid=appid, error=str(exc))
            continue
        for item in neighbours:
            if item.steam_appid in payload.liked_steam_appids:
                continue
            keep = aggregated.get(item.steam_appid)
            if keep is None or item.score > keep.score:
                aggregated[item.steam_appid] = item

    items: list[RecommendationItem] = sorted(
        aggregated.values(), key=lambda x: x.score, reverse=True
    )[:limit]
    if not items:
        items = store.global_top(limit)
        served = "global_fallback"
    else:
        served = "onboarding_pgvector"
        cache.set(payload.user_id, items)

    latency = (time.perf_counter() - started) * 1000
    return RecommendationResponse(
        user_id=payload.user_id,
        served_from=served,
        latency_ms=round(latency, 2),
        items=items,
    )


@app.get("/similar/{steam_appid}", response_model=RecommendationResponse)
async def similar(
    steam_appid: int,
    store: Annotated[RecommendationStore, Depends(get_store)],
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    started = time.perf_counter()
    items = store.similar_games(steam_appid, limit=limit)
    if not items:
        raise HTTPException(status_code=404, detail=f"no embedding for appid {steam_appid}")
    latency = (time.perf_counter() - started) * 1000
    return RecommendationResponse(
        user_id=str(steam_appid),
        served_from="pgvector",
        latency_ms=round(latency, 2),
        items=items,
    )


@app.get("/global", response_model=RecommendationResponse)
async def global_top(
    store: Annotated[RecommendationStore, Depends(get_store)],
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    started = time.perf_counter()
    items = store.global_top(limit=limit)
    latency = (time.perf_counter() - started) * 1000
    return RecommendationResponse(
        user_id="__global__",
        served_from="postgres",
        latency_ms=round(latency, 2),
        items=items,
    )


@app.exception_handler(Exception)
async def unhandled_error(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.error("api.unhandled", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "internal error"})
