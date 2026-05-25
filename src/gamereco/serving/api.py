"""FastAPI application exposing the recommendation service.

P95 latency target: 185 ms. Two paths feed it:

  * **personalised** : Postgres lookup -> Redis read-through cache
  * **similar items**: pgvector cosine search on the 64-D embedding index

Both endpoints log structured request metadata so latency can be
attributed to cache, DB, or vector search at runtime.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from gamereco.common.logging import configure_logging, get_logger
from gamereco.common.schemas import RecommendationResponse
from gamereco.serving.cache import RecommendationCache
from gamereco.serving.store import RecommendationStore, init_schema

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - bootstrap only
    configure_logging()
    store = RecommendationStore()
    try:
        init_schema(store._engine)
    except Exception as exc:  # noqa: BLE001 - logged for visibility
        log.warning("api.schema_init_failed", error=str(exc))
    app.state.store = store
    app.state.cache = RecommendationCache()
    yield


app = FastAPI(
    title="Game Stream Recommender",
    version="0.2.0",
    description="Hybrid ALS + NCF + KMeans + XGBoost recommendation service",
    lifespan=lifespan,
)


def get_store() -> RecommendationStore:
    return app.state.store  # type: ignore[no-any-return]


def get_cache() -> RecommendationCache:
    return app.state.cache  # type: ignore[no-any-return]


@app.get("/health")
async def health(cache: Annotated[RecommendationCache, Depends(get_cache)]) -> dict[str, str]:
    return {"status": "ok", "redis": "up" if cache.ping() else "down"}


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def recommendations(
    user_id: str,
    store: Annotated[RecommendationStore, Depends(get_store)],
    cache: Annotated[RecommendationCache, Depends(get_cache)],
    limit: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    started = time.perf_counter()
    cached = cache.get(user_id)
    if cached is not None:
        latency = (time.perf_counter() - started) * 1000
        return RecommendationResponse(
            user_id=user_id,
            served_from="cache",
            latency_ms=round(latency, 2),
            items=cached[:limit],
        )

    items = store.fetch_user_recommendations(user_id, limit=limit)
    if not items:
        raise HTTPException(status_code=404, detail=f"no recommendations for user '{user_id}'")

    cache.set(user_id, items)
    latency = (time.perf_counter() - started) * 1000
    return RecommendationResponse(
        user_id=user_id,
        served_from="postgres",
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
        raise HTTPException(
            status_code=404, detail=f"no embedding for appid {steam_appid}"
        )
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
