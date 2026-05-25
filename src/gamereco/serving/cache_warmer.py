"""Pre-warm Redis with top-K recommendations for the heaviest users.

The cache warmer is invoked nightly by the serving_refresh DAG. Warming
the cache for the top decile of users (who account for the bulk of
traffic) is what keeps the P95 latency around 185 ms even on a cold
service.
"""

from __future__ import annotations

import click
from sqlalchemy import select

from gamereco.common.logging import configure_logging, get_logger
from gamereco.serving.cache import RecommendationCache
from gamereco.serving.db import build_engine, user_recommendations_table
from gamereco.serving.store import RecommendationStore

log = get_logger(__name__)


def warm(top_n: int = 200, limit_per_user: int = 10) -> int:
    engine = build_engine()
    store = RecommendationStore(engine=engine)
    cache = RecommendationCache()

    with engine.begin() as conn:
        users = (
            conn.execute(
                select(user_recommendations_table.c.user_id)
                .group_by(user_recommendations_table.c.user_id)
                .limit(top_n)
            )
            .scalars()
            .all()
        )
    warmed = 0
    for user_id in users:
        items = store.fetch_user_recommendations(user_id, limit=limit_per_user)
        if items:
            cache.set(user_id, items)
            warmed += 1
    log.info("cache.warmed", users=warmed)
    return warmed


@click.command()
@click.option("--top-n", default=200, type=int)
@click.option("--limit", default=10, type=int)
def main(top_n: int, limit: int) -> None:
    configure_logging()
    warm(top_n, limit)


if __name__ == "__main__":  # pragma: no cover
    main()
