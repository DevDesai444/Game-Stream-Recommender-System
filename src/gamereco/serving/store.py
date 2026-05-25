"""Postgres-backed recommendation store, including pgvector similarity search."""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from gamereco.common.schemas import RecommendationItem
from gamereco.serving.db import (
    build_engine,
    games_table,
    user_cohorts_table,
    user_recommendations_table,
)


class RecommendationStore:
    """Read-side wrapper that knows how to combine recs with game metadata."""

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or build_engine()

    def fetch_user_recommendations(
        self, user_id: str, *, limit: int = 10
    ) -> list[RecommendationItem]:
        stmt = (
            select(
                user_recommendations_table.c.steam_appid,
                user_recommendations_table.c.score,
                games_table.c.name,
                games_table.c.header_image,
            )
            .select_from(
                user_recommendations_table.join(
                    games_table,
                    games_table.c.steam_appid == user_recommendations_table.c.steam_appid,
                )
            )
            .where(user_recommendations_table.c.user_id == user_id)
            .order_by(user_recommendations_table.c.rank.asc())
            .limit(limit)
        )
        with Session(self._engine) as session:
            rows = session.execute(stmt).all()
        return [
            RecommendationItem(
                steam_appid=int(r.steam_appid),
                name=str(r.name),
                header_image=r.header_image,
                score=float(r.score),
            )
            for r in rows
        ]

    def similar_games(self, steam_appid: int, *, limit: int = 10) -> list[RecommendationItem]:
        """pgvector cosine similarity search over the game-embedding table."""
        query = text(
            """
            SELECT g.steam_appid, g.name, g.header_image,
                   1 - (e.embedding <=> base.embedding) AS score
              FROM game_embeddings e
              JOIN games g ON g.game_idx = e.game_idx
              JOIN game_embeddings base ON base.steam_appid = :appid
             WHERE e.steam_appid <> :appid
             ORDER BY e.embedding <=> base.embedding
             LIMIT :limit
            """
        )
        with Session(self._engine) as session:
            rows = session.execute(query, {"appid": steam_appid, "limit": limit}).all()
        return [
            RecommendationItem(
                steam_appid=int(r.steam_appid),
                name=str(r.name),
                header_image=r.header_image,
                score=float(r.score),
            )
            for r in rows
        ]

    def user_cohort(self, user_id: str) -> int | None:
        stmt = (
            select(user_cohorts_table.c.cohort_id)
            .where(user_cohorts_table.c.user_id == user_id)
            .limit(1)
        )
        with Session(self._engine) as session:
            row = session.execute(stmt).first()
        return int(row.cohort_id) if row else None

    def cohort_top(self, cohort_id: int, *, limit: int = 10) -> list[RecommendationItem]:
        """Per-cohort top-K. Used as a cold-start fallback for users
        whose history is too thin to support a personalised list."""
        stmt = text(
            """
            SELECT c.steam_appid, c.score, g.name, g.header_image
              FROM cohort_top c
              JOIN games g ON g.steam_appid = c.steam_appid
             WHERE c.cohort_id = :cohort_id
             ORDER BY c.rank ASC
             LIMIT :limit
            """
        )
        with Session(self._engine) as session:
            rows = session.execute(stmt, {"cohort_id": cohort_id, "limit": limit}).all()
        return [
            RecommendationItem(
                steam_appid=int(r.steam_appid),
                name=str(r.name),
                header_image=r.header_image,
                score=float(r.score),
            )
            for r in rows
        ]

    def global_top(self, limit: int = 10) -> list[RecommendationItem]:
        stmt = text(
            """
            SELECT r.steam_appid, g.name, g.header_image, AVG(r.score) AS score
              FROM user_recommendations r
              JOIN games g ON g.steam_appid = r.steam_appid
             GROUP BY r.steam_appid, g.name, g.header_image
             ORDER BY score DESC
             LIMIT :limit
            """
        )
        with Session(self._engine) as session:
            rows = session.execute(stmt, {"limit": limit}).all()
        return [
            RecommendationItem(
                steam_appid=int(r.steam_appid),
                name=str(r.name),
                header_image=r.header_image,
                score=float(r.score),
            )
            for r in rows
        ]


def init_schema(engine: Engine) -> None:
    """Create the pgvector extension and tables when the API boots."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from gamereco.serving.db import metadata

    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS game_embeddings_cosine_idx "
                "ON game_embeddings USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )
