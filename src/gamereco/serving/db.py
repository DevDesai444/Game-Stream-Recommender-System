"""SQLAlchemy + pgvector model definitions for the recommendation API."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

try:
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover - pgvector optional at test time
    Vector = None  # type: ignore[assignment]
    PGVECTOR_AVAILABLE = False

from gamereco.common.config import postgres_settings

metadata = MetaData()


games_table = Table(
    "games",
    metadata,
    Column("game_idx", Integer, primary_key=True),
    Column("steam_appid", BigInteger, unique=True, nullable=False),
    Column("name", Text, nullable=False),
    Column("header_image", Text),
    Column("short_description", Text),
)


game_embeddings_table = Table(
    "game_embeddings",
    metadata,
    Column("game_idx", Integer, ForeignKey("games.game_idx"), primary_key=True),
    Column("steam_appid", BigInteger, nullable=False),
    Column("embedding", Vector(64) if PGVECTOR_AVAILABLE else Text, nullable=False),
)


user_recommendations_table = Table(
    "user_recommendations",
    metadata,
    Column("user_id", String(32), primary_key=True),
    Column("rank", Integer, primary_key=True),
    Column("steam_appid", BigInteger, nullable=False),
    Column("score", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
)


user_cohorts_table = Table(
    "user_cohorts",
    metadata,
    Column("user_id", String(32), primary_key=True),
    Column("cohort_id", Integer, nullable=False, index=True),
)


cohort_top_table = Table(
    "cohort_top",
    metadata,
    Column("cohort_id", Integer, primary_key=True),
    Column("rank", Integer, primary_key=True),
    Column("steam_appid", BigInteger, nullable=False),
    Column("score", Float, nullable=False),
)


def build_engine(echo: bool = False) -> Engine:
    cfg = postgres_settings()
    return create_engine(cfg.dsn, future=True, echo=echo, pool_pre_ping=True)


def session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or build_engine(), expire_on_commit=False, future=True)


def upsert_recommendations(
    session: Session,
    user_id: str,
    rows: list[dict[str, object]],
    model_version: str,
) -> None:
    """Idempotent upsert of (user_id, rank) -> recommendation rows."""
    if not rows:
        return
    stmt = pg_insert(user_recommendations_table).values(
        [
            {
                "user_id": user_id,
                "rank": row["rank"],
                "steam_appid": row["steam_appid"],
                "score": row["score"],
                "model_version": model_version,
            }
            for row in rows
        ]
    )
    excluded = stmt.excluded
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id", "rank"],
        set_={
            "steam_appid": excluded.steam_appid,
            "score": excluded.score,
            "model_version": excluded.model_version,
        },
    )
    session.execute(stmt)
    session.commit()
