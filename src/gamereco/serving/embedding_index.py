"""Publish game embeddings from the latest ALS / NCF artifacts into pgvector."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from gamereco.common.config import spark_settings
from gamereco.common.logging import configure_logging, get_logger
from gamereco.common.paths import LakePaths
from gamereco.etl.session import build_spark
from gamereco.serving.db import build_engine, game_embeddings_table
from gamereco.serving.store import init_schema

log = get_logger(__name__)


EMBEDDING_DIM = 64


def _als_item_factors(als_model: ALSModel) -> dict[int, np.ndarray]:
    factors = als_model.itemFactors.collect()
    out: dict[int, np.ndarray] = {}
    for row in factors:
        vec = np.array(row["features"], dtype=np.float32)
        out[int(row["id"])] = vec
    return out


def _resize(vec: np.ndarray, target: int = EMBEDDING_DIM) -> np.ndarray:
    if vec.size == target:
        return vec
    if vec.size > target:
        return vec[:target]
    padded = np.zeros(target, dtype=np.float32)
    padded[: vec.size] = vec
    return padded


def publish_embeddings(spark, lake: LakePaths, engine: Engine) -> int:
    """Sync the latest ALS item factors into the pgvector index."""
    init_schema(engine)

    als_model = ALSModel.load(str(Path(lake.root) / "models" / "als"))
    games = (
        spark.read.format("delta")
        .load(str(lake.silver_games))
        .select("game_idx", "steam_appid", "name", "header_image", "short_description")
        .dropna(subset=["game_idx", "steam_appid"])
    )

    item_factors = _als_item_factors(als_model)
    game_rows = games.collect()

    inserted = 0
    with Session(engine) as session:
        session.execute(text("TRUNCATE TABLE game_embeddings"))
        session.execute(
            text(
                """
                INSERT INTO games (game_idx, steam_appid, name, header_image, short_description)
                VALUES (:game_idx, :steam_appid, :name, :header_image, :short_description)
                ON CONFLICT (game_idx) DO UPDATE SET
                    steam_appid = EXCLUDED.steam_appid,
                    name = EXCLUDED.name,
                    header_image = EXCLUDED.header_image,
                    short_description = EXCLUDED.short_description
                """
            ),
            [
                {
                    "game_idx": int(r["game_idx"]),
                    "steam_appid": int(r["steam_appid"]),
                    "name": str(r["name"]),
                    "header_image": r["header_image"],
                    "short_description": r["short_description"],
                }
                for r in game_rows
            ],
        )
        for row in game_rows:
            game_idx = int(row["game_idx"])
            factors = item_factors.get(game_idx)
            if factors is None:
                continue
            vec = _resize(factors)
            session.execute(
                game_embeddings_table.insert().values(
                    game_idx=game_idx,
                    steam_appid=int(row["steam_appid"]),
                    embedding=vec.tolist(),
                )
            )
            inserted += 1
        session.commit()
    log.info("embedding_index.published", count=inserted)
    return inserted


@click.command()
@click.option("--refresh/--no-refresh", default=True, help="Truncate and rebuild the index")
def main(refresh: bool) -> None:
    configure_logging()
    spark = build_spark("gamereco-embedding-index")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    engine = build_engine()
    if refresh:
        publish_embeddings(spark, lake, engine)
    else:
        log.info("embedding_index.skipped")


if __name__ == "__main__":  # pragma: no cover
    main()
