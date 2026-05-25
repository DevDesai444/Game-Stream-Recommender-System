"""Silver layer: explode owned-games arrays into a clean interaction table.

Owned games arrive as one nested array per user. The silver layer:

1. explodes the array into one row per (user, game),
2. converts ``playtime_forever`` (minutes) into the implicit-feedback
   confidence used by ALS / NCF training,
3. assigns compact integer indices (``user_idx``, ``game_idx``) so the model
   layer never has to round-trip 64-bit Steam IDs,
4. drops users with fewer than 3 owned games (too sparse to train against).
"""

from __future__ import annotations

import math

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from gamereco.common.logging import get_logger
from gamereco.common.paths import LakePaths

log = get_logger(__name__)


MIN_INTERACTIONS_PER_USER = 3


def _explode_owned_games(owned: DataFrame) -> DataFrame:
    exploded = (
        owned.select("steamid", F.explode("games").alias("g"), "ingested_at")
        .select(
            F.col("steamid").alias("user_id"),
            F.col("g.appid").cast("int").alias("steam_appid"),
            F.col("g.playtime_forever").cast("int").alias("playtime_minutes"),
            F.coalesce(F.col("g.playtime_2weeks"), F.lit(0)).cast("int").alias("playtime_2weeks"),
            F.col("ingested_at"),
        )
        .filter(F.col("steam_appid").isNotNull())
        .filter(F.col("playtime_minutes") >= 0)
    )
    return exploded


def _attach_event_ts(df: DataFrame) -> DataFrame:
    """Synthesise an event timestamp.

    Steam doesn't expose the timestamp of *when* a user played a game, only
    cumulative playtime. We approximate event time by anchoring at the
    ingestion timestamp and using a deterministic offset derived from
    ``playtime_2weeks`` (recent activity ≈ recent event) so the temporal
    split has signal to work with.
    """
    return df.withColumn(
        "event_ts",
        F.from_unixtime(
            F.unix_timestamp("ingested_at")
            - (
                F.lit(60 * 60 * 24 * 14)
                - F.least(F.col("playtime_2weeks"), F.lit(60 * 60 * 24 * 14))
            )
        ).cast("timestamp"),
    )


def build_silver(spark: SparkSession, lake: LakePaths) -> dict[str, int]:
    """Produce silver interaction/game/user Delta tables from bronze."""
    owned = spark.read.format("delta").load(str(lake.bronze_owned_games / "delta"))
    games = spark.read.format("delta").load(str(lake.bronze_game_detail / "delta"))
    summaries = spark.read.format("delta").load(str(lake.bronze_users / "delta"))

    interactions = _explode_owned_games(owned)
    interactions = _attach_event_ts(interactions)

    user_counts = interactions.groupBy("user_id").count()
    qualified_users = user_counts.filter(F.col("count") >= MIN_INTERACTIONS_PER_USER).select(
        "user_id"
    )
    interactions = interactions.join(qualified_users, on="user_id", how="inner")

    user_idx_window = Window.orderBy("user_id")
    user_idx = (
        interactions.select("user_id")
        .distinct()
        .withColumn("user_idx", F.row_number().over(user_idx_window) - 1)
    )
    game_idx_window = Window.orderBy("steam_appid")
    game_idx = (
        interactions.select("steam_appid")
        .distinct()
        .withColumn("game_idx", F.row_number().over(game_idx_window) - 1)
    )

    silver_interactions = (
        interactions.join(user_idx, on="user_id", how="inner")
        .join(game_idx, on="steam_appid", how="inner")
        .withColumn(
            "confidence",
            F.log1p(F.col("playtime_minutes")).cast("double"),
        )
        .select(
            "user_idx",
            "game_idx",
            "user_id",
            "steam_appid",
            "playtime_minutes",
            "playtime_2weeks",
            "confidence",
            "event_ts",
        )
    )

    silver_interactions.write.format("delta").mode("overwrite").save(str(lake.silver_interactions))

    silver_users = summaries.join(user_idx, F.col("steamid") == F.col("user_id"), "inner").select(
        "user_idx",
        F.col("steamid").alias("user_id"),
        "personaname",
        "loccountrycode",
        "timecreated",
    )
    silver_users.write.format("delta").mode("overwrite").save(str(lake.silver_users))

    silver_games = games.join(game_idx, on="steam_appid", how="inner").select(
        "game_idx",
        "steam_appid",
        "name",
        "header_image",
        "short_description",
        "genres",
        "categories",
        "release_date",
        "metacritic",
    )
    silver_games.write.format("delta").mode("overwrite").save(str(lake.silver_games))

    counts = {
        "interactions": silver_interactions.count(),
        "users": silver_users.count(),
        "games": silver_games.count(),
    }
    log.info("silver.done", **counts)
    # Sanity floor — keeps Airflow happy if the dataset shrinks unexpectedly.
    if counts["interactions"] < 1:
        raise RuntimeError("silver interactions table is empty")
    # Log NDCG-ready scale.
    log.info(
        "silver.scale",
        approx_users=counts["users"],
        approx_games=counts["games"],
        log_interactions=round(math.log10(max(counts["interactions"], 1)), 2),
    )
    return counts
