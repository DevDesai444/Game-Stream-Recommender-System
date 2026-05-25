"""Bronze layer: land raw NDJSON ingestion outputs into Delta tables."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from gamereco.common.logging import get_logger
from gamereco.common.paths import LakePaths

log = get_logger(__name__)


def _read_jsonl(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.option("mode", "PERMISSIVE").json(path)


def land_user_summary(spark: SparkSession, lake: LakePaths) -> int:
    src = str(lake.bronze_users / "user_summary.jsonl")
    df = _read_jsonl(spark, src).withColumn("ingested_at", F.current_timestamp())
    target = str(lake.bronze_users / "delta")
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(target)
    return df.count()


def land_owned_games(spark: SparkSession, lake: LakePaths) -> int:
    src = str(lake.bronze_owned_games / "owned_games.jsonl")
    df = (
        _read_jsonl(spark, src)
        .withColumn("ingested_at", F.current_timestamp())
        .filter(F.col("steamid").isNotNull())
    )
    target = str(lake.bronze_owned_games / "delta")
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(target)
    return df.count()


def land_recently_played(spark: SparkSession, lake: LakePaths) -> int:
    src = str(lake.bronze_recently_played / "recently_played.jsonl")
    df = _read_jsonl(spark, src).withColumn("ingested_at", F.current_timestamp())
    target = str(lake.bronze_recently_played / "delta")
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(target)
    return df.count()


def land_game_details(spark: SparkSession, lake: LakePaths) -> int:
    src = str(lake.bronze_game_detail / "game_detail.jsonl")
    df = (
        _read_jsonl(spark, src)
        .withColumn("ingested_at", F.current_timestamp())
        .dropDuplicates(["steam_appid"])
    )
    target = str(lake.bronze_game_detail / "delta")
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(target)
    return df.count()


def land_friends(spark: SparkSession, lake: LakePaths) -> int:
    src = str(lake.bronze_friends / "friends.jsonl")
    df = _read_jsonl(spark, src).withColumn("ingested_at", F.current_timestamp())
    target = str(lake.bronze_friends / "delta")
    df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(target)
    return df.count()


def run_bronze(spark: SparkSession, lake: LakePaths) -> dict[str, int]:
    """Land all bronze NDJSON files into Delta tables."""
    counts = {
        "user_summary": land_user_summary(spark, lake),
        "owned_games": land_owned_games(spark, lake),
        "recently_played": land_recently_played(spark, lake),
        "game_details": land_game_details(spark, lake),
        "friends": land_friends(spark, lake),
    }
    log.info("bronze.done", **counts)
    return counts
