"""Gold layer: temporal train/val/test interactions ready for training."""

from __future__ import annotations

from pyspark.sql import SparkSession

from gamereco.common.logging import get_logger
from gamereco.common.paths import LakePaths
from gamereco.etl.splits import SplitFractions, temporal_split

log = get_logger(__name__)


def build_gold(
    spark: SparkSession,
    lake: LakePaths,
    fractions: SplitFractions = SplitFractions(),
) -> dict[str, int]:
    interactions = spark.read.format("delta").load(str(lake.silver_interactions))
    train, val, test = temporal_split(interactions, fractions)

    train.write.format("delta").mode("overwrite").save(str(lake.gold_train))
    val.write.format("delta").mode("overwrite").save(str(lake.gold_val))
    test.write.format("delta").mode("overwrite").save(str(lake.gold_test))

    counts = {"train": train.count(), "val": val.count(), "test": test.count()}
    log.info("gold.done", **counts)
    return counts
