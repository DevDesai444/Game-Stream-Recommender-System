"""Temporal train/val/test splitting for interaction data.

We do not use random splitting: ranking metrics like NDCG@10 are only
meaningful when the validation/test splits live in the future of the
training data. The temporal split below ranks each user's interactions
by ``event_ts`` and holds out the most recent ``test_frac`` for testing
and the next most recent ``val_frac`` for validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


@dataclass(frozen=True)
class SplitFractions:
    val: float = 0.10
    test: float = 0.10

    def __post_init__(self) -> None:
        if not (0 < self.val < 1) or not (0 < self.test < 1):
            raise ValueError("split fractions must be in (0, 1)")
        if self.val + self.test >= 1:
            raise ValueError("val + test must be < 1")


def temporal_split(
    interactions: DataFrame, fractions: SplitFractions = SplitFractions()
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Return (train, val, test) DataFrames with per-user temporal hold-out."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("user_idx").orderBy(F.col("event_ts").desc())
    ranked = interactions.withColumn("rn", F.row_number().over(w))
    per_user_count = interactions.groupBy("user_idx").count().withColumnRenamed("count", "n")
    ranked = ranked.join(per_user_count, on="user_idx", how="inner")

    ranked = ranked.withColumn(
        "test_cut", F.greatest(F.lit(1), F.ceil(F.col("n") * F.lit(fractions.test)))
    ).withColumn(
        "val_cut",
        F.greatest(F.lit(1), F.ceil(F.col("n") * F.lit(fractions.test + fractions.val))),
    )

    test = ranked.filter(F.col("rn") <= F.col("test_cut")).drop("rn", "n", "test_cut", "val_cut")
    val = ranked.filter((F.col("rn") > F.col("test_cut")) & (F.col("rn") <= F.col("val_cut"))).drop(
        "rn", "n", "test_cut", "val_cut"
    )
    train = ranked.filter(F.col("rn") > F.col("val_cut")).drop("rn", "n", "test_cut", "val_cut")
    return train, val, test
