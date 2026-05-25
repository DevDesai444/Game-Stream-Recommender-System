"""Ranking metrics used to score recommendation models.

Two implementations live here:

* :func:`ndcg_at_k_numpy` is a fast, pure-Python NDCG@K used for the
  small evaluation slices we keep in memory.
* :func:`ndcg_at_k_spark` is the same metric on a Spark
  ``DataFrame`` of (user_idx, recs) and ground-truth so the metric can
  be computed without collecting the per-user top-K into the driver.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from pyspark.sql import DataFrame


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k_numpy(
    predictions: dict[int, Sequence[int]],
    ground_truth: dict[int, Iterable[int]],
    k: int = 10,
) -> float:
    """NDCG@K over per-user ranked candidate lists.

    Items in ground truth count as binary relevance 1, everything else 0.
    Users with an empty ground-truth list are skipped (consistent with the
    way Spark MLlib's RankingMetrics handles them).
    """
    scores: list[float] = []
    for user, preds in predictions.items():
        truth = set(ground_truth.get(user, []))
        if not truth:
            continue
        top_k = list(preds)[:k]
        gains = [1.0 if item in truth else 0.0 for item in top_k]
        ideal = [1.0] * min(len(truth), k)
        ideal_dcg = _dcg(ideal)
        if ideal_dcg == 0:
            continue
        scores.append(_dcg(gains) / ideal_dcg)
    return float(np.mean(scores)) if scores else 0.0


def collect_top_k(recommendations: DataFrame, k: int = 10) -> dict[int, list[int]]:
    """Collect (user_idx -> [game_idx, ...]) sorted by score desc."""
    from pyspark.sql import functions as F

    out: dict[int, list[int]] = {}
    rows = recommendations.orderBy("user_idx", F.col("score").desc()).collect()
    for row in rows:
        out.setdefault(int(row["user_idx"]), []).append(int(row["game_idx"]))
        if len(out[int(row["user_idx"])]) > k:
            out[int(row["user_idx"])] = out[int(row["user_idx"])][:k]
    return out


def ground_truth_from_holdout(holdout: DataFrame) -> dict[int, list[int]]:
    rows = holdout.select("user_idx", "game_idx").collect()
    truth: dict[int, list[int]] = {}
    for row in rows:
        truth.setdefault(int(row["user_idx"]), []).append(int(row["game_idx"]))
    return truth
