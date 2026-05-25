"""Lightweight non-personalised baselines used to anchor the hybrid lift.

The two recommenders below run in pure pandas / NumPy. A serious hybrid
recommender has to beat both of them — if it only beats raw popularity
it is overfitting to head-of-distribution titles, and if it can't beat
item-cooccurrence on a dense subset it isn't actually learning the
collaborative structure.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


def popularity_recommender(
    train_df: pd.DataFrame, *, n_items: int, k: int = 50
) -> dict[int, list[int]]:
    """Predict the same top-K most-played games for every user (minus known)."""
    popular = (
        train_df.groupby("game_idx")["confidence"]
        .sum()
        .sort_values(ascending=False)
        .index.to_numpy()
    )
    known: dict[int, set[int]] = {}
    for user_idx, group in train_df.groupby("user_idx"):
        known[int(user_idx)] = set(group["game_idx"].astype(int).tolist())

    users = train_df["user_idx"].unique()
    out: dict[int, list[int]] = {}
    for u in users:
        excluded = known.get(int(u), set())
        recs: list[int] = []
        for item in popular:
            if int(item) in excluded:
                continue
            recs.append(int(item))
            if len(recs) >= k:
                break
        out[int(u)] = recs
    return out


def item_cooccurrence_recommender(
    train_df: pd.DataFrame,
    *,
    n_items: int,
    k: int = 50,
    binary: bool = True,
) -> dict[int, list[int]]:
    """Predict via item-item co-occurrence (the classic Amazon-style baseline).

    For each user we sum the columns of the co-occurrence matrix
    corresponding to their training items, then take top-K of the
    resulting score vector after masking known items.
    """
    n_users = int(train_df["user_idx"].max() + 1)
    n_items = int(n_items)
    rows = train_df["user_idx"].to_numpy(dtype=np.int64)
    cols = train_df["game_idx"].to_numpy(dtype=np.int64)
    data = np.ones_like(rows, dtype=np.float64) if binary else train_df["confidence"].to_numpy()
    user_item = sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()
    item_user = user_item.T.tocsr()
    cooc = (item_user @ user_item).tocsr()
    cooc.setdiag(0.0)

    out: dict[int, list[int]] = {}
    for user_idx in range(n_users):
        start = user_item.indptr[user_idx]
        end = user_item.indptr[user_idx + 1]
        items = user_item.indices[start:end]
        if items.size == 0:
            out[user_idx] = []
            continue
        scores = np.asarray(cooc[items].sum(axis=0)).ravel()
        scores[items] = -np.inf
        top = np.argpartition(-scores, min(k, scores.size - 1))[:k]
        top = top[np.argsort(-scores[top])]
        out[user_idx] = [int(i) for i in top]
    return out


def known_items_by_user(train_df: pd.DataFrame) -> dict[int, set[int]]:
    return {
        int(user): set(int(g) for g in group["game_idx"])
        for user, group in train_df.groupby("user_idx")
    }


def item_popularity(train_df: pd.DataFrame) -> dict[int, float]:
    return {int(g): float(c) for g, c in train_df.groupby("game_idx").size().items()}


def truncate_predictions(predictions: dict[int, Iterable[int]], k: int) -> dict[int, list[int]]:
    return {u: list(items)[:k] for u, items in predictions.items()}
