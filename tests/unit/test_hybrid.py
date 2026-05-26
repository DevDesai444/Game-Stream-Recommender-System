"""Smoke tests for the hybrid recommender harness.

The end-to-end metrics for the hybrid are produced by the benchmark
runner against real Steam data (see benchmarks/results.md). These
unit tests just pin the *contract* of the building blocks so a future
refactor can't silently break the harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamereco.training.als_inmem import ALSInMemConfig, train_als_inmem
from gamereco.training.hybrid import (
    HybridBundle,
    als_topk,
    assemble_candidates,
    hybrid_rerank,
    ncf_topk,
    train_hybrid,
)


@pytest.fixture
def silver() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(60):
        block = user // 20
        for game in range(block * 6, block * 6 + 6):
            rows.append(
                {
                    "user_idx": user,
                    "game_idx": game,
                    "user_id": f"u{user}",
                    "game_name": f"g{game}",
                    "play_hours": float(rng.uniform(1, 5)),
                    "playtime_minutes": int(rng.integers(60, 600)),
                    "confidence": float(np.log1p(rng.integers(60, 600))),
                    "purchased": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def train_val(silver: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Hold out one item per user as val; the rest train.
    held: list[int] = []
    for user, group in silver.groupby("user_idx"):
        held.append(int(group.index[-1]))
    val = silver.loc[held].reset_index(drop=True)
    train = silver.drop(held).reset_index(drop=True)
    return train, val


def test_als_topk_excludes_known_and_returns_k_items(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    als = train_als_inmem(
        train, n_users=n_users, n_items=n_items, config=ALSInMemConfig(factors=8, iterations=4)
    )
    users = val["user_idx"].unique()
    recs = als_topk(als, train, users, k=5)
    for user, items in recs.items():
        known = set(train[train["user_idx"] == user]["game_idx"].astype(int))
        assert known.isdisjoint(items)
        assert len(items) == 5


def test_assemble_candidates_attaches_all_features(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    als = train_als_inmem(
        train, n_users=n_users, n_items=n_items, config=ALSInMemConfig(factors=8, iterations=4)
    )
    clusters = np.zeros(n_users, dtype=np.int32)
    cand = assemble_candidates(
        train,
        val,
        als=als,
        user_clusters=clusters,
        n_candidates=20,
        ncf_user_emb=None,
        ncf_item_emb=None,
    )
    for col in (
        "als_score",
        "ncf_score",
        "user_cluster",
        "cluster_popularity",
        "log_playtime_user",
        "log_global_popularity",
        "label",
    ):
        assert col in cand.df.columns


def test_train_hybrid_returns_runnable_bundle(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    bundle = train_hybrid(
        train,
        val,
        n_users=n_users,
        n_items=n_items,
        als_config=ALSInMemConfig(factors=8, iterations=4),
        kmeans_k=4,
        n_candidates=20,
        use_ncf=False,
    )
    assert isinstance(bundle, HybridBundle)
    assert bundle.booster is not None
    recs = hybrid_rerank(bundle, train, val["user_idx"].unique(), n_candidates=20, k=5)
    for user, items in recs.items():
        known = set(train[train["user_idx"] == user]["game_idx"].astype(int))
        assert known.isdisjoint(items)


def test_ncf_topk_returns_user_to_list_map(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    # Use random embeddings as a stand-in for trained NCF embeddings.
    rng = np.random.default_rng(0)
    user_emb = rng.normal(0, 0.1, size=(n_users, 8))
    item_emb = rng.normal(0, 0.1, size=(n_items, 8))
    recs = ncf_topk(user_emb, item_emb, train, val["user_idx"].unique(), k=5)
    assert set(recs.keys()) == set(int(u) for u in val["user_idx"].unique())
    assert all(len(v) == 5 for v in recs.values())


def test_train_hybrid_accepts_pretrained_ncf(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Pre-trained NCF embeddings flow into the bundle without an inline train."""
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    rng = np.random.default_rng(0)
    pre_user = rng.normal(0, 0.1, size=(n_users, 8)).astype(np.float32)
    pre_item = rng.normal(0, 0.1, size=(n_items, 8)).astype(np.float32)
    bundle = train_hybrid(
        train,
        val,
        n_users=n_users,
        n_items=n_items,
        als_config=ALSInMemConfig(factors=8, iterations=4),
        kmeans_k=4,
        n_candidates=20,
        pretrained_ncf=(pre_user, pre_item),
    )
    # Bundle reuses the supplied embeddings rather than re-training.
    assert bundle.ncf_user_emb is pre_user
    assert bundle.ncf_item_emb is pre_item
    # Ranker still scores cleanly.
    recs = hybrid_rerank(bundle, train, val["user_idx"].unique(), n_candidates=20, k=5)
    assert len(recs) > 0


def test_train_hybrid_unions_ncf_candidates(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """ncf_candidate_k > 0 mixes NCF candidates into the candidate pool."""
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    rng = np.random.default_rng(0)
    pre_user = rng.normal(0, 0.1, size=(n_users, 8)).astype(np.float32)
    pre_item = rng.normal(0, 0.1, size=(n_items, 8)).astype(np.float32)
    bundle = train_hybrid(
        train,
        val,
        n_users=n_users,
        n_items=n_items,
        als_config=ALSInMemConfig(factors=8, iterations=4),
        kmeans_k=4,
        n_candidates=10,
        pretrained_ncf=(pre_user, pre_item),
        ncf_candidate_k=10,
    )
    # The richer feature set is selected when NCF retrieval is enabled.
    for col in ("ncf_rank", "ncf_in_top_k", "source_als", "source_ncf"):
        assert col in bundle.feature_columns
    assert bundle.ncf_candidate_k == 10
    recs = hybrid_rerank(bundle, train, val["user_idx"].unique(), n_candidates=10, k=5)
    for user, items in recs.items():
        known = set(train[train["user_idx"] == user]["game_idx"].astype(int))
        assert known.isdisjoint(items)


def test_assemble_candidates_ncf_union_adds_source_flags(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, val = train_val
    n_users = int(train["user_idx"].max() + 1)
    n_items = int(train["game_idx"].max() + 1)
    als = train_als_inmem(
        train, n_users=n_users, n_items=n_items, config=ALSInMemConfig(factors=8, iterations=4)
    )
    clusters = np.zeros(n_users, dtype=np.int32)
    rng = np.random.default_rng(0)
    pre_user = rng.normal(0, 0.1, size=(n_users, 8)).astype(np.float32)
    pre_item = rng.normal(0, 0.1, size=(n_items, 8)).astype(np.float32)
    cand = assemble_candidates(
        train,
        val,
        als=als,
        user_clusters=clusters,
        n_candidates=10,
        ncf_user_emb=pre_user,
        ncf_item_emb=pre_item,
        ncf_candidate_k=10,
    )
    for col in (
        "als_score",
        "ncf_score",
        "ncf_rank",
        "ncf_in_top_k",
        "source_als",
        "source_ncf",
        "label",
    ):
        assert col in cand.df.columns
    # At least one row must be flagged by each retriever.
    assert (cand.df["source_als"] == 1).any()
    # Every row must be sourced by at least one retriever.
    assert ((cand.df["source_als"] + cand.df["source_ncf"]) > 0).all()
