"""Tests for the in-memory implicit-ALS implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamereco.training.als_inmem import (
    ALSInMemConfig,
    ALSInMemModel,
    train_als_inmem,
)


def _toy_silver(seed: int = 0) -> pd.DataFrame:
    """Three taste groups of users, each preferring one block of games."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(30):
        block = user // 10
        favourites = range(block * 5, block * 5 + 5)
        for game in favourites:
            playtime = rng.integers(60, 600)
            rows.append(
                {
                    "user_idx": user,
                    "game_idx": game,
                    "user_id": str(user),
                    "game_name": f"g{game}",
                    "play_hours": playtime / 60,
                    "playtime_minutes": playtime,
                    "confidence": float(np.log1p(playtime)),
                    "purchased": True,
                }
            )
        # Sprinkle one random outside-block game per user
        outside = rng.integers(15, 25)
        rows.append(
            {
                "user_idx": user,
                "game_idx": int(outside),
                "user_id": str(user),
                "game_name": f"g{outside}",
                "play_hours": 0.5,
                "playtime_minutes": 30,
                "confidence": float(np.log1p(30)),
                "purchased": True,
            }
        )
    return pd.DataFrame(rows)


def test_model_returns_expected_shape() -> None:
    silver = _toy_silver()
    model = train_als_inmem(
        silver,
        n_users=silver["user_idx"].max() + 1,
        n_items=silver["game_idx"].max() + 1,
        config=ALSInMemConfig(factors=8, iterations=5),
    )
    assert model.user_factors.shape == (30, 8)
    assert model.item_factors.shape[1] == 8


def test_predict_single_and_batch_agree() -> None:
    silver = _toy_silver()
    model = train_als_inmem(silver, config=ALSInMemConfig(factors=4, iterations=3))
    users = np.array([0, 1, 2])
    items = np.array([0, 5, 10])
    batch = model.predict_batch(users, items)
    for i, (u, g) in enumerate(zip(users, items, strict=True)):
        assert model.predict(int(u), int(g)) == pytest.approx(batch[i])


def test_score_all_items_returns_per_item_vector() -> None:
    silver = _toy_silver()
    n_items = silver["game_idx"].max() + 1
    model = train_als_inmem(silver, config=ALSInMemConfig(factors=4, iterations=3))
    scores = model.score_all_items(0)
    assert scores.shape[0] == n_items


def test_recommend_top_k_descending() -> None:
    silver = _toy_silver()
    model = train_als_inmem(silver, config=ALSInMemConfig(factors=8, iterations=8))
    recs = model.recommend(0, k=5)
    assert len(recs) == 5
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_excludes_known_items() -> None:
    silver = _toy_silver()
    model = train_als_inmem(silver, config=ALSInMemConfig(factors=8, iterations=8))
    known = silver[silver["user_idx"] == 0]["game_idx"].tolist()
    recs = model.recommend(0, k=10, exclude=known)
    assert all(item not in known for item, _ in recs)


def test_als_users_in_same_block_share_more_recs_than_across_blocks() -> None:
    """ALS should embed same-block users closer than cross-block users.

    The toy data has three clean taste blocks. After training, the
    overlap between two same-block users' top-K recommendations
    should be larger than between two cross-block users'.
    """
    silver = _toy_silver(seed=0)
    model = train_als_inmem(
        silver,
        n_users=silver["user_idx"].max() + 1,
        n_items=silver["game_idx"].max() + 1,
        config=ALSInMemConfig(factors=16, iterations=20, reg=0.01, alpha=40.0),
    )

    def top_k(user: int, k: int = 10) -> set[int]:
        return {
            item
            for item, _ in model.recommend(
                user, k=k, exclude=silver[silver["user_idx"] == user]["game_idx"].tolist()
            )
        }

    same_block = len(top_k(0) & top_k(5))  # both block-0
    cross_block = len(top_k(0) & top_k(20))  # block-0 vs block-2
    assert same_block >= cross_block


def test_handles_empty_user_row() -> None:
    silver = _toy_silver()
    # Drop user 29 entirely; their factor row should remain zeros.
    silver = silver[silver["user_idx"] != 29]
    model = train_als_inmem(
        silver,
        n_users=30,
        n_items=silver["game_idx"].max() + 1,
        config=ALSInMemConfig(factors=4, iterations=3),
    )
    assert np.allclose(model.user_factors[29], 0.0)


def test_config_is_propagated_to_model() -> None:
    silver = _toy_silver()
    cfg = ALSInMemConfig(factors=6, iterations=2, reg=0.5, alpha=2.0)
    model = train_als_inmem(silver, config=cfg)
    assert model.config.factors == 6
    assert model.config.alpha == 2.0
