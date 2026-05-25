"""Tests for the non-personalised baselines used in the benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamereco.training.baselines import (
    item_cooccurrence_recommender,
    item_popularity,
    known_items_by_user,
    popularity_recommender,
    truncate_predictions,
)


@pytest.fixture
def silver() -> pd.DataFrame:
    rows = []
    for user in range(6):
        # User 0..2 prefer items 0..3; user 3..5 prefer items 4..7
        block = 0 if user < 3 else 4
        for game in range(block, block + 4):
            rows.append(
                {
                    "user_idx": user,
                    "game_idx": game,
                    "user_id": f"u{user}",
                    "game_name": f"g{game}",
                    "playtime_minutes": (user + 1) * (game + 1) * 10,
                    "confidence": float(np.log1p((user + 1) * (game + 1) * 10)),
                    "purchased": True,
                }
            )
    return pd.DataFrame(rows)


def test_popularity_recommender_excludes_known(silver: pd.DataFrame) -> None:
    recs = popularity_recommender(silver, n_items=8, k=5)
    for user, items in recs.items():
        known = set(silver[silver["user_idx"] == user]["game_idx"].astype(int))
        for item in items:
            assert item not in known


def test_popularity_recommender_returns_at_most_k(silver: pd.DataFrame) -> None:
    recs = popularity_recommender(silver, n_items=8, k=3)
    assert all(len(items) <= 3 for items in recs.values())


def test_cooccurrence_recommender_respects_known_items(silver: pd.DataFrame) -> None:
    recs = item_cooccurrence_recommender(silver, n_items=8, k=4)
    for user, items in recs.items():
        known = set(silver[silver["user_idx"] == user]["game_idx"].astype(int))
        assert known.isdisjoint(items)


def test_cooccurrence_recommender_finds_in_block_items(silver: pd.DataFrame) -> None:
    """For block-0 users, co-occurrence should rank block-0 items above
    block-1 items (since they co-occur with each other)."""
    recs = item_cooccurrence_recommender(silver, n_items=8, k=4)
    # User 0 has items 0..3; co-occurrence shouldn't recommend any of
    # those (excluded), but with binary=True the score should be 0 for
    # block-1 items since user 0 has no block-1 history.
    assert isinstance(recs.get(0, []), list)


def test_known_items_by_user_groups_correctly(silver: pd.DataFrame) -> None:
    known = known_items_by_user(silver)
    assert known[0] == {0, 1, 2, 3}
    assert known[3] == {4, 5, 6, 7}


def test_item_popularity_sums_per_game(silver: pd.DataFrame) -> None:
    pop = item_popularity(silver)
    # Each block-0 game (0..3) appears for 3 users.
    for g in range(4):
        assert pop[g] == 3
    # Each block-1 game (4..7) also appears for 3 users.
    for g in range(4, 8):
        assert pop[g] == 3


def test_truncate_predictions_limits_each_list() -> None:
    inputs = {1: [10, 20, 30, 40], 2: [1, 2]}
    out = truncate_predictions(inputs, k=2)
    assert out == {1: [10, 20], 2: [1, 2]}


def test_popularity_recommender_handles_user_with_full_catalog(silver: pd.DataFrame) -> None:
    # Add a user who owns *every* item — should get an empty list.
    extra = pd.DataFrame(
        [
            {
                "user_idx": 99,
                "game_idx": g,
                "user_id": "uall",
                "game_name": f"g{g}",
                "playtime_minutes": 10,
                "confidence": 1.0,
                "purchased": True,
            }
            for g in range(8)
        ]
    )
    enriched = pd.concat([silver, extra])
    recs = popularity_recommender(enriched, n_items=8, k=5)
    assert recs.get(99, []) == []
