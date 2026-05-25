"""Loader for the public Steam-200k dataset (200K real user-game interactions).

The raw CSV has four columns:

    user_id, game_name, behavior, value

where ``behavior`` is either ``"purchase"`` (value=1.0) or ``"play"``
(value = hours played). For implicit-feedback modelling we collapse
each (user, game) pair into a single record carrying the maximum play
hours observed, and assign compact integer indices over the user and
game vocabularies.

The pandas-backed silver/gold shape is intentionally a 1:1 mirror of
the Spark silver/gold contract — the same columns, the same temporal
split semantics, the same ``confidence = log1p(playtime_minutes)`` —
so models trained either way share a single evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Steam200kRecord:
    """A single collapsed user-game interaction."""

    user_id: str
    game_name: str
    purchased: bool
    play_hours: float


def load_steam_200k(path: str | Path) -> pd.DataFrame:
    """Read the raw Steam-200k CSV into a tidy ``DataFrame``.

    The CSV has no header in the canonical dump, so we name columns
    explicitly. The trailing zero column is dropped (it is unused).
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["user_id", "game_name", "behavior", "value", "_zero"],
        dtype={"user_id": "string", "game_name": "string", "behavior": "string"},
    )
    df = df.drop(columns=["_zero"])
    df["user_id"] = df["user_id"].str.strip()
    df["game_name"] = df["game_name"].str.strip()
    df["behavior"] = df["behavior"].str.lower().str.strip()
    return df


def _collapse_to_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse purchase + play rows into one interaction per (user, game)."""
    plays = (
        df[df["behavior"] == "play"]
        .groupby(["user_id", "game_name"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "play_hours"})
    )
    purchases = (
        df[df["behavior"] == "purchase"][["user_id", "game_name"]]
        .drop_duplicates()
        .assign(purchased=True)
    )
    merged = purchases.merge(plays, on=["user_id", "game_name"], how="outer")
    merged["purchased"] = merged["purchased"].fillna(False).astype(bool)
    merged["play_hours"] = merged["play_hours"].fillna(0.0).astype(float)
    return merged


def materialise_silver(
    df: pd.DataFrame,
    *,
    min_interactions_per_user: int = 3,
    min_interactions_per_game: int = 2,
) -> pd.DataFrame:
    """Produce the silver-layer interaction table from raw rows.

    Mirrors the Spark silver contract: integer user/game indices, log1p
    confidence, and the same per-user activity floor. Users / games
    that fall below the activity floor are pruned before indices are
    assigned, so the index space is dense.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "user_idx",
                "game_idx",
                "user_id",
                "game_name",
                "play_hours",
                "playtime_minutes",
                "confidence",
                "purchased",
            ]
        )

    collapsed = _collapse_to_interactions(df)

    user_counts = collapsed.groupby("user_id").size()
    game_counts = collapsed.groupby("game_name").size()
    qualified_users = user_counts[user_counts >= min_interactions_per_user].index
    qualified_games = game_counts[game_counts >= min_interactions_per_game].index
    filtered = collapsed[
        collapsed["user_id"].isin(qualified_users) & collapsed["game_name"].isin(qualified_games)
    ].copy()

    user_index = {u: i for i, u in enumerate(sorted(filtered["user_id"].unique()))}
    game_index = {g: i for i, g in enumerate(sorted(filtered["game_name"].unique()))}
    filtered["user_idx"] = filtered["user_id"].map(user_index).astype(np.int32)
    filtered["game_idx"] = filtered["game_name"].map(game_index).astype(np.int32)

    filtered["playtime_minutes"] = (filtered["play_hours"] * 60.0).round().astype(np.int64)
    filtered["confidence"] = np.log1p(filtered["playtime_minutes"].to_numpy(dtype=np.float64))
    # A purchase with zero playtime is still a positive signal — give it
    # a small floor so ALS doesn't ignore those interactions.
    purchase_only_mask = (filtered["confidence"] == 0.0) & filtered["purchased"]
    filtered.loc[purchase_only_mask, "confidence"] = np.log1p(1.0)

    return filtered[
        [
            "user_idx",
            "game_idx",
            "user_id",
            "game_name",
            "play_hours",
            "playtime_minutes",
            "confidence",
            "purchased",
        ]
    ].reset_index(drop=True)


def temporal_split_pandas(
    silver: pd.DataFrame,
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified-by-user holdout that imitates a temporal split.

    Steam-200k carries no event timestamps; in lieu of real time we use
    each user's *play_hours* ordering as a deterministic stand-in
    (higher playtime ≈ later, since cumulative playtime can only grow).
    The behaviour at the (user, game) granularity is identical to the
    Spark temporal_split contract — most-recent N% held out, next N%
    for validation, rest for training. Users with fewer than three
    interactions go entirely into training (no holdout is meaningful).
    """
    if not 0.0 < val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")

    rng = np.random.default_rng(seed)
    silver = silver.copy()
    # Stable tiebreaker: hash of (user, game) so equal-playtime rows
    # still split deterministically.
    silver["_tiebreak"] = silver["user_idx"].astype(int) * 1_000_003 + silver["game_idx"].astype(
        int
    )
    silver = silver.sort_values(
        ["user_idx", "play_hours", "_tiebreak"], ascending=[True, True, True]
    )

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in silver.groupby("user_idx", sort=False):
        n = len(group)
        if n < 3:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:
            n_val = max(1, n - n_test - 1)
        # Most recent at the tail because we sorted ascending by play_hours.
        test = group.tail(n_test)
        val = group.iloc[-(n_test + n_val) : -n_test] if n_test > 0 else group.tail(n_val)
        train = group.iloc[: n - n_test - n_val]
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    _ = rng  # rng reserved for future shuffling extensions
    drop = ["_tiebreak"]
    train_df = pd.concat(train_parts).drop(columns=drop).reset_index(drop=True)
    val_df = (
        pd.concat(val_parts).drop(columns=drop).reset_index(drop=True)
        if val_parts
        else silver.iloc[0:0].drop(columns=drop)
    )
    test_df = (
        pd.concat(test_parts).drop(columns=drop).reset_index(drop=True)
        if test_parts
        else silver.iloc[0:0].drop(columns=drop)
    )
    return train_df, val_df, test_df
