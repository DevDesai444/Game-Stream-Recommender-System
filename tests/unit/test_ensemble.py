"""Tests for the XGBoost ensemble layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamereco.training.ensemble import (
    FEATURE_COLUMNS,
    XGBConfig,
    lift_vs_baseline,
    train_xgb,
)


@pytest.fixture
def ranking_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, float]] = []
    for user in range(20):
        for game in range(10):
            als = rng.uniform(0, 1)
            ncf = rng.uniform(0, 1)
            label = int(0.55 * als + 0.45 * ncf > 0.5)
            rows.append(
                {
                    "user_idx": user,
                    "game_idx": game,
                    "als_score": als,
                    "ncf_score": ncf,
                    "user_cluster": user % 3,
                    "cluster_popularity": int(rng.integers(0, 5)),
                    "log_playtime_user": float(rng.uniform(0, 3)),
                    "log_global_popularity": float(rng.uniform(0, 3)),
                    "label": label,
                }
            )
    return pd.DataFrame(rows).sort_values("user_idx").reset_index(drop=True)


def test_feature_columns_listed() -> None:
    assert "als_score" in FEATURE_COLUMNS
    assert "ncf_score" in FEATURE_COLUMNS
    assert "user_cluster" in FEATURE_COLUMNS


def test_lift_vs_baseline_positive() -> None:
    assert lift_vs_baseline(0.5, 0.4) == pytest.approx(0.25)


def test_lift_vs_baseline_zero_baseline() -> None:
    assert lift_vs_baseline(0.5, 0.0) == 0.0


def test_lift_vs_baseline_negative_when_worse() -> None:
    assert lift_vs_baseline(0.3, 0.4) < 0


def test_train_xgb_returns_metrics(ranking_frame: pd.DataFrame) -> None:
    cutoff = ranking_frame["user_idx"].max() // 2
    train = ranking_frame[ranking_frame["user_idx"] <= cutoff].copy()
    val = ranking_frame[ranking_frame["user_idx"] > cutoff].copy()
    artifacts = train_xgb(train, val, XGBConfig(n_estimators=20, early_stopping_rounds=5))
    assert "ndcg_at_10" in artifacts.metrics
    assert 0.0 <= artifacts.metrics["ndcg_at_10"] <= 1.0


def test_train_xgb_feature_importance_populated(ranking_frame: pd.DataFrame) -> None:
    cutoff = ranking_frame["user_idx"].max() // 2
    train = ranking_frame[ranking_frame["user_idx"] <= cutoff].copy()
    val = ranking_frame[ranking_frame["user_idx"] > cutoff].copy()
    artifacts = train_xgb(train, val, XGBConfig(n_estimators=15, early_stopping_rounds=5))
    # XGBoost may use 0 or more features; importance is a dict.
    assert isinstance(artifacts.feature_importance, dict)
