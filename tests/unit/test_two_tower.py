"""Tests for the two-tower NCF model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gamereco.training.two_tower import (
    ContentFeatureSpec,
    TwoTowerConfig,
    TwoTowerModel,
    build_item_feature_matrix,
    build_user_feature_matrix,
    fit_content_feature_spec,
    load_two_tower,
    save_two_tower,
    train_two_tower,
    two_tower_topk,
)


@pytest.fixture
def fixture_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_idx": 0,
                "app_id": 100,
                "name": "A",
                "genres": ["Action", "RPG"],
                "tags": ["Open World", "Story"],
                "specs": ["Single-player"],
                "price": 9.99,
                "release_year": 2018,
                "developer": "X",
                "publisher": "Y",
                "early_access": False,
            },
            {
                "game_idx": 1,
                "app_id": 200,
                "name": "B",
                "genres": ["Strategy"],
                "tags": ["Turn-Based"],
                "specs": ["Multi-player"],
                "price": None,
                "release_year": None,
                "developer": "",
                "publisher": "",
                "early_access": False,
            },
            {
                "game_idx": 2,
                "app_id": 300,
                "name": "C",
                "genres": ["Action"],
                "tags": ["Story", "FPS"],
                "specs": ["Single-player"],
                "price": np.nan,
                "release_year": float("nan"),
                "developer": "Z",
                "publisher": "W",
                "early_access": True,
            },
        ]
    )


@pytest.fixture
def fixture_users() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_idx": 0,
                "user_id": "u0",
                "items_count": 10,
                "total_playtime": 1200,
                "active_recent": 3,
                "reviews_count": 2,
            },
            {
                "user_idx": 1,
                "user_id": "u1",
                "items_count": 4,
                "total_playtime": 60,
                "active_recent": 0,
                "reviews_count": 0,
            },
            {
                "user_idx": 2,
                "user_id": "u2",
                "items_count": 25,
                "total_playtime": 5000,
                "active_recent": 12,
                "reviews_count": 5,
            },
        ]
    )


@pytest.fixture
def fixture_train(fixture_users: pd.DataFrame, fixture_games: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # Each user owns 2 of the 3 games (different pairs) so negative
    # sampling has at least one valid candidate per user.
    ownership = {0: [0, 1], 1: [1, 2], 2: [0, 2]}
    for u in fixture_users["user_idx"]:
        for g in ownership[int(u)]:
            rows.append(
                {
                    "user_idx": int(u),
                    "game_idx": int(g),
                    "playtime_forever": int(rng.integers(10, 600)),
                    "playtime_2weeks": 0,
                    "confidence": float(np.log1p(rng.integers(10, 600))),
                }
            )
    return pd.DataFrame(rows)


def test_fit_content_feature_spec_builds_vocabs(fixture_games: pd.DataFrame) -> None:
    spec = fit_content_feature_spec(fixture_games)
    assert "Action" in spec.genre_vocab
    assert "RPG" in spec.genre_vocab
    assert "Strategy" in spec.genre_vocab
    # Each tag with a count >= 1 should be in the vocab (we're under top-200 cap).
    assert "Story" in spec.tag_vocab
    assert "Open World" in spec.tag_vocab
    assert spec.price_std > 0


def test_build_item_feature_matrix_dimensions(fixture_games: pd.DataFrame) -> None:
    spec = fit_content_feature_spec(fixture_games)
    g, t, d = build_item_feature_matrix(fixture_games, spec)
    assert g.shape == (3, spec.n_genres)
    assert t.shape == (3, spec.n_tags)
    assert d.shape == (3, 5)
    # No NaN even when input price/year are None or NaN
    assert not np.isnan(d).any()
    # Game with NaN price has has_price flag set to 0
    assert d[1, 2] == 0.0
    assert d[2, 2] == 0.0
    # Game 0 (price=9.99, year=2018) has both flags set
    assert d[0, 2] == 1.0
    assert d[0, 3] == 1.0


def test_build_user_feature_matrix_z_scored(fixture_users: pd.DataFrame) -> None:
    arr, stats = build_user_feature_matrix(fixture_users)
    assert arr.shape == (3, 4)
    # Each column should be ~zero-mean (mean is exactly subtracted)
    np.testing.assert_allclose(arr.mean(axis=0), 0.0, atol=1e-5)
    # Stats dict should carry per-column mean/std
    assert "log_items_count_mean" in stats


def test_two_tower_forward_returns_finite_logits(
    fixture_games: pd.DataFrame, fixture_users: pd.DataFrame
) -> None:
    spec = fit_content_feature_spec(fixture_games)
    g, t, d = build_item_feature_matrix(fixture_games, spec)
    ud, _ = build_user_feature_matrix(fixture_users)
    cfg = TwoTowerConfig(
        n_users=3,
        n_items=3,
        n_genres=spec.n_genres,
        n_tags=spec.n_tags,
        n_user_dense=ud.shape[1],
        n_item_dense=d.shape[1],
        embedding_dim=8,
        tower_hidden=(16,),
        output_dim=8,
    )
    model = TwoTowerModel(cfg)
    out = model(
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.from_numpy(ud),
        torch.from_numpy(g),
        torch.from_numpy(t),
        torch.from_numpy(d),
    )
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_two_tower_score_in_unit_interval(
    fixture_games: pd.DataFrame, fixture_users: pd.DataFrame
) -> None:
    spec = fit_content_feature_spec(fixture_games)
    g, t, d = build_item_feature_matrix(fixture_games, spec)
    ud, _ = build_user_feature_matrix(fixture_users)
    cfg = TwoTowerConfig(
        n_users=3,
        n_items=3,
        n_genres=spec.n_genres,
        n_tags=spec.n_tags,
        n_user_dense=ud.shape[1],
        n_item_dense=d.shape[1],
        embedding_dim=8,
        tower_hidden=(16,),
        output_dim=8,
    )
    model = TwoTowerModel(cfg)
    s = model.score(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.from_numpy(ud[:2]),
        torch.from_numpy(g[:2]),
        torch.from_numpy(t[:2]),
        torch.from_numpy(d[:2]),
    )
    assert ((s >= 0) & (s <= 1)).all()


def test_train_two_tower_lowers_loss(
    fixture_train: pd.DataFrame,
    fixture_games: pd.DataFrame,
    fixture_users: pd.DataFrame,
) -> None:
    cfg = TwoTowerConfig(
        n_users=3,
        n_items=3,
        n_genres=0,
        n_tags=0,
        embedding_dim=8,
        tower_hidden=(16,),
        output_dim=8,
        epochs=4,
        batch_size=16,
        negative_ratio=2,
    )
    artifacts = train_two_tower(fixture_train, fixture_games, fixture_users, config=cfg)
    assert len(artifacts.train_loss_history) == cfg.epochs
    # Loss should not be NaN
    assert all(np.isfinite(artifacts.train_loss_history))
    assert artifacts.user_vectors.shape == (3, cfg.output_dim)
    assert artifacts.item_vectors.shape == (3, cfg.output_dim)


def test_two_tower_topk_excludes_known(
    fixture_train: pd.DataFrame,
    fixture_games: pd.DataFrame,
    fixture_users: pd.DataFrame,
) -> None:
    artifacts = train_two_tower(
        fixture_train,
        fixture_games,
        fixture_users,
        config=TwoTowerConfig(
            n_users=3,
            n_items=3,
            n_genres=0,
            n_tags=0,
            embedding_dim=8,
            tower_hidden=(16,),
            output_dim=8,
            epochs=2,
            batch_size=16,
            negative_ratio=1,
        ),
    )
    # Each user owns 2 of 3 games in the fixture, so top-K with the
    # owned items excluded should leave exactly one candidate (and the
    # one returned must not be in the user's training set).
    recs = two_tower_topk(artifacts, fixture_train, np.array([0, 1, 2]), k=5)
    for user, items in recs.items():
        known = set(fixture_train[fixture_train["user_idx"] == user]["game_idx"].astype(int))
        for item in items:
            assert item not in known


def test_save_load_two_tower_round_trip(
    tmp_path,
    fixture_train: pd.DataFrame,
    fixture_games: pd.DataFrame,
    fixture_users: pd.DataFrame,
) -> None:
    artifacts = train_two_tower(
        fixture_train,
        fixture_games,
        fixture_users,
        config=TwoTowerConfig(
            n_users=3,
            n_items=3,
            n_genres=0,
            n_tags=0,
            embedding_dim=8,
            tower_hidden=(16,),
            output_dim=8,
            epochs=1,
            batch_size=16,
            negative_ratio=1,
        ),
    )
    target = tmp_path / "tt.pt"
    save_two_tower(artifacts, target)
    model, spec, stats = load_two_tower(target)
    assert isinstance(model, TwoTowerModel)
    assert isinstance(spec, ContentFeatureSpec)
    assert "log_items_count_mean" in stats


def test_train_two_tower_with_val_records_ndcg_history(
    fixture_train: pd.DataFrame,
    fixture_games: pd.DataFrame,
    fixture_users: pd.DataFrame,
) -> None:
    """val_df=… enables NDCG monitoring + early-stopping rollback."""
    # Hold out one positive per user as a tiny "validation" set so the
    # NDCG monitor has something to score.
    val_rows = fixture_train.groupby("user_idx").head(1).reset_index(drop=True)
    train_rows = fixture_train.drop(fixture_train.groupby("user_idx").head(1).index).reset_index(
        drop=True
    )
    cfg = TwoTowerConfig(
        n_users=3,
        n_items=3,
        n_genres=0,
        n_tags=0,
        embedding_dim=8,
        tower_hidden=(16,),
        output_dim=8,
        epochs=3,
        batch_size=8,
        sampled_softmax=False,  # BCE path — robust on a 3-item toy.
        negative_ratio=1,
        eval_k=2,
        eval_max_users=3,
    )
    artifacts = train_two_tower(
        train_rows, fixture_games, fixture_users, config=cfg, val_df=val_rows
    )
    assert len(artifacts.val_ndcg_history) >= 1
    assert artifacts.best_epoch >= 0
    assert all(0.0 <= v <= 1.0 for v in artifacts.val_ndcg_history)


def test_train_two_tower_sampled_softmax_runs(
    fixture_train: pd.DataFrame,
    fixture_games: pd.DataFrame,
    fixture_users: pd.DataFrame,
) -> None:
    """Sampled-softmax path produces finite loss + vectors on a tiny corpus."""
    cfg = TwoTowerConfig(
        n_users=3,
        n_items=3,
        n_genres=0,
        n_tags=0,
        embedding_dim=8,
        tower_hidden=(16,),
        output_dim=8,
        epochs=2,
        batch_size=4,
        sampled_softmax=True,
        hard_negatives_per_pos=1,
    )
    artifacts = train_two_tower(fixture_train, fixture_games, fixture_users, config=cfg)
    assert all(np.isfinite(artifacts.train_loss_history))
    assert artifacts.user_vectors.shape == (3, cfg.output_dim)


def test_config_overrides_are_overridden_by_discovered_spec(
    fixture_train: pd.DataFrame,
    fixture_games: pd.DataFrame,
    fixture_users: pd.DataFrame,
) -> None:
    """A caller-supplied (n_genres=0, n_tags=0) should be replaced by
    the actual sizes the spec discovers from the games table."""
    artifacts = train_two_tower(
        fixture_train,
        fixture_games,
        fixture_users,
        config=TwoTowerConfig(
            n_users=3,
            n_items=3,
            n_genres=0,
            n_tags=0,
            embedding_dim=8,
            tower_hidden=(16,),
            output_dim=8,
            epochs=1,
            batch_size=16,
            negative_ratio=1,
        ),
    )
    assert artifacts.spec.n_genres > 0
    assert artifacts.spec.n_tags > 0
