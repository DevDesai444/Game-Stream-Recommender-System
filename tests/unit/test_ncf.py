"""Tests for the PyTorch NeuMF model and dataset."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from gamereco.training.ncf import (
    InteractionDataset,
    NCFConfig,
    NCFGrid,
    NCFModel,
    iter_configs,
    load_ncf,
    save_ncf,
)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "game_idx": [0, 1, 1, 2, 0, 2],
        }
    )


def test_ncf_model_forward_shape() -> None:
    cfg = NCFConfig(num_users=4, num_items=6, embedding_dim=8, mlp_layers=(16, 8))
    model = NCFModel(cfg)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    out = model(users, items)
    assert out.shape == (3,)
    assert torch.all((out >= 0) & (out <= 1))


def test_ncf_item_embeddings_shape() -> None:
    cfg = NCFConfig(num_users=2, num_items=3, embedding_dim=4)
    model = NCFModel(cfg)
    embeddings = model.item_embeddings()
    assert embeddings.shape == (3, 8)


def test_ncf_user_embeddings_shape() -> None:
    cfg = NCFConfig(num_users=5, num_items=3, embedding_dim=4)
    model = NCFModel(cfg)
    assert model.user_embeddings().shape == (5, 8)


def test_dataset_length_includes_negatives(tiny_df: pd.DataFrame) -> None:
    ds = InteractionDataset(tiny_df, num_items=3, negative_ratio=2)
    assert len(ds) == len(tiny_df) * 3


def test_dataset_positive_first(tiny_df: pd.DataFrame) -> None:
    ds = InteractionDataset(tiny_df, num_items=3, negative_ratio=0)
    user, item, label = ds[0]
    assert label == 1.0
    assert user == tiny_df.iloc[0]["user_idx"]


def test_dataset_negative_excludes_positive(tiny_df: pd.DataFrame) -> None:
    ds = InteractionDataset(tiny_df, num_items=3, negative_ratio=1, seed=1)
    pos = ds.user_pos
    for idx in range(len(ds)):
        user, item, label = ds[idx]
        if label == 0.0:
            assert item not in pos[user]


def test_grid_total_configs_is_product() -> None:
    grid = NCFGrid(
        embedding_dims=[16, 32],
        mlp_layers=[(64,), (128,)],
        learning_rates=[1e-3],
        negative_ratios=[4],
    )
    assert grid.total_configs == 4


def test_iter_configs_yields_total_configs() -> None:
    base = NCFConfig(num_users=2, num_items=3)
    grid = NCFGrid(
        embedding_dims=[8],
        mlp_layers=[(16,)],
        learning_rates=[1e-3, 5e-4],
        negative_ratios=[2, 4],
    )
    configs = list(iter_configs(grid, num_users=2, num_items=3, base=base))
    assert len(configs) == 4
    assert configs[0].num_users == 2


def test_save_and_load_ncf(tmp_path) -> None:
    cfg = NCFConfig(num_users=4, num_items=4, embedding_dim=4, mlp_layers=(8,))
    model = NCFModel(cfg)
    target = tmp_path / "ncf.pt"
    save_ncf(model, cfg, target)
    loaded, loaded_cfg = load_ncf(target)
    assert loaded_cfg.num_items == cfg.num_items
    # Sanity-check the loaded model can still forward.
    out = loaded(torch.tensor([0]), torch.tensor([0]))
    assert out.shape == (1,)
