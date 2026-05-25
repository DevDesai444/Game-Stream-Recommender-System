"""Neural Collaborative Filtering (NeuMF) in PyTorch.

The architecture follows He et al. 2017: two parallel branches fuse a
generalised matrix factorisation (GMF) component with a multi-layer
perceptron (MLP). Both share user/item embedding tables which are
trained jointly with binary cross-entropy on observed-vs-negative pairs.

The :class:`NCFTuner` cross-validates an MLP-width × embedding-dim ×
learning-rate × negative-ratio grid (3 × 2 × 2 × 2 = *24 configurations*
to match the ALS arm — 48 total across both models).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gamereco.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class NCFConfig:
    num_users: int
    num_items: int
    embedding_dim: int = 32
    mlp_layers: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 4096
    epochs: int = 8
    negative_ratio: int = 4
    weight_decay: float = 1e-6
    device: str = "cpu"
    seed: int = 42


class NCFModel(nn.Module):
    """NeuMF fuses GMF and MLP branches."""

    def __init__(self, cfg: NCFConfig) -> None:
        super().__init__()
        self.user_gmf = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_gmf = nn.Embedding(cfg.num_items, cfg.embedding_dim)
        self.user_mlp = nn.Embedding(cfg.num_users, cfg.embedding_dim)
        self.item_mlp = nn.Embedding(cfg.num_items, cfg.embedding_dim)

        mlp_in = cfg.embedding_dim * 2
        layers: list[nn.Module] = []
        prev = mlp_in
        for hidden in cfg.mlp_layers:
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            prev = hidden
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(cfg.embedding_dim + prev, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.output.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp_in = torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1)
        mlp = self.mlp(mlp_in)
        fused = torch.cat([gmf, mlp], dim=-1)
        return torch.sigmoid(self.output(fused)).squeeze(-1)

    @torch.no_grad()
    def item_embeddings(self) -> np.ndarray:
        """Return the concatenated [GMF | MLP] item embedding matrix."""
        return torch.cat([self.item_gmf.weight, self.item_mlp.weight], dim=-1).cpu().numpy()

    @torch.no_grad()
    def user_embeddings(self) -> np.ndarray:
        return torch.cat([self.user_gmf.weight, self.user_mlp.weight], dim=-1).cpu().numpy()


class InteractionDataset(Dataset):
    """Yields positive + sampled-negative (user, item, label) triples."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        num_items: int,
        negative_ratio: int = 4,
        user_pos: dict[int, set[int]] | None = None,
        seed: int = 42,
    ) -> None:
        self.users = interactions["user_idx"].to_numpy(dtype=np.int64)
        self.items = interactions["game_idx"].to_numpy(dtype=np.int64)
        self.num_items = num_items
        self.negative_ratio = negative_ratio
        self._rng = np.random.default_rng(seed)
        self.user_pos = user_pos or self._build_pos_map(self.users, self.items)

    @staticmethod
    def _build_pos_map(users: np.ndarray, items: np.ndarray) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for u, i in zip(users.tolist(), items.tolist(), strict=False):
            out.setdefault(int(u), set()).add(int(i))
        return out

    def __len__(self) -> int:
        return len(self.users) * (1 + self.negative_ratio)

    def __getitem__(self, idx: int) -> tuple[int, int, float]:
        bucket = idx % (1 + self.negative_ratio)
        anchor = idx // (1 + self.negative_ratio)
        user = int(self.users[anchor])
        if bucket == 0:
            return user, int(self.items[anchor]), 1.0
        pos = self.user_pos[user]
        while True:
            candidate = int(self._rng.integers(0, self.num_items))
            if candidate not in pos:
                return user, candidate, 0.0


def _train_one(
    cfg: NCFConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[NCFModel, dict[str, float]]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = NCFModel(cfg).to(device)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    dataset = InteractionDataset(train_df, cfg.num_items, cfg.negative_ratio, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    history: list[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        seen = 0
        for users, items, labels in loader:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            optim.zero_grad()
            preds = model(users, items)
            loss = loss_fn(preds, labels)
            loss.backward()
            optim.step()
            running += float(loss.detach().cpu()) * users.size(0)
            seen += users.size(0)
        epoch_loss = running / max(seen, 1)
        history.append(epoch_loss)
        log.info("ncf.epoch", epoch=epoch, loss=epoch_loss)

    ndcg = _evaluate_ndcg(model, train_df, val_df, cfg, k=10)
    return model, {"loss_final": history[-1] if history else 0.0, "ndcg_at_10": ndcg}


def _evaluate_ndcg(
    model: NCFModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: NCFConfig,
    k: int = 10,
) -> float:
    """Score every item for each val user, then compute NDCG@K vs the holdout."""
    device = torch.device(cfg.device)
    model.eval()
    train_pos: dict[int, set[int]] = {}
    for u, i in zip(
        train_df["user_idx"].to_numpy(),
        train_df["game_idx"].to_numpy(),
        strict=False,
    ):
        train_pos.setdefault(int(u), set()).add(int(i))

    truth: dict[int, set[int]] = {}
    for u, i in zip(
        val_df["user_idx"].to_numpy(),
        val_df["game_idx"].to_numpy(),
        strict=False,
    ):
        truth.setdefault(int(u), set()).add(int(i))

    if not truth:
        return 0.0

    all_items = torch.arange(cfg.num_items, device=device)
    scores: list[float] = []
    with torch.no_grad():
        for user_idx, gt in truth.items():
            users_t = torch.full((cfg.num_items,), user_idx, dtype=torch.long, device=device)
            preds = model(users_t, all_items).cpu().numpy()
            for already in train_pos.get(user_idx, set()):
                preds[already] = -1.0
            top_k = np.argpartition(-preds, k)[:k]
            top_k = top_k[np.argsort(-preds[top_k])]
            gains = [1.0 if int(idx) in gt else 0.0 for idx in top_k]
            ideal = [1.0] * min(len(gt), k)
            ideal_dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
            if ideal_dcg == 0:
                continue
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(gains))
            scores.append(dcg / ideal_dcg)
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class NCFGrid:
    embedding_dims: list[int] = field(default_factory=lambda: [16, 32, 64])
    mlp_layers: list[tuple[int, ...]] = field(default_factory=lambda: [(64, 32, 16), (128, 64, 32)])
    learning_rates: list[float] = field(default_factory=lambda: [1e-3, 5e-4])
    negative_ratios: list[int] = field(default_factory=lambda: [4, 8])

    @property
    def total_configs(self) -> int:
        return (
            len(self.embedding_dims)
            * len(self.mlp_layers)
            * len(self.learning_rates)
            * len(self.negative_ratios)
        )


def iter_configs(
    grid: NCFGrid, *, num_users: int, num_items: int, base: NCFConfig
) -> Iterable[NCFConfig]:
    for emb in grid.embedding_dims:
        for layers in grid.mlp_layers:
            for lr in grid.learning_rates:
                for neg in grid.negative_ratios:
                    yield NCFConfig(
                        num_users=num_users,
                        num_items=num_items,
                        embedding_dim=emb,
                        mlp_layers=layers,
                        learning_rate=lr,
                        negative_ratio=neg,
                        batch_size=base.batch_size,
                        epochs=base.epochs,
                        dropout=base.dropout,
                        device=base.device,
                        seed=base.seed,
                    )


def cross_validate_ncf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    grid: NCFGrid = NCFGrid(),
    base: NCFConfig | None = None,
) -> tuple[NCFModel, NCFConfig, dict[str, float], list[dict[str, float]]]:
    """Grid search NCF; return best (model, config, metrics, history)."""
    base = base or NCFConfig(num_users=num_users, num_items=num_items)
    history: list[dict[str, float]] = []
    best_metrics: dict[str, float] | None = None
    best_model: NCFModel | None = None
    best_cfg: NCFConfig | None = None
    for cfg in iter_configs(grid, num_users=num_users, num_items=num_items, base=base):
        log.info(
            "ncf.config",
            embedding_dim=cfg.embedding_dim,
            mlp_layers=str(cfg.mlp_layers),
            lr=cfg.learning_rate,
            neg=cfg.negative_ratio,
        )
        model, metrics = _train_one(cfg, train_df, val_df)
        history.append({"config": cfg.__dict__, **metrics})
        if best_metrics is None or metrics["ndcg_at_10"] > best_metrics["ndcg_at_10"]:
            best_metrics = metrics
            best_model = model
            best_cfg = cfg
    assert best_model is not None and best_cfg is not None and best_metrics is not None
    return best_model, best_cfg, best_metrics, history


def save_ncf(model: NCFModel, cfg: NCFConfig, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, target)


def load_ncf(path: Path) -> tuple[NCFModel, NCFConfig]:
    payload = torch.load(path, map_location="cpu")
    cfg = NCFConfig(**payload["config"])
    model = NCFModel(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg
