"""Two-tower neural collaborative filter with real content features.

Architecture:

  user tower:    [user_id_embedding ⊕ continuous_user_features]
                 → Linear → ReLU → Dropout → Linear → user_vector ∈ R^d

  item tower:    [item_id_embedding ⊕ multi_hot_genres
                  ⊕ multi_hot_tags ⊕ continuous_item_features]
                 → Linear → ReLU → Dropout → Linear → item_vector ∈ R^d

  score(u, i) = sigmoid( user_vector(u) · item_vector(i) )

Trained with binary cross-entropy on observed positives + sampled
negatives. Unlike the bare-embeddings NeuMF in
:mod:`gamereco.training.ncf`, this model lets brand-new items (or new
users) be scored from their features alone — the content towers
produce a vector even when the id-embedding is freshly initialised,
which is exactly the cold-item story we couldn't tell on steam-200k.

The two tower vectors are also useful artifacts independent of the
score head: they go into the pgvector item-similarity index and
become a candidate generator for the XGBoost ranker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------- Content feature builders ----------


@dataclass
class ContentFeatureSpec:
    """Realised vocabularies + scalers for an item-feature column."""

    genre_vocab: dict[str, int]
    tag_vocab: dict[str, int]
    price_mean: float
    price_std: float
    year_mean: float
    year_std: float

    @property
    def n_genres(self) -> int:
        return len(self.genre_vocab)

    @property
    def n_tags(self) -> int:
        return len(self.tag_vocab)


def fit_content_feature_spec(
    games: pd.DataFrame,
    *,
    top_n_tags: int = 200,
) -> ContentFeatureSpec:
    """Build vocabularies and scalers from the game metadata table."""
    genre_counter: dict[str, int] = {}
    tag_counter: dict[str, int] = {}
    for row_genres in games["genres"]:
        for g in row_genres or ():
            genre_counter[str(g)] = genre_counter.get(str(g), 0) + 1
    for row_tags in games["tags"]:
        for t in row_tags or ():
            tag_counter[str(t)] = tag_counter.get(str(t), 0) + 1

    # Genres are already a short curated list — keep all.
    genre_vocab = {g: i for i, (g, _) in enumerate(sorted(genre_counter.items()))}
    # Tags are long-tail — cap to the top N by frequency.
    top_tags = sorted(tag_counter.items(), key=lambda kv: -kv[1])[:top_n_tags]
    tag_vocab = {t: i for i, t in enumerate(sorted(t for t, _ in top_tags))}

    prices = pd.to_numeric(games["price"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    years = (
        pd.to_numeric(games["release_year"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    )
    return ContentFeatureSpec(
        genre_vocab=genre_vocab,
        tag_vocab=tag_vocab,
        price_mean=float(prices.mean()) if prices.size else 0.0,
        price_std=float(prices.std()) if prices.size and prices.std() > 0 else 1.0,
        year_mean=float(years.mean()) if years.size else 2010.0,
        year_std=float(years.std()) if years.size and years.std() > 0 else 5.0,
    )


def build_item_feature_matrix(
    games: pd.DataFrame, spec: ContentFeatureSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render an (n_items, ...) feature triple: (genres_multi_hot, tags_multi_hot, dense).

    dense columns: [price_z, release_year_z, has_price (0/1), has_year (0/1), early_access]
    """
    n = len(games)
    genres = np.zeros((n, spec.n_genres), dtype=np.float32)
    tags = np.zeros((n, spec.n_tags), dtype=np.float32)
    dense = np.zeros((n, 5), dtype=np.float32)

    def _real(value) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    sorted_games = games.sort_values("game_idx").reset_index(drop=True)
    for row in sorted_games.itertuples(index=False):
        idx = int(row.game_idx)
        for g in row.genres or ():
            j = spec.genre_vocab.get(str(g))
            if j is not None:
                genres[idx, j] = 1.0
        for t in row.tags or ():
            j = spec.tag_vocab.get(str(t))
            if j is not None:
                tags[idx, j] = 1.0
        price = _real(row.price)
        year = _real(row.release_year)
        dense[idx, 0] = (price - spec.price_mean) / spec.price_std if price is not None else 0.0
        dense[idx, 1] = (year - spec.year_mean) / spec.year_std if year is not None else 0.0
        dense[idx, 2] = 1.0 if price is not None else 0.0
        dense[idx, 3] = 1.0 if year is not None else 0.0
        dense[idx, 4] = 1.0 if getattr(row, "early_access", False) else 0.0
    return genres, tags, dense


def build_user_feature_matrix(
    users: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render an (n_users, 4) dense user-feature matrix + scaler stats.

    Columns: [log_items_count, log_total_playtime, log_active_recent, log_reviews_count]
    Each is log1p'd then z-scored across the population.
    """
    n = len(users)
    sorted_users = users.sort_values("user_idx").reset_index(drop=True)
    arr = np.zeros((n, 4), dtype=np.float32)
    items = np.log1p(sorted_users["items_count"].astype(float).to_numpy())
    play = np.log1p(sorted_users["total_playtime"].astype(float).to_numpy())
    active = np.log1p(sorted_users["active_recent"].astype(float).to_numpy())
    revs = np.log1p(
        sorted_users.get("reviews_count", pd.Series(np.zeros(n))).astype(float).to_numpy()
    )

    stats: dict[str, float] = {}
    for col_idx, (col, raw) in enumerate(
        [
            ("log_items_count", items),
            ("log_total_playtime", play),
            ("log_active_recent", active),
            ("log_reviews_count", revs),
        ]
    ):
        mean = float(raw.mean()) if raw.size else 0.0
        std = float(raw.std()) if raw.size and raw.std() > 0 else 1.0
        stats[f"{col}_mean"] = mean
        stats[f"{col}_std"] = std
        arr[:, col_idx] = (raw - mean) / std
    return arr, stats


# ---------- Model ----------


@dataclass
class TwoTowerConfig:
    n_users: int
    n_items: int
    n_genres: int
    n_tags: int
    n_user_dense: int = 4
    n_item_dense: int = 5
    embedding_dim: int = 32
    tower_hidden: tuple[int, ...] = (128, 64)
    output_dim: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 8192
    epochs: int = 4
    negative_ratio: int = 4
    seed: int = 42


class _Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTowerModel(nn.Module):
    """Two-tower NCF: dot-product score between user and item vectors."""

    def __init__(self, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(cfg.n_users, cfg.embedding_dim)
        self.item_emb = nn.Embedding(cfg.n_items, cfg.embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        user_in = cfg.embedding_dim + cfg.n_user_dense
        item_in = cfg.embedding_dim + cfg.n_genres + cfg.n_tags + cfg.n_item_dense
        self.user_tower = _Tower(user_in, cfg.tower_hidden, cfg.output_dim, cfg.dropout)
        self.item_tower = _Tower(item_in, cfg.tower_hidden, cfg.output_dim, cfg.dropout)
        # Learnable temperature for the cosine score. Initialised so
        # the early-training logits sit at ~log(20) = ~3 in absolute
        # value, which is a comfortable BCE regime.
        self.scale = nn.Parameter(torch.tensor(10.0))

    def encode_user(self, user_idx: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        emb = self.user_emb(user_idx)
        out = self.user_tower(torch.cat([emb, user_dense], dim=-1))
        # L2-normalise so the dot product is bounded (turns into a
        # cosine similarity scaled by self.scale). This is the standard
        # two-tower-retrieval setup that keeps BCE losses from blowing
        # up when feature inputs are high-dimensional one-hots.
        return nn.functional.normalize(out, p=2, dim=-1)

    def encode_item(
        self,
        item_idx: torch.Tensor,
        item_genres: torch.Tensor,
        item_tags: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.item_emb(item_idx)
        out = self.item_tower(torch.cat([emb, item_genres, item_tags, item_dense], dim=-1))
        return nn.functional.normalize(out, p=2, dim=-1)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        user_dense: torch.Tensor,
        item_genres: torch.Tensor,
        item_tags: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw logits. Caller applies sigmoid (or uses
        ``BCEWithLogitsLoss`` for training) — pairing logits with
        the logits-aware loss is numerically stable, whereas
        sigmoid + BCELoss can round outside ``[0, 1]``."""
        u = self.encode_user(user_idx, user_dense)
        v = self.encode_item(item_idx, item_genres, item_tags, item_dense)
        cosine = (u * v).sum(dim=-1)
        return self.scale * cosine

    def score(self, *args, **kwargs) -> torch.Tensor:
        """Convenience wrapper applying sigmoid for inference."""
        return torch.sigmoid(self.forward(*args, **kwargs))

    @torch.no_grad()
    def all_user_vectors(self, user_dense: torch.Tensor) -> np.ndarray:
        idxs = torch.arange(self.cfg.n_users, dtype=torch.long, device=user_dense.device)
        return self.encode_user(idxs, user_dense).cpu().numpy()

    @torch.no_grad()
    def all_item_vectors(
        self,
        item_genres: torch.Tensor,
        item_tags: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> np.ndarray:
        idxs = torch.arange(self.cfg.n_items, dtype=torch.long, device=item_dense.device)
        return self.encode_item(idxs, item_genres, item_tags, item_dense).cpu().numpy()


# ---------- Training ----------


@dataclass
class TwoTowerArtifacts:
    model: TwoTowerModel
    user_vectors: np.ndarray
    item_vectors: np.ndarray
    spec: ContentFeatureSpec
    user_stats: dict[str, float]
    train_loss_history: list[float] = field(default_factory=list)


def train_two_tower(
    train_df: pd.DataFrame,
    games: pd.DataFrame,
    users: pd.DataFrame,
    *,
    config: TwoTowerConfig | None = None,
) -> TwoTowerArtifacts:
    """Train the two-tower model end-to-end and return artifacts.

    The training loop is the same vectorised positive + sampled-negative
    BCE scheme used by :func:`gamereco.training.hybrid._train_ncf_quick`,
    so it stays under a minute on a laptop on the UCSD slice the
    benchmark runs against.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    spec = fit_content_feature_spec(games)
    item_genres_np, item_tags_np, item_dense_np = build_item_feature_matrix(games, spec)
    user_dense_np, user_stats = build_user_feature_matrix(users)
    if config is None:
        cfg = TwoTowerConfig(
            n_users=int(users["user_idx"].max() + 1),
            n_items=int(games["game_idx"].max() + 1),
            n_genres=spec.n_genres,
            n_tags=spec.n_tags,
            n_user_dense=user_dense_np.shape[1],
            n_item_dense=item_dense_np.shape[1],
        )
    else:
        # Always derive vocab sizes / dense widths from the discovered
        # spec so a caller-supplied config can't desync with the data.
        cfg = TwoTowerConfig(
            **{
                **config.__dict__,
                "n_genres": spec.n_genres,
                "n_tags": spec.n_tags,
                "n_user_dense": user_dense_np.shape[1],
                "n_item_dense": item_dense_np.shape[1],
            }
        )

    torch.manual_seed(cfg.seed)
    model = TwoTowerModel(cfg)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    user_dense_t = torch.from_numpy(user_dense_np)
    item_genres_t = torch.from_numpy(item_genres_np)
    item_tags_t = torch.from_numpy(item_tags_np)
    item_dense_t = torch.from_numpy(item_dense_np)

    pos_users = train_df["user_idx"].to_numpy(dtype=np.int64)
    pos_items = train_df["game_idx"].to_numpy(dtype=np.int64)
    user_pos: dict[int, set[int]] = {}
    for u, i in zip(pos_users.tolist(), pos_items.tolist(), strict=False):
        user_pos.setdefault(u, set()).add(i)

    rng = np.random.default_rng(cfg.seed)
    n_pos = pos_users.size
    n_neg = n_pos * cfg.negative_ratio
    n_items = cfg.n_items

    history: list[float] = []
    for _epoch in range(cfg.epochs):
        neg_users = np.repeat(pos_users, cfg.negative_ratio)
        sampled = rng.integers(0, n_items, size=int(n_neg * 1.2))
        accepted_neg = np.empty(n_neg, dtype=np.int64)
        write = 0
        read = 0
        while write < n_neg and read < sampled.size:
            u = int(neg_users[write])
            cand = int(sampled[read])
            if cand not in user_pos.get(u, ()):
                accepted_neg[write] = cand
                write += 1
            read += 1
        # Top up if oversampling underdelivered. Bail out after a
        # reasonable number of attempts — in pathological tiny-corpus
        # tests where users already own every item, no valid negative
        # exists and we'd otherwise spin forever.
        retries = 0
        max_retries = max(10 * n_neg, 1000)
        while write < n_neg and retries < max_retries:
            u = int(neg_users[write])
            cand = int(rng.integers(0, n_items))
            if cand not in user_pos.get(u, ()):
                accepted_neg[write] = cand
                write += 1
            retries += 1
        if write < n_neg:
            # Truncate to what we did sample.
            accepted_neg = accepted_neg[:write]
            neg_users = neg_users[:write]
            n_neg = write

        users_all = np.concatenate([pos_users, neg_users])
        items_all = np.concatenate([pos_items, accepted_neg])
        labels_all = np.concatenate(
            [np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)]
        )
        perm = rng.permutation(users_all.size)
        users_all, items_all, labels_all = users_all[perm], items_all[perm], labels_all[perm]

        model.train()
        running = 0.0
        seen_rows = 0
        for start in range(0, users_all.size, cfg.batch_size):
            end = start + cfg.batch_size
            u_idx = torch.from_numpy(users_all[start:end]).long()
            i_idx = torch.from_numpy(items_all[start:end]).long()
            labels = torch.from_numpy(labels_all[start:end]).float()
            u_dense_batch = user_dense_t[u_idx]
            i_genres_batch = item_genres_t[i_idx]
            i_tags_batch = item_tags_t[i_idx]
            i_dense_batch = item_dense_t[i_idx]
            optim.zero_grad()
            preds = model(u_idx, i_idx, u_dense_batch, i_genres_batch, i_tags_batch, i_dense_batch)
            loss = loss_fn(preds, labels)
            loss.backward()
            optim.step()
            running += float(loss.detach()) * u_idx.size(0)
            seen_rows += u_idx.size(0)
        history.append(running / max(seen_rows, 1))

    model.eval()
    user_vectors = model.all_user_vectors(user_dense_t)
    item_vectors = model.all_item_vectors(item_genres_t, item_tags_t, item_dense_t)
    return TwoTowerArtifacts(
        model=model,
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        spec=spec,
        user_stats=user_stats,
        train_loss_history=history,
    )


def two_tower_topk(
    artifacts: TwoTowerArtifacts,
    train_df: pd.DataFrame,
    user_indices: np.ndarray,
    *,
    k: int = 10,
) -> dict[int, list[int]]:
    """Top-K predictions from the two-tower dot-product scores."""
    known: dict[int, set[int]] = {}
    for u, group in train_df.groupby("user_idx"):
        known[int(u)] = set(int(g) for g in group["game_idx"])

    item_vectors = artifacts.item_vectors
    out: dict[int, list[int]] = {}
    for user in user_indices:
        scores = item_vectors @ artifacts.user_vectors[int(user)]
        for item in known.get(int(user), ()):
            if 0 <= item < scores.size:
                scores[int(item)] = -np.inf
        n = min(k, int(np.isfinite(scores).sum()))
        if n <= 0:
            out[int(user)] = []
            continue
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        out[int(user)] = [int(i) for i in top]
    return out


def save_two_tower(artifacts: TwoTowerArtifacts, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": artifacts.model.state_dict(),
            "config": artifacts.model.cfg.__dict__,
            "user_stats": artifacts.user_stats,
            "spec": {
                "genre_vocab": artifacts.spec.genre_vocab,
                "tag_vocab": artifacts.spec.tag_vocab,
                "price_mean": artifacts.spec.price_mean,
                "price_std": artifacts.spec.price_std,
                "year_mean": artifacts.spec.year_mean,
                "year_std": artifacts.spec.year_std,
            },
        },
        target,
    )


def load_two_tower(path: Path) -> tuple[TwoTowerModel, ContentFeatureSpec, dict[str, float]]:
    payload = torch.load(path, map_location="cpu")
    cfg = TwoTowerConfig(**payload["config"])
    model = TwoTowerModel(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    spec = ContentFeatureSpec(**payload["spec"])
    return model, spec, payload["user_stats"]
