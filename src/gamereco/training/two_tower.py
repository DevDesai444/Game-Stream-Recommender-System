"""Two-tower neural collaborative filter with real content features.

Architecture:

  user tower:    [user_id_embedding ⊕ continuous_user_features]
                 → Linear → LayerNorm → ReLU → Dropout → ... → user_vector ∈ R^d

  item tower:    [item_id_embedding ⊕ multi_hot_genres
                  ⊕ multi_hot_tags ⊕ continuous_item_features]
                 → Linear → LayerNorm → ReLU → Dropout → ... → item_vector ∈ R^d

  score(u, i) = scale * cos(user_vector(u), item_vector(i))

The training loss is **sampled softmax with in-batch negatives** plus an
optional pool of *hard* popularity-sampled negatives, with a logQ
correction so popular items aren't unfairly penalised (the YouTube
recsys paper, Bengio et al.). This replaces the older
positives-vs-random-negatives BCE — sampled softmax directly
approximates listwise ranking, which is what NDCG measures.

Validation NDCG@K is monitored every epoch with early stopping on a
plateau, so we don't ship a model that has overfit train loss.

The two tower vectors are also useful artifacts independent of the
score head: they feed the pgvector item-similarity index and become a
candidate generator for the XGBoost ranker in
:mod:`gamereco.training.hybrid`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    embedding_dim: int = 64
    tower_hidden: tuple[int, ...] = (256, 128)
    output_dim: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    # 1024 keeps the in-batch positive mask under ~5MB per step (the
    # mask is B × (B + hard_negatives_per_pos × B) bool). With 8192,
    # the same mask was ~335MB per step and dominated wall clock /
    # memory pressure on real-sized datasets.
    batch_size: int = 1024
    epochs: int = 15
    # Loss / sampling. With sampled_softmax=True every positive in the
    # batch uses the other positives' items as in-batch negatives, plus
    # ``hard_negatives_per_pos`` extra negatives drawn proportional to
    # item popularity (hard negatives). This is the standard
    # YouTube/two-tower retrieval recipe.
    sampled_softmax: bool = True
    hard_negatives_per_pos: int = 4
    # Random (uniform) negatives are only used when sampled_softmax=False
    # — the legacy BCE path is retained for the steam-200k benchmark
    # which has tiny n_items and where in-batch negatives degenerate.
    negative_ratio: int = 4
    # logQ popularity correction strength (1.0 = full correction).
    logq_correction: float = 1.0
    # Learnable score scale init. ~10 gives early logits in [-10, 10]
    # which is a comfortable softmax / BCE regime.
    init_scale: float = 10.0
    # Training-data hygiene knobs.
    min_playtime_minutes: float = 0.0
    use_confidence_weights: bool = True
    # Cosine LR schedule + early stopping on validation NDCG@K.
    use_lr_schedule: bool = True
    early_stopping_patience: int = 3
    eval_k: int = 10
    eval_max_users: int = 512
    seed: int = 42


class _Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
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
    """Two-tower NCF: scaled-cosine score between user and item vectors."""

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
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))

    def encode_user(self, user_idx: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        emb = self.user_emb(user_idx)
        out = self.user_tower(torch.cat([emb, user_dense], dim=-1))
        return F.normalize(out, p=2, dim=-1)

    def encode_item(
        self,
        item_idx: torch.Tensor,
        item_genres: torch.Tensor,
        item_tags: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.item_emb(item_idx)
        out = self.item_tower(torch.cat([emb, item_genres, item_tags, item_dense], dim=-1))
        return F.normalize(out, p=2, dim=-1)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        user_dense: torch.Tensor,
        item_genres: torch.Tensor,
        item_tags: torch.Tensor,
        item_dense: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw logits. Caller applies sigmoid or uses
        ``BCEWithLogitsLoss`` / sampled-softmax cross-entropy for
        training — pairing logits with the logits-aware loss is
        numerically stable."""
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
    val_ndcg_history: list[float] = field(default_factory=list)
    best_epoch: int = 0


def _ndcg_at_k(scores: np.ndarray, truth: set[int], k: int) -> float:
    """NDCG@K from a score vector for one user, ignoring untruthed items."""
    if not truth:
        return 0.0
    n = min(k, scores.size)
    if n <= 0:
        return 0.0
    finite = np.isfinite(scores)
    if not finite.any():
        return 0.0
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    dcg = 0.0
    for rank, item in enumerate(top.tolist()):
        if item in truth:
            dcg += 1.0 / math.log2(rank + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / ideal if ideal > 0 else 0.0


def _val_ndcg(
    model: TwoTowerModel,
    user_dense_t: torch.Tensor,
    item_genres_t: torch.Tensor,
    item_tags_t: torch.Tensor,
    item_dense_t: torch.Tensor,
    truth: dict[int, set[int]],
    seen: dict[int, set[int]],
    user_ids: np.ndarray,
    k: int,
) -> float:
    model.eval()
    with torch.no_grad():
        item_vecs = model.all_item_vectors(item_genres_t, item_tags_t, item_dense_t)
        user_vecs = model.all_user_vectors(user_dense_t)
    ndcgs: list[float] = []
    for u in user_ids:
        u = int(u)
        if u not in truth:
            continue
        scores = item_vecs @ user_vecs[u]
        for blocked in seen.get(u, ()):
            if 0 <= blocked < scores.size:
                scores[int(blocked)] = -np.inf
        ndcgs.append(_ndcg_at_k(scores, truth[u], k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def train_two_tower(
    train_df: pd.DataFrame,
    games: pd.DataFrame,
    users: pd.DataFrame,
    *,
    config: TwoTowerConfig | None = None,
    val_df: pd.DataFrame | None = None,
) -> TwoTowerArtifacts:
    """Train the two-tower model end-to-end and return artifacts.

    Defaults to sampled-softmax with in-batch + hard popularity-sampled
    negatives (see ``TwoTowerConfig``). When ``val_df`` is supplied,
    validation NDCG@K is monitored each epoch and early stopping kicks
    in after ``early_stopping_patience`` epochs without improvement —
    the model state is rolled back to the best-NDCG epoch.
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

    # Optional playtime filter on training positives only. The benchmark
    # docs flag UCSD's "owned but never played" rows as the dominant
    # source of NCF noise; dropping them at the threshold of meaningful
    # engagement gives the model cleaner positives without affecting
    # the eval splits.
    if cfg.min_playtime_minutes > 0 and "playtime_minutes" in train_df.columns:
        train_df = train_df[train_df["playtime_minutes"] >= cfg.min_playtime_minutes].reset_index(
            drop=True
        )
    elif cfg.min_playtime_minutes > 0 and "playtime_forever" in train_df.columns:
        train_df = train_df[train_df["playtime_forever"] >= cfg.min_playtime_minutes].reset_index(
            drop=True
        )

    torch.manual_seed(cfg.seed)
    model = TwoTowerModel(cfg)
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 1))
        if cfg.use_lr_schedule
        else None
    )

    user_dense_t = torch.from_numpy(user_dense_np)
    item_genres_t = torch.from_numpy(item_genres_np)
    item_tags_t = torch.from_numpy(item_tags_np)
    item_dense_t = torch.from_numpy(item_dense_np)

    pos_users = train_df["user_idx"].to_numpy(dtype=np.int64)
    pos_items = train_df["game_idx"].to_numpy(dtype=np.int64)

    # Per-positive confidence weights from playtime. Normalised to mean 1
    # so the overall loss magnitude is unchanged and the learning rate
    # stays comparable across configs with/without weighting.
    if cfg.use_confidence_weights:
        pt_col = (
            "playtime_minutes"
            if "playtime_minutes" in train_df.columns
            else ("playtime_forever" if "playtime_forever" in train_df.columns else None)
        )
        if pt_col is not None:
            raw_w = np.log1p(train_df[pt_col].to_numpy(dtype=np.float64))
            if raw_w.mean() > 0:
                pos_weights = (raw_w / raw_w.mean()).astype(np.float32)
            else:
                pos_weights = np.ones(pos_users.size, dtype=np.float32)
        else:
            pos_weights = np.ones(pos_users.size, dtype=np.float32)
    else:
        pos_weights = np.ones(pos_users.size, dtype=np.float32)

    user_pos: dict[int, set[int]] = {}
    for u, i in zip(pos_users.tolist(), pos_items.tolist(), strict=False):
        user_pos.setdefault(u, set()).add(i)

    # CSR view of (user, item) positives for vectorised in-batch
    # collision masking inside the sampled-softmax loop. Building this
    # once outside the loop is the difference between O(B^2) Python
    # membership checks per batch and a single sparse slice.
    from scipy.sparse import csr_matrix as _csr

    pos_csr = _csr(
        (
            np.ones(pos_items.size, dtype=np.bool_),
            (pos_users, pos_items),
        ),
        shape=(cfg.n_users, cfg.n_items),
    )

    # Item popularity used for both the logQ correction and for sampling
    # hard negatives. We add a smoothing prior so brand-new items still
    # get sampled with non-zero probability.
    item_counts = np.zeros(cfg.n_items, dtype=np.float64)
    counts = np.bincount(pos_items, minlength=cfg.n_items)
    item_counts[: counts.size] = counts
    item_counts += 1.0  # Laplace prior
    item_prob = item_counts / item_counts.sum()
    log_item_prob = np.log(item_prob)

    # Pre-cache validation truth + seen for NDCG monitoring.
    val_truth: dict[int, set[int]] = {}
    val_user_sample: np.ndarray = np.empty(0, dtype=np.int64)
    if val_df is not None and len(val_df) > 0:
        val_truth = {
            int(u): set(int(g) for g in group["game_idx"])
            for u, group in val_df.groupby("user_idx")
        }
        rng_val = np.random.default_rng(cfg.seed)
        all_val_users = np.array(sorted(val_truth.keys()), dtype=np.int64)
        if all_val_users.size > cfg.eval_max_users:
            val_user_sample = rng_val.choice(all_val_users, cfg.eval_max_users, replace=False)
        else:
            val_user_sample = all_val_users

    rng = np.random.default_rng(cfg.seed)
    n_pos = pos_users.size

    loss_history: list[float] = []
    ndcg_history: list[float] = []
    best_ndcg = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_since_improvement = 0

    log_item_prob_t = torch.from_numpy(log_item_prob.astype(np.float32))

    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(n_pos)
        users_ep = pos_users[perm]
        items_ep = pos_items[perm]
        weights_ep = pos_weights[perm]

        if cfg.sampled_softmax:
            running_loss = 0.0
            seen_rows = 0
            for start in range(0, n_pos, cfg.batch_size):
                end = start + cfg.batch_size
                u_batch = users_ep[start:end]
                i_batch = items_ep[start:end]
                w_batch = weights_ep[start:end]
                B = u_batch.size

                # Hard popularity-sampled negatives for the whole batch.
                # We share negatives across all positives in the batch
                # (the standard sampled-softmax trick) so the cost is
                # O(B + H) item-tower evals per step, not O(B * (1 + H)).
                H = cfg.hard_negatives_per_pos * B if cfg.hard_negatives_per_pos > 0 else 0
                if H > 0:
                    hard_items = rng.choice(cfg.n_items, size=H, replace=True, p=item_prob)
                    all_items = np.concatenate([i_batch, hard_items.astype(np.int64)])
                else:
                    all_items = i_batch

                u_idx_t = torch.from_numpy(u_batch).long()
                i_idx_t = torch.from_numpy(all_items).long()
                u_dense_batch = user_dense_t[u_idx_t]
                i_g_batch = item_genres_t[i_idx_t]
                i_t_batch = item_tags_t[i_idx_t]
                i_d_batch = item_dense_t[i_idx_t]
                w_t = torch.from_numpy(w_batch).float()

                optim.zero_grad()
                u_vecs = model.encode_user(u_idx_t, u_dense_batch)  # [B, d]
                i_vecs = model.encode_item(i_idx_t, i_g_batch, i_t_batch, i_d_batch)  # [B+H, d]
                logits = model.scale * (u_vecs @ i_vecs.T)  # [B, B+H]
                # logQ correction: subtract log(p_item) for each column
                # so popular items don't dominate the softmax. This is
                # the YouTube/Bengio sampling correction.
                if cfg.logq_correction > 0:
                    logits = logits - cfg.logq_correction * log_item_prob_t[i_idx_t].unsqueeze(0)
                # Mask out columns where the candidate item is actually
                # a positive for the user in the row (in-batch / hard-
                # negative collision). Without this, an unlucky batch
                # teaches the model to *down*-weight items the user
                # really does like. Vectorised via the sparse CSR.
                mask_np = pos_csr[u_batch][:, all_items].toarray()
                # Don't mask the row's own true positive (column r).
                mask_np[np.arange(B), np.arange(B)] = False
                mask = torch.from_numpy(mask_np)
                logits = logits.masked_fill(mask, float("-inf"))

                labels = torch.arange(B, dtype=torch.long)
                per_row_loss = F.cross_entropy(logits, labels, reduction="none")
                loss = (per_row_loss * w_t).mean()
                loss.backward()
                optim.step()
                running_loss += float(loss.detach()) * B
                seen_rows += B
            loss_history.append(running_loss / max(seen_rows, 1))
        else:
            # Legacy BCE path with mixed random + popularity-sampled
            # negatives. Retained for the steam-200k benchmark where
            # the catalogue is tiny (~4K items) and in-batch negatives
            # degenerate into the same handful of repeats.
            n_neg = n_pos * cfg.negative_ratio
            neg_users = np.repeat(users_ep, cfg.negative_ratio)
            n_hard = int(n_neg * 0.5)
            hard_neg = rng.choice(cfg.n_items, size=n_hard, replace=True, p=item_prob)
            rand_neg = rng.integers(0, cfg.n_items, size=int((n_neg - n_hard) * 1.3))
            sampled = np.concatenate([hard_neg, rand_neg])
            rng.shuffle(sampled)
            accepted = np.empty(n_neg, dtype=np.int64)
            write = 0
            read = 0
            while write < n_neg and read < sampled.size:
                u = int(neg_users[write])
                cand = int(sampled[read])
                if cand not in user_pos.get(u, ()):
                    accepted[write] = cand
                    write += 1
                read += 1
            retries = 0
            max_retries = max(10 * n_neg, 1000)
            while write < n_neg and retries < max_retries:
                u = int(neg_users[write])
                cand = int(rng.integers(0, cfg.n_items))
                if cand not in user_pos.get(u, ()):
                    accepted[write] = cand
                    write += 1
                retries += 1
            if write < n_neg:
                accepted = accepted[:write]
                neg_users = neg_users[:write]
                n_neg = write

            users_all = np.concatenate([users_ep, neg_users])
            items_all = np.concatenate([items_ep, accepted])
            labels_all = np.concatenate(
                [np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)]
            )
            weights_all = np.concatenate([weights_ep, np.ones(n_neg, dtype=np.float32)])
            perm2 = rng.permutation(users_all.size)
            users_all, items_all, labels_all, weights_all = (
                users_all[perm2],
                items_all[perm2],
                labels_all[perm2],
                weights_all[perm2],
            )

            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            running_loss = 0.0
            seen_rows = 0
            for start in range(0, users_all.size, cfg.batch_size):
                end = start + cfg.batch_size
                u_idx = torch.from_numpy(users_all[start:end]).long()
                i_idx = torch.from_numpy(items_all[start:end]).long()
                labels = torch.from_numpy(labels_all[start:end]).float()
                w_t = torch.from_numpy(weights_all[start:end]).float()
                u_dense_batch = user_dense_t[u_idx]
                i_g_batch = item_genres_t[i_idx]
                i_t_batch = item_tags_t[i_idx]
                i_d_batch = item_dense_t[i_idx]
                optim.zero_grad()
                preds = model(u_idx, i_idx, u_dense_batch, i_g_batch, i_t_batch, i_d_batch)
                loss = (loss_fn(preds, labels) * w_t).mean()
                loss.backward()
                optim.step()
                running_loss += float(loss.detach()) * u_idx.size(0)
                seen_rows += u_idx.size(0)
            loss_history.append(running_loss / max(seen_rows, 1))

        if scheduler is not None:
            scheduler.step()

        # Validation NDCG monitoring + early stopping.
        if val_user_sample.size > 0:
            ndcg = _val_ndcg(
                model,
                user_dense_t,
                item_genres_t,
                item_tags_t,
                item_dense_t,
                truth=val_truth,
                seen=user_pos,
                user_ids=val_user_sample,
                k=cfg.eval_k,
            )
            ndcg_history.append(ndcg)
            if ndcg > best_ndcg + 1e-6:
                best_ndcg = ndcg
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= cfg.early_stopping_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    user_vectors = model.all_user_vectors(user_dense_t)
    item_vectors = model.all_item_vectors(item_genres_t, item_tags_t, item_dense_t)
    return TwoTowerArtifacts(
        model=model,
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        spec=spec,
        user_stats=user_stats,
        train_loss_history=loss_history,
        val_ndcg_history=ndcg_history,
        best_epoch=best_epoch,
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
