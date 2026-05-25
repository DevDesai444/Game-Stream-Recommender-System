"""End-to-end hybrid recommender harness used by the benchmark runner.

This module wires the laptop-runnable building blocks (in-memory ALS,
PyTorch NeuMF, KMeans over ALS factors, XGBoost ranker) together so the
benchmark report has a single ``run_hybrid_benchmark`` entrypoint that
reproduces the headline numbers end-to-end on Steam-200k.

The Spark / MLflow versions of these stages live next to this module
and are interface-compatible; the benchmark uses the laptop path
because Spark / MLflow / pgvector are not appropriate dependencies for
a reproducibility script that anyone should be able to run locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans

from gamereco.training.als_inmem import (
    ALSInMemConfig,
    ALSInMemModel,
    train_als_inmem,
)


@dataclass
class HybridCandidates:
    """One row per (user, candidate) pair fed into the XGBoost ranker."""

    df: pd.DataFrame
    n_items: int


@dataclass
class HybridBundle:
    """Trained artifacts produced by :func:`train_hybrid`."""

    als: ALSInMemModel
    ncf_user_emb: np.ndarray | None
    ncf_item_emb: np.ndarray | None
    user_clusters: np.ndarray
    booster: xgb.Booster
    feature_columns: list[str] = field(default_factory=list)


_FEATURES = [
    "als_score",
    "ncf_score",
    "user_cluster",
    "cluster_popularity",
    "log_playtime_user",
    "log_global_popularity",
]


def _als_top_candidates(
    model: ALSInMemModel,
    user_indices: np.ndarray,
    *,
    known: dict[int, set[int]],
    n_candidates: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for user in user_indices:
        scores = model.score_all_items(int(user))
        for item in known.get(int(user), ()):
            if 0 <= item < scores.size:
                scores[item] = -np.inf
        finite_count = int(np.isfinite(scores).sum())
        n = min(n_candidates, finite_count)
        if n <= 0:
            continue
        top = np.argpartition(-scores, n - 1)[:n]
        for item in top:
            score = float(scores[int(item)])
            if not np.isfinite(score):
                continue
            rows.append(
                {
                    "user_idx": int(user),
                    "game_idx": int(item),
                    "als_score": score,
                }
            )
    return pd.DataFrame(rows)


def _ncf_score(
    user_emb: np.ndarray | None,
    item_emb: np.ndarray | None,
    candidates: pd.DataFrame,
) -> np.ndarray:
    if user_emb is None or item_emb is None:
        return np.zeros(len(candidates), dtype=np.float64)
    u = user_emb[candidates["user_idx"].to_numpy()]
    v = item_emb[candidates["game_idx"].to_numpy()]
    return np.einsum("ij,ij->i", u, v)


def _user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby("user_idx")["playtime_minutes"]
        .sum()
        .pipe(np.log1p)
        .rename("log_playtime_user")
        .reset_index()
    )


def _global_pop(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby("game_idx")["playtime_minutes"]
        .sum()
        .pipe(np.log1p)
        .rename("log_global_popularity")
        .reset_index()
    )


def _cluster_popularity(train_df: pd.DataFrame, user_clusters: np.ndarray) -> pd.DataFrame:
    annotated = train_df.assign(user_cluster=user_clusters[train_df["user_idx"]])
    return (
        annotated.groupby(["user_cluster", "game_idx"])
        .size()
        .rename("cluster_popularity")
        .reset_index()
    )


def _train_ncf_quick(
    train_df: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    epochs: int = 4,
    batch_size: int = 8192,
    embedding_dim: int = 16,
    negative_ratio: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a short PyTorch NeuMF and return (user_emb, item_emb).

    Uses a pure-NumPy negative sampler (vectorised across the whole
    epoch in one shot) rather than a per-sample DataLoader so the
    laptop benchmark finishes in seconds. The production grid-search
    NCF in :mod:`gamereco.training.ncf` still uses the configurable
    InteractionDataset for full sweeps.
    """
    import os

    import torch

    from gamereco.training.ncf import NCFConfig, NCFModel

    # On macOS the default PyTorch OMP / MKL threadpool occasionally
    # deadlocks during the first Embedding lookup against many users
    # at once. Pinning the thread count avoids the deadlock and is a
    # no-op when the env is already constrained.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    cfg = NCFConfig(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        mlp_layers=(32, 16),
        learning_rate=1e-3,
        batch_size=batch_size,
        epochs=epochs,
        negative_ratio=negative_ratio,
    )
    torch.manual_seed(cfg.seed)
    model = NCFModel(cfg)
    loss_fn = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    pos_users = train_df["user_idx"].to_numpy(dtype=np.int64)
    pos_items = train_df["game_idx"].to_numpy(dtype=np.int64)
    user_pos: dict[int, set[int]] = {}
    for u, i in zip(pos_users.tolist(), pos_items.tolist(), strict=False):
        user_pos.setdefault(u, set()).add(i)

    rng = np.random.default_rng(cfg.seed)
    n_pos = pos_users.size
    n_neg = n_pos * negative_ratio

    for epoch in range(epochs):
        # Vectorised epoch-level negative sampling. We over-sample by
        # 20% and rejection-filter against per-user positives in one
        # numpy pass; for a 92K-row training set the rejection rate is
        # negligible (catalogue has ~4K items, avg user has ~20).
        neg_users = np.repeat(pos_users, negative_ratio)
        sampled = rng.integers(0, n_items, size=int(n_neg * 1.2))
        # Lazy per-user rejection — only the first n_neg accepted.
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
        # Top up with random items if we ran out (vanishingly rare).
        while write < n_neg:
            u = int(neg_users[write])
            cand = int(rng.integers(0, n_items))
            if cand not in user_pos.get(u, ()):
                accepted_neg[write] = cand
                write += 1

        users_all = np.concatenate([pos_users, neg_users])
        items_all = np.concatenate([pos_items, accepted_neg])
        labels_all = np.concatenate(
            [np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)]
        )
        perm = rng.permutation(users_all.size)
        users_all = users_all[perm]
        items_all = items_all[perm]
        labels_all = labels_all[perm]

        model.train()
        for start in range(0, users_all.size, batch_size):
            end = start + batch_size
            u_batch = torch.from_numpy(users_all[start:end]).long()
            i_batch = torch.from_numpy(items_all[start:end]).long()
            l_batch = torch.from_numpy(labels_all[start:end]).float()
            optim.zero_grad()
            preds = model(u_batch, i_batch)
            loss = loss_fn(preds, l_batch)
            loss.backward()
            optim.step()
    model.eval()
    return model.user_embeddings(), model.item_embeddings()


def assemble_candidates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    als: ALSInMemModel,
    user_clusters: np.ndarray,
    n_candidates: int,
    ncf_user_emb: np.ndarray | None,
    ncf_item_emb: np.ndarray | None,
) -> HybridCandidates:
    known = {
        int(u): set(int(g) for g in group["game_idx"]) for u, group in train_df.groupby("user_idx")
    }
    user_indices = val_df["user_idx"].unique()
    cand = _als_top_candidates(als, user_indices, known=known, n_candidates=n_candidates)
    cand["user_cluster"] = user_clusters[cand["user_idx"].to_numpy()]
    cand = cand.merge(_user_features(train_df), on="user_idx", how="left")
    cand = cand.merge(_global_pop(train_df), on="game_idx", how="left")
    cand = cand.merge(
        _cluster_popularity(train_df, user_clusters),
        on=["user_cluster", "game_idx"],
        how="left",
    )
    cand["cluster_popularity"] = cand["cluster_popularity"].fillna(0)
    cand["log_playtime_user"] = cand["log_playtime_user"].fillna(0.0)
    cand["log_global_popularity"] = cand["log_global_popularity"].fillna(0.0)
    cand["ncf_score"] = _ncf_score(ncf_user_emb, ncf_item_emb, cand)

    truth = val_df.set_index(["user_idx", "game_idx"]).index
    cand["label"] = [
        1 if (u, g) in truth else 0
        for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
    ]
    cand = cand.sort_values("user_idx").reset_index(drop=True)
    return HybridCandidates(df=cand, n_items=int(als.n_items))


def train_hybrid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    als_config: ALSInMemConfig = ALSInMemConfig(),
    kmeans_k: int = 16,
    n_candidates: int = 200,
    use_ncf: bool = True,
    ncf_epochs: int = 4,
) -> HybridBundle:
    """Train ALS → KMeans → NCF → XGBoost ranker end-to-end."""
    als = train_als_inmem(train_df, n_users=n_users, n_items=n_items, config=als_config)
    km = KMeans(n_clusters=kmeans_k, random_state=als_config.seed, n_init=4)
    user_clusters = km.fit_predict(als.user_factors).astype(np.int32)

    if use_ncf:
        ncf_user, ncf_item = _train_ncf_quick(
            train_df, n_users=n_users, n_items=n_items, epochs=ncf_epochs
        )
    else:
        ncf_user = ncf_item = None

    cand = assemble_candidates(
        train_df,
        val_df,
        als=als,
        user_clusters=user_clusters,
        n_candidates=n_candidates,
        ncf_user_emb=ncf_user,
        ncf_item_emb=ncf_item,
    )

    X = cand.df[_FEATURES].copy()
    X["user_cluster"] = X["user_cluster"].astype("category")
    y = cand.df["label"].to_numpy()
    group = cand.df.groupby("user_idx").size().to_numpy()
    dmatrix = xgb.DMatrix(X, label=y, group=group, enable_categorical=True)
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": als_config.seed,
    }
    booster = xgb.train(params, dmatrix, num_boost_round=200, verbose_eval=False)

    return HybridBundle(
        als=als,
        ncf_user_emb=ncf_user,
        ncf_item_emb=ncf_item,
        user_clusters=user_clusters,
        booster=booster,
        feature_columns=_FEATURES,
    )


def hybrid_rerank(
    bundle: HybridBundle,
    train_df: pd.DataFrame,
    user_indices: np.ndarray,
    *,
    n_candidates: int = 200,
    k: int = 10,
) -> dict[int, list[int]]:
    """Score and re-rank ALS candidates with the XGBoost ranker."""
    known = {
        int(u): set(int(g) for g in group["game_idx"]) for u, group in train_df.groupby("user_idx")
    }
    cand = _als_top_candidates(bundle.als, user_indices, known=known, n_candidates=n_candidates)
    cand["user_cluster"] = bundle.user_clusters[cand["user_idx"].to_numpy()]
    cand = cand.merge(_user_features(train_df), on="user_idx", how="left")
    cand = cand.merge(_global_pop(train_df), on="game_idx", how="left")
    cand = cand.merge(
        _cluster_popularity(train_df, bundle.user_clusters),
        on=["user_cluster", "game_idx"],
        how="left",
    )
    cand["cluster_popularity"] = cand["cluster_popularity"].fillna(0)
    cand["log_playtime_user"] = cand["log_playtime_user"].fillna(0.0)
    cand["log_global_popularity"] = cand["log_global_popularity"].fillna(0.0)
    cand["ncf_score"] = _ncf_score(bundle.ncf_user_emb, bundle.ncf_item_emb, cand)
    X = cand[bundle.feature_columns].copy()
    X["user_cluster"] = X["user_cluster"].astype("category")
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    cand["score"] = bundle.booster.predict(dmatrix)
    out: dict[int, list[int]] = {}
    for user_idx, group in cand.sort_values("score", ascending=False).groupby("user_idx"):
        out[int(user_idx)] = group["game_idx"].astype(int).head(k).tolist()
    return out


def als_topk(
    als: ALSInMemModel,
    train_df: pd.DataFrame,
    user_indices: np.ndarray,
    *,
    k: int = 10,
) -> dict[int, list[int]]:
    """Top-K predictions from a bare ALS model (no re-ranker)."""
    known = {
        int(u): set(int(g) for g in group["game_idx"]) for u, group in train_df.groupby("user_idx")
    }
    out: dict[int, list[int]] = {}
    for user in user_indices:
        recs = als.recommend(int(user), k=k, exclude=known.get(int(user), ()))
        out[int(user)] = [item for item, _ in recs]
    return out


def ncf_topk(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    train_df: pd.DataFrame,
    user_indices: np.ndarray,
    *,
    k: int = 10,
) -> dict[int, list[int]]:
    """Top-K predictions from raw NCF dot-product scores."""
    known = {
        int(u): set(int(g) for g in group["game_idx"]) for u, group in train_df.groupby("user_idx")
    }
    out: dict[int, list[int]] = {}
    for user in user_indices:
        scores = item_emb @ user_emb[int(user)]
        for item in known.get(int(user), ()):
            if 0 <= item < scores.size:
                scores[int(item)] = -np.inf
        top = np.argpartition(-scores, min(k, scores.size - 1))[:k]
        top = top[np.argsort(-scores[top])]
        out[int(user)] = [int(i) for i in top]
    return out
