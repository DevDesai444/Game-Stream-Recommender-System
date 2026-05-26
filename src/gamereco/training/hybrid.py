"""End-to-end hybrid recommender harness used by the benchmark runner.

This module wires the laptop-runnable building blocks (in-memory ALS,
PyTorch NeuMF / two-tower, KMeans over ALS factors, XGBoost ranker)
together so the benchmark report has a single ``train_hybrid``
entrypoint that reproduces the headline numbers end-to-end.

The Spark / MLflow versions of these stages live next to this module
and are interface-compatible; the benchmark uses the laptop path
because Spark / MLflow / pgvector are not appropriate dependencies for
a reproducibility script that anyone should be able to run locally.

The hybrid composes signals at three layers:

  1. **Retrieval** — top-K candidates per user from ALS, optionally
     unioned with top-K candidates from the trained two-tower. Mixing
     retrievers surfaces items where ALS and NCF disagree, which is
     where the ranker has the most to combine.
  2. **Featurisation** — ALS score, NCF cosine, NCF rank within the
     user, an "in NCF top-K" flag, KMeans cluster popularity, log
     global popularity, and the user's log total playtime.
  3. **Ranking** — XGBoost ``rank:ndcg`` directly optimises the
     headline metric using the candidates' label = (item ∈ val truth).
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
    # Number of NCF candidates to mix in at re-rank time. 0 means the
    # ranker was trained on ALS-only candidates and we should keep
    # serving that way at inference (mixing in untrained-on candidates
    # would skew the score distribution).
    ncf_candidate_k: int = 0


_FEATURES_BASE = [
    "als_score",
    "ncf_score",
    "user_cluster",
    "cluster_popularity",
    "log_playtime_user",
    "log_global_popularity",
]
_FEATURES_WITH_NCF = _FEATURES_BASE + [
    "ncf_rank",
    "ncf_in_top_k",
    "source_als",
    "source_ncf",
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


def _ncf_top_candidates(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_indices: np.ndarray,
    *,
    known: dict[int, set[int]],
    n_candidates: int,
) -> pd.DataFrame:
    """Top-K candidates from the two-tower cosine scores.

    Returned rows carry ``ncf_score`` and ``ncf_rank`` for each user,
    so the caller doesn't have to recompute either downstream.
    """
    rows: list[dict[str, float]] = []
    for user in user_indices:
        u = int(user)
        if u >= user_emb.shape[0]:
            continue
        scores = item_emb @ user_emb[u]
        for item in known.get(u, ()):
            if 0 <= item < scores.size:
                scores[int(item)] = -np.inf
        finite = int(np.isfinite(scores).sum())
        n = min(n_candidates, finite)
        if n <= 0:
            continue
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        for rank, item in enumerate(top.tolist()):
            rows.append(
                {
                    "user_idx": u,
                    "game_idx": int(item),
                    "ncf_score_retr": float(scores[int(item)]),
                    "ncf_rank": int(rank),
                }
            )
    return pd.DataFrame(rows)


def _ncf_score(
    user_emb: np.ndarray | None,
    item_emb: np.ndarray | None,
    candidates: pd.DataFrame,
) -> np.ndarray:
    if user_emb is None or item_emb is None or len(candidates) == 0:
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

    Used by the steam-200k benchmark where the catalogue is tiny (~4K
    items) and where training a full content-aware two-tower is
    overkill. For UCSD the production path is to pass a trained
    ``TwoTowerArtifacts`` into :func:`train_hybrid` directly and skip
    this entirely.
    """
    import os

    import torch

    from gamereco.training.ncf import NCFConfig, NCFModel

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

    for _epoch in range(epochs):
        neg_users = np.repeat(pos_users, negative_ratio)
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
    ncf_candidate_k: int = 0,
) -> HybridCandidates:
    """Assemble candidate rows for the ranker.

    When ``ncf_candidate_k > 0`` and NCF embeddings are provided, the
    candidate set is the **union** of ALS top-N and NCF top-K per user.
    Items missing an ALS score (NCF-only candidates) are filled by
    re-scoring against ALS so the ranker sees both signals on every
    row. ``source_als`` / ``source_ncf`` boolean features tell the
    ranker which retriever surfaced each row (or both).
    """
    known = {
        int(u): set(int(g) for g in group["game_idx"]) for u, group in train_df.groupby("user_idx")
    }
    user_indices = val_df["user_idx"].unique()
    use_ncf_retrieval = (
        ncf_candidate_k > 0 and ncf_user_emb is not None and ncf_item_emb is not None
    )

    als_cand = _als_top_candidates(als, user_indices, known=known, n_candidates=n_candidates)
    if use_ncf_retrieval:
        ncf_cand = _ncf_top_candidates(
            ncf_user_emb,
            ncf_item_emb,
            user_indices,
            known=known,
            n_candidates=ncf_candidate_k,
        )
        if not ncf_cand.empty:
            # Union by (user, game). When the same item is surfaced by
            # both retrievers we keep the ALS row (which already has
            # als_score populated) and merge in ncf_rank / source flags.
            als_cand["source_als"] = 1
            als_cand["source_ncf"] = 0
            ncf_only = ncf_cand.merge(
                als_cand[["user_idx", "game_idx"]],
                on=["user_idx", "game_idx"],
                how="left",
                indicator=True,
            )
            ncf_only = ncf_only[ncf_only["_merge"] == "left_only"].drop(columns="_merge")
            if not ncf_only.empty:
                # Score NCF-only candidates against ALS in one vectorised
                # pass: stacked factor lookups + einsum is ~100× faster
                # than per-row score_all_items on a 100K-row union.
                u_arr = ncf_only["user_idx"].to_numpy(dtype=np.int64)
                g_arr = ncf_only["game_idx"].to_numpy(dtype=np.int64)
                u_vecs = als.user_factors[u_arr]
                v_vecs = als.item_factors[g_arr]
                ncf_only_scores = np.einsum("ij,ij->i", u_vecs, v_vecs).astype(np.float64)
                ncf_only = ncf_only.assign(
                    als_score=ncf_only_scores,
                    source_als=0,
                    source_ncf=1,
                )[["user_idx", "game_idx", "als_score", "source_als", "source_ncf"]]
                cand = pd.concat([als_cand, ncf_only], ignore_index=True)
            else:
                cand = als_cand

            # Annotate ncf_rank for ALL rows (default = ncf_candidate_k for
            # ALS-only rows that NCF didn't surface in its top-K).
            rank_lookup = ncf_cand.set_index(["user_idx", "game_idx"])["ncf_rank"].to_dict()
            cand["ncf_rank"] = [
                rank_lookup.get((int(u), int(g)), ncf_candidate_k)
                for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
            ]
            cand["ncf_in_top_k"] = (cand["ncf_rank"] < ncf_candidate_k).astype(int)
            # Mark rows that ALS surfaced even if originally tagged 0.
            als_set = set(
                zip(
                    als_cand["user_idx"].astype(int),
                    als_cand["game_idx"].astype(int),
                    strict=False,
                )
            )
            cand["source_als"] = [
                1 if (int(u), int(g)) in als_set else 0
                for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
            ]
            ncf_set = set(
                zip(
                    ncf_cand["user_idx"].astype(int),
                    ncf_cand["game_idx"].astype(int),
                    strict=False,
                )
            )
            cand["source_ncf"] = [
                1 if (int(u), int(g)) in ncf_set else 0
                for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
            ]
        else:
            cand = als_cand
            cand["source_als"] = 1
            cand["source_ncf"] = 0
            cand["ncf_rank"] = ncf_candidate_k
            cand["ncf_in_top_k"] = 0
    else:
        cand = als_cand

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
    pretrained_ncf: tuple[np.ndarray, np.ndarray] | None = None,
    ncf_candidate_k: int = 0,
    xgb_objective: str = "rank:ndcg",
    xgb_num_boost_round: int = 300,
) -> HybridBundle:
    """Train ALS → KMeans → (NCF) → XGBoost ranker end-to-end.

    Args:
        pretrained_ncf: optional ``(user_emb, item_emb)`` to use
            instead of training a quick NeuMF inline. Pass
            ``(artifacts.user_vectors, artifacts.item_vectors)`` from
            :func:`gamereco.training.two_tower.train_two_tower` to
            re-use a strong, content-aware two-tower without spending
            another training run inside this function.
        ncf_candidate_k: when > 0, union ALS top-N candidates with NCF
            top-K candidates per user. The ranker then learns when to
            trust each retriever from the ``source_als`` / ``source_ncf``
            flags. Requires NCF embeddings (either pretrained or trained
            inline).
        xgb_objective: ``rank:ndcg`` (default) directly optimises NDCG.
            ``rank:pairwise`` is the older objective.
    """
    als = train_als_inmem(train_df, n_users=n_users, n_items=n_items, config=als_config)
    km = KMeans(n_clusters=kmeans_k, random_state=als_config.seed, n_init=4)
    user_clusters = km.fit_predict(als.user_factors).astype(np.int32)

    if pretrained_ncf is not None:
        ncf_user, ncf_item = pretrained_ncf
    elif use_ncf:
        ncf_user, ncf_item = _train_ncf_quick(
            train_df, n_users=n_users, n_items=n_items, epochs=ncf_epochs
        )
    else:
        ncf_user = ncf_item = None

    effective_ncf_candidate_k = (
        ncf_candidate_k if ncf_user is not None and ncf_item is not None else 0
    )

    cand = assemble_candidates(
        train_df,
        val_df,
        als=als,
        user_clusters=user_clusters,
        n_candidates=n_candidates,
        ncf_user_emb=ncf_user,
        ncf_item_emb=ncf_item,
        ncf_candidate_k=effective_ncf_candidate_k,
    )

    features = _FEATURES_WITH_NCF if effective_ncf_candidate_k > 0 else _FEATURES_BASE
    X = cand.df[features].copy()
    X["user_cluster"] = X["user_cluster"].astype("category")
    y = cand.df["label"].to_numpy()
    group = cand.df.groupby("user_idx").size().to_numpy()
    dmatrix = xgb.DMatrix(X, label=y, group=group, enable_categorical=True)
    params = {
        "objective": xgb_objective,
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": als_config.seed,
    }
    booster = xgb.train(params, dmatrix, num_boost_round=xgb_num_boost_round, verbose_eval=False)

    return HybridBundle(
        als=als,
        ncf_user_emb=ncf_user,
        ncf_item_emb=ncf_item,
        user_clusters=user_clusters,
        booster=booster,
        feature_columns=features,
        ncf_candidate_k=effective_ncf_candidate_k,
    )


def hybrid_rerank(
    bundle: HybridBundle,
    train_df: pd.DataFrame,
    user_indices: np.ndarray,
    *,
    n_candidates: int = 200,
    k: int = 10,
) -> dict[int, list[int]]:
    """Score and re-rank candidates with the trained XGBoost ranker.

    Mirrors :func:`assemble_candidates` at training time: if the bundle
    was trained on ALS+NCF unioned candidates, the same union is built
    here at inference so the feature distribution matches what the
    ranker was fit on.
    """
    known = {
        int(u): set(int(g) for g in group["game_idx"]) for u, group in train_df.groupby("user_idx")
    }
    use_ncf_retrieval = (
        bundle.ncf_candidate_k > 0
        and bundle.ncf_user_emb is not None
        and bundle.ncf_item_emb is not None
    )

    als_cand = _als_top_candidates(bundle.als, user_indices, known=known, n_candidates=n_candidates)
    if use_ncf_retrieval:
        ncf_cand = _ncf_top_candidates(
            bundle.ncf_user_emb,
            bundle.ncf_item_emb,
            user_indices,
            known=known,
            n_candidates=bundle.ncf_candidate_k,
        )
        if not ncf_cand.empty:
            als_cand["source_als"] = 1
            als_cand["source_ncf"] = 0
            ncf_only = ncf_cand.merge(
                als_cand[["user_idx", "game_idx"]],
                on=["user_idx", "game_idx"],
                how="left",
                indicator=True,
            )
            ncf_only = ncf_only[ncf_only["_merge"] == "left_only"].drop(columns="_merge")
            if not ncf_only.empty:
                u_arr = ncf_only["user_idx"].to_numpy(dtype=np.int64)
                g_arr = ncf_only["game_idx"].to_numpy(dtype=np.int64)
                u_vecs = bundle.als.user_factors[u_arr]
                v_vecs = bundle.als.item_factors[g_arr]
                ncf_only_scores = np.einsum("ij,ij->i", u_vecs, v_vecs).astype(np.float64)
                ncf_only = ncf_only.assign(
                    als_score=ncf_only_scores,
                    source_als=0,
                    source_ncf=1,
                )[["user_idx", "game_idx", "als_score", "source_als", "source_ncf"]]
                cand = pd.concat([als_cand, ncf_only], ignore_index=True)
            else:
                cand = als_cand
            rank_lookup = ncf_cand.set_index(["user_idx", "game_idx"])["ncf_rank"].to_dict()
            cand["ncf_rank"] = [
                rank_lookup.get((int(u), int(g)), bundle.ncf_candidate_k)
                for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
            ]
            cand["ncf_in_top_k"] = (cand["ncf_rank"] < bundle.ncf_candidate_k).astype(int)
            als_set = set(
                zip(
                    als_cand["user_idx"].astype(int),
                    als_cand["game_idx"].astype(int),
                    strict=False,
                )
            )
            cand["source_als"] = [
                1 if (int(u), int(g)) in als_set else 0
                for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
            ]
            ncf_set = set(
                zip(
                    ncf_cand["user_idx"].astype(int),
                    ncf_cand["game_idx"].astype(int),
                    strict=False,
                )
            )
            cand["source_ncf"] = [
                1 if (int(u), int(g)) in ncf_set else 0
                for u, g in zip(cand["user_idx"], cand["game_idx"], strict=False)
            ]
        else:
            cand = als_cand
            cand["source_als"] = 1
            cand["source_ncf"] = 0
            cand["ncf_rank"] = bundle.ncf_candidate_k
            cand["ncf_in_top_k"] = 0
    else:
        cand = als_cand

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
