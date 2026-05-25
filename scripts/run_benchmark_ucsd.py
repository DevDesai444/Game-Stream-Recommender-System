"""Run the full hybrid benchmark on the UCSD Steam dataset.

Headlining the two-tower NCF that consumes real content metadata
(genres, tags, price, release year, etc.) instead of the bare-id
NeuMF the steam-200k benchmark could afford.

Usage:
    bash scripts/download_ucsd_dataset.sh
    python scripts/run_benchmark_ucsd.py --out benchmarks/results_ucsd.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gamereco.datasets.steam_ucsd import (
    UCSDLoadConfig,
    load_ucsd,
    temporal_split_ucsd,
)
from gamereco.training.als_inmem import ALSInMemConfig, train_als_inmem
from gamereco.training.baselines import (
    item_cooccurrence_recommender,
    item_popularity,
    popularity_recommender,
    truncate_predictions,
)
from gamereco.training.evaluation import evaluate, relative_lift
from gamereco.training.hybrid import (
    als_topk,
    hybrid_rerank,
    train_hybrid,
)
from gamereco.training.two_tower import (
    TwoTowerConfig,
    train_two_tower,
    two_tower_topk,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out", default="benchmarks/results_ucsd.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-users", type=int, default=10000, help="Cap for laptop runs")
    p.add_argument("--als-factors", type=int, default=48)
    p.add_argument("--als-iters", type=int, default=10)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--kmeans-k", type=int, default=16)
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--two-tower-epochs", type=int, default=4)
    p.add_argument("--two-tower-embedding-dim", type=int, default=32)
    p.add_argument("--two-tower-hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--two-tower-output-dim", type=int, default=32)
    return p.parse_args()


def _truth_map(df: pd.DataFrame) -> dict[int, list[int]]:
    return {int(u): list(int(g) for g in group["game_idx"]) for u, group in df.groupby("user_idx")}


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    games_path = data_dir / "steam_games.json.gz"
    users_path = data_dir / "australian_users_items.json.gz"
    reviews_path = data_dir / "australian_user_reviews.json.gz"
    for path in (games_path, users_path, reviews_path):
        if not path.exists():
            print(f"Missing {path}. Run scripts/download_ucsd_dataset.sh first.")
            return 1

    t_total = time.time()
    print("loading UCSD dataset ...")
    result = load_ucsd(
        UCSDLoadConfig(
            games_path=games_path,
            users_items_path=users_path,
            reviews_path=reviews_path,
            max_users=args.max_users,
        )
    )
    train_df, val_df, test_df = temporal_split_ucsd(result.interactions)
    n_users = result.n_users
    n_items = result.n_games
    print(
        f"  users={n_users:,}  games={n_items:,}  "
        f"interactions={len(result.interactions):,}  "
        f"reviews={len(result.reviews):,}  "
        f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    pop_items = item_popularity(train_df)
    truth_val = _truth_map(val_df)
    truth_test = _truth_map(test_df)

    results = {}
    timings = {}

    # 1. Popularity
    print("training: popularity baseline")
    t0 = time.time()
    pop_preds = truncate_predictions(
        popularity_recommender(train_df, n_items=n_items, k=50), args.k
    )
    timings["popularity"] = round(time.time() - t0, 2)
    results["popularity"] = evaluate(
        pop_preds,
        truth_val,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
    )

    # 2. Item co-occurrence
    print("training: item co-occurrence baseline")
    t0 = time.time()
    cooc_preds = truncate_predictions(
        item_cooccurrence_recommender(train_df, n_items=n_items, k=50), args.k
    )
    timings["cooccurrence"] = round(time.time() - t0, 2)
    results["cooccurrence"] = evaluate(
        cooc_preds,
        truth_val,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
    )

    # 3. ALS
    print(f"training: implicit ALS (factors={args.als_factors}, iters={args.als_iters})")
    t0 = time.time()
    als_model = train_als_inmem(
        train_df,
        n_users=n_users,
        n_items=n_items,
        config=ALSInMemConfig(
            factors=args.als_factors,
            iterations=args.als_iters,
            reg=args.reg,
            alpha=args.alpha,
        ),
    )
    timings["als"] = round(time.time() - t0, 2)
    val_users = val_df["user_idx"].unique()
    test_users = test_df["user_idx"].unique()
    als_preds = als_topk(als_model, train_df, val_users, k=args.k)
    results["als"] = evaluate(
        als_preds,
        truth_val,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
        item_embeddings=als_model.item_factors,
    )

    # 4. Two-tower NCF (with real content features)
    print("training: two-tower NCF with content features")
    t0 = time.time()
    tt_artifacts = train_two_tower(
        train_df,
        result.games,
        result.users,
        config=TwoTowerConfig(
            n_users=n_users,
            n_items=n_items,
            n_genres=0,
            n_tags=0,
            embedding_dim=args.two_tower_embedding_dim,
            tower_hidden=tuple(args.two_tower_hidden),
            output_dim=args.two_tower_output_dim,
            epochs=args.two_tower_epochs,
        ),
    )
    timings["two_tower"] = round(time.time() - t0, 2)
    tt_preds = two_tower_topk(tt_artifacts, train_df, val_users, k=args.k)
    results["two_tower"] = evaluate(
        tt_preds,
        truth_val,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
        item_embeddings=tt_artifacts.item_vectors,
    )

    # 5. Hybrid (ALS candidates + two-tower NCF score + KMeans + XGBoost)
    print(
        f"training: hybrid (kmeans={args.kmeans_k}, candidates={args.candidates}) "
        "with two-tower as ncf signal"
    )
    t0 = time.time()
    # Reuse the hybrid harness — but inject the two-tower scores
    # as the 'ncf' channel by passing pre-trained user/item vectors
    # in place of the quick NeuMF default.
    bundle = train_hybrid(
        train_df,
        val_df,
        n_users=n_users,
        n_items=n_items,
        als_config=ALSInMemConfig(
            factors=args.als_factors,
            iterations=args.als_iters,
            reg=args.reg,
            alpha=args.alpha,
        ),
        kmeans_k=args.kmeans_k,
        n_candidates=args.candidates,
        use_ncf=False,  # we'll inject the two-tower embeddings directly
    )
    bundle.ncf_user_emb = tt_artifacts.user_vectors
    bundle.ncf_item_emb = tt_artifacts.item_vectors
    # The bundle already has ALS + clusters + XGBoost trained on
    # ALS-only candidates. Retrain XGBoost with the two-tower channel
    # now populated by re-running the candidate assembly + booster.
    from gamereco.training.hybrid import assemble_candidates
    import xgboost as xgb

    cand = assemble_candidates(
        train_df,
        val_df,
        als=bundle.als,
        user_clusters=bundle.user_clusters,
        n_candidates=args.candidates,
        ncf_user_emb=tt_artifacts.user_vectors,
        ncf_item_emb=tt_artifacts.item_vectors,
    )
    X = cand.df[bundle.feature_columns].copy()
    X["user_cluster"] = X["user_cluster"].astype("category")
    y = cand.df["label"].to_numpy()
    group = cand.df.groupby("user_idx").size().to_numpy()
    dmatrix = xgb.DMatrix(X, label=y, group=group, enable_categorical=True)
    bundle.booster = xgb.train(
        {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@10",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "tree_method": "hist",
            "seed": 42,
        },
        dmatrix,
        num_boost_round=300,
        verbose_eval=False,
    )
    timings["hybrid"] = round(time.time() - t0, 2)

    hybrid_preds = hybrid_rerank(
        bundle, train_df, val_users, n_candidates=args.candidates, k=args.k
    )
    results["hybrid"] = evaluate(
        hybrid_preds,
        truth_val,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
        item_embeddings=bundle.als.item_factors,
    )

    # 6. Same hybrid on the held-out test split for the headline number.
    hybrid_test = hybrid_rerank(
        bundle, train_df, test_users, n_candidates=args.candidates, k=args.k
    )
    results["hybrid_test"] = evaluate(
        hybrid_test,
        truth_test,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
        item_embeddings=bundle.als.item_factors,
    )
    als_test = als_topk(als_model, train_df, test_users, k=args.k)
    results["als_test"] = evaluate(
        als_test,
        truth_test,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
        item_embeddings=als_model.item_factors,
    )

    report = {
        "dataset": {
            "name": "UCSD Steam (Australian users)",
            "raw_files": [str(games_path.name), str(users_path.name), str(reviews_path.name)],
            "n_users": int(n_users),
            "n_games": int(n_items),
            "n_interactions": int(len(result.interactions)),
            "n_reviews": int(len(result.reviews)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "config": {
            "k": args.k,
            "max_users": args.max_users,
            "als_factors": args.als_factors,
            "als_iters": args.als_iters,
            "alpha": args.alpha,
            "reg": args.reg,
            "kmeans_k": args.kmeans_k,
            "candidates": args.candidates,
            "two_tower": {
                "embedding_dim": args.two_tower_embedding_dim,
                "tower_hidden": args.two_tower_hidden,
                "output_dim": args.two_tower_output_dim,
                "epochs": args.two_tower_epochs,
                "n_genres": tt_artifacts.spec.n_genres,
                "n_tags": tt_artifacts.spec.n_tags,
            },
        },
        "timings_seconds": timings,
        "metrics": {name: result.as_dict() for name, result in results.items()},
        "lift": {
            "hybrid_vs_als_val": round(relative_lift(results["hybrid"], results["als"]), 4),
            "hybrid_vs_als_test": round(
                relative_lift(results["hybrid_test"], results["als_test"]), 4
            ),
            "hybrid_vs_popularity_val": round(
                relative_lift(results["hybrid"], results["popularity"]), 4
            ),
            "two_tower_vs_als_val": round(relative_lift(results["two_tower"], results["als"]), 4),
        },
        "wall_clock_seconds": round(time.time() - t_total, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nreport written -> {out_path}")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True, default=str))
    print(json.dumps(report["lift"], indent=2, sort_keys=True))
    print(f"\ntotal wall clock: {report['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
