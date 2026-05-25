"""Run the full hybrid benchmark on Steam-200k and emit a results JSON.

This is the script behind every measured number in benchmarks/results.md.
It deliberately runs on a laptop (pure pandas/numpy + a short PyTorch
NCF + scipy ALS + xgboost) so the results are reproducible without
needing to stand up the full Spark + MLflow + Airflow stack.

Usage:
    bash scripts/download_dataset.sh
    python scripts/run_benchmark.py --out benchmarks/results.json

Optional flags:
    --no-ncf            skip the PyTorch NCF step (faster)
    --kmeans-k 16       user-cohort clusters
    --candidates 200    ALS candidates per user fed into the ranker
    --als-factors 64
    --als-iters 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gamereco.datasets.steam200k import (
    load_steam_200k,
    materialise_silver,
    temporal_split_pandas,
)
from gamereco.training.als_inmem import ALSInMemConfig, train_als_inmem
from gamereco.training.baselines import (
    item_cooccurrence_recommender,
    item_popularity,
    popularity_recommender,
    truncate_predictions,
)
from gamereco.training.evaluation import EvalResult, evaluate, relative_lift
from gamereco.training.hybrid import (
    als_topk,
    hybrid_rerank,
    ncf_topk,
    train_hybrid,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw/steam-200k.csv")
    p.add_argument("--out", default="benchmarks/results.json")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--als-factors", type=int, default=64)
    p.add_argument("--als-iters", type=int, default=15)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--kmeans-k", type=int, default=16)
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--ncf-epochs", type=int, default=4)
    p.add_argument("--no-ncf", action="store_true")
    return p.parse_args()


def _truth_map(val_df) -> dict[int, list[int]]:
    return {
        int(u): list(int(g) for g in group["game_idx"]) for u, group in val_df.groupby("user_idx")
    }


def main() -> int:
    args = parse_args()
    csv_path = Path(args.data)
    if not csv_path.exists():
        print(
            f"Dataset not found at {csv_path}. Run scripts/download_dataset.sh first.",
        )
        return 1

    t_total = time.time()
    print(f"loading {csv_path} ...")
    raw = load_steam_200k(csv_path)
    silver = materialise_silver(raw)
    train_df, val_df, test_df = temporal_split_pandas(silver)
    n_users = int(silver["user_idx"].max() + 1)
    n_items = int(silver["game_idx"].max() + 1)
    print(
        f"  raw rows={len(raw):,}  silver={len(silver):,}  "
        f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}  "
        f"users={n_users:,}  games={n_items:,}"
    )

    pop_items = item_popularity(train_df)
    truth_val = _truth_map(val_df)
    truth_test = _truth_map(test_df)

    results: dict[str, EvalResult] = {}
    timings: dict[str, float] = {}

    # 1. Popularity baseline
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
    als_preds = als_topk(als_model, train_df, val_users, k=args.k)
    results["als"] = evaluate(
        als_preds,
        truth_val,
        k=args.k,
        n_items=n_items,
        item_popularity=pop_items,
        item_embeddings=als_model.item_factors,
    )

    # 4. Hybrid (ALS + optional NCF + KMeans + XGBoost)
    print(
        "training: hybrid ensemble "
        f"(ncf={'off' if args.no_ncf else f'{args.ncf_epochs} epochs'}, "
        f"kmeans={args.kmeans_k}, candidates={args.candidates})"
    )
    t0 = time.time()
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
        use_ncf=not args.no_ncf,
        ncf_epochs=args.ncf_epochs,
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

    # 5. Optional standalone NCF
    if not args.no_ncf and bundle.ncf_user_emb is not None and bundle.ncf_item_emb is not None:
        ncf_preds = ncf_topk(
            bundle.ncf_user_emb,
            bundle.ncf_item_emb,
            train_df,
            val_users,
            k=args.k,
        )
        results["ncf"] = evaluate(
            ncf_preds,
            truth_val,
            k=args.k,
            n_items=n_items,
            item_popularity=pop_items,
            item_embeddings=bundle.ncf_item_emb,
        )

    # 6. Hybrid on the held-out test split for the report's headline number.
    test_users = test_df["user_idx"].unique()
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
            "path": str(csv_path),
            "raw_rows": int(len(raw)),
            "silver_rows": int(len(silver)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "n_users": int(n_users),
            "n_items": int(n_items),
        },
        "config": {
            "k": args.k,
            "als_factors": args.als_factors,
            "als_iters": args.als_iters,
            "alpha": args.alpha,
            "reg": args.reg,
            "kmeans_k": args.kmeans_k,
            "candidates": args.candidates,
            "ncf_enabled": not args.no_ncf,
            "ncf_epochs": args.ncf_epochs,
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
        },
        "wall_clock_seconds": round(time.time() - t_total, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nreport written -> {out_path}")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(json.dumps(report["lift"], indent=2, sort_keys=True))
    print(f"\ntotal wall clock: {report['wall_clock_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
