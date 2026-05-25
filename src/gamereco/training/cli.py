"""CLI entrypoint: ``gamereco-train``."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F

from gamereco.common.config import spark_settings
from gamereco.common.logging import configure_logging, get_logger
from gamereco.common.paths import LakePaths
from gamereco.etl.session import build_spark
from gamereco.training.als import ALSGrid, train_and_log_als
from gamereco.training.clustering import (
    KMeansConfig,
    fit_user_kmeans,
    silhouette_score_proxy,
)
from gamereco.training.ensemble import (
    XGBConfig,
    lift_vs_baseline,
    save_booster,
    train_xgb,
)
from gamereco.training.mlflow_utils import log_metrics, log_params, start_run
from gamereco.training.ncf import (
    NCFConfig,
    NCFGrid,
    cross_validate_ncf,
    save_ncf,
)

log = get_logger(__name__)


@click.group()
def cli() -> None:
    """Model training CLI."""
    configure_logging()


@cli.command("als")
def train_als_cmd() -> None:
    """Train ALS with CrossValidator and log to MLflow."""
    spark = build_spark("gamereco-train-als")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    metrics = train_and_log_als(spark, lake)
    click.echo(metrics)


@cli.command("ncf")
@click.option("--epochs", default=8, type=int)
@click.option("--batch-size", default=4096, type=int)
@click.option("--device", default="cpu", type=str)
def train_ncf_cmd(epochs: int, batch_size: int, device: str) -> None:
    """Train PyTorch NeuMF over the temporal train/val gold tables."""
    spark = build_spark("gamereco-train-ncf")
    lake = LakePaths(root=Path(spark_settings().delta_root))

    train_pd = (
        spark.read.format("delta").load(str(lake.gold_train)).select("user_idx", "game_idx").toPandas()
    )
    val_pd = (
        spark.read.format("delta").load(str(lake.gold_val)).select("user_idx", "game_idx").toPandas()
    )
    num_users = int(max(train_pd["user_idx"].max(), val_pd["user_idx"].max()) + 1)
    num_items = int(max(train_pd["game_idx"].max(), val_pd["game_idx"].max()) + 1)
    base = NCFConfig(
        num_users=num_users,
        num_items=num_items,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    grid = NCFGrid()
    with start_run("ncf-cv", tags={"model": "ncf"}):
        log_params({"grid_size": grid.total_configs, "num_users": num_users, "num_items": num_items})
        best_model, best_cfg, metrics, history = cross_validate_ncf(
            train_pd, val_pd, num_users, num_items, grid=grid, base=base
        )
        log_params({f"best_{k}": v for k, v in asdict(best_cfg).items()})
        log_metrics(metrics)
        target = Path(lake.root) / "models" / "ncf.pt"
        save_ncf(best_model, best_cfg, target)
        mlflow.log_artifact(str(target), artifact_path="ncf")
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/ncf",
            name="gamereco-ncf",
        )
    click.echo(metrics)


@cli.command("kmeans")
@click.option("--k", default=16, type=int)
def train_kmeans_cmd(k: int) -> None:
    """Fit user-cohort KMeans on ALS factors."""
    spark = build_spark("gamereco-train-kmeans")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    als_path = Path(lake.root) / "models" / "als"
    als_model = ALSModel.load(str(als_path))
    with start_run("kmeans-cohorts", tags={"model": "kmeans"}):
        model, assignments = fit_user_kmeans(spark, als_model, KMeansConfig(k=k))
        entropy = silhouette_score_proxy(assignments)
        log_metrics({"cluster_entropy": entropy})
        assignments.write.format("delta").mode("overwrite").save(str(lake.gold_user_clusters))
        model.write().overwrite().save(str(Path(lake.root) / "models" / "kmeans"))
    click.echo({"cluster_entropy": entropy, "k": k})


def _candidate_frame(
    spark, lake: LakePaths, *, top_n_candidates: int
) -> pd.DataFrame:
    """Stitch ALS top-N candidates with NCF scores, cohort features, and labels."""
    als_path = Path(lake.root) / "models" / "als"
    als_model = ALSModel.load(str(als_path))
    train = spark.read.format("delta").load(str(lake.gold_train))
    val = spark.read.format("delta").load(str(lake.gold_val))
    cohorts = spark.read.format("delta").load(str(lake.gold_user_clusters))

    candidates = (
        als_model.recommendForAllUsers(top_n_candidates)
        .select("user_idx", F.explode("recommendations").alias("rec"))
        .select(
            "user_idx",
            F.col("rec.game_idx").alias("game_idx"),
            F.col("rec.rating").alias("als_score"),
        )
    )
    candidates = candidates.join(cohorts, on="user_idx", how="left")
    global_pop = (
        train.groupBy("game_idx")
        .agg(F.sum("playtime_minutes").alias("global_playtime"))
        .withColumn("log_global_popularity", F.log1p("global_playtime"))
        .drop("global_playtime")
    )
    user_pop = (
        train.groupBy("user_idx")
        .agg(F.sum("playtime_minutes").alias("user_playtime"))
        .withColumn("log_playtime_user", F.log1p("user_playtime"))
        .drop("user_playtime")
    )
    cluster_pop = (
        train.join(cohorts, on="user_idx", how="inner")
        .groupBy("user_cluster", "game_idx")
        .agg(F.count("*").alias("cluster_popularity"))
    )
    labels = (
        val.select("user_idx", "game_idx")
        .withColumn("label", F.lit(1))
    )
    enriched = (
        candidates.join(global_pop, on="game_idx", how="left")
        .join(user_pop, on="user_idx", how="left")
        .join(cluster_pop, on=["user_cluster", "game_idx"], how="left")
        .join(labels, on=["user_idx", "game_idx"], how="left")
        .fillna({
            "log_global_popularity": 0.0,
            "log_playtime_user": 0.0,
            "cluster_popularity": 0,
            "label": 0,
            "user_cluster": -1,
        })
    )
    return enriched.toPandas()


@cli.command("ensemble")
@click.option("--top-n", default=200, type=int, help="ALS candidate depth per user")
def train_ensemble_cmd(top_n: int) -> None:
    """Train the XGBoost ensemble over ALS + NCF + cohort features."""
    spark = build_spark("gamereco-train-ensemble")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    df = _candidate_frame(spark, lake, top_n_candidates=top_n)
    ncf_path = Path(lake.root) / "models" / "ncf.pt"
    if ncf_path.exists():
        from gamereco.training.ncf import load_ncf

        ncf_model, ncf_cfg = load_ncf(ncf_path)
        import torch

        with torch.no_grad():
            users_t = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
            items_t = torch.tensor(df["game_idx"].to_numpy(), dtype=torch.long)
            df["ncf_score"] = ncf_model(users_t, items_t).cpu().numpy()
    else:
        log.warning("ensemble.ncf_missing", path=str(ncf_path))
        df["ncf_score"] = 0.0

    df["user_cluster"] = df["user_cluster"].astype(int)
    # Per-user split (80/20) for the ranker.
    rng = np.random.default_rng(seed=42)
    user_ids = df["user_idx"].unique()
    rng.shuffle(user_ids)
    cutoff = int(len(user_ids) * 0.8)
    train_users = set(user_ids[:cutoff])
    train_df = df[df["user_idx"].isin(train_users)].sort_values("user_idx").reset_index(drop=True)
    val_df = df[~df["user_idx"].isin(train_users)].sort_values("user_idx").reset_index(drop=True)

    with start_run("ensemble-xgb", tags={"model": "xgb-ensemble"}):
        artifacts = train_xgb(train_df, val_df, XGBConfig())
        baseline_ndcg = _baseline_ndcg(val_df)
        lift = lift_vs_baseline(artifacts.metrics["ndcg_at_10"], baseline_ndcg)
        log_metrics(
            {
                "ndcg_at_10": artifacts.metrics["ndcg_at_10"],
                "ndcg_at_10_baseline_als": baseline_ndcg,
                "ndcg_at_10_lift_vs_als": lift,
            }
        )
        target = Path(lake.root) / "models" / "xgb_ensemble.json"
        save_booster(artifacts, target)
        mlflow.log_artifact(str(target), artifact_path="xgb")
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/xgb",
            name="gamereco-xgb-ensemble",
        )
    click.echo({"ndcg_at_10": artifacts.metrics["ndcg_at_10"], "lift_vs_als": lift})


def _baseline_ndcg(val_df: pd.DataFrame) -> float:
    from gamereco.training.metrics import ndcg_at_k_numpy

    sorted_df = val_df.sort_values(["user_idx", "als_score"], ascending=[True, False])
    preds = sorted_df.groupby("user_idx")["game_idx"].apply(list).to_dict()
    truth = (
        val_df[val_df["label"] == 1].groupby("user_idx")["game_idx"].apply(list).to_dict()
    )
    return ndcg_at_k_numpy(preds, truth, k=10)


@cli.command("full")
def full_cmd() -> None:
    """Run the full hybrid training pipeline end-to-end."""
    spark = build_spark("gamereco-train-full")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    train_and_log_als(spark, lake, ALSGrid())
    train_ncf_cmd.callback(epochs=8, batch_size=4096, device="cpu")
    train_kmeans_cmd.callback(k=16)
    train_ensemble_cmd.callback(top_n=200)


def main() -> None:  # pragma: no cover
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
