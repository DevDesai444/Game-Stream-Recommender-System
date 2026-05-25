"""ALS collaborative filtering with Spark MLlib + CrossValidator tuning.

The hyperparameter grid (rank × regParam × alpha × maxIter) crosses to
*24 ALS configurations*. Paired with the 24-config NCF grid downstream
this gives the *48 configurations* total tuned via CrossValidator that
the project advertises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import mlflow.spark
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from gamereco.common.logging import get_logger
from gamereco.common.paths import LakePaths
from gamereco.training.metrics import (
    collect_top_k,
    ground_truth_from_holdout,
    ndcg_at_k_numpy,
)
from gamereco.training.mlflow_utils import log_metrics, log_params, start_run

log = get_logger(__name__)


@dataclass
class ALSGrid:
    ranks: list[int] = field(default_factory=lambda: [32, 64, 128])
    reg_params: list[float] = field(default_factory=lambda: [0.01, 0.05])
    alphas: list[float] = field(default_factory=lambda: [10.0, 40.0])
    max_iters: list[int] = field(default_factory=lambda: [10, 15])
    num_folds: int = 3

    @property
    def total_configs(self) -> int:
        return len(self.ranks) * len(self.reg_params) * len(self.alphas) * len(self.max_iters)


def _build_param_grid(als: ALS, grid: ALSGrid):
    builder = ParamGridBuilder()
    builder = builder.addGrid(als.rank, grid.ranks)
    builder = builder.addGrid(als.regParam, grid.reg_params)
    builder = builder.addGrid(als.alpha, grid.alphas)
    builder = builder.addGrid(als.maxIter, grid.max_iters)
    return builder.build()


def train_als(
    train: DataFrame,
    val: DataFrame,
    grid: ALSGrid = ALSGrid(),
    *,
    rating_col: str = "confidence",
    user_col: str = "user_idx",
    item_col: str = "game_idx",
    seed: int = 42,
) -> tuple[ALSModel, dict[str, float], dict[str, object]]:
    """Train ALS with CrossValidator and return (best_model, metrics, best_params)."""
    als = ALS(
        userCol=user_col,
        itemCol=item_col,
        ratingCol=rating_col,
        implicitPrefs=True,
        nonnegative=True,
        coldStartStrategy="drop",
        seed=seed,
    )
    param_grid = _build_param_grid(als, grid)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol=rating_col, predictionCol="prediction"
    )
    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=grid.num_folds,
        parallelism=2,
        seed=seed,
    )
    log.info("als.cv.start", total_configs=grid.total_configs, folds=grid.num_folds)
    cv_model = cv.fit(train)
    best_model: ALSModel = cv_model.bestModel  # type: ignore[assignment]
    best_params = {
        "rank": best_model._java_obj.parent().getRank(),
        "regParam": best_model._java_obj.parent().getRegParam(),
        "alpha": best_model._java_obj.parent().getAlpha(),
        "maxIter": best_model._java_obj.parent().getMaxIter(),
    }

    val_preds = best_model.transform(val).na.drop(subset=["prediction"])
    rmse = evaluator.evaluate(val_preds)

    user_recs = (
        best_model.recommendForAllUsers(50)
        .select(
            F.col(user_col),
            F.explode("recommendations").alias("rec"),
        )
        .select(
            F.col(user_col),
            F.col("rec.game_idx").alias(item_col),
            F.col("rec.rating").alias("score"),
        )
    )
    truth = ground_truth_from_holdout(val)
    preds = collect_top_k(user_recs, k=10)
    ndcg10 = ndcg_at_k_numpy(preds, truth, k=10)

    metrics = {"rmse": float(rmse), "ndcg_at_10": float(ndcg10)}
    log.info("als.cv.done", **best_params, **metrics)
    return best_model, metrics, best_params


def train_and_log_als(
    spark: SparkSession,
    lake: LakePaths,
    grid: ALSGrid = ALSGrid(),
    *,
    register: bool = True,
    model_name: str = "gamereco-als",
) -> dict[str, float]:
    """Train ALS over gold train/val and log everything to MLflow."""
    train = spark.read.format("delta").load(str(lake.gold_train))
    val = spark.read.format("delta").load(str(lake.gold_val))

    with start_run("als-cv", tags={"model": "als"}):
        log_params({"grid_size": grid.total_configs, "folds": grid.num_folds})
        best_model, metrics, best_params = train_als(train, val, grid)
        log_params({f"best_{k}": v for k, v in best_params.items()})
        log_metrics(metrics)
        artifact_dir = Path(lake.root) / "models" / "als"
        artifact_dir.parent.mkdir(parents=True, exist_ok=True)
        best_model.write().overwrite().save(str(artifact_dir))
        mlflow.spark.log_model(best_model, artifact_path="als")
        if register:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/als",
                name=model_name,
            )
    return metrics
