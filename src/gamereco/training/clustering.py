"""K-Means clustering of user latent factors.

We fit K-Means on the ALS user-factor matrix to assign each user to a
cohort. The cohort id becomes a categorical feature in the XGBoost
ensemble — it captures coarse taste segments (e.g. *MOBA-heavy*,
*single-player RPG*, *casual indie*) that neither ALS nor NCF expose
explicitly through their continuous embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from gamereco.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class KMeansConfig:
    k: int = 16
    max_iter: int = 30
    seed: int = 42
    features_col: str = "features"
    prediction_col: str = "user_cluster"


def _vectorise(factors: DataFrame) -> DataFrame:
    """ALS .userFactors returns (id, features: array<float>). Convert to ml Vector."""
    array_to_vector = F.udf(lambda arr: Vectors.dense(arr), "vector")
    return factors.withColumn("features", array_to_vector(F.col("features")))


def fit_user_kmeans(
    spark: SparkSession,
    als_model: ALSModel,
    config: KMeansConfig = KMeansConfig(),
) -> tuple[KMeansModel, DataFrame]:
    factors = _vectorise(als_model.userFactors)
    kmeans = KMeans(
        k=config.k,
        maxIter=config.max_iter,
        seed=config.seed,
        featuresCol=config.features_col,
        predictionCol=config.prediction_col,
    )
    model = kmeans.fit(factors)
    assignments = model.transform(factors).select(
        F.col("id").alias("user_idx"), F.col(config.prediction_col)
    )
    cost = float(model.summary.trainingCost)
    log.info("kmeans.fit", k=config.k, training_cost=cost)
    return model, assignments


def silhouette_score_proxy(assignments: DataFrame) -> float:
    """A cheap proxy for cluster balance — Shannon entropy of the cohort sizes.

    A real silhouette would require pairwise distances; for the
    cohort-features role we only need to confirm the clustering isn't
    collapsed onto a single bucket.
    """
    counts = (
        assignments.groupBy("user_cluster").count().toPandas()["count"].to_numpy(dtype=np.float64)
    )
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return entropy
