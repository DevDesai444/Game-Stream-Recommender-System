"""SparkSession factory wired for Delta Lake 3.x on Spark 3.5."""

from __future__ import annotations

from pyspark.sql import SparkSession

from gamereco.common.config import spark_settings


def build_spark(app_name: str = "gamereco") -> SparkSession:
    """Construct a Spark 3.5 + Delta Lake session.

    Delta is plugged in via the catalog extension and a custom catalog. The
    rest of the project assumes Delta is the default table format for the
    silver / gold layers.
    """
    cfg = spark_settings()
    builder = (
        SparkSession.builder.appName(app_name)
        .master(cfg.master)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return builder.getOrCreate()
