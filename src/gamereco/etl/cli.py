"""CLI entrypoint: ``gamereco-etl``."""

from __future__ import annotations

from pathlib import Path

import click

from gamereco.common.config import spark_settings
from gamereco.common.logging import configure_logging
from gamereco.common.paths import LakePaths
from gamereco.etl.bronze import run_bronze
from gamereco.etl.gold import build_gold
from gamereco.etl.session import build_spark
from gamereco.etl.silver import build_silver
from gamereco.etl.splits import SplitFractions


@click.group()
def cli() -> None:
    """PySpark + Delta Lake ETL CLI."""
    configure_logging()


@cli.command("bronze")
def cmd_bronze() -> None:
    """Land NDJSON ingestion outputs as bronze Delta tables."""
    spark = build_spark("gamereco-bronze")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    counts = run_bronze(spark, lake)
    click.echo(counts)


@cli.command("silver")
def cmd_silver() -> None:
    """Build silver interactions/users/games tables."""
    spark = build_spark("gamereco-silver")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    counts = build_silver(spark, lake)
    click.echo(counts)


@cli.command("gold")
@click.option("--val-frac", default=0.10, type=float)
@click.option("--test-frac", default=0.10, type=float)
def cmd_gold(val_frac: float, test_frac: float) -> None:
    """Produce temporal train/val/test gold tables."""
    spark = build_spark("gamereco-gold")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    counts = build_gold(spark, lake, SplitFractions(val=val_frac, test=test_frac))
    click.echo(counts)


@cli.command("all")
@click.option("--val-frac", default=0.10, type=float)
@click.option("--test-frac", default=0.10, type=float)
def cmd_all(val_frac: float, test_frac: float) -> None:
    """Run bronze + silver + gold end-to-end."""
    spark = build_spark("gamereco-etl-all")
    lake = LakePaths(root=Path(spark_settings().delta_root))
    bronze_counts = run_bronze(spark, lake)
    silver_counts = build_silver(spark, lake)
    gold_counts = build_gold(spark, lake, SplitFractions(val=val_frac, test=test_frac))
    click.echo({"bronze": bronze_counts, "silver": silver_counts, "gold": gold_counts})


def main() -> None:  # pragma: no cover
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
