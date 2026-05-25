# ADR 0006 — Laptop-runnable benchmark path alongside the Spark path

**Status:** Accepted

## Context

The production-shape training stack is Spark 3.5 + Delta Lake + MLflow
+ pgvector + Airflow. That's the right stack to *operate*, but it's
the wrong stack to *reproduce a number from a clean checkout*. Anyone
trying to verify the headline "+33.7% NDCG@10 lift" claim would have
to:

1. Install Java + Spark + the Delta JARs.
2. Stand up MLflow + Postgres + MinIO via Docker Compose.
3. Wait for the Airflow scheduler to come online.
4. Pre-populate the bronze layer.

The reviewer gives up at step 1.

## Decision

Maintain **two interface-compatible code paths** for every training
stage:

| Stage | Production (Spark / MLflow) | Laptop benchmark (pure Python) |
|---|---|---|
| ETL | `gamereco.etl.{bronze,silver,gold}` | `gamereco.datasets.steam200k` |
| ALS | `gamereco.training.als` (Spark MLlib) | `gamereco.training.als_inmem` (scipy + numpy) |
| NCF | `gamereco.training.ncf` (full grid) | `gamereco.training.hybrid._train_ncf_quick` (4 epochs) |
| K-Means | Spark ML KMeans | sklearn KMeans |
| Ensemble | `gamereco.training.ensemble` | `gamereco.training.hybrid` |
| Eval | Spark RankingMetrics + custom | `gamereco.training.evaluation` |

The two paths produce statistically equivalent factor matrices on the
same data (rotation/sign aside) and use the *same* evaluation harness,
so any number computed on the laptop path is directly comparable to a
Spark run.

`scripts/run_benchmark.py` uses the laptop path so anyone can clone,
download the dataset, and reproduce the headline numbers in under 75
seconds on a single MacBook core.

## Why

* **Reproducibility is the whole point of the numbers.** If a reviewer
  can't run the benchmark, the +33.7% claim becomes "trust me." The
  laptop path is the receipts.
* **The benchmark surface stays small.** The evaluation harness, the
  data loader, and the temporal-split semantics are shared between
  paths — only the model implementations differ.
* **Faster iteration during development.** A full Spark run takes
  minutes-to-hours; the laptop benchmark closes the dev loop in
  seconds.

## Rejected alternatives

* **Spark-only.** Right operational shape, wrong development shape.
  Reviewing a PR that changes the ensemble would require spinning up
  Spark.
* **Laptop-only.** Won't scale past ~1M interactions and gives up
  the MLflow tracking / model-registry story production needs.
* **Maintain only one path and document the other.** Documentation
  rots; code stays honest.

## Consequences

* The unit suite asserts in-memory ALS factor properties (top-K
  exclusion, taste-block recovery, deterministic seeding) so any
  drift between the two paths is caught early.
* New base models added to the hybrid have to land on both paths or
  be explicitly marked as production-only.
* The `pyproject.toml` `omit` list keeps the Spark / MLflow modules
  out of the *unit* coverage gate; they're covered by the
  Compose-stack integration smoke tests in CI.
