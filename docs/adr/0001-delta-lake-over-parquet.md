# ADR 0001 — Delta Lake over plain Parquet for the medallion ETL

**Status:** Accepted

## Context

The training data lake holds five logical tables (bronze raw + silver
interactions + silver users + silver games + gold splits). Each one
is rewritten end-to-end every training cycle, and the splits are
sometimes re-issued mid-cycle when a hyperparameter change happens
without re-ingesting. The lake also feeds two downstream consumers
(the training jobs and the embedding-index publisher) that need
consistent reads while a writer is mid-flight.

## Decision

Land every table under `data/delta/{bronze,silver,gold}/` as a
**Delta Lake** table rather than plain Parquet.

## Why

* **Atomic writes.** A failed Spark write to a plain Parquet directory
  leaves a half-written table that breaks the next read. Delta's
  transaction log makes the rewrite atomic — readers either see the
  pre-write snapshot or the new one, never both.
* **`mergeSchema` on schema drift.** New columns (e.g. `playtime_2weeks`
  appeared mid-project) can be added without rewriting historical
  bronze partitions.
* **Time travel for debugging.** `VERSION AS OF` lets us re-run a
  training job against the exact silver snapshot it used last week
  when investigating a metric regression.
* **OPTIMIZE + Z-ORDER.** Bin-packing small files and clustering by
  `user_idx` keeps the silver scan from blowing up after a few weeks of
  daily ingestion.

## Rejected alternatives

* **Plain Parquet.** Cheap, but the atomic-write and schema-evolution
  story has to be hand-rolled. Multiple writers would also race.
* **Apache Iceberg.** Functionally equivalent, but Delta's Spark
  integration is more turnkey and Databricks-friendly (likely
  production target), and the project doesn't need Iceberg's Hive
  catalog story.
* **Postgres as the single source of truth.** Acceptable for the
  serving side but unworkable for 200K → millions of rows of ETL —
  Spark scan throughput would be 10–100× lower.

## Consequences

The `gamereco.etl` modules carry the small overhead of constructing a
SparkSession with `DeltaSparkSessionExtension` and `DeltaCatalog`. In
exchange the rest of the pipeline never has to worry about partial
writes or schema migrations.
