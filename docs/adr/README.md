# Architecture Decision Records

Each file in this directory captures one consequential design choice
the project made — what was decided, why, and what was rejected.
Numbering is monotonic; an ADR is never edited in place, it's
superseded by a new file that links back to the original.

| # | Title | Status |
|---|---|---|
| [0001](0001-delta-lake-over-parquet.md) | Delta Lake over plain Parquet for the medallion ETL | Accepted |
| [0002](0002-hybrid-ranker-not-single-model.md) | Hybrid XGBoost ranker over a single recsys model | Accepted |
| [0003](0003-temporal-split-not-random.md) | Per-user temporal split over random holdout | Accepted |
| [0004](0004-pgvector-not-separate-vector-db.md) | pgvector inside Postgres over a separate vector DB | Accepted |
| [0005](0005-cold-start-cascade-not-404.md) | Personal → cohort → global cascade over 404 on missing users | Accepted |
| [0006](0006-laptop-runnable-benchmark-path.md) | Laptop-runnable benchmark path alongside the Spark path | Accepted |
| [0007](0007-ucsd-dataset-and-two-tower.md) | UCSD Steam dataset and a content-aware two-tower NCF | Accepted |
