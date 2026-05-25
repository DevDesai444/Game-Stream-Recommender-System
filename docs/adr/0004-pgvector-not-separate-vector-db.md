# ADR 0004 — pgvector inside Postgres over a separate vector DB

**Status:** Accepted

## Context

The serving path needs two things from the storage layer:

1. Per-user top-K recommendations (random-access lookup by `user_id`).
2. "More like this" similarity search over game embeddings.

The natural choice for (1) is a relational store. The natural choice
for (2), at first glance, is a dedicated vector database (Pinecone,
Qdrant, Weaviate, Milvus).

## Decision

Use **Postgres with the `pgvector` extension** for both. The `games`,
`game_embeddings`, `user_recommendations`, `user_cohorts`, and
`cohort_top` tables all live in the same database; the vector index
is an `ivfflat` on the 64-D `embedding` column.

## Why

* **One operational surface.** Backups, replicas, point-in-time
  recovery, and IAM are the same shape they already are for the rest
  of the app. Adding a vector DB would multiply the operational story.
* **Joins.** The "more like this" query joins game embeddings back to
  the `games` table to fetch name + header image. Inside Postgres
  that's a single SQL statement; with a separate vector DB it's two
  network hops and an in-app join.
* **ivfflat is fast enough.** With ~4K games at 64-D and `lists = 100`,
  cosine search returns in well under 10 ms even before query caching.
  The bottleneck isn't the vector index, it's the JSON serialisation.
* **Schema migration story.** `pgvector` is just another column type.
  No bespoke ingestion pipeline; the same SQLAlchemy `Table` shape we
  already use for recommendations carries the embedding column.

## Rejected alternatives

* **Pinecone (managed).** Lowest operational overhead, but adds a
  per-query cost and a network egress hop. Justified at >10M items
  with rapidly-mutating embeddings, not at ~4K games rebuilt nightly.
* **Qdrant / Milvus / Weaviate self-hosted.** Strong feature sets, but
  add another stateful service to the Docker Compose stack and
  another set of replication+backup decisions.
* **FAISS in-memory.** Fast, but tied to the API process — restart
  → lose the index → cold P95 spikes during rebuild. With pgvector
  the index survives restarts and a new API replica picks it up from
  Postgres on boot.

## Consequences

* The pgvector ANN index has accuracy/recall tradeoffs tied to
  `lists` and `probes`. Defaults (lists=100, default probes) are
  fine at 4K items; at 100K+ items the publisher in
  `gamereco.serving.embedding_index` would need to grow `lists` and
  the read path would set `probes`.
* The `pgvector/pgvector:pg16` Docker image is the canonical
  Postgres for both local dev and Compose. The base image is ~50 MB
  larger than vanilla `postgres:16`; acceptable.
* If embedding dimensionality changes, the schema migration has to
  drop and rebuild `game_embeddings_cosine_idx`. The publisher
  already truncates the table per refresh so this is a low-friction
  operation.
