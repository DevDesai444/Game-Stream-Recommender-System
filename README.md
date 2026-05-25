# Large-Scale Game Recommendation Engine

**Author:** Dev Desai (`DevDesai-444`)

A production-shaped Steam game recommender built on PySpark 3.5 + Delta Lake, PyTorch, MLflow, Apache Airflow, FastAPI, pgvector, and Redis. Every headline number in this README is **measured** against the public [Steam-200k](https://www.kaggle.com/datasets/tamber/steam-video-games) dataset and reproducible from a clean checkout in under two minutes.

---

## TL;DR — measured, not claimed

| Claim | Measured value | Where |
|---|---|---|
| Implicit interactions processed | **200,000 raw → 119,005 collapsed** across 5,217 users / 3,978 games | [`benchmarks/results.md`](benchmarks/results.md) |
| Hybrid NDCG@10 lift over tuned ALS (held-out test split) | **+33.7%** (0.328 vs 0.245) | [`benchmarks/results.md`](benchmarks/results.md) |
| Recall@10 / HitRate@10 lift (test) | **+36.3% / +33.3%** | [`benchmarks/results.md`](benchmarks/results.md) |
| Hyperparameter configs swept during training | **48** (24 ALS × 24 NCF) | `src/gamereco/training/{als,ncf}.py` |
| FastAPI P95 latency, single-worker steady state (16 concurrent) | **57 ms** | [`benchmarks/latency.md`](benchmarks/latency.md) |
| FastAPI P95 latency, single-worker overloaded (50 concurrent) | **197 ms** | [`benchmarks/latency.md`](benchmarks/latency.md) |
| Docker Compose services | **7** (postgres+pgvector, redis, mlflow, minio, spark, airflow, api) | `docker-compose.yml` |
| Pytest unit tests / branch coverage | **151 tests / 82%** | `make test-cov` |
| End-to-end benchmark wall clock | **72 s** on a single laptop core | [`scripts/run_benchmark.py`](scripts/run_benchmark.py) |

> Reproduce:
> ```bash
> make data         # downloads the 200k-row Steam dataset
> make benchmark    # writes benchmarks/results.json + results.md
> make loadtest     # writes benchmarks/latency.json + latency.md
> ```

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Async Steam ingestion (50K+ users target)"]
        S1[Steam Web API] --> A1[aiohttp + tenacity]
        S2[Steam Storefront] --> A1
        A1 --> B1[NDJSON bronze]
    end

    subgraph Lake["PySpark 3.5 + Delta Lake (medallion)"]
        B1 --> BR[Bronze Delta]
        BR --> SI[Silver interactions<br/>+ users + games]
        SI --> GO[Gold: temporal<br/>train / val / test]
    end

    subgraph Models["Hybrid recommender (48-config CV sweep)"]
        GO --> ALS[Spark ALS<br/>24 configs]
        GO --> NCF[PyTorch NeuMF<br/>24 configs]
        ALS --> KM[K-Means cohorts]
        ALS --> XGB[XGBoost ranker<br/>rank:pairwise]
        NCF --> XGB
        KM --> XGB
    end

    subgraph Track["MLflow"]
        ALS --> ML[Experiments<br/>+ registry]
        NCF --> ML
        XGB --> ML
    end

    subgraph Serve["FastAPI service — 57 ms P95"]
        XGB --> PG[(Postgres + pgvector)]
        PG --> CASCADE[personal -> cohort<br/>-> global cascade]
        CASCADE --> API[FastAPI]
        API --> RD[(Redis cache)]
        API --> METRICS[/metrics, /health,<br/>X-Request-ID]
    end

    Air[Airflow<br/>3 DAGs] -.orchestrates.-> Ingest
    Air -.orchestrates.-> Lake
    Air -.orchestrates.-> Models
    Air -.orchestrates.-> Serve
```

---

## What's measured (and how)

### 1. Ranking quality — `benchmarks/results.md`

Seven-axis evaluation harness (NDCG, Recall, MAP, Hit-rate, Coverage, Novelty, Diversity) scored at K=10 against a **per-user temporal holdout** of Steam-200k. Five recommenders compared head-to-head:

| Model | NDCG@10 (val) | Recall@10 | MAP@10 | HitRate@10 | Coverage |
|---|---:|---:|---:|---:|---:|
| Popularity baseline | 0.138 | 0.222 | 0.097 | 0.320 | 0.006 |
| Item co-occurrence | 0.259 | 0.352 | 0.215 | 0.437 | 0.156 |
| Tuned ALS | 0.274 | 0.374 | 0.226 | 0.476 | 0.281 |
| NCF (quick, laptop) | 0.027 | 0.047 | 0.017 | 0.084 | 0.381 |
| **Hybrid ensemble** | **0.386** | **0.488** | **0.328** | **0.613** | 0.227 |

**On the held-out test split — never used for training or tuning — the hybrid posts NDCG@10 = 0.328 vs ALS at 0.245: a +33.7% lift.** Recall@10 climbs 36%, hit rate 33%, MAP 30%. Diversity@10 *also* climbs (0.797 → 0.855), which means the win isn't a popularity collapse — see [`benchmarks/results.md`](benchmarks/results.md) for the full analysis.

### 2. Latency — `benchmarks/latency.md`

In-process uvicorn with stubbed Postgres + Redis, 2,000 concurrent requests across four routes:

| Operating point | P50 | **P95** | P99 | Max |
|---|---:|---:|---:|---:|
| Steady-state (concurrency 16) | 20 ms | **57 ms** | 95 ms | 183 ms |
| Overloaded (concurrency 50) | 30 ms | **197 ms** | 345 ms | 551 ms |

`/recommendations/{user_id}` — the primary surface — is the fastest endpoint by a wide margin because cache hits short-circuit the cold-start cascade.

### 3. Coverage — `make test-cov`

**151 unit tests at 82% branch coverage**, gated in CI at 73%. See `pyproject.toml [tool.coverage.run]` for the omit list (Spark / MLflow / pgvector heavy modules are covered by the Compose-stack integration tests).

---

## Hybrid recommender

The serving model is an **XGBoost `rank:pairwise` ranker** trained with `eval_metric=ndcg@10` and early stopping. It blends six signals per (user, candidate) pair:

| Feature | Source |
|---|---|
| `als_score` | Spark ALS (implicit feedback, log1p confidence) |
| `ncf_score` | PyTorch NeuMF (GMF + MLP towers, shared embeddings) |
| `user_cluster` | K-Means cohort over ALS user factors |
| `cluster_popularity` | Per-cohort × per-game interaction count |
| `log_playtime_user` | Per-user popularity prior |
| `log_global_popularity` | Per-game popularity prior |

The ALS and NCF arms are each tuned over a **24-config grid via CrossValidator** (rank × regParam × alpha × maxIter for ALS; embedding × layers × lr × negative-ratio for NCF) — 48 configurations total. Every run is tracked in MLflow with `ndcg_at_10_lift_vs_als` as a first-class registered metric.

---

## Cold-start cascade

`/recommendations/{user_id}` never 404s on unknown users. The endpoint walks a deterministic cascade and reports which layer answered via `served_from` in the response and an `X-Served-From` header:

```
cache  ──►  personal (Postgres)  ──►  cohort (K-Means top)  ──►  global fallback
```

Plus a `POST /onboard` endpoint for brand-new users: given a few liked appids, it blends the pgvector nearest-neighbours into an instant top-K and warms the cache for the next read. See [ADR 0005](docs/adr/0005-cold-start-cascade-not-404.md) for the rationale.

---

## Data pipeline

### 1. Ingestion (`gamereco.ingestion`)

`asyncio` + `aiohttp` Steam client with bounded-concurrency semaphore and tenacity exponential backoff on 429/5xx. Three CLI subcommands:

```bash
gamereco-ingest discover --pages 250 --target 50000
gamereco-ingest users    --seed data/delta/bronze/users/seed.jsonl
gamereco-ingest games    --limit 20000
```

### 2. ETL (`gamereco.etl`)

PySpark 3.5 + Delta Lake medallion: **bronze** lands raw NDJSON with `mergeSchema=true`; **silver** explodes owned-games arrays into one row per `(user, game)` with `confidence = log1p(playtime_minutes)`, compact integer indices, and an activity floor (≥3 interactions/user); **gold** runs a per-user **temporal split** into train / val / test (the only correct setup for ranking metrics — see [ADR 0003](docs/adr/0003-temporal-split-not-random.md)).

```bash
gamereco-etl all --val-frac 0.10 --test-frac 0.10
```

### 3. Training (`gamereco.training`)

```bash
gamereco-train als        # 24-config Spark CrossValidator
gamereco-train ncf        # 24-config PyTorch grid
gamereco-train kmeans     # cohort labels over ALS user factors
gamereco-train ensemble   # XGBoost rank:pairwise blender
```

Every run logs params / metrics / artifacts to **MLflow** and registers the best model under `gamereco-als`, `gamereco-ncf`, `gamereco-xgb-ensemble`.

A laptop-runnable mirror of the same pipeline lives in `gamereco.datasets.steam200k` + `gamereco.training.als_inmem` + `gamereco.training.hybrid` — interface-compatible with the Spark path, fits the same factor matrices, and is what `scripts/run_benchmark.py` uses to produce the headline numbers in 72 s ([ADR 0006](docs/adr/0006-laptop-runnable-benchmark-path.md)).

---

## Serving — 57 ms P95, observable end-to-end

```bash
docker compose up -d
curl http://localhost:8000/recommendations/76561198000000000?limit=10
```

| Endpoint | What it serves |
|---|---|
| `GET /recommendations/{user_id}` | Personal → cohort → global cascade |
| `GET /similar/{steam_appid}` | pgvector cosine search over the 64-D embedding index |
| `POST /onboard` | Blend pgvector neighbours of a seed list of liked appids |
| `GET /global` | Catalog-wide top, the cold-start floor |
| `GET /health` | Liveness probe (also pings Redis) |
| `GET /metrics` | Prometheus scrape format |

**Observability is baked in:**

* Every request gets an `X-Request-ID` (UUID4 if the client didn't supply one), echoed in the response and bound into the structlog context.
* One JSON access-log line per request with method, route, status, latency, and `served_from`.
* Prometheus `/metrics` exposes `gamereco_requests_total{method,route,status}`, `gamereco_request_latency_seconds` histogram (with a 185 ms bucket sized to the project's P95 target), and `gamereco_recs_served_from_total{served_from}` so cache vs personal vs cohort vs global hit rate is queryable.

---

## Orchestration — three Airflow DAGs

| DAG | Schedule | What |
|---|---|---|
| `gamereco_ingestion_daily` | `0 3 * * *` | async user + game ingestion |
| `gamereco_training_weekly` | `0 4 * * 0` | bronze → silver → gold → ALS, NCF, KMeans → XGBoost |
| `gamereco_serving_refresh` | `30 5 * * *` | publish pgvector embeddings + warm Redis cache |

All tasks shell out to the gamereco CLI so DAGs are portable to `KubernetesPodOperator` without rewrites.

---

## Docker Compose — 7 services

```bash
cp .env.example .env
docker compose up -d
```

| # | Service | Role |
|---|---|---|
| 1 | `postgres` | `pgvector/pgvector:pg16` — recs + game catalog + embeddings + cohorts |
| 2 | `redis` | Read-through cache with TTL |
| 3 | `mlflow` | Tracking server + model registry, Postgres-backed |
| 4 | `minio` | S3-compatible artifact store (Delta + MLflow artifacts) |
| 5 | `spark` | Spark 3.5 runtime image |
| 6 | `airflow` | LocalExecutor + DAGs mounted in |
| 7 | `api` | FastAPI recommendation service |

The Postgres init script (`infra/postgres/init.sql`) provisions `pgvector`, the schema (games, embeddings, user_recommendations, user_cohorts, cohort_top), and an `ivfflat` cosine index on the embedding column. See [ADR 0004](docs/adr/0004-pgvector-not-separate-vector-db.md) for why we keep vectors in Postgres instead of a separate vector DB.

---

## Demo

```bash
docker compose up -d api postgres redis
streamlit run demo/streamlit_app.py
```

A four-tab Streamlit page that hits the live API: personalised recs (with a colour-coded `served_from` badge), pgvector similarity search, brand-new-user onboarding, and the global top.

---

## Repository layout

```text
.
├── airflow/dags/                  # 3 Airflow DAGs (ingest / train / serve refresh)
├── benchmarks/                    # MEASURED RESULTS (results.md, latency.md, JSON snapshots)
├── demo/streamlit_app.py          # Streamlit interactive demo
├── docker-compose.yml             # 7-service stack
├── docs/adr/                      # Architecture decision records
├── infra/
│   ├── docker/                    # Dockerfile.api, .spark, .airflow
│   └── postgres/init.sql          # pgvector schema bootstrap
├── Makefile                       # canonical task runner
├── pyproject.toml                 # gamereco package + console scripts
├── scripts/
│   ├── download_dataset.sh        # fetch Steam-200k from a public mirror
│   ├── run_benchmark.py           # produces benchmarks/results.json
│   └── run_loadtest.py            # produces benchmarks/latency.json
├── src/gamereco/
│   ├── common/                    # config, logging, paths, pydantic schemas
│   ├── datasets/                  # Steam-200k loader (pure pandas)
│   ├── ingestion/                 # async aiohttp Steam client + pipeline + CLI
│   ├── etl/                       # Spark bronze → silver → gold + temporal split
│   ├── training/                  # ALS (Spark + in-memory), NCF, KMeans, ensemble,
│   │                              # baselines, evaluation harness, hybrid harness
│   └── serving/                   # FastAPI, pgvector store, Redis cache, cold-start
│                                  # cascade, observability, embedding publisher
└── tests/unit/                    # 151 unit tests, 82% branch coverage
```

---

## End-to-end run (real data, real metrics)

```bash
# 1. Local infra (optional — skip for the laptop-only benchmark)
docker compose up -d postgres redis mlflow minio

# 2. Download the 200K Steam interactions dataset
make data
# -> data/raw/steam-200k.csv  (8.5 MB)

# 3. Run the benchmark — produces measured NDCG@10 lift in 72 seconds
make benchmark
# -> benchmarks/results.json + results.md
#    Hybrid NDCG@10 = 0.328 (test)   vs ALS = 0.245   ->   +33.7% lift

# 4. Run the latency benchmark — produces P95 numbers
make loadtest
# -> benchmarks/latency.json + latency.md
#    P95 @ concurrency=16: 57 ms

# 5. Boot the demo
docker compose up -d api
make demo
# -> http://localhost:8501
```

---

## Architecture decisions

Six ADRs document the load-bearing choices:

* [ADR 0001 — Delta Lake over plain Parquet for the medallion ETL](docs/adr/0001-delta-lake-over-parquet.md)
* [ADR 0002 — Hybrid XGBoost ranker over a single recsys model](docs/adr/0002-hybrid-ranker-not-single-model.md)
* [ADR 0003 — Per-user temporal split over random holdout](docs/adr/0003-temporal-split-not-random.md)
* [ADR 0004 — pgvector inside Postgres over a separate vector DB](docs/adr/0004-pgvector-not-separate-vector-db.md)
* [ADR 0005 — Personal → cohort → global cascade over 404 on missing users](docs/adr/0005-cold-start-cascade-not-404.md)
* [ADR 0006 — Laptop-runnable benchmark path alongside the Spark path](docs/adr/0006-laptop-runnable-benchmark-path.md)

---

## License

No license file is currently included. Add one before public redistribution or external reuse.
