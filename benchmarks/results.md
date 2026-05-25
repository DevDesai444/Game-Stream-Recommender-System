# Benchmark Results — Steam-200k

**Dataset:** Tamber's [Steam-200k](https://www.kaggle.com/datasets/tamber/steam-video-games) — 200,000 real Steam user-game interactions (purchase + play events) collected from public Steam profiles. After collapsing `purchase + play` rows per `(user, game)` and dropping users with fewer than three interactions, the working set is:

| | Count |
|---|---:|
| Raw rows | 200,000 |
| Silver interactions (deduped) | 119,005 |
| Training rows | 92,245 |
| Validation rows | 13,380 |
| Test rows | 13,380 |
| Users | 5,217 |
| Games | 3,978 |

Splits are **per-user temporal holdouts** (most-recent N% per user reserved for val + test, with cumulative `play_hours` as the temporal axis — Steam-200k carries no event timestamps). All numbers below are produced by `scripts/run_benchmark.py` and are reproducible from this commit.

## Headline

> **The hybrid ALS + NCF + KMeans + XGBoost ensemble lifts NDCG@10 by +33.7% over the tuned ALS baseline on the held-out test split** (0.328 vs 0.245), and by +40.8% on validation.

Reproduce:

```bash
bash scripts/download_dataset.sh
PYTHONPATH=src python scripts/run_benchmark.py \
  --als-factors 64 --als-iters 12 --alpha 20 --reg 0.05 \
  --kmeans-k 16 --candidates 200 --ncf-epochs 4 \
  --out benchmarks/results.json
```

Total wall clock on a single MacBook core: **72 seconds**.

## Per-metric comparison @ K=10 (validation split)

| Model | NDCG | Recall | MAP | Hit rate | Coverage | Novelty | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity baseline | 0.138 | 0.222 | 0.097 | 0.320 | 0.006 | 7.71 | 0.000 |
| Item co-occurrence | 0.259 | 0.352 | 0.215 | 0.437 | 0.156 | 7.91 | 0.000 |
| Tuned ALS | 0.274 | 0.374 | 0.226 | 0.476 | 0.281 | 9.08 | 0.797 |
| NCF (3-epoch, 16-dim, laptop) | 0.027 | 0.047 | 0.017 | 0.084 | 0.381 | 10.15 | 0.202 |
| **Hybrid ensemble** | **0.386** | **0.488** | **0.328** | **0.613** | 0.227 | 8.56 | 0.855 |

## Test split (held out — never used in training, never used for tuning)

| Model | NDCG@10 | Recall@10 | MAP@10 | Hit rate@10 |
|---|---:|---:|---:|---:|
| Tuned ALS | 0.245 | 0.362 | 0.191 | 0.468 |
| **Hybrid ensemble** | **0.328** | **0.494** | **0.247** | **0.624** |
| **Lift** | **+33.7%** | **+36.3%** | **+29.6%** | **+33.3%** |

## What the numbers say

* **Hybrid > ALS at every operating point on test, not just NDCG.** Recall climbs 36%, hit rate climbs 33%, MAP climbs 30%. The ranker isn't gaming a single metric — the underlying retrieval quality is genuinely better.
* **The popularity baseline is what you have to beat.** It posts NDCG@10 = 0.138 (recommending the same global top-10 to everyone), which means **the hybrid is ~2.8× better than a non-personalised system**. Co-occurrence is a stronger anchor (0.259) and ALS edges past it; the hybrid pulls clearly ahead of both.
* **NCF on its own is weak here — and that's the right reason to keep it.** With 4 epochs and 16-dim embeddings it doesn't beat ALS, but as a *feature* inside the XGBoost ranker it still carries non-trivial gain (`feature_importance` in the run log shows NCF score among the top-3 splits). This is the whole point of the ensemble — different models cover different failure modes.
* **Coverage drops from 0.281 (ALS) to 0.227 (hybrid).** The ranker focuses recommendations on the head of the catalog where the signal is strongest. Diversity goes up (0.797 → 0.855), so the head it focuses on isn't collapsed onto a single niche — but if catalog coverage matters for the product, the K-Means cohort feature can be re-weighted to push the ranker toward longer-tail items.
* **Novelty drops slightly (9.08 → 8.56).** Same story: the ranker prefers slightly more popular items because they pay off more in NDCG. Acceptable trade for +33.7%.

## Timings

| Stage | Seconds |
|---|---:|
| Popularity | 0.2 |
| Item co-occurrence (binarised) | 1.0 |
| In-memory ALS (64 factors, 12 iters, 92K rows) | 19.9 |
| Hybrid (ALS + NCF + KMeans + XGBoost + candidate assembly) | 38.7 |
| **End-to-end** | **72.0** |

The benchmark deliberately uses the laptop-runnable code path
(`gamereco.training.als_inmem` + `gamereco.training.hybrid`) so anyone
can reproduce it without Spark / MLflow / pgvector. The Spark training
modules in `gamereco.training.als` / `gamereco.training.ncf` are
interface-compatible and run the same models at production scale.

## What's not measured here

* **Cold users / cold items.** The activity floor (`min_interactions=3`)
  filters out the tail. The fallback chain in `gamereco.serving.api`
  exercises that path separately.
* **Latency.** Served-side P95 is measured by `benchmarks/loadtest.py`
  (Locust) and lives in `benchmarks/latency.md`.
* **Spark parity.** The in-memory ALS and Spark ALS agree to within
  rotation/sign on the same inputs (verified via the unit suite); the
  hybrid metrics above are computed on the laptop path.
