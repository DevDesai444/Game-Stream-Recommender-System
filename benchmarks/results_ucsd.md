# Benchmark Results — UCSD Steam (real two-tower NCF)

Dataset: Julian McAuley's UCSD Steam release. Three source files
(steam_games metadata, australian_users_items, australian_user_reviews)
joined into a single interaction table with content metadata on both
sides of the (user, item) graph. See ADR 0007 for the migration
rationale.

| | Count |
|---|---:|
| Users (capped for laptop reproducibility) | 3,946 |
| Games | 7,208 |
| Interactions | 442,170 |
| Reviews (with recommend label) | 4,913 |
| Training rows | 353,684 |
| Validation rows | 44,243 |
| Test rows | 44,243 |
| Item-tower genre vocab | 21 |
| Item-tower tag vocab (top-N capped) | 200 |

Per-user temporal holdout using cumulative `playtime_forever` as the
time proxy. All numbers come from `scripts/run_benchmark_ucsd.py` on
the configuration `--max-users 4000 --als-factors 32 --als-iters 8
--kmeans-k 12 --candidates 150 --two-tower-epochs 3
--two-tower-embedding-dim 24 --two-tower-output-dim 24`.

## Headline

The hybrid ranker (ALS + two-tower NCF + KMeans cohort + XGBoost) lifts
NDCG@10 by **+142.7%** over the tuned ALS baseline on the held-out
test split (0.311 vs 0.128). Hit rate goes from 51% to 88%.

Reproduce:

```bash
bash scripts/download_ucsd_dataset.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/run_benchmark_ucsd.py \
    --max-users 4000 --als-factors 32 --als-iters 8 \
    --kmeans-k 12 --candidates 150 \
    --two-tower-epochs 3 \
    --out benchmarks/results_ucsd.json
```

Total wall clock on a single MacBook core: **181 seconds**.

## Per-metric comparison @ K=10 (validation split)

| Model | NDCG | Recall | MAP | Hit rate | Coverage | Novelty | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Item co-occurrence | 0.096 | 0.093 | 0.042 | 0.480 | 0.009 | 8.16 | 0.000 |
| Popularity | 0.167 | 0.142 | 0.086 | 0.647 | 0.005 | 8.31 | 0.000 |
| Two-tower NCF alone | 0.109 | 0.102 | 0.052 | 0.499 | 0.029 | 8.30 | 0.036 |
| Tuned ALS | 0.126 | 0.145 | 0.066 | 0.536 | 0.197 | 10.02 | 0.409 |
| **Hybrid ensemble** | **0.301** | **0.291** | **0.174** | **0.879** | 0.080 | 8.92 | 0.122 |

## Test split (held out — never used in training or tuning)

| Model | NDCG@10 | Recall@10 | MAP@10 | Hit rate@10 |
|---|---:|---:|---:|---:|
| Tuned ALS | 0.128 | 0.146 | 0.070 | 0.510 |
| **Hybrid ensemble** | **0.311** | **0.313** | **0.178** | **0.883** |
| **Lift** | **+142.7%** | **+114.5%** | **+156.4%** | **+73.2%** |

## Reading the numbers honestly

* **UCSD is a harder dataset than Steam-200k.** The popularity
  baseline posts 0.167 NDCG here vs 0.138 on Steam-200k, but ALS
  alone is *lower* (0.128 vs 0.245). The reason is the catalogue is
  ~2× larger (7,208 vs 3,978 games) and the implicit signal is
  noisier — UCSD includes a lot of "owned but never played" rows
  that drag the ALS confidence values down.
* **The hybrid wins by a larger margin precisely because the base
  signals disagree more.** When popularity, ALS, and two-tower all
  carry partial information, the XGBoost ranker has more to combine.
  That's why the lift is +143% here vs +34% on the simpler dataset.
* **Two-tower NCF alone (NDCG = 0.109) underperforms ALS (0.128).**
  This is fine for an ensemble component — the towers contribute
  complementary content signal, not absolute ranking quality. Their
  output is high-quality as a *feature*, which is why hybrid hit
  rate jumps to 88%.
* **Coverage drops to 0.080.** The ranker is confidently leaning on
  the head of the catalogue. If catalogue coverage is a product
  requirement, the cohort-popularity feature can be re-weighted, or
  the XGBoost objective can switch to `rank:ndcg` with a diversity
  penalty — neither is implemented here.
* **Diversity also drops (0.41 ALS → 0.12 Hybrid).** Same root
  cause. The L2-normalised two-tower vectors land in a narrow region
  of the embedding space when training is short (3 epochs); a longer
  NCF run widens the spread but doesn't change the headline lift.

## Two-tower content-feature dimensions

| Tower | Input | Hidden | Output |
|---|---|---|---|
| User | `user_emb(24) ⊕ dense(4) = 28` | `(64, 32)` | `24` |
| Item | `item_emb(24) ⊕ genres(21) ⊕ tags(200) ⊕ dense(5) = 250` | `(64, 32)` | `24` |

Score: `10 × cos(user_vec, item_vec)`. Trained with
`BCEWithLogitsLoss` on observed positives + 3 sampled negatives per
positive. Vectors are L2-normalised inside each tower so the cosine
is well-bounded.

## Timings

| Stage | Seconds |
|---|---:|
| Dataset load (parse 3 gzipped repr files) | ~12 |
| Popularity | <1 |
| Item co-occurrence | ~1 |
| Implicit ALS (32 factors, 8 iters, 354K rows) | ~70 |
| Two-tower NCF (3 epochs) | ~25 |
| Hybrid (KMeans + candidate assembly + XGBoost) | ~60 |
| **End-to-end** | **181 s** |

## What's not measured here

* Real Spark ALS / Spark KMeans run on the full ~88K-user UCSD set
  — that lives behind the production Airflow DAG, not the laptop
  benchmark.
* User-side cold-start. The two-tower's user tower input is dense
  user features (items_count, total_playtime, etc.) which all
  require at least *some* history. The cohort and global fallback
  layers in `gamereco.serving.coldstart` cover the brand-new-user
  case at serve time.
* Reviews as supervision. The 4.9K explicit-recommend labels in the
  UCSD reviews file are loaded by the dataset module but not yet
  consumed by the ranker — future work.
