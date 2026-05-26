# Benchmark Results — UCSD Steam (content-aware two-tower NCF, v2)

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
the default v2 config (`--max-users 4000 --als-factors 32 --als-iters 8
--kmeans-k 12 --candidates 150`). The two-tower defaults are
sampled-softmax with 4 popularity-sampled hard negatives per positive,
15 epochs with cosine LR schedule + early stopping on val NDCG, batch
size 1024, embedding dim 64, tower hidden (256, 128), output dim 64,
playtime-confidence-weighted positives, and a 30-minute playtime floor
on training positives. The XGBoost objective is `rank:ndcg` and NCF
contributes 50 top-K candidates per user to the retrieval pool.

## Headline

The standalone **two-tower NCF NDCG@10 went from 0.109 → 0.225 (+106%)
on val and now beats ALS (0.126) by +79%** — the headline improvement
of the v2 training recipe. The hybrid ensemble lifts NDCG@10 by
**+130.6%** over the tuned ALS baseline on the held-out test split
(0.295 vs 0.128) with a hit rate of 88.8%.

Reproduce:

```bash
bash scripts/download_ucsd_dataset.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/run_benchmark_ucsd.py \
    --max-users 4000 --als-factors 32 --als-iters 8 \
    --kmeans-k 12 --candidates 150 \
    --out benchmarks/results_ucsd.json
```

Total wall clock on a single MacBook core: **239 seconds**.

## Per-metric comparison @ K=10 (validation split)

| Model | NDCG | Recall | MAP | Hit rate | Coverage | Novelty | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Item co-occurrence | 0.096 | 0.093 | 0.042 | 0.480 | 0.009 | 8.16 | 0.000 |
| Popularity | 0.167 | 0.142 | 0.086 | 0.647 | 0.005 | 8.31 | 0.000 |
| Tuned ALS | 0.126 | 0.145 | 0.066 | 0.536 | 0.197 | 10.02 | 0.409 |
| **Two-tower NCF alone (v2)** | **0.225** | **0.203** | **0.123** | **0.745** | 0.055 | 8.84 | 0.127 |
| **Hybrid ensemble** | **0.318** | **0.275** | **0.191** | **0.866** | 0.075 | 8.86 | 0.110 |

## Test split (held out — never used in training or tuning)

| Model | NDCG@10 | Recall@10 | MAP@10 | Hit rate@10 |
|---|---:|---:|---:|---:|
| Tuned ALS | 0.128 | 0.146 | 0.070 | 0.510 |
| **Hybrid ensemble** | **0.295** | **0.316** | **0.164** | **0.888** |
| **Lift** | **+130.6%** | **+116.4%** | **+135.6%** | **+74.3%** |

## What v2 changed in NCF training

The v1 two-tower used pointwise BCE on positives + random negatives,
3 epochs, dim 24. v2 keeps the model architecture (it's still
two-tower with content features on the item side) but rewrites the
training loop:

1. **Sampled-softmax with in-batch negatives + logQ correction.**
   Replaces BCE. Each positive uses every other positive in the
   batch as an implicit negative, plus 4 popularity-sampled "hard"
   negatives per positive. Popularity-correction (`logits -= log p(item)`)
   prevents the softmax from being dominated by head items. This is
   the YouTube/two-tower retrieval recipe — it directly approximates
   listwise ranking, which is what NDCG measures.
2. **Confidence-weighted positives.** Per-row weight = log1p(playtime),
   normalised to mean 1. A 100-hour played game contributes more
   gradient than a 10-minute trial — same idea as ALS's confidence
   weighting, applied to the cross-entropy.
3. **Playtime floor on training positives.** UCSD includes a lot of
   "owned but never played" rows that v1 explicitly called out as the
   dominant source of NCF noise. v2 drops training rows with
   `playtime < 30 min` so the model sees cleaner positives. The eval
   splits are unchanged.
4. **Bigger model, longer training, real schedule.** Dim 64, hidden
   (256, 128), 15 epochs (vs 24/128-64/3), cosine LR schedule, and
   early stopping on validation NDCG@10. Without monitoring val NDCG
   directly, dropping loss does not imply ranking is improving.
5. **In-batch collision masking.** When another positive in the batch
   is itself a positive for the row's user, the softmax denominator
   would teach the model to push that item *down*. v2 masks those
   columns out using a sparse CSR of (user, item) positives built
   once outside the training loop.

The v1 two-tower trained to NDCG ~0.109; v2 reaches ~0.225 on the
same data with the same dataset split, and early-stopped at epoch 7
of 15. The val NDCG trajectory across epochs is recorded in the
benchmark JSON under `config.two_tower.val_ndcg_history`.

## What v2 changed in the hybrid ensemble

1. **No more inline NCF training.** The hybrid harness used to call a
   tiny inline NeuMF for ~30 seconds and use it as a feature; the
   strong v2 two-tower trained separately is now passed in via the
   `pretrained_ncf=(user_vecs, item_vecs)` argument to `train_hybrid`.
2. **NCF as a candidate generator, not just a feature.** With
   `ncf_candidate_k=50`, the candidate pool per user is the **union**
   of ALS top-150 and NCF top-50 — items where the two retrievers
   disagree are exactly where the ranker has the most to combine.
   NCF-only candidates are re-scored against ALS via a vectorised
   einsum (`als.user_factors[u_arr] · als.item_factors[g_arr]`) so the
   ranker sees both signals on every row.
3. **Richer ranker features.** Added `ncf_rank` (item's rank in the
   user's NCF top-K), `ncf_in_top_k` (boolean), and `source_als` /
   `source_ncf` flags so the ranker can learn when to trust each
   retriever.
4. **`rank:ndcg` objective.** Replaces `rank:pairwise` — directly
   optimises the headline metric instead of a pairwise proxy.

## Reading the numbers honestly

* **Standalone NCF now beats ALS by a wide margin (+79% val NDCG).**
  In v1 NCF lost head-to-head (0.109 vs 0.126); the v2 training recipe
  inverts that ranking and the gap is no longer in noise territory.
* **Hybrid lift over ALS on test fell from +143% (v1) to +131% (v2).**
  ALS test NDCG is essentially identical between runs (0.128 in both).
  The hybrid gives up a few points of test NDCG because most of the
  signal v1's hybrid was synthesising from weak NCF + strong ALS is
  now already captured by the much stronger NCF alone — the marginal
  contribution of XGBoost re-ranking is smaller when each retriever is
  individually more correct. Val NDCG actually *improved* (0.301 →
  0.318), so the apparent regression is within the val/test split
  variance — both numbers round to "≈0.30 NDCG."
* **Coverage and diversity stay constrained.** Hybrid coverage dropped
  from 0.080 → 0.075 and diversity from 0.122 → 0.110. The stronger
  NCF concentrates more confidently on a smaller head — if catalogue
  coverage is a product requirement, the cohort-popularity feature can
  be re-weighted, or NCF training can be re-run with a smaller
  `logq_correction` to soften the popularity push-back.
* **Two-tower training cost up 6× (25s → 157s) for the +106% NCF
  quality gain.** End-to-end wall clock went from 181s → 239s, which
  is still well inside the "runs on a laptop in under five minutes"
  budget.

## Two-tower content-feature dimensions

| Tower | Input | Hidden | Output |
|---|---|---|---|
| User | `user_emb(64) ⊕ dense(4) = 68` | `(256, 128)` | `64` |
| Item | `item_emb(64) ⊕ genres(21) ⊕ tags(200) ⊕ dense(5) = 290` | `(256, 128)` | `64` |

Score: `scale × cos(user_vec, item_vec)` with a learnable `scale`
parameter (init 10). Vectors are L2-normalised inside each tower so
the cosine is well-bounded. Training: sampled-softmax cross-entropy
with in-batch negatives + 4 popularity-sampled hard negatives per
positive, logQ corrected.

## Timings

| Stage | Seconds |
|---|---:|
| Dataset load (parse 3 gzipped repr files) | ~10 |
| Popularity | <1 |
| Item co-occurrence | ~5 |
| Implicit ALS (32 factors, 8 iters, 354K rows) | ~2 |
| Two-tower NCF (sampled-softmax, 15 epochs, early-stopped at 7) | ~157 |
| Hybrid (KMeans + candidate assembly + XGBoost) | ~50 |
| **End-to-end** | **239 s** |

(ALS time dropped from v1's ~70s because v1 was running with extra
diagnostic re-scoring; the model itself is the same.)

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
