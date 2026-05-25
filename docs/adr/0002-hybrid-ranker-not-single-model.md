# ADR 0002 — Hybrid XGBoost ranker over a single recsys model

**Status:** Accepted

## Context

The classic "train ALS, serve top-K" architecture is the simpler
deployment, and ALS does score well on the implicit-feedback dataset
the project targets (`benchmarks/results.md` measures it at NDCG@10 =
0.245 on the held-out test split). But ALS has known failure modes:

* It can't see content (a brand-new game with no interactions has no
  factors).
* It collapses long-tail items into the top of the catalog because
  the rare-item factor estimates have high variance.
* It can't combine signals from outside the (user, item, rating)
  triple — playtime curves, cohort membership, popularity priors.

## Decision

Layer an **XGBoost `rank:pairwise` ranker** on top of ALS top-N
candidates. The ranker consumes six features per (user, candidate)
pair:

```
[als_score, ncf_score, user_cluster, cluster_popularity,
 log_playtime_user, log_global_popularity]
```

The XGBoost output replaces ALS as the served score. Other models
(NCF, K-Means cohorts) exist only as feature inputs.

## Why

Measured on the held-out test split of Steam-200k:

| Model | NDCG@10 |
|---|---:|
| Tuned ALS | 0.245 |
| **Hybrid (ALS + NCF + cohort + popularity, XGBoost ranked)** | **0.328** |
| Lift | **+33.7%** |

The hybrid wins at every operating point — Recall@10 climbs 36%, hit
rate 33%, MAP@10 30%. Diversity@10 *also* climbs (0.80 → 0.86), so
the win isn't from collapsing onto popular titles.

The ensemble form also lets us drop individual signals (e.g. disable
NCF, swap K-Means for HDBSCAN) without re-architecting the serving
path — the ranker just sees a different feature vector.

## Rejected alternatives

* **Pure NCF.** Strong on dense subsets of MovieLens-shape datasets,
  but the implicit Steam dataset is heavy-tail and a short NCF run
  underperforms ALS. A *very long* NCF run might catch up, but the
  hybrid wins anyway and is faster to retrain.
* **Two-tower retrieval (e.g. SASRec).** Production-grade, but the
  inference path needs a vector DB hop *and* a cross-encoder for
  re-ranking; the hybrid hits NDCG parity with one Postgres lookup.
* **LightGBM.** Functionally equivalent ranker. We use XGBoost because
  its `rank:pairwise` objective has been more stable in our experience
  on small-to-mid datasets.

## Consequences

* The serving path has to materialise candidates *before* the ranker
  scores them — ALS top-200 per user, written into the
  `user_recommendations` table after every training cycle.
* The feature pipeline has to stay in sync across train and serve.
  The same `assemble_candidates` function in `gamereco.training.hybrid`
  is the single source of truth.
* MLflow tracks `ndcg_at_10_lift_vs_als` as a first-class metric so
  the lift is part of the model registry contract.
