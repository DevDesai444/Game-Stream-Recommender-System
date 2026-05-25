# ADR 0007 — UCSD Steam dataset and a content-aware two-tower NCF

**Status:** Accepted

## Context

The original benchmark dataset was Tamber's Steam-200k Kaggle dump:
four columns (`user_id`, `game_name`, `behavior`, `value`) and nothing
else. That's enough for a collaborative-only NeuMF that just learns
embeddings from `user_idx × item_idx`, but it cannot support a real
**two-tower** model because there are no side features to feed the
towers — no genres, no tags, no price, no release date, no user
profile.

ADR 0002 picked the hybrid XGBoost ranker over a single recsys model,
and the rationale there assumed the NCF arm would eventually become
content-aware. With Steam-200k that promise was empty.

## Decision

Switch the canonical benchmark dataset to **Julian McAuley's UCSD
Steam dataset** (the version with `steam_games.json.gz`,
`australian_users_items.json.gz`, and `australian_user_reviews.json.gz`)
and build a content-aware two-tower NeuMF that consumes the metadata
the new dataset carries:

* **Item tower**  ::  `[item_embedding ⊕ multi_hot_genres ⊕ multi_hot_tags ⊕ dense_features]`
                       → MLP → L2-normalised item vector
* **User tower**  ::  `[user_embedding ⊕ dense_user_features]`
                       → MLP → L2-normalised user vector
* **Score**       ::  `scale × cosine(user_vector, item_vector)`

Trained with `BCEWithLogitsLoss` on observed positives + sampled
negatives. The Steam-200k loader (`gamereco.datasets.steam200k`) is
kept around as a regression target for the simpler benchmark.

## Why

1. **A two-tower is only meaningful with side features.** On
   Steam-200k both towers would collapse to id-lookup embeddings,
   which is just NeuMF. UCSD provides ~20 genres, ~200 tags, prices,
   release years, developer/publisher strings, plus per-user
   `items_count`, `total_playtime`, `active_recent`, and
   `reviews_count`. The item tower output now generalises to brand-new
   items (their content features survive without any interactions).
2. **It enables real cold-item handling.** ADR 0005's cold-start
   cascade can now actually serve a never-played game with content
   features alone — the pgvector index publisher consumes the
   two-tower item vectors instead of just ALS factors.
3. **Bigger and more realistic dataset.** UCSD has ~88K users, ~30K
   games, and several million interactions. Capping to 10K users for
   the laptop benchmark still yields ~10K users × ~10K games × ~1.1M
   interactions — roughly 10× the Steam-200k working set.
4. **Reviews carry recommend labels.** The reviews file ties an
   explicit "would-recommend" Y/N signal to ~25K user-game pairs.
   That's a richer ground-truth supervisor than implicit ownership
   and is available as future work for re-ranker training.

## Rejected alternatives

* **Stay on Steam-200k.** Faster to load, but two-tower architecture
  has nothing to learn from.
* **Crawl real Steam users via the Web API for side features.** Works
  in principle (the `gamereco.ingestion` async crawler already does
  this), but requires a Steam API key and several hours of crawl
  time, neither of which we can ask of a reviewer trying to reproduce
  numbers.
* **Manufacture content features.** Genre/tag/price are domain
  primitives — a synthetic side-feature dataset would be deceptive
  and easy to overfit to.

## Consequences

* The loader is Python-`repr` format rather than JSON; parsing uses
  `ast.literal_eval` rather than `json.loads`.
* Dataset download is ~80 MB compressed (vs 8.5 MB for Steam-200k).
* The two-tower output vectors are stored alongside ALS item factors;
  the pgvector publisher in `gamereco.serving.embedding_index` can
  use either as the source of truth — production should prefer the
  two-tower output because it's content-aware.
* The hybrid harness (`gamereco.training.hybrid`) now accepts
  pre-trained two-tower user/item vectors and injects them as the
  `ncf_score` channel of the XGBoost ranker.
* `scripts/run_benchmark.py` (Steam-200k) and
  `scripts/run_benchmark_ucsd.py` (UCSD) both live in the repo so
  Steam-200k stays as a smaller regression target.
