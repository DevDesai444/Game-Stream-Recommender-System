# ADR 0003 — Per-user temporal split over random holdout

**Status:** Accepted

## Context

Every NDCG@10 number in `benchmarks/results.md` lives or dies on the
train/val/test split. The default in many MovieLens-shaped tutorials
is a uniform 80/10/10 random split — easy, but leaky in a serving
context because it lets the model peek at "future" interactions when
predicting "past" ones for the same user.

## Decision

Use a **per-user temporal split**: sort each user's interactions by
event time (or, in the Steam-200k loader, by cumulative playtime as a
proxy), hold out the most-recent N% for testing, the next N% for
validation, and use the rest for training. Users with fewer than three
interactions go entirely into training (no holdout would be
meaningful).

## Why

* **Matches the serving contract.** At serve time the model only sees
  a user's past; the metric should evaluate that exact regime.
* **Catches popularity overfit.** A random split lets a model "learn"
  that a user owns `Dota 2` from one of their owns-it rows and
  "predict" the same fact in the holdout. The lift over the popularity
  baseline collapses under a random split because the popularity
  baseline gets the same illegal peek.
* **Stable comparison.** Different model variants are compared on the
  same per-user holdout indices, so an NDCG delta isn't an artifact
  of which 10% of interactions ended up in test.

## Rejected alternatives

* **Random split.** Easy to implement, easy to over-interpret. Inflates
  NDCG by 10–30% across all models, washing out the model deltas we
  actually care about.
* **Leave-one-out (LOO) per user.** Better than random, but
  computationally heavier (one eval pass per held-out item) and
  noisier on users with few interactions.
* **Global temporal split.** Trains on "everything before T", tests on
  "everything after T". Most realistic, but Steam-200k lacks
  per-event timestamps so we can't implement it faithfully.

## Consequences

* Steam-200k carries no event timestamps, so the loader uses
  cumulative `play_hours` as a deterministic temporal proxy (higher
  playtime ≈ later, since cumulative playtime monotonically grows).
* The split fractions must respect `val + test < 1` (enforced by
  `SplitFractions.__post_init__`).
* The implementation lives in two parity-checked places:
  `gamereco.etl.splits.temporal_split` (Spark) and
  `gamereco.datasets.steam200k.temporal_split_pandas` (laptop).
