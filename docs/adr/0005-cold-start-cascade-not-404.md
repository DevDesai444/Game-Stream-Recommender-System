# ADR 0005 — Personal → cohort → global cascade over 404 on missing users

**Status:** Accepted

## Context

The original `/recommendations/{user_id}` endpoint returned 404 when
the personal-recs table had no rows for the requested user. This is
technically correct REST behaviour but is the wrong product behaviour:
unknown users are *exactly the population we want to convert* and
404-ing them is the worst possible UX.

There are three layers we could plausibly serve from:

1. **Personal** — per-user table written after every training cycle.
2. **Cohort top** — K-Means cohort-level top-K, useful for users we
   know enough about to assign a cohort but not enough to score
   personally.
3. **Global top** — the catalog-wide top-K, useful for brand-new
   users with zero history.

## Decision

Implement `gamereco.serving.coldstart.resolve` as a deterministic
**cascade**:

```
personal  ──►  cohort  ──►  global
```

Whichever layer first produces ≥ 1 item is the answer; the response
reports `served_from` so observability can attribute hit rates to each
layer. A 503 is only returned if *every* layer is empty (i.e. the
backend itself isn't provisioned), never just because the user is
unknown.

Add a `POST /onboard` endpoint that takes a few liked appids and
returns blended pgvector neighbours — the "first-touch" experience
for users we know nothing about yet.

## Why

* **Never 404 on missing users.** A new install always gets *some*
  list, so the client never has to special-case empty state.
* **Layer-attributable telemetry.** Prometheus counter
  `gamereco_recs_served_from_total{served_from="..."}` makes the
  cache/personal/cohort/global split a first-class metric. If cohort
  hit rate jumps overnight, that's a signal worth pursuing.
* **Cheap to add.** The cascade is one function with three branches;
  the data already exists (the K-Means stage writes `user_cohorts` and
  `cohort_top`).

## Rejected alternatives

* **404 on missing.** Maximal correctness, worst product behaviour.
* **Always global.** Simple, but throws away the cohort signal we
  already compute during training. Cohort recs are measurably better
  than global for users with at least a little history.
* **On-the-fly KNN at request time.** Possible but costly; pgvector
  search per request would dominate latency for users we've already
  classified.

## Consequences

* The cascade has to be defensive — every layer is wrapped in a
  Protocol so each can be stubbed/replaced for unit tests. See
  `tests/unit/test_coldstart.py` for the seven scenarios pinned.
* `served_from` becomes part of the public response shape. Clients
  can display "We're showing global popular games because we're new
  here" copy without inspecting headers.
* The `X-Served-From` response header is set by the route handler so
  the access-log middleware can fold it into Prometheus labels.
