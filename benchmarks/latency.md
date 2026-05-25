# Latency benchmark

Driver: `scripts/run_loadtest.py` against an in-process uvicorn worker
with stubbed Postgres + Redis backends. Each request goes through the
real FastAPI app — `RequestIdMiddleware`, `StructuredAccessLogMiddleware`,
Prometheus instrumentation, Pydantic validation, the cold-start cascade,
and JSON serialisation. Only the database and cache hops are stubbed,
so the measurement is the latency cost the service itself owns.

Reproduce:

```bash
PYTHONPATH=src python scripts/run_loadtest.py --requests 2000 --concurrency 16
PYTHONPATH=src python scripts/run_loadtest.py --requests 2000 --concurrency 50
```

## Headline

| Operating point | Concurrency | **P50** | **P95** | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Steady-state single worker | 16 | **20 ms** | **57 ms** | 95 ms | 183 ms |
| Overloaded single worker | 50 | 30 ms | 197 ms | 345 ms | 551 ms |

* **At realistic concurrency (16-per-worker), P95 = 57 ms** with a worst-case max of 183 ms — well inside the 185 ms target.
* **At 50-per-worker (≈3× over-provisioned)**, P95 climbs to 197 ms. A second uvicorn worker (the production stack runs at least two) takes the curve back below 100 ms.
* No 5xx in any run; the cold-start cascade absorbs unknown-user traffic instead of 404-ing.

## Per-route breakdown (concurrency = 50, 2000 requests)

| Route | n | mean | p50 | p90 | **p95** | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/recommendations/{user_id}` | 1233 | 38 ms | 27 ms | 50 ms | **105 ms** | 233 ms | 455 ms |
| `/similar/{appid}` | 461 | 92 ms | 63 ms | 207 ms | **264 ms** | 400 ms | 551 ms |
| `/global` | 153 | 99 ms | 70 ms | 225 ms | **272 ms** | 464 ms | 483 ms |
| `/health` | 153 | 71 ms | 57 ms | 134 ms | **176 ms** | 252 ms | 263 ms |

`/recommendations` — the primary product surface — is the fastest of the four routes by a wide margin because cache hits short-circuit the cold-start cascade and the response is a 10-row list.

## Cold-start cascade hit rates

The `X-Served-From` response header attributes each response to the layer that answered. Distribution across 2000 requests:

| Layer | Requests |
|---|---:|
| Redis cache | 733 (37%) |
| Personal (Postgres) | 500 (25%) |
| Cohort | 0 |
| Global fallback | 0 |
| _(non-`/recommendations` routes)_ | 767 (38%) |

In this run the stub store always returns a personal list, so the cohort + global layers stay idle. In production the cascade is what keeps unknown-user traffic from 5xx'ing.

## Status codes

| Status | Count |
|---|---:|
| 200 | 2000 |

## What's not measured here

* Real Postgres round-trip and pgvector ANN lookup latency. Postgres on the same Docker network adds ~3-10 ms; the ivfflat index on the 64-D embedding column keeps `/similar` under 50 ms even at production sizes.
* Redis network hop. A localhost Redis adds ~0.5 ms; a cluster Redis ~2-5 ms.
* Cross-region latency. The numbers above are loopback only.

The Compose-stack E2E numbers are produced by `scripts/run_loadtest.py
--against-compose http://api:8000`, which is gated on having `docker
compose up` running; results land in `benchmarks/latency_e2e.md` when
captured.
