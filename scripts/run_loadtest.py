"""Measure real served-side P95 latency against the FastAPI service.

Boots uvicorn in a subprocess with the live FastAPI app, swaps in
in-memory stubs for the Postgres store and Redis cache so the
benchmark is purely a measurement of *FastAPI + middleware + JSON
serialisation + Python overhead* (i.e. the slice of latency the
service code itself owns), then drives concurrent HTTPX requests
against it. Writes a JSON + Markdown report to ``benchmarks/``.

This is the script that produces the headline P95 number in
benchmarks/latency.md. It does not require Docker, Postgres, or Redis
— the database and cache are replaced with deterministic stubs so
the run is self-contained and reproducible from a clean checkout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_stub_app(path: Path) -> None:
    """Render an ASGI module that boots the real app with stub backends."""
    path.write_text(
        """from gamereco.serving import api as api_module
from gamereco.serving.cache import RecommendationCache
from gamereco.common.schemas import RecommendationItem


class StubStore:
    def fetch_user_recommendations(self, user_id, *, limit=10):
        # Deterministic per-user list so cache hits and misses behave.
        seed = sum(ord(c) for c in user_id) % 50
        return [
            RecommendationItem(
                steam_appid=seed + i,
                name=f"game-{seed + i}",
                header_image=None,
                score=1.0 - 0.01 * i,
            )
            for i in range(limit)
        ]

    def user_cohort(self, user_id):
        return sum(ord(c) for c in user_id) % 8

    def cohort_top(self, cohort_id, *, limit=10):
        return [
            RecommendationItem(
                steam_appid=1000 + cohort_id * 10 + i,
                name=f"cohort-{cohort_id}-{i}",
                header_image=None,
                score=0.8 - 0.01 * i,
            )
            for i in range(limit)
        ]

    def similar_games(self, steam_appid, *, limit=10):
        return [
            RecommendationItem(
                steam_appid=steam_appid + 1 + i,
                name=f"sim-{steam_appid}-{i}",
                header_image=None,
                score=0.9 - 0.01 * i,
            )
            for i in range(limit)
        ]

    def global_top(self, limit=10):
        return [
            RecommendationItem(
                steam_appid=42 + i,
                name=f"global-{i}",
                header_image=None,
                score=0.7 - 0.01 * i,
            )
            for i in range(limit)
        ]


class StubCache:
    def __init__(self):
        self._d = {}

    def get(self, user_id):
        return self._d.get(user_id)

    def set(self, user_id, items):
        self._d[user_id] = list(items)

    def invalidate(self, user_id):
        self._d.pop(user_id, None)

    def ping(self):
        return True


api_module.app.state.store = StubStore()
api_module.app.state.cache = StubCache()
app = api_module.app
"""
    )


async def hammer(
    base_url: str,
    *,
    concurrency: int,
    total_requests: int,
    user_pool: int,
) -> dict[str, float | int | dict[str, float]]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    per_route: dict[str, list[float]] = defaultdict(list)
    served_from: Counter[str] = Counter()
    status_counts: Counter[int] = Counter()
    routes = [
        ("/recommendations/{user}", 8),
        ("/similar/{appid}", 3),
        ("/global", 1),
        ("/health", 1),
    ]
    total_weight = sum(w for _, w in routes)
    plan: list[str] = []
    for route, weight in routes:
        plan.extend([route] * int(total_requests * weight / total_weight))
    while len(plan) < total_requests:
        plan.append("/recommendations/{user}")

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        # Warm up: 3 quick calls so the first-request torch / connection
        # pool setup doesn't pollute the P95.
        for _ in range(3):
            await client.get("/health")

        async def one(route_template: str, idx: int) -> None:
            if route_template == "/recommendations/{user}":
                path = f"/recommendations/u_{idx % user_pool:05d}"
                tag = "/recommendations/{user_id}"
            elif route_template == "/similar/{appid}":
                path = f"/similar/{440 + (idx % 50)}"
                tag = "/similar/{appid}"
            elif route_template == "/global":
                path = "/global"
                tag = "/global"
            else:
                path = "/health"
                tag = "/health"
            async with semaphore:
                started = time.perf_counter()
                resp = await client.get(path)
                duration_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(duration_ms)
            per_route[tag].append(duration_ms)
            status_counts[int(resp.status_code)] += 1
            served_from[resp.headers.get("X-Served-From", "n/a")] += 1

        await asyncio.gather(*(one(t, i) for i, t in enumerate(plan)))

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0}
        sorted_v = sorted(values)
        return {
            "count": float(len(sorted_v)),
            "mean_ms": round(statistics.fmean(sorted_v), 3),
            "p50_ms": round(sorted_v[int(len(sorted_v) * 0.50)], 3),
            "p90_ms": round(sorted_v[int(len(sorted_v) * 0.90)], 3),
            "p95_ms": round(sorted_v[int(len(sorted_v) * 0.95)], 3),
            "p99_ms": round(sorted_v[min(int(len(sorted_v) * 0.99), len(sorted_v) - 1)], 3),
            "max_ms": round(sorted_v[-1], 3),
        }

    return {
        "total_requests": len(latencies),
        "concurrency": concurrency,
        "user_pool": user_pool,
        "overall": stats(latencies),
        "per_route": {route: stats(values) for route, values in per_route.items()},
        "status_codes": {str(k): v for k, v in status_counts.items()},
        "served_from": dict(served_from),
    }


def render_markdown(report: dict[str, object]) -> str:
    overall = report["overall"]  # type: ignore[index]
    rows = []
    for route, s in report["per_route"].items():  # type: ignore[union-attr]
        rows.append(
            f"| `{route}` | {int(s['count'])} | {s['mean_ms']} | "
            f"{s['p50_ms']} | {s['p90_ms']} | {s['p95_ms']} | "
            f"{s['p99_ms']} | {s['max_ms']} |"
        )
    routes_md = "\n".join(rows)
    served_from = report["served_from"]
    served_md = "\n".join(f"- `{k}`: {v}" for k, v in served_from.items())
    status_md = ", ".join(f"`{k}`: {v}" for k, v in report["status_codes"].items())
    return f"""# Latency benchmark

Driver: `scripts/run_loadtest.py` against an in-process uvicorn worker
with stubbed Postgres / Redis. The measured numbers below are the
latency the FastAPI service itself adds — middleware (request id +
access log + Prometheus), Pydantic validation, JSON serialisation,
and the cold-start cascade traversal — without any database or
network hop. The Compose-stack E2E latency is strictly higher; this
number is the lower-bound on what the service can serve.

## Overall

| Metric | Value |
|---|---|
| Total requests | {report['total_requests']} |
| Concurrency | {report['concurrency']} |
| User pool | {report['user_pool']} |
| Mean | {overall['mean_ms']} ms |
| **P50** | **{overall['p50_ms']} ms** |
| **P95** | **{overall['p95_ms']} ms** |
| **P99** | **{overall['p99_ms']} ms** |
| Max | {overall['max_ms']} ms |

## Per route

| Route | n | mean | p50 | p90 | **p95** | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
{routes_md}

## Served-from distribution

{served_md}

## Status codes

{status_md}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--user-pool", type=int, default=500)
    parser.add_argument("--port", type=int, default=0, help="0 = pick free port")
    parser.add_argument("--out-json", default="benchmarks/latency.json")
    parser.add_argument("--out-md", default="benchmarks/latency.md")
    args = parser.parse_args()

    port = args.port or free_port()
    stub_module = Path("scripts/_loadtest_stub_app.py")
    write_stub_app(stub_module)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts._loadtest_stub_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        # Wait for the server to come up.
        deadline = time.time() + 30
        ready = False
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{base}/health", timeout=0.5)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.2)
        if not ready:
            print("uvicorn did not come up in 30s", file=sys.stderr)
            return 2

        report = asyncio.run(
            hammer(
                base,
                concurrency=args.concurrency,
                total_requests=args.requests,
                user_pool=args.user_pool,
            )
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(args.out_md).write_text(render_markdown(report))
    print(f"wrote {out_json} and {args.out_md}")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
