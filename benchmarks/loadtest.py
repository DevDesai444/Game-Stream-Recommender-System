"""Locust load test for the recommendation API.

Run:
    # 1. Bring up the API (e.g. against a stubbed store/cache for a pure
    #    latency test, or the docker-compose stack for a true E2E test):
    PYTHONPATH=src uvicorn gamereco.serving.api:app \
        --host 127.0.0.1 --port 8000 --workers 1 &
    # 2. Drive load:
    locust -f benchmarks/loadtest.py --headless \
        --host http://127.0.0.1:8000 \
        --users 50 --spawn-rate 10 -t 30s

Or use scripts/run_loadtest.py to do both steps in one shot.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


_USERS = [f"u_{i:05d}" for i in range(500)]
_APPIDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 730, 440, 570, 4000]


class RecommendationUser(HttpUser):
    """Simulates a player browsing the recommendation surface."""

    wait_time = between(0.05, 0.25)

    @task(8)
    def get_recommendations(self) -> None:
        user = random.choice(_USERS)
        with self.client.get(
            f"/recommendations/{user}?limit=10",
            name="/recommendations/{user_id}",
            catch_response=True,
        ) as resp:
            # 5xx is a real failure; cohort/global fallbacks are
            # *success* (cold-start still served sensible data).
            if resp.status_code >= 500:
                resp.failure(f"server error {resp.status_code}")

    @task(3)
    def get_similar(self) -> None:
        appid = random.choice(_APPIDS)
        with self.client.get(
            f"/similar/{appid}?limit=10",
            name="/similar/{appid}",
            catch_response=True,
        ) as resp:
            if resp.status_code == 404:
                # Not all appids have embeddings; not a load-test failure.
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"server error {resp.status_code}")

    @task(1)
    def get_global(self) -> None:
        self.client.get("/global?limit=10", name="/global")

    @task(1)
    def get_health(self) -> None:
        self.client.get("/health", name="/health")
