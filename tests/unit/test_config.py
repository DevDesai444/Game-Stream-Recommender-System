"""Tests for the typed settings layer."""

from __future__ import annotations

import os

import pytest

from gamereco.common import config


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    for fn in (
        config.steam_settings,
        config.spark_settings,
        config.mlflow_settings,
        config.postgres_settings,
        config.redis_settings,
        config.api_settings,
    ):
        fn.cache_clear()


def test_steam_settings_defaults(env_overrides) -> None:
    os.environ["STEAM_API_KEY"] = "abc"
    os.environ.pop("STEAM_INGEST_CONCURRENCY", None)
    cfg = config.SteamSettings()
    assert cfg.api_key == "abc"
    assert cfg.concurrency == 64
    assert cfg.user_target == 50_000


def test_steam_settings_override(env_overrides) -> None:
    os.environ["STEAM_API_KEY"] = "key"
    os.environ["STEAM_INGEST_CONCURRENCY"] = "32"
    cfg = config.SteamSettings()
    assert cfg.concurrency == 32


def test_postgres_dsn_format(env_overrides) -> None:
    os.environ.update(
        {
            "POSTGRES_HOST": "db",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "x",
            "POSTGRES_USER": "u",
            "POSTGRES_PASSWORD": "p",
        }
    )
    cfg = config.PostgresSettings()
    assert cfg.dsn == "postgresql://u:p@db:5432/x"
    assert cfg.async_dsn.startswith("postgresql+psycopg://")


def test_redis_settings_defaults(env_overrides) -> None:
    cfg = config.RedisSettings()
    assert cfg.port == 6379
    assert cfg.ttl_seconds > 0


def test_api_settings_defaults(env_overrides) -> None:
    cfg = config.ApiSettings()
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000


def test_mlflow_settings_defaults(env_overrides) -> None:
    cfg = config.MLflowSettings()
    assert cfg.experiment == "gamereco"
    assert cfg.tracking_uri.startswith("http")


def test_spark_settings_defaults(env_overrides) -> None:
    cfg = config.SparkSettings()
    assert cfg.master.startswith("local")
    assert cfg.driver_memory.endswith("g")


def test_steam_settings_requires_key_blank_ok(env_overrides) -> None:
    os.environ["STEAM_API_KEY"] = ""
    cfg = config.SteamSettings()
    assert cfg.api_key == ""
