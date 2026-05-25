"""Centralised settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SteamSettings(BaseSettings):
    api_key: str = Field(default="", alias="STEAM_API_KEY")
    concurrency: int = Field(default=64, alias="STEAM_INGEST_CONCURRENCY")
    user_target: int = Field(default=50_000, alias="STEAM_INGEST_USER_TARGET")
    request_timeout_s: float = Field(default=10.0, alias="STEAM_REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class SparkSettings(BaseSettings):
    master: str = Field(default="local[*]", alias="SPARK_MASTER")
    driver_memory: str = Field(default="4g", alias="SPARK_DRIVER_MEMORY")
    executor_memory: str = Field(default="4g", alias="SPARK_EXECUTOR_MEMORY")
    delta_root: str = Field(default="./data/delta", alias="DELTA_ROOT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MLflowSettings(BaseSettings):
    tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    registry_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_REGISTRY_URI")
    experiment: str = Field(default="gamereco", alias="MLFLOW_EXPERIMENT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PostgresSettings(BaseSettings):
    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    database: str = Field(default="gamereco", alias="POSTGRES_DB")
    user: str = Field(default="gamereco", alias="POSTGRES_USER")
    password: str = Field(default="gamereco", alias="POSTGRES_PASSWORD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}" f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def async_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(BaseSettings):
    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    ttl_seconds: int = Field(default=900, alias="REDIS_TTL_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ApiSettings(BaseSettings):
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="info", alias="API_LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def steam_settings() -> SteamSettings:
    return SteamSettings()


@lru_cache
def spark_settings() -> SparkSettings:
    return SparkSettings()


@lru_cache
def mlflow_settings() -> MLflowSettings:
    return MLflowSettings()


@lru_cache
def postgres_settings() -> PostgresSettings:
    return PostgresSettings()


@lru_cache
def redis_settings() -> RedisSettings:
    return RedisSettings()


@lru_cache
def api_settings() -> ApiSettings:
    return ApiSettings()
