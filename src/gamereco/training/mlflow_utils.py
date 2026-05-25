"""Thin wrappers around the MLflow tracking / registry APIs."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from gamereco.common.config import mlflow_settings


def _bootstrap() -> None:
    cfg = mlflow_settings()
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_registry_uri(cfg.registry_uri)
    mlflow.set_experiment(cfg.experiment)


@contextlib.contextmanager
def start_run(run_name: str, *, nested: bool = False, tags: dict[str, str] | None = None) -> Iterator[Any]:
    _bootstrap()
    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags or {}) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def register_model(model_uri: str, name: str) -> ModelVersion:
    """Register a logged model artifact under `name` in the MLflow registry."""
    _bootstrap()
    client = MlflowClient()
    try:
        client.get_registered_model(name)
    except mlflow.exceptions.RestException:
        client.create_registered_model(name)
    return mlflow.register_model(model_uri=model_uri, name=name)


def promote_to_stage(name: str, version: str | int, stage: str = "Production") -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
        archive_existing_versions=True,
    )
