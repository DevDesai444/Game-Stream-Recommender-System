"""Refresh the pgvector candidate index + Redis warm cache from the latest gold/model artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "gamereco",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="gamereco_serving_refresh",
    description="Push latest game embeddings into pgvector and warm Redis top-K cache",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="30 5 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["gamereco", "serving"],
) as dag:
    publish_embeddings = BashOperator(
        task_id="publish_pgvector_embeddings",
        bash_command="python -m gamereco.serving.embedding_index --refresh",
    )

    warm_cache = BashOperator(
        task_id="warm_redis_top_k",
        bash_command="python -m gamereco.serving.cache_warmer --top-n 200",
    )

    publish_embeddings >> warm_cache
