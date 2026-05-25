"""Daily Steam ingestion DAG.

Discovers Steam IDs from community pages, fans out async API ingestion
for summary / owned / recent / friends, and refreshes the game-detail
catalog. All steps run as BashOperators against the installed CLI so
they can be ported to KubernetesPodOperator without changing the DAG."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "gamereco",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}

with DAG(
    dag_id="gamereco_ingestion_daily",
    description="Ingest Steam users and game catalog into the bronze NDJSON store",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["gamereco", "ingestion"],
) as dag:
    discover = BashOperator(
        task_id="discover_steam_ids",
        bash_command=(
            "gamereco-ingest discover "
            "--pages 250 --target 50000 "
            "--out /opt/gamereco/data/delta/bronze/users/seed_steam_ids.jsonl"
        ),
    )

    ingest_users = BashOperator(
        task_id="ingest_user_endpoints",
        bash_command=(
            "gamereco-ingest users "
            "--seed /opt/gamereco/data/delta/bronze/users/seed_steam_ids.jsonl"
        ),
    )

    ingest_games = BashOperator(
        task_id="ingest_game_catalog",
        bash_command="gamereco-ingest games --limit 20000",
    )

    discover >> ingest_users
    discover >> ingest_games
