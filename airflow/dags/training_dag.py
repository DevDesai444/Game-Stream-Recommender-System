"""Weekly training DAG: bronze -> silver -> gold -> ALS/NCF/KMeans -> XGBoost ensemble.

Each task shells out to the gamereco CLI. MLflow tracking + registry
calls happen inside the training tasks, so the DAG itself doesn't need
to know which model is being promoted."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "gamereco",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="gamereco_training_weekly",
    description="Refit ALS + NCF + KMeans + XGBoost ensemble against the latest gold tables",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 4 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["gamereco", "training"],
) as dag:
    wait_for_ingest = ExternalTaskSensor(
        task_id="wait_for_ingest",
        external_dag_id="gamereco_ingestion_daily",
        external_task_id="ingest_user_endpoints",
        allowed_states=["success"],
        timeout=60 * 60 * 6,
        mode="reschedule",
        poke_interval=5 * 60,
    )

    bronze = BashOperator(task_id="bronze", bash_command="gamereco-etl bronze")
    silver = BashOperator(task_id="silver", bash_command="gamereco-etl silver")
    gold = BashOperator(task_id="gold", bash_command="gamereco-etl gold")
    als = BashOperator(task_id="train_als", bash_command="gamereco-train als")
    ncf = BashOperator(
        task_id="train_ncf",
        bash_command="gamereco-train ncf --epochs 8 --batch-size 4096",
    )
    kmeans = BashOperator(
        task_id="train_kmeans",
        bash_command="gamereco-train kmeans --k 16",
    )
    ensemble = BashOperator(
        task_id="train_ensemble",
        bash_command="gamereco-train ensemble --top-n 200",
    )

    wait_for_ingest >> bronze >> silver >> gold
    gold >> als
    gold >> ncf
    als >> kmeans
    [als, ncf, kmeans] >> ensemble
