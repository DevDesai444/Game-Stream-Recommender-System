"""Tests for the async ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gamereco.common.paths import LakePaths
from gamereco.ingestion.pipeline import IngestionPipeline


def _stub_client() -> MagicMock:
    client = MagicMock()
    client.player_summary = AsyncMock(return_value={"steamid": "1", "personaname": "a"})
    client.owned_games = AsyncMock(
        return_value={"steamid": "1", "game_count": 1, "games": [{"appid": 10}]}
    )
    client.recently_played = AsyncMock(
        return_value={"steamid": "1", "total_count": 1, "games": [{"appid": 10}]}
    )
    client.friend_list = AsyncMock(return_value={"steamid": "1", "friends": []})
    client.app_details = AsyncMock(return_value={"steam_appid": 10, "name": "X", "success": True})
    return client


@pytest.mark.asyncio
async def test_pipeline_ingest_users_writes_all_layers(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    pipeline = IngestionPipeline(_stub_client(), lake)
    stats = await pipeline.ingest_users(["1", "2", "3"])
    assert stats.users == 3
    assert (lake.bronze_users / "user_summary.jsonl").exists()
    assert (lake.bronze_owned_games / "owned_games.jsonl").exists()
    assert (lake.bronze_friends / "friends.jsonl").exists()


@pytest.mark.asyncio
async def test_pipeline_ingest_users_handles_empty_input(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    pipeline = IngestionPipeline(_stub_client(), lake)
    stats = await pipeline.ingest_users([])
    assert stats.users == 0


@pytest.mark.asyncio
async def test_pipeline_ingest_games_skips_missing(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    client = _stub_client()
    client.app_details = AsyncMock(side_effect=[{"steam_appid": 1, "name": "A"}, None])
    pipeline = IngestionPipeline(client, lake)
    count = await pipeline.ingest_game_details([1, 2])
    assert count == 1


@pytest.mark.asyncio
async def test_pipeline_writes_owned_games_lines(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    pipeline = IngestionPipeline(_stub_client(), lake)
    await pipeline.ingest_users(["1"])
    rows = (lake.bronze_owned_games / "owned_games.jsonl").read_text().splitlines()
    obj = json.loads(rows[0])
    assert obj["steamid"] == "1"
    assert obj["games"][0]["appid"] == 10
