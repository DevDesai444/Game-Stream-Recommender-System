"""Tests for the medallion path helper."""

from __future__ import annotations

from pathlib import Path

from gamereco.common.paths import LakePaths, from_env


def test_lake_paths_bronze_silver_gold(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    assert lake.bronze == tmp_path / "bronze"
    assert lake.silver == tmp_path / "silver"
    assert lake.gold == tmp_path / "gold"


def test_lake_paths_subpaths(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    assert lake.bronze_users.name == "users"
    assert lake.silver_interactions.parent == tmp_path / "silver"
    assert lake.gold_train.name == "interactions_train"
    assert lake.gold_user_clusters.name == "user_clusters"


def test_lake_ensure_creates_dirs(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path / "lake")
    lake.ensure()
    for sub in ("bronze", "silver", "gold"):
        assert (tmp_path / "lake" / sub).exists()


def test_from_env_constructs(tmp_path: Path) -> None:
    lake = from_env(str(tmp_path))
    assert isinstance(lake, LakePaths)
    assert lake.root == tmp_path


def test_lake_paths_are_unique(tmp_path: Path) -> None:
    lake = LakePaths(root=tmp_path)
    seen = {
        lake.bronze_users,
        lake.bronze_owned_games,
        lake.bronze_recently_played,
        lake.bronze_friends,
        lake.bronze_game_detail,
        lake.silver_interactions,
        lake.silver_games,
        lake.silver_users,
        lake.gold_train,
        lake.gold_val,
        lake.gold_test,
        lake.gold_user_clusters,
        lake.gold_game_embeddings,
    }
    assert len(seen) == 13
