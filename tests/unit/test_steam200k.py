"""Tests for the Steam-200k loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gamereco.datasets.steam200k import (
    Steam200kRecord,
    load_steam_200k,
    materialise_silver,
    temporal_split_pandas,
)


@pytest.fixture
def fake_csv(tmp_path: Path) -> Path:
    rows = [
        "1,GameA,purchase,1.0,0",
        "1,GameA,play,10.0,0",
        "1,GameB,purchase,1.0,0",
        "1,GameB,play,5.0,0",
        "1,GameC,purchase,1.0,0",
        "1,GameC,play,2.0,0",
        "1,GameD,purchase,1.0,0",
        "2,GameA,purchase,1.0,0",
        "2,GameA,play,30.0,0",
        "2,GameB,purchase,1.0,0",
        "2,GameC,purchase,1.0,0",
        "2,GameC,play,1.0,0",
        # User 3 has too few interactions; should be filtered.
        "3,GameA,purchase,1.0,0",
    ]
    p = tmp_path / "fake.csv"
    p.write_text("\n".join(rows))
    return p


def test_load_parses_columns(fake_csv: Path) -> None:
    df = load_steam_200k(fake_csv)
    assert list(df.columns) == ["user_id", "game_name", "behavior", "value"]
    assert {"purchase", "play"}.issuperset(set(df["behavior"].unique()))


def test_materialise_silver_filters_inactive_users(fake_csv: Path) -> None:
    silver = materialise_silver(load_steam_200k(fake_csv))
    assert (silver["user_id"] != "3").all()


def test_materialise_silver_indices_are_dense(fake_csv: Path) -> None:
    silver = materialise_silver(load_steam_200k(fake_csv))
    n_users = silver["user_idx"].nunique()
    n_games = silver["game_idx"].nunique()
    assert set(silver["user_idx"]) == set(range(n_users))
    assert set(silver["game_idx"]) == set(range(n_games))


def test_materialise_silver_purchase_only_has_floor_confidence(fake_csv: Path) -> None:
    silver = materialise_silver(load_steam_200k(fake_csv))
    purchase_only = silver[(silver["purchased"]) & (silver["play_hours"] == 0)]
    if len(purchase_only):
        assert (purchase_only["confidence"] > 0.0).all()


def test_materialise_silver_play_hours_to_minutes(fake_csv: Path) -> None:
    silver = materialise_silver(load_steam_200k(fake_csv))
    play = silver[silver["play_hours"] > 0].iloc[0]
    assert play["playtime_minutes"] == round(play["play_hours"] * 60)


def test_materialise_silver_handles_empty() -> None:
    empty = pd.DataFrame(
        columns=["user_id", "game_name", "behavior", "value"],
        dtype="string",
    )
    silver = materialise_silver(empty)
    assert silver.empty


def test_temporal_split_partitions_disjoint(fake_csv: Path) -> None:
    silver = materialise_silver(load_steam_200k(fake_csv))
    train, val, test = temporal_split_pandas(silver, val_frac=0.2, test_frac=0.2)
    total = len(train) + len(val) + len(test)
    assert total == len(silver)
    # No row appears in more than one split.
    joined = pd.concat(
        [
            train.assign(_split="train"),
            val.assign(_split="val"),
            test.assign(_split="test"),
        ]
    )
    counts = joined.groupby(["user_idx", "game_idx"]).size()
    assert (counts == 1).all()


def test_temporal_split_rejects_bad_fractions() -> None:
    silver = pd.DataFrame(
        {
            "user_idx": [0, 0, 0],
            "game_idx": [0, 1, 2],
            "user_id": ["x"] * 3,
            "game_name": ["a", "b", "c"],
            "play_hours": [1.0, 2.0, 3.0],
            "playtime_minutes": [60, 120, 180],
            "confidence": [0.0, 0.0, 0.0],
            "purchased": [True, True, True],
        }
    )
    with pytest.raises(ValueError):
        temporal_split_pandas(silver, val_frac=0.0, test_frac=0.1)
    with pytest.raises(ValueError):
        temporal_split_pandas(silver, val_frac=0.6, test_frac=0.5)


def test_dataclass_record_is_frozen() -> None:
    rec = Steam200kRecord(user_id="u", game_name="g", purchased=True, play_hours=1.0)
    with pytest.raises(AttributeError):
        rec.user_id = "y"  # type: ignore[misc]
