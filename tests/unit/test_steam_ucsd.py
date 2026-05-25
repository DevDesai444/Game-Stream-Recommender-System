"""Tests for the UCSD Steam loader."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from gamereco.datasets.steam_ucsd import (
    UCSDLoadConfig,
    _parse_price,
    _parse_release_year,
    load_ucsd,
    temporal_split_ucsd,
)


def _write_repr_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(repr(row) + "\n")


@pytest.fixture
def ucsd_fixture(tmp_path: Path) -> UCSDLoadConfig:
    games_path = tmp_path / "games.json.gz"
    users_path = tmp_path / "users_items.json.gz"
    reviews_path = tmp_path / "reviews.json.gz"

    _write_repr_gz(
        games_path,
        [
            {
                "id": "100",
                "app_name": "Game A",
                "genres": ["Action", "RPG"],
                "tags": ["Open World", "Story Rich"],
                "specs": ["Single-player"],
                "price": "9.99",
                "release_date": "2018-05-01",
                "developer": "Dev A",
                "publisher": "Pub A",
                "sentiment": "Very Positive",
                "early_access": False,
            },
            {
                "id": "200",
                "app_name": "Game B",
                "genres": ["Strategy"],
                "tags": ["Turn-Based"],
                "specs": ["Multi-player"],
                "price": "Free",
                "release_date": "2020-01-15",
                "developer": "Dev B",
                "publisher": "Pub B",
            },
            {
                "id": "300",
                "app_name": "Game C",
                "genres": ["Indie"],
                "tags": ["Casual"],
                "release_date": "2021",
                "price": 4.99,
            },
        ],
    )
    _write_repr_gz(
        users_path,
        [
            {
                "user_id": "u1",
                "items_count": 3,
                "items": [
                    {"item_id": "100", "playtime_forever": 600, "playtime_2weeks": 60},
                    {"item_id": "200", "playtime_forever": 120, "playtime_2weeks": 0},
                    {"item_id": "300", "playtime_forever": 0, "playtime_2weeks": 0},
                    {"item_id": "999", "playtime_forever": 5, "playtime_2weeks": 0},
                    {"item_id": "888", "playtime_forever": 10, "playtime_2weeks": 0},
                ],
            },
            {
                "user_id": "u2",
                "items_count": 4,
                "items": [
                    {"item_id": "100", "playtime_forever": 200, "playtime_2weeks": 0},
                    {"item_id": "200", "playtime_forever": 50, "playtime_2weeks": 0},
                    {"item_id": "999", "playtime_forever": 1, "playtime_2weeks": 0},
                    {"item_id": "888", "playtime_forever": 7, "playtime_2weeks": 0},
                    {"item_id": "300", "playtime_forever": 3, "playtime_2weeks": 0},
                ],
            },
            {
                "user_id": "u3",
                "items_count": 5,
                "items": [
                    {"item_id": "100", "playtime_forever": 30, "playtime_2weeks": 0},
                    {"item_id": "300", "playtime_forever": 90, "playtime_2weeks": 0},
                    {"item_id": "999", "playtime_forever": 2, "playtime_2weeks": 0},
                    {"item_id": "888", "playtime_forever": 4, "playtime_2weeks": 0},
                    {"item_id": "200", "playtime_forever": 6, "playtime_2weeks": 0},
                ],
            },
        ],
    )
    _write_repr_gz(
        reviews_path,
        [
            {
                "user_id": "u1",
                "reviews": [
                    {"item_id": "100", "recommend": True, "review": "great"},
                    {"item_id": "300", "recommend": False, "review": "meh"},
                ],
            },
            {
                "user_id": "u2",
                "reviews": [{"item_id": "200", "recommend": True, "review": "ok"}],
            },
        ],
    )

    return UCSDLoadConfig(
        games_path=games_path,
        users_items_path=users_path,
        reviews_path=reviews_path,
        min_interactions_per_user=2,
        min_interactions_per_game=2,
    )


def test_parse_release_year_from_string() -> None:
    assert _parse_release_year("2018-05-01") == 2018
    assert _parse_release_year("Coming soon") is None
    assert _parse_release_year(None) is None
    assert _parse_release_year(2019) == 2019


def test_parse_release_year_rejects_garbage() -> None:
    # Year too far in the past or future is rejected
    assert _parse_release_year("1850") is None
    assert _parse_release_year("3000-01-01") is None


def test_parse_price_free_text() -> None:
    assert _parse_price("Free to Play") == 0.0
    assert _parse_price("Free Demo") == 0.0


def test_parse_price_numeric() -> None:
    assert _parse_price(4.99) == 4.99
    assert _parse_price("9.99") == 9.99
    assert _parse_price("$14.99") == 14.99


def test_parse_price_unknown_returns_none() -> None:
    assert _parse_price(None) is None
    assert _parse_price("Unknown") is None


def test_load_ucsd_produces_dense_indices(ucsd_fixture: UCSDLoadConfig) -> None:
    result = load_ucsd(ucsd_fixture)
    n_users = result.n_users
    n_games = result.n_games
    assert set(result.interactions["user_idx"]) == set(range(n_users))
    assert set(result.interactions["game_idx"]) == set(range(n_games))


def test_load_ucsd_carries_game_metadata(ucsd_fixture: UCSDLoadConfig) -> None:
    result = load_ucsd(ucsd_fixture)
    # Every game has a genres list (possibly empty)
    assert "genres" in result.games.columns
    assert "tags" in result.games.columns
    assert "price" in result.games.columns
    assert "release_year" in result.games.columns


def test_load_ucsd_carries_user_features(ucsd_fixture: UCSDLoadConfig) -> None:
    result = load_ucsd(ucsd_fixture)
    assert "items_count" in result.users.columns
    assert "total_playtime" in result.users.columns
    assert "reviews_count" in result.users.columns
    # Reviews counter is populated for the two users with reviews
    assert (result.users["reviews_count"] > 0).sum() >= 2


def test_load_ucsd_review_indices_align_with_interactions(ucsd_fixture: UCSDLoadConfig) -> None:
    result = load_ucsd(ucsd_fixture)
    assert set(result.reviews["user_idx"]).issubset(set(result.interactions["user_idx"]))
    assert set(result.reviews["game_idx"]).issubset(set(result.interactions["game_idx"]))


def test_load_ucsd_confidence_is_log1p_playtime(ucsd_fixture: UCSDLoadConfig) -> None:
    import numpy as np

    result = load_ucsd(ucsd_fixture)
    expected = np.log1p(result.interactions["playtime_forever"].to_numpy(dtype=np.float64))
    # Zero-playtime rows get a floor of log1p(1) so they still contribute.
    expected[expected == 0.0] = np.log1p(1.0)
    np.testing.assert_allclose(result.interactions["confidence"].to_numpy(), expected)


def test_temporal_split_ucsd_partitions_disjoint(ucsd_fixture: UCSDLoadConfig) -> None:
    result = load_ucsd(ucsd_fixture)
    train, val, test = temporal_split_ucsd(result.interactions, val_frac=0.2, test_frac=0.2)
    total = len(train) + len(val) + len(test)
    assert total == len(result.interactions)


def test_temporal_split_ucsd_rejects_invalid_fractions(ucsd_fixture: UCSDLoadConfig) -> None:
    result = load_ucsd(ucsd_fixture)
    with pytest.raises(ValueError):
        temporal_split_ucsd(result.interactions, val_frac=0.0, test_frac=0.1)
    with pytest.raises(ValueError):
        temporal_split_ucsd(result.interactions, val_frac=0.6, test_frac=0.5)


def test_load_ucsd_drops_inactive_users(tmp_path: Path) -> None:
    games_path = tmp_path / "g.json.gz"
    users_path = tmp_path / "u.json.gz"
    reviews_path = tmp_path / "r.json.gz"
    _write_repr_gz(
        games_path,
        [{"id": str(i * 100), "app_name": f"Game {i}", "genres": ["X"]} for i in range(1, 11)],
    )
    _write_repr_gz(
        users_path,
        [
            {
                "user_id": f"u{u}",
                "items_count": 1,
                "items": [{"item_id": "100", "playtime_forever": 10}],
            }
            for u in range(5)
        ],
    )
    _write_repr_gz(reviews_path, [])
    cfg = UCSDLoadConfig(
        games_path=games_path,
        users_items_path=users_path,
        reviews_path=reviews_path,
        min_interactions_per_user=5,
        min_interactions_per_game=2,
    )
    with pytest.raises(ValueError):
        load_ucsd(cfg)
