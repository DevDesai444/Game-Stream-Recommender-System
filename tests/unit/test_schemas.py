"""Tests for the pydantic schemas."""

from __future__ import annotations

from datetime import datetime

import pytest

from gamereco.common.schemas import (
    FriendEdge,
    FriendList,
    GameDetail,
    Interaction,
    OwnedGame,
    OwnedGamesResponse,
    PlayerSummary,
    RecentlyPlayedResponse,
    RecommendationItem,
    RecommendationResponse,
)


def test_player_summary_minimal() -> None:
    s = PlayerSummary(steamid="76561")
    assert s.steamid == "76561"
    assert s.personaname is None


def test_owned_game_default_zero() -> None:
    g = OwnedGame(appid=440)
    assert g.playtime_forever == 0
    assert g.playtime_2weeks == 0


def test_owned_games_response_default_empty() -> None:
    r = OwnedGamesResponse(steamid="x")
    assert r.game_count == 0
    assert r.games == []


def test_recently_played_response_default_empty() -> None:
    r = RecentlyPlayedResponse(steamid="x")
    assert r.total_count == 0
    assert r.games == []


def test_friend_list_default_empty() -> None:
    fl = FriendList(steamid="x")
    assert fl.friends == []


def test_friend_edge_optional_fields() -> None:
    edge = FriendEdge(steamid="y")
    assert edge.relationship is None


def test_game_detail_from_payload_basic() -> None:
    payload = {
        "steam_appid": 730,
        "name": "Counter-Strike",
        "header_image": "img",
        "short_description": "fps",
        "genres": [{"description": "Action"}],
        "categories": [{"description": "Multi-player"}],
        "release_date": {"date": "2012"},
        "metacritic": {"score": 83},
    }
    detail = GameDetail.from_steam_payload(payload)
    assert detail.steam_appid == 730
    assert detail.genres == ["Action"]
    assert detail.categories == ["Multi-player"]
    assert detail.release_date == "2012"
    assert detail.metacritic_score == 83


def test_game_detail_from_payload_missing_optionals() -> None:
    detail = GameDetail.from_steam_payload({"steam_appid": 1, "name": "X"})
    assert detail.release_date is None
    assert detail.metacritic_score is None
    assert detail.genres == []


def test_interaction_validates() -> None:
    inter = Interaction(user_idx=1, game_idx=2, playtime_minutes=10, event_ts=datetime.utcnow())
    assert inter.confidence == pytest.approx(1.0)


def test_interaction_rejects_negative_playtime() -> None:
    with pytest.raises(ValueError):
        Interaction(user_idx=1, game_idx=2, playtime_minutes=-1, event_ts=datetime.utcnow())


def test_recommendation_item_round_trip() -> None:
    item = RecommendationItem(steam_appid=440, name="TF2", header_image=None, score=0.9)
    assert item.model_dump()["score"] == 0.9


def test_recommendation_response_packs_items() -> None:
    item = RecommendationItem(steam_appid=1, name="x", header_image=None, score=1.0)
    resp = RecommendationResponse(user_id="u", served_from="cache", latency_ms=12.5, items=[item])
    assert len(resp.items) == 1
    assert resp.served_from == "cache"
