"""Tests for the async Steam client."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gamereco.ingestion.steam_client import SteamClient, SteamClientConfig


def _client() -> SteamClient:
    return SteamClient(SteamClientConfig(api_key="test", concurrency=2, max_attempts=2))


def test_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        SteamClient(SteamClientConfig(api_key=""))


def test_client_session_raises_outside_context() -> None:
    c = _client()
    with pytest.raises(RuntimeError):
        _ = c.session


@pytest.mark.asyncio
async def test_player_summary_returns_first_player() -> None:
    client = _client()

    async def fake_get(self, url, params=None):  # noqa: ARG001
        return {"response": {"players": [{"steamid": "1", "personaname": "alice"}]}}

    with patch.object(SteamClient, "_get_json", new=fake_get):
        async with client:
            summary = await client.player_summary("1")
            assert summary["personaname"] == "alice"


@pytest.mark.asyncio
async def test_player_summary_empty_response_falls_back() -> None:
    client = _client()

    async def fake_get(self, url, params=None):  # noqa: ARG001
        return {"response": {"players": []}}

    with patch.object(SteamClient, "_get_json", new=fake_get):
        async with client:
            summary = await client.player_summary("42")
            assert summary == {"steamid": "42"}


@pytest.mark.asyncio
async def test_owned_games_passes_through_fields() -> None:
    client = _client()

    async def fake_get(self, url, params=None):  # noqa: ARG001
        return {"response": {"game_count": 3, "games": [{"appid": 1}, {"appid": 2}]}}

    with patch.object(SteamClient, "_get_json", new=fake_get):
        async with client:
            owned = await client.owned_games("1")
            assert owned["game_count"] == 3
            assert len(owned["games"]) == 2


@pytest.mark.asyncio
async def test_recently_played_handles_missing_response() -> None:
    client = _client()

    async def fake_get(self, url, params=None):  # noqa: ARG001
        return {"response": {}}

    with patch.object(SteamClient, "_get_json", new=fake_get):
        async with client:
            recent = await client.recently_played("1")
            assert recent["total_count"] == 0
            assert recent["games"] == []


@pytest.mark.asyncio
async def test_friend_list_returns_friends() -> None:
    client = _client()

    async def fake_get(self, url, params=None):  # noqa: ARG001
        return {"friendslist": {"friends": [{"steamid": "2"}]}}

    with patch.object(SteamClient, "_get_json", new=fake_get):
        async with client:
            friends = await client.friend_list("1")
            assert len(friends["friends"]) == 1


@pytest.mark.asyncio
async def test_app_list_returns_apps() -> None:
    client = _client()

    async def fake_get(self, url, params=None):  # noqa: ARG001
        return {"applist": {"apps": [{"appid": 10, "name": "A"}]}}

    with patch.object(SteamClient, "_get_json", new=fake_get):
        async with client:
            apps = await client.app_list()
            assert apps[0]["appid"] == 10
