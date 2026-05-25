"""Async client for the Steam Web API and Storefront API.

This client uses aiohttp + a semaphore to keep concurrency under control and
tenacity to retry transient errors. Steam's free-tier rate limits are strict
(roughly 100k calls / day, with bursts throttled at the IP level), so a
modest concurrency level — combined with exponential backoff on 429/5xx —
is enough to sustain ingestion of 50K+ users without being banned.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from gamereco.common.logging import get_logger

log = get_logger(__name__)

WEB_API = "https://api.steampowered.com"
STORE_API = "https://store.steampowered.com/api"

DEFAULT_HEADERS = {
    "User-Agent": "gamereco/0.2 (+https://github.com/DevDesai-444/Game-Stream-Recommender-System)",
    "Accept": "application/json",
}


class RetryableSteamError(RuntimeError):
    """Raised when a Steam endpoint returns 429/5xx and the caller should back off."""


@dataclass(slots=True)
class SteamClientConfig:
    api_key: str
    concurrency: int = 64
    request_timeout_s: float = 10.0
    max_attempts: int = 5


class SteamClient:
    """Async Steam client with bounded concurrency and retry/backoff."""

    def __init__(self, config: SteamClientConfig) -> None:
        if not config.api_key:
            raise ValueError("Steam API key is required")
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def __aenter__(self) -> SteamClient:
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout, headers=DEFAULT_HEADERS)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("SteamClient must be used as an async context manager")
        return self._session

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = {**(params or {}), "key": self._config.api_key}
        async with self._semaphore:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._config.max_attempts),
                    wait=wait_exponential_jitter(initial=0.5, max=15),
                    retry=retry_if_exception_type(
                        (RetryableSteamError, aiohttp.ClientError, asyncio.TimeoutError)
                    ),
                    reraise=True,
                ):
                    with attempt:
                        async with self.session.get(url, params=params) as resp:
                            if resp.status == 429 or 500 <= resp.status < 600:
                                raise RetryableSteamError(f"{url} -> HTTP {resp.status}")
                            resp.raise_for_status()
                            return await resp.json(content_type=None)
            except RetryError as exc:
                log.warning("steam.retry_exhausted", url=url, error=str(exc))
                raise
        return {}

    async def player_summary(self, steamid: str) -> dict[str, Any]:
        url = f"{WEB_API}/ISteamUser/GetPlayerSummaries/v0002/"
        payload = await self._get_json(url, params={"steamids": steamid})
        players = payload.get("response", {}).get("players", [])
        if not players:
            return {"steamid": steamid}
        return players[0]

    async def owned_games(self, steamid: str) -> dict[str, Any]:
        url = f"{WEB_API}/IPlayerService/GetOwnedGames/v0001/"
        payload = await self._get_json(
            url,
            params={
                "steamid": steamid,
                "include_appinfo": 1,
                "include_played_free_games": 1,
            },
        )
        resp = payload.get("response", {})
        return {
            "steamid": steamid,
            "game_count": resp.get("game_count", 0),
            "games": resp.get("games", []),
        }

    async def recently_played(self, steamid: str) -> dict[str, Any]:
        url = f"{WEB_API}/IPlayerService/GetRecentlyPlayedGames/v0001/"
        payload = await self._get_json(url, params={"steamid": steamid})
        resp = payload.get("response", {})
        return {
            "steamid": steamid,
            "total_count": resp.get("total_count", 0),
            "games": resp.get("games", []),
        }

    async def friend_list(self, steamid: str) -> dict[str, Any]:
        url = f"{WEB_API}/ISteamUser/GetFriendList/v0001/"
        try:
            payload = await self._get_json(
                url, params={"steamid": steamid, "relationship": "friend"}
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status == 401:
                return {"steamid": steamid, "friends": []}
            raise
        friends = payload.get("friendslist", {}).get("friends", [])
        return {"steamid": steamid, "friends": friends}

    async def app_list(self) -> list[dict[str, Any]]:
        url = f"{WEB_API}/ISteamApps/GetAppList/v2/"
        payload = await self._get_json(url)
        return payload.get("applist", {}).get("apps", [])

    async def app_details(self, appid: int) -> dict[str, Any] | None:
        url = f"{STORE_API}/appdetails"
        params = {"appids": str(appid)}
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._config.max_attempts),
                wait=wait_exponential_jitter(initial=0.5, max=15),
                retry=retry_if_exception_type(
                    (RetryableSteamError, aiohttp.ClientError, asyncio.TimeoutError)
                ),
                reraise=True,
            ):
                with attempt:
                    async with self.session.get(url, params=params) as resp:
                        if resp.status == 429 or 500 <= resp.status < 600:
                            raise RetryableSteamError(f"appdetails {appid} -> HTTP {resp.status}")
                        resp.raise_for_status()
                        payload = await resp.json(content_type=None)
        node = payload.get(str(appid))
        if not node or not node.get("success"):
            return None
        return node.get("data")
