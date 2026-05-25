"""High-throughput async ingestion pipeline targeting 50K+ Steam users."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from gamereco.common.logging import get_logger
from gamereco.common.paths import LakePaths
from gamereco.ingestion.jsonl_sink import JsonlSink
from gamereco.ingestion.steam_client import SteamClient

log = get_logger(__name__)


@dataclass(slots=True)
class IngestionStats:
    users: int = 0
    owned_games: int = 0
    recently_played: int = 0
    friends: int = 0
    game_details: int = 0


class IngestionPipeline:
    """Coordinates async fans-out to the Steam Web API and lands NDJSON files."""

    def __init__(self, client: SteamClient, lake: LakePaths) -> None:
        self._client = client
        self._lake = lake
        self._lake.bronze.mkdir(parents=True, exist_ok=True)

    async def ingest_users(self, steam_ids: Iterable[str]) -> IngestionStats:
        """Fan out async requests for all 4 user-level endpoints in parallel."""
        steam_ids = list(steam_ids)
        if not steam_ids:
            return IngestionStats()

        summary_path = self._lake.bronze_users / "user_summary.jsonl"
        owned_path = self._lake.bronze_owned_games / "owned_games.jsonl"
        recent_path = self._lake.bronze_recently_played / "recently_played.jsonl"
        friends_path = self._lake.bronze_friends / "friends.jsonl"
        for p in (summary_path, owned_path, recent_path, friends_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        stats = IngestionStats()
        with (
            JsonlSink(summary_path) as summary_sink,
            JsonlSink(owned_path) as owned_sink,
            JsonlSink(recent_path) as recent_sink,
            JsonlSink(friends_path) as friends_sink,
        ):

            async def _process_user(steamid: str) -> None:
                summary, owned, recent, friends = await asyncio.gather(
                    self._client.player_summary(steamid),
                    self._client.owned_games(steamid),
                    self._client.recently_played(steamid),
                    self._client.friend_list(steamid),
                    return_exceptions=False,
                )
                summary_sink.write(summary)
                owned_sink.write(owned)
                recent_sink.write(recent)
                friends_sink.write(friends)
                stats.users += 1
                stats.owned_games += len(owned.get("games", []))
                stats.recently_played += len(recent.get("games", []))
                stats.friends += len(friends.get("friends", []))
                if stats.users % 500 == 0:
                    log.info("ingest.progress", **asdict(stats))

            await asyncio.gather(*(_process_user(sid) for sid in steam_ids))

        log.info("ingest.complete", **asdict(stats))
        return stats

    async def ingest_game_details(
        self, app_ids: Iterable[int], *, sink_path: Path | None = None
    ) -> int:
        sink_path = sink_path or (self._lake.bronze_game_detail / "game_detail.jsonl")
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with JsonlSink(sink_path) as sink:

            async def _process(appid: int) -> None:
                nonlocal count
                data = await self._client.app_details(appid)
                if data is None:
                    return
                sink.write(data)
                count += 1

            await asyncio.gather(*(_process(int(a)) for a in app_ids))

        log.info("ingest.game_details_complete", count=count)
        return count
