"""CLI entrypoint: ``gamereco-ingest``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from gamereco.common.config import spark_settings, steam_settings
from gamereco.common.logging import configure_logging, get_logger
from gamereco.common.paths import LakePaths
from gamereco.ingestion.discovery import discover_steam_ids
from gamereco.ingestion.pipeline import IngestionPipeline
from gamereco.ingestion.steam_client import SteamClient, SteamClientConfig

log = get_logger(__name__)


@click.group()
def cli() -> None:
    """Steam ingestion CLI."""
    configure_logging()


@cli.command("discover")
@click.option("--pages", default=200, type=int, help="Community member pages to scan")
@click.option("--target", default=None, type=int, help="Stop once this many ids are collected")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/delta/bronze/users/seed_steam_ids.jsonl"),
)
def discover(pages: int, target: int | None, out: Path) -> None:
    """Scrape Steam community pages to seed an ingestion run."""

    async def _run() -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        seen = 0
        with out.open("w", encoding="utf-8") as fh:
            async for sid in discover_steam_ids(pages=pages, target=target):
                fh.write(json.dumps({"steamid": sid}))
                fh.write("\n")
                seen += 1
        return seen

    n = asyncio.run(_run())
    click.echo(f"discovered {n} steamids -> {out}")


@cli.command("users")
@click.option(
    "--seed",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file with {'steamid': '...'} lines",
)
def ingest_users(seed: Path) -> None:
    """Fetch summary/owned/recent/friends for each steam id in `seed`."""
    cfg = steam_settings()
    lake = LakePaths(root=Path(spark_settings().delta_root))

    async def _run() -> None:
        client = SteamClient(SteamClientConfig(api_key=cfg.api_key, concurrency=cfg.concurrency))
        async with client:
            pipeline = IngestionPipeline(client, lake)
            ids = [json.loads(line)["steamid"] for line in seed.read_text().splitlines() if line]
            await pipeline.ingest_users(ids)

    asyncio.run(_run())


@cli.command("games")
@click.option("--limit", default=10000, type=int, help="Cap of app ids to enrich")
def ingest_games(limit: int) -> None:
    """Enrich `limit` app ids with game-detail metadata."""
    cfg = steam_settings()
    lake = LakePaths(root=Path(spark_settings().delta_root))

    async def _run() -> None:
        client = SteamClient(
            SteamClientConfig(api_key=cfg.api_key, concurrency=max(8, cfg.concurrency // 4))
        )
        async with client:
            pipeline = IngestionPipeline(client, lake)
            apps = await client.app_list()
            app_ids = [a["appid"] for a in apps[:limit]]
            await pipeline.ingest_game_details(app_ids)

    asyncio.run(_run())


def main() -> None:  # pragma: no cover - thin wrapper
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
