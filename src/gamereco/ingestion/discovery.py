"""Steam user discovery via the public community member pages."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

import aiohttp
from bs4 import BeautifulSoup
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from gamereco.common.logging import get_logger

log = get_logger(__name__)

MEMBER_PAGE = "https://steamcommunity.com/games/steam/members?p={page}"
STEAMID_RE = re.compile(r'"steamid":"(\d+)"')
PROFILE_RE = re.compile(r"top\.location\.href='(https://steamcommunity\.com/[^']+)'")


async def _fetch_text(session: aiohttp.ClientSession, url: str) -> str:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=0.5, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    ):
        with attempt:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.text()
    return ""


async def _resolve_steamid(session: aiohttp.ClientSession, profile_url: str) -> str | None:
    text = await _fetch_text(session, profile_url)
    match = STEAMID_RE.search(text)
    return match.group(1) if match else None


async def discover_steam_ids(
    pages: int,
    *,
    concurrency: int = 16,
    target: int | None = None,
) -> AsyncIterator[str]:
    """Yield Steam IDs scraped from `pages` community member pages.

    Stops early once `target` IDs have been emitted.
    """
    timeout = aiohttp.ClientTimeout(total=20)
    headers = {"User-Agent": "gamereco/0.2 (+research)"}
    seen: set[str] = set()
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for page in range(1, pages + 1):
            html = await _fetch_text(session, MEMBER_PAGE.format(page=page))
            soup = BeautifulSoup(html, "html.parser")
            profiles: list[str] = []
            for node in soup.find_all("div", onclick=PROFILE_RE):
                m = PROFILE_RE.search(node.get("onclick", ""))
                if m:
                    profiles.append(m.group(1))

            async def _process(profile_url: str) -> str | None:
                async with semaphore:
                    return await _resolve_steamid(session, profile_url)

            results = await asyncio.gather(*(_process(p) for p in profiles))
            for sid in results:
                if sid and sid not in seen:
                    seen.add(sid)
                    yield sid
                    if target is not None and len(seen) >= target:
                        log.info("discovery.target_reached", target=target)
                        return

            log.info("discovery.page_complete", page=page, total=len(seen))
