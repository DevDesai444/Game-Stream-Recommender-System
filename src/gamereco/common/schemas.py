"""Pydantic models shared across ingestion, training, and serving layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlayerSummary(BaseModel):
    steamid: str
    personaname: str | None = None
    profileurl: str | None = None
    avatar: str | None = None
    loccountrycode: str | None = None
    timecreated: int | None = None

    model_config = ConfigDict(extra="ignore")


class OwnedGame(BaseModel):
    appid: int
    name: str | None = None
    playtime_forever: int = 0
    playtime_2weeks: int = 0
    img_icon_url: str | None = None

    model_config = ConfigDict(extra="ignore")


class OwnedGamesResponse(BaseModel):
    steamid: str
    game_count: int = 0
    games: list[OwnedGame] = Field(default_factory=list)


class RecentlyPlayedResponse(BaseModel):
    steamid: str
    total_count: int = 0
    games: list[OwnedGame] = Field(default_factory=list)


class FriendEdge(BaseModel):
    steamid: str
    relationship: str | None = None
    friend_since: int | None = None


class FriendList(BaseModel):
    steamid: str
    friends: list[FriendEdge] = Field(default_factory=list)


class GameDetail(BaseModel):
    steam_appid: int
    name: str
    header_image: str | None = None
    short_description: str | None = None
    genres: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    release_date: str | None = None
    metacritic_score: int | None = None

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_steam_payload(cls, payload: dict[str, Any]) -> "GameDetail":
        genres = [g["description"] for g in payload.get("genres", []) if "description" in g]
        categories = [
            c["description"] for c in payload.get("categories", []) if "description" in c
        ]
        release_date = payload.get("release_date", {}).get("date") if payload.get(
            "release_date"
        ) else None
        metacritic = payload.get("metacritic", {}).get("score") if payload.get("metacritic") else None
        return cls(
            steam_appid=int(payload["steam_appid"]),
            name=payload.get("name", ""),
            header_image=payload.get("header_image"),
            short_description=payload.get("short_description"),
            genres=genres,
            categories=categories,
            release_date=release_date,
            metacritic_score=metacritic,
        )


class Interaction(BaseModel):
    """Implicit-feedback interaction record used during ETL."""

    user_idx: int
    game_idx: int
    playtime_minutes: int = Field(ge=0)
    event_ts: datetime
    confidence: float = Field(ge=0.0, default=1.0)


class RecommendationItem(BaseModel):
    steam_appid: int
    name: str
    header_image: str | None = None
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    served_from: str
    latency_ms: float
    items: list[RecommendationItem]
