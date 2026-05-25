"""Loader for Julian McAuley's UCSD Steam dataset.

The UCSD release carries genuine content metadata on both sides of
the (user, item) graph — genres, tags, specs, price, release date,
developer/publisher on the item side; owned-game count, total
playtime, review count, average review helpfulness on the user side.
This is what makes the content-aware two-tower model possible (steam-200k only
had user_id, game_name, behavior, value — embedding-only territory).

Three source files, all Python ``repr``-style (single-quoted,
``u'...'`` literals) rather than JSON, parsed with
:func:`ast.literal_eval`:

  steam_games.json.gz              ~32,000 games × ~10 content fields
  australian_users_items.json.gz   ~88,000 users × owned-games arrays
  australian_user_reviews.json.gz  ~26,000 users × reviews (recommend Y/N)

The :func:`load_ucsd` entrypoint returns four DataFrames that match
the contract expected by the rest of :mod:`gamereco.training`:

  interactions   user_idx, game_idx, playtime_forever, playtime_2weeks,
                 confidence, recommended (nullable bool)
  games          game_idx, app_id, name, genres (list), tags (list),
                 specs (list), price, release_year, developer, publisher
  users          user_idx, user_id, items_count, total_playtime,
                 reviews_count
  reviews        user_idx, game_idx, recommend (bool), review_text

Compact integer indices are dense (``set(user_idx) == range(n_users)``)
so they can drop straight into the existing ALS / NCF / XGBoost
pipeline.
"""

from __future__ import annotations

import ast
import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gamereco.common.logging import get_logger

log = get_logger(__name__)


_DEFAULT_MIN_USER_ITEMS = 5
_DEFAULT_MIN_GAME_USERS = 3
_RELEASE_YEAR_RE = re.compile(r"(\d{4})")


@dataclass(frozen=True)
class UCSDLoadConfig:
    games_path: Path
    users_items_path: Path
    reviews_path: Path
    min_interactions_per_user: int = _DEFAULT_MIN_USER_ITEMS
    min_interactions_per_game: int = _DEFAULT_MIN_GAME_USERS
    max_users: int | None = None  # cap for laptop runs


@dataclass(frozen=True)
class UCSDLoadResult:
    interactions: pd.DataFrame
    games: pd.DataFrame
    users: pd.DataFrame
    reviews: pd.DataFrame
    n_users: int
    n_games: int


def _iter_repr_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a gzipped Python-repr file, one dict per line."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                # A handful of rows in the UCSD dump are broken; skip them
                # rather than abort the whole load.
                continue
            if isinstance(obj, dict):
                yield obj


def _parse_release_year(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        year = int(raw)
        if 1980 <= year <= datetime.utcnow().year + 2:
            return year
        return None
    if isinstance(raw, str):
        m = _RELEASE_YEAR_RE.search(raw)
        if m:
            year = int(m.group(1))
            if 1980 <= year <= datetime.utcnow().year + 2:
                return year
    return None


def _parse_price(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        # Common patterns: "Free", "Free to Play", "Free Demo", "$4.99", ...
        s = raw.strip().lower()
        if "free" in s:
            return 0.0
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
    return None


def _load_games(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in _iter_repr_lines(path):
        app_id = obj.get("id") or obj.get("app_id")
        if app_id is None:
            continue
        try:
            app_id = int(app_id)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "app_id": app_id,
                "name": obj.get("app_name") or obj.get("title") or "",
                "genres": list(obj.get("genres") or []),
                "tags": list(obj.get("tags") or []),
                "specs": list(obj.get("specs") or []),
                "price": _parse_price(obj.get("price")),
                "release_year": _parse_release_year(obj.get("release_date")),
                "developer": obj.get("developer") or "",
                "publisher": obj.get("publisher") or "",
                "sentiment": obj.get("sentiment") or "",
                "early_access": bool(obj.get("early_access") or False),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["app_id"]).reset_index(drop=True)
    log.info("ucsd.games_loaded", n=len(df))
    return df


def _load_users_items(path: Path, max_users: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen = 0
    for obj in _iter_repr_lines(path):
        user_id = obj.get("user_id") or obj.get("steam_id")
        if not user_id:
            continue
        items = obj.get("items") or []
        if not isinstance(items, list) or not items:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            app_id = item.get("item_id") or item.get("appid")
            if app_id is None:
                continue
            try:
                app_id = int(app_id)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "user_id": str(user_id),
                    "app_id": app_id,
                    "playtime_forever": int(item.get("playtime_forever") or 0),
                    "playtime_2weeks": int(item.get("playtime_2weeks") or 0),
                }
            )
        seen += 1
        if max_users is not None and seen >= max_users:
            break
    df = pd.DataFrame(rows)
    log.info("ucsd.users_items_loaded", n_users=seen, n_rows=len(df))
    return df


def _load_reviews(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in _iter_repr_lines(path):
        user_id = obj.get("user_id")
        if not user_id:
            continue
        reviews = obj.get("reviews") or []
        if not isinstance(reviews, list):
            continue
        for r in reviews:
            if not isinstance(r, dict):
                continue
            app_id = r.get("item_id")
            if app_id is None:
                continue
            try:
                app_id = int(app_id)
            except (TypeError, ValueError):
                continue
            recommend = r.get("recommend")
            rows.append(
                {
                    "user_id": str(user_id),
                    "app_id": app_id,
                    "recommend": bool(recommend) if recommend is not None else None,
                    "review_text": (r.get("review") or "")[:1000],
                }
            )
    df = pd.DataFrame(rows)
    log.info("ucsd.reviews_loaded", n=len(df))
    return df


def _assign_indices(
    interactions: pd.DataFrame,
    *,
    min_users: int,
    min_games: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, int]]:
    """Iteratively prune sparse rows then assign dense integer indices."""
    df = interactions.copy()
    # One pass is enough in practice (the second pass converges instantly).
    for _ in range(3):
        u_counts = df.groupby("user_id").size()
        g_counts = df.groupby("app_id").size()
        qualified_users = u_counts[u_counts >= min_users].index
        qualified_games = g_counts[g_counts >= min_games].index
        before = len(df)
        df = df[df["user_id"].isin(qualified_users) & df["app_id"].isin(qualified_games)]
        if len(df) == before:
            break

    user_index = {u: i for i, u in enumerate(sorted(df["user_id"].unique()))}
    game_index = {g: i for i, g in enumerate(sorted(df["app_id"].unique()))}
    df["user_idx"] = df["user_id"].map(user_index).astype(np.int32)
    df["game_idx"] = df["app_id"].map(game_index).astype(np.int32)
    return df.reset_index(drop=True), user_index, game_index


def _build_user_features(df: pd.DataFrame, user_index: dict[str, int]) -> pd.DataFrame:
    rows = []
    for user_id, group in df.groupby("user_id"):
        idx = user_index.get(user_id)
        if idx is None:
            continue
        rows.append(
            {
                "user_idx": idx,
                "user_id": user_id,
                "items_count": int(len(group)),
                "total_playtime": int(group["playtime_forever"].sum()),
                "active_recent": int((group["playtime_2weeks"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("user_idx").reset_index(drop=True)


def _build_game_features(
    interactions: pd.DataFrame,
    game_meta: pd.DataFrame,
    game_index: dict[int, int],
) -> pd.DataFrame:
    meta = game_meta[game_meta["app_id"].isin(game_index.keys())].copy()
    meta["game_idx"] = meta["app_id"].map(game_index).astype(np.int32)
    cols = [
        "game_idx",
        "app_id",
        "name",
        "genres",
        "tags",
        "specs",
        "price",
        "release_year",
        "developer",
        "publisher",
        "sentiment",
        "early_access",
    ]
    # Games that appear in interactions but not in game_meta: synthesise a
    # placeholder row so every game_idx has metadata.
    present_app_ids = set(meta["app_id"].astype(int))
    missing = []
    for app_id, idx in game_index.items():
        if app_id not in present_app_ids:
            missing.append(
                {
                    "game_idx": idx,
                    "app_id": app_id,
                    "name": f"app_{app_id}",
                    "genres": [],
                    "tags": [],
                    "specs": [],
                    "price": None,
                    "release_year": None,
                    "developer": "",
                    "publisher": "",
                    "sentiment": "",
                    "early_access": False,
                }
            )
    if missing:
        missing_df = pd.DataFrame(missing)[cols]
        # Match dtypes so concat doesn't warn about object/string coercion.
        for col in cols:
            if col in meta.columns:
                missing_df[col] = missing_df[col].astype(meta[col].dtype, errors="ignore")
        meta = pd.concat([meta[cols], missing_df], ignore_index=True)
    else:
        meta = meta[cols]
    return meta.sort_values("game_idx").reset_index(drop=True)


def load_ucsd(config: UCSDLoadConfig) -> UCSDLoadResult:
    games = _load_games(config.games_path)
    interactions = _load_users_items(config.users_items_path, config.max_users)
    if interactions.empty:
        raise ValueError("no user-items rows parsed from UCSD dataset")

    # Confidence — implicit feedback. log1p(playtime) gives ALS / NCF
    # something proportional to play depth without letting one whale
    # user dominate the loss.
    interactions["confidence"] = np.log1p(
        interactions["playtime_forever"].to_numpy(dtype=np.float64)
    )
    # Floor for the "owned-but-never-played" rows so they still register.
    interactions.loc[interactions["confidence"] == 0.0, "confidence"] = np.log1p(1.0)
    # Alias to the column name used by the Spark-side silver schema so
    # the same hybrid harness ingests both datasets without a branch.
    # 'playtime_forever' as returned by the Steam Web API is in minutes.
    interactions["playtime_minutes"] = interactions["playtime_forever"]

    interactions, user_index, game_index = _assign_indices(
        interactions,
        min_users=config.min_interactions_per_user,
        min_games=config.min_interactions_per_game,
    )
    if interactions.empty or not user_index or not game_index:
        raise ValueError(
            "no interactions survived the activity floor "
            f"(min_user_interactions={config.min_interactions_per_user}, "
            f"min_game_users={config.min_interactions_per_game})"
        )

    users = _build_user_features(interactions, user_index)
    games_out = _build_game_features(interactions, games, game_index)

    reviews_raw = _load_reviews(config.reviews_path)
    if not reviews_raw.empty:
        reviews_raw["user_idx"] = reviews_raw["user_id"].map(user_index)
        reviews_raw["game_idx"] = reviews_raw["app_id"].map(game_index)
        reviews = (
            reviews_raw.dropna(subset=["user_idx", "game_idx"])
            .astype({"user_idx": np.int32, "game_idx": np.int32})
            .reset_index(drop=True)
        )
        reviews_count = reviews.groupby("user_idx").size().rename("reviews_count").reset_index()
        users = users.merge(reviews_count, on="user_idx", how="left")
        users["reviews_count"] = users["reviews_count"].fillna(0).astype(int)
    else:
        reviews = reviews_raw
        users["reviews_count"] = 0

    log.info(
        "ucsd.loaded",
        n_users=len(user_index),
        n_games=len(game_index),
        n_interactions=len(interactions),
        n_reviews=len(reviews),
    )
    return UCSDLoadResult(
        interactions=interactions,
        games=games_out,
        users=users,
        reviews=reviews,
        n_users=len(user_index),
        n_games=len(game_index),
    )


def temporal_split_ucsd(
    interactions: pd.DataFrame,
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split using cumulative playtime as the time axis.

    The UCSD dump has no per-event timestamps, so we use the same
    proxy as :func:`gamereco.datasets.steam200k.temporal_split_pandas`
    — sort each user's interactions by ``playtime_forever`` ascending
    (cumulative playtime monotonically grows over time), then hold
    out the tail.
    """
    if not 0.0 < val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    df = interactions.copy()
    df["_tie"] = df["user_idx"].astype(int) * 1_000_003 + df["game_idx"].astype(int)
    df = df.sort_values(["user_idx", "playtime_forever", "_tie"], ascending=True)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("user_idx", sort=False):
        n = len(group)
        if n < 3:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:
            n_val = max(1, n - n_test - 1)
        test_parts.append(group.tail(n_test))
        if n_val > 0:
            val_parts.append(group.iloc[-(n_test + n_val) : -n_test])
        train_parts.append(group.iloc[: n - n_test - n_val])
    drop = ["_tie"]
    return (
        pd.concat(train_parts).drop(columns=drop).reset_index(drop=True),
        (
            pd.concat(val_parts).drop(columns=drop).reset_index(drop=True)
            if val_parts
            else df.iloc[0:0].drop(columns=drop)
        ),
        (
            pd.concat(test_parts).drop(columns=drop).reset_index(drop=True)
            if test_parts
            else df.iloc[0:0].drop(columns=drop)
        ),
    )
