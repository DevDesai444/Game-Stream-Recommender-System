"""Canonical paths for the medallion data lake layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LakePaths:
    """Medallion layout under a Delta Lake root."""

    root: Path

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def bronze_users(self) -> Path:
        return self.bronze / "users"

    @property
    def bronze_owned_games(self) -> Path:
        return self.bronze / "owned_games"

    @property
    def bronze_recently_played(self) -> Path:
        return self.bronze / "recently_played"

    @property
    def bronze_friends(self) -> Path:
        return self.bronze / "friends"

    @property
    def bronze_game_detail(self) -> Path:
        return self.bronze / "game_detail"

    @property
    def silver_interactions(self) -> Path:
        return self.silver / "interactions"

    @property
    def silver_games(self) -> Path:
        return self.silver / "games"

    @property
    def silver_users(self) -> Path:
        return self.silver / "users"

    @property
    def gold_train(self) -> Path:
        return self.gold / "interactions_train"

    @property
    def gold_val(self) -> Path:
        return self.gold / "interactions_val"

    @property
    def gold_test(self) -> Path:
        return self.gold / "interactions_test"

    @property
    def gold_user_clusters(self) -> Path:
        return self.gold / "user_clusters"

    @property
    def gold_game_embeddings(self) -> Path:
        return self.gold / "game_embeddings"

    def ensure(self) -> None:
        for path in (self.bronze, self.silver, self.gold):
            path.mkdir(parents=True, exist_ok=True)


def from_env(root: str | Path) -> LakePaths:
    return LakePaths(root=Path(root))
