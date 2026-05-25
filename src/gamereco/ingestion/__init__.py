"""Async Steam Web API ingestion."""

from gamereco.ingestion.pipeline import IngestionPipeline
from gamereco.ingestion.steam_client import SteamClient

__all__ = ["SteamClient", "IngestionPipeline"]
