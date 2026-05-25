"""Public dataset loaders (Steam-200k, etc.)."""

from gamereco.datasets.steam200k import (
    Steam200kRecord,
    load_steam_200k,
    materialise_silver,
    temporal_split_pandas,
)
from gamereco.datasets.steam_ucsd import (
    UCSDLoadConfig,
    UCSDLoadResult,
    load_ucsd,
    temporal_split_ucsd,
)

__all__ = [
    "Steam200kRecord",
    "UCSDLoadConfig",
    "UCSDLoadResult",
    "load_steam_200k",
    "load_ucsd",
    "materialise_silver",
    "temporal_split_pandas",
    "temporal_split_ucsd",
]
