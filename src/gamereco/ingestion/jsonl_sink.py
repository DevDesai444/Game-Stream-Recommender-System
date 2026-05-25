"""Newline-delimited JSON sink used by the ingestion stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlSink:
    """Append-only writer that lands one JSON object per line.

    The bronze layer is intentionally raw newline-delimited JSON. Spark reads
    these files natively via `spark.read.json(...)` and we keep them around
    for replay/debugging without any schema enforcement.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._count = 0

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str))
        self._fh.write("\n")
        self._count += 1

    def flush(self) -> None:
        self._fh.flush()

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
