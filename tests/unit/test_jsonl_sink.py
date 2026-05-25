"""Tests for the NDJSON ingestion sink."""

from __future__ import annotations

import json
from pathlib import Path

from gamereco.ingestion.jsonl_sink import JsonlSink


def test_sink_appends_records(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "a.jsonl")
    sink.write({"x": 1})
    sink.write({"x": 2})
    sink.close()
    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert rows == [{"x": 1}, {"x": 2}]


def test_sink_count(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "a.jsonl")
    for _ in range(5):
        sink.write({"v": 1})
    assert sink.count == 5
    sink.close()


def test_sink_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    with JsonlSink(path) as sink:
        sink.write({"k": "v"})
    assert path.exists()
    assert path.read_text().strip() == '{"k": "v"}'


def test_sink_creates_parent(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "a.jsonl"
    sink = JsonlSink(path)
    sink.write({})
    sink.close()
    assert path.exists()
