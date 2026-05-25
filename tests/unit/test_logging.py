"""Tests for the structured logging bootstrap."""

from __future__ import annotations

import logging

from gamereco.common.logging import configure_logging, get_logger


def test_configure_logging_sets_root_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_get_logger_returns_bound_logger() -> None:
    configure_logging("INFO")
    log = get_logger("gamereco.tests")
    assert hasattr(log, "info")


def test_configure_defaults_to_info(env_overrides) -> None:
    import os

    os.environ.pop("LOG_LEVEL", None)
    configure_logging()
    assert logging.getLogger().level == logging.INFO
