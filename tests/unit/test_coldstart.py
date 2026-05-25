"""Tests for the cold-start fallback chain."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gamereco.common.schemas import RecommendationItem
from gamereco.serving.coldstart import ResolvedRecommendations, resolve


def _item(appid: int) -> RecommendationItem:
    return RecommendationItem(steam_appid=appid, name=f"g-{appid}", header_image=None, score=0.5)


@pytest.fixture
def store() -> MagicMock:
    return MagicMock()


def test_personal_layer_short_circuits(store: MagicMock) -> None:
    store.fetch_user_recommendations.return_value = [_item(1), _item(2)]
    resolved = resolve(store, "u", limit=10)
    assert resolved.served_from == "personal"
    assert [i.steam_appid for i in resolved.items] == [1, 2]
    store.user_cohort.assert_not_called()
    store.cohort_top.assert_not_called()
    store.global_top.assert_not_called()


def test_falls_through_to_cohort_when_personal_empty(store: MagicMock) -> None:
    store.fetch_user_recommendations.return_value = []
    store.user_cohort.return_value = 7
    store.cohort_top.return_value = [_item(11), _item(12)]
    resolved = resolve(store, "u", limit=10)
    assert resolved.served_from == "cohort"
    assert resolved.cohort_id == 7
    assert len(resolved.items) == 2
    store.global_top.assert_not_called()


def test_falls_through_to_global_when_cohort_empty(store: MagicMock) -> None:
    store.fetch_user_recommendations.return_value = []
    store.user_cohort.return_value = 7
    store.cohort_top.return_value = []
    store.global_top.return_value = [_item(99)]
    resolved = resolve(store, "u", limit=10)
    assert resolved.served_from == "global_fallback"
    assert resolved.cohort_id == 7
    assert resolved.items[0].steam_appid == 99


def test_falls_through_to_global_when_cohort_unknown(store: MagicMock) -> None:
    store.fetch_user_recommendations.return_value = []
    store.user_cohort.return_value = None
    store.global_top.return_value = [_item(50)]
    resolved = resolve(store, "u", limit=10)
    assert resolved.served_from == "global_fallback"
    assert resolved.cohort_id is None
    store.cohort_top.assert_not_called()


def test_returns_empty_when_nothing_anywhere(store: MagicMock) -> None:
    store.fetch_user_recommendations.return_value = []
    store.user_cohort.return_value = None
    store.global_top.return_value = []
    resolved = resolve(store, "u", limit=10)
    assert resolved.items == []
    assert resolved.served_from == "global_fallback"


def test_limit_is_forwarded_to_each_layer(store: MagicMock) -> None:
    store.fetch_user_recommendations.return_value = []
    store.user_cohort.return_value = 1
    store.cohort_top.return_value = []
    store.global_top.return_value = [_item(1)]
    resolve(store, "u", limit=25)
    store.fetch_user_recommendations.assert_called_with("u", limit=25)
    store.cohort_top.assert_called_with(1, limit=25)
    store.global_top.assert_called_with(25)


def test_resolved_is_frozen_dataclass() -> None:
    resolved = ResolvedRecommendations(items=[], served_from="personal", cohort_id=None)
    with pytest.raises(AttributeError):
        resolved.served_from = "x"  # type: ignore[misc]
