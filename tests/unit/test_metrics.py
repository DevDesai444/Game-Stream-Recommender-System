"""Tests for the ranking-metric helpers."""

from __future__ import annotations

import math

import pytest

from gamereco.training.metrics import _dcg, ndcg_at_k_numpy


def test_dcg_first_position_no_discount() -> None:
    assert _dcg([1.0]) == pytest.approx(1.0)


def test_dcg_decreasing_with_rank() -> None:
    assert _dcg([1.0, 1.0]) == pytest.approx(1.0 + 1.0 / math.log2(3))


def test_ndcg_perfect_ranking_is_one() -> None:
    preds = {1: [10, 20, 30]}
    truth = {1: [10, 20]}
    assert ndcg_at_k_numpy(preds, truth, k=3) == pytest.approx(1.0)


def test_ndcg_completely_wrong_is_zero() -> None:
    preds = {1: [99, 98, 97]}
    truth = {1: [1, 2]}
    assert ndcg_at_k_numpy(preds, truth, k=3) == 0.0


def test_ndcg_partial_credit() -> None:
    preds = {1: [10, 99, 20]}
    truth = {1: [10, 20]}
    ndcg = ndcg_at_k_numpy(preds, truth, k=3)
    assert 0 < ndcg < 1


def test_ndcg_skips_user_with_empty_truth() -> None:
    preds = {1: [10, 20], 2: [30, 40]}
    truth = {1: [10], 2: []}
    ndcg = ndcg_at_k_numpy(preds, truth, k=2)
    assert ndcg == pytest.approx(1.0)


def test_ndcg_returns_zero_when_no_users() -> None:
    assert ndcg_at_k_numpy({}, {}, k=10) == 0.0


def test_ndcg_truncates_to_k() -> None:
    preds = {1: list(range(100))}
    truth = {1: [50]}
    assert ndcg_at_k_numpy(preds, truth, k=10) == 0.0


def test_ndcg_handles_short_truth() -> None:
    preds = {1: [1, 2, 3]}
    truth = {1: [3]}
    ndcg = ndcg_at_k_numpy(preds, truth, k=3)
    assert 0 < ndcg < 1
