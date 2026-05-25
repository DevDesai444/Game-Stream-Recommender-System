"""Tests for the expanded recsys evaluation harness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gamereco.training.evaluation import (
    EvalResult,
    evaluate,
    relative_lift,
)


def test_perfect_predictions_score_one_across_ranking_metrics() -> None:
    preds = {1: [10, 20, 30]}
    truth = {1: [10, 20, 30]}
    result = evaluate(preds, truth, k=3, n_items=100)
    assert result.ndcg == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.map_score == pytest.approx(1.0)
    assert result.hit_rate == pytest.approx(1.0)


def test_completely_wrong_predictions_score_zero() -> None:
    preds = {1: [1, 2, 3]}
    truth = {1: [10, 20, 30]}
    result = evaluate(preds, truth, k=3, n_items=100)
    assert result.ndcg == 0.0
    assert result.recall == 0.0
    assert result.map_score == 0.0
    assert result.hit_rate == 0.0


def test_partial_credit_reflected_in_metrics() -> None:
    preds = {1: [10, 99, 20]}
    truth = {1: [10, 20]}
    result = evaluate(preds, truth, k=3, n_items=100)
    assert 0 < result.ndcg < 1
    assert result.recall == pytest.approx(1.0)
    assert result.hit_rate == pytest.approx(1.0)


def test_coverage_counts_unique_recommended_items() -> None:
    preds = {1: [1, 2, 3], 2: [3, 4, 5]}
    truth = {1: [1], 2: [5]}
    result = evaluate(preds, truth, k=3, n_items=10)
    assert result.coverage == pytest.approx(5 / 10)


def test_coverage_handles_zero_catalog() -> None:
    result = evaluate({1: [1]}, {1: [1]}, k=1, n_items=0)
    assert result.coverage == 0.0


def test_novelty_higher_for_rare_items() -> None:
    popularity = {0: 100, 1: 50, 2: 1}
    result = evaluate(
        {1: [2, 1, 0]},
        {1: [2]},
        k=3,
        n_items=3,
        item_popularity=popularity,
    )
    rare_novelty = -math.log2(1 / 151)
    assert result.novelty == pytest.approx(
        (rare_novelty + -math.log2(50 / 151) + -math.log2(100 / 151)) / 3, rel=1e-4
    )


def test_novelty_handles_unseen_item_floor() -> None:
    popularity = {0: 10}
    result = evaluate(
        {1: [99]},
        {1: [99]},
        k=1,
        n_items=100,
        item_popularity=popularity,
    )
    # Floor of 1/10 gives a finite novelty.
    assert result.novelty > 0


def test_diversity_drops_for_near_identical_embeddings() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0001, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    similar = evaluate({1: [0, 1]}, {1: [0]}, k=2, n_items=3, item_embeddings=embeddings)
    diverse = evaluate({1: [0, 2]}, {1: [0]}, k=2, n_items=3, item_embeddings=embeddings)
    assert diverse.diversity > similar.diversity


def test_diversity_returns_zero_when_no_embeddings() -> None:
    result = evaluate({1: [0, 1]}, {1: [0]}, k=2, n_items=3)
    assert result.diversity == 0.0


def test_skips_users_with_empty_truth() -> None:
    preds = {1: [1, 2, 3], 2: [4, 5, 6]}
    truth = {1: [1], 2: []}
    result = evaluate(preds, truth, k=3, n_items=10)
    assert result.n_users_scored == 1


def test_map_decreases_when_hit_moves_down_the_list() -> None:
    truth = {1: [10]}
    top_hit = evaluate({1: [10, 99, 100]}, truth, k=3, n_items=200)
    deep_hit = evaluate({1: [99, 100, 10]}, truth, k=3, n_items=200)
    assert top_hit.map_score > deep_hit.map_score


def test_relative_lift_zero_baseline_returns_zero() -> None:
    a = EvalResult(
        k=10,
        ndcg=0.5,
        recall=0,
        map_score=0,
        hit_rate=0,
        coverage=0,
        novelty=0,
        diversity=0,
        n_users_scored=1,
    )
    b = EvalResult(
        k=10,
        ndcg=0.0,
        recall=0,
        map_score=0,
        hit_rate=0,
        coverage=0,
        novelty=0,
        diversity=0,
        n_users_scored=1,
    )
    assert relative_lift(a, b) == 0.0


def test_relative_lift_positive() -> None:
    a = EvalResult(
        k=10,
        ndcg=0.5,
        recall=0,
        map_score=0,
        hit_rate=0,
        coverage=0,
        novelty=0,
        diversity=0,
        n_users_scored=1,
    )
    b = EvalResult(
        k=10,
        ndcg=0.4,
        recall=0,
        map_score=0,
        hit_rate=0,
        coverage=0,
        novelty=0,
        diversity=0,
        n_users_scored=1,
    )
    assert relative_lift(a, b) == pytest.approx(0.25)


def test_as_dict_includes_extras() -> None:
    result = evaluate(
        {1: [1, 2]},
        {1: [1]},
        k=2,
        n_items=5,
        extras={"train_seconds": 7.5},
    )
    d = result.as_dict()
    assert d["train_seconds"] == 7.5
    assert d["k"] == 2
