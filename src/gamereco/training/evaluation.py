"""Recsys-grade evaluation harness.

A single recommender is scored across seven complementary axes:

  * **NDCG@K**      — ranking quality with logarithmic position discount
  * **Recall@K**    — fraction of held-out items the system surfaces
  * **MAP@K**       — mean average precision (precision at every hit, averaged)
  * **HitRate@K**   — fraction of users with ≥ 1 hit
  * **Coverage@K**  — fraction of the *catalog* recommended to ≥ 1 user
  * **Novelty@K**   — mean ``-log2(p(i))`` of recommended items (rare item = high novelty)
  * **Diversity@K** — 1 minus mean pairwise cosine similarity within each user's list

NDCG is the headline metric used to compare model variants; the rest
keep us honest about *what kind of bad* a model is when it isn't best.
A model that wins on NDCG but tanks coverage or diversity is a popularity
collapse — and you want to know before shipping it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

_ItemList = Sequence[int]


@dataclass(frozen=True)
class EvalResult:
    """Container for all measured metrics at a single K."""

    k: int
    ndcg: float
    recall: float
    map_score: float
    hit_rate: float
    coverage: float
    novelty: float
    diversity: float
    n_users_scored: int
    extras: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            "k": self.k,
            "n_users_scored": self.n_users_scored,
            "ndcg@k": round(self.ndcg, 6),
            "recall@k": round(self.recall, 6),
            "map@k": round(self.map_score, 6),
            "hit_rate@k": round(self.hit_rate, 6),
            "coverage@k": round(self.coverage, 6),
            "novelty@k": round(self.novelty, 6),
            "diversity@k": round(self.diversity, 6),
        }
        for key, value in self.extras.items():
            out[key] = value
        return out


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_for_user(predicted: _ItemList, truth: set[int], k: int) -> float | None:
    if not truth:
        return None
    top_k = list(predicted)[:k]
    gains = [1.0 if item in truth else 0.0 for item in top_k]
    ideal = [1.0] * min(len(truth), k)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return None
    return _dcg(gains) / ideal_dcg


def _ap_for_user(predicted: _ItemList, truth: set[int], k: int) -> float | None:
    if not truth:
        return None
    top_k = list(predicted)[:k]
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in truth:
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / min(len(truth), k)


def _recall_for_user(predicted: _ItemList, truth: set[int], k: int) -> float | None:
    if not truth:
        return None
    top_k = set(list(predicted)[:k])
    return len(top_k & truth) / len(truth)


def _hit_for_user(predicted: _ItemList, truth: set[int], k: int) -> float | None:
    if not truth:
        return None
    top_k = set(list(predicted)[:k])
    return 1.0 if (top_k & truth) else 0.0


def _coverage(all_predictions: Mapping[int, _ItemList], n_items: int, k: int) -> float:
    if n_items <= 0:
        return 0.0
    seen: set[int] = set()
    for items in all_predictions.values():
        seen.update(list(items)[:k])
    return len(seen) / n_items


def _novelty(
    all_predictions: Mapping[int, _ItemList],
    item_popularity: Mapping[int, float],
    k: int,
) -> float:
    if not item_popularity:
        return 0.0
    total_interactions = sum(item_popularity.values()) or 1.0
    novelties: list[float] = []
    for items in all_predictions.values():
        for item in list(items)[:k]:
            count = item_popularity.get(item, 0.0)
            if count <= 0:
                # Treat unseen items as the rarest possible — log2 of a
                # single phantom interaction over the corpus.
                count = 1.0
            p = count / total_interactions
            novelties.append(-math.log2(p))
    if not novelties:
        return 0.0
    return float(np.mean(novelties))


def _diversity(
    all_predictions: Mapping[int, _ItemList],
    item_embeddings: np.ndarray | None,
    k: int,
) -> float:
    """1 - mean pairwise cosine similarity inside each user's top-K."""
    if item_embeddings is None or item_embeddings.size == 0:
        return 0.0
    normalised = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-12)
    diversities: list[float] = []
    for items in all_predictions.values():
        top_k = [int(i) for i in list(items)[:k] if 0 <= int(i) < normalised.shape[0]]
        if len(top_k) < 2:
            continue
        vecs = normalised[top_k]
        sims = vecs @ vecs.T
        # Mask the diagonal.
        n = sims.shape[0]
        off_diag = sims.sum() - np.trace(sims)
        mean_sim = off_diag / (n * (n - 1))
        diversities.append(1.0 - float(mean_sim))
    if not diversities:
        return 0.0
    return float(np.mean(diversities))


def evaluate(
    predictions: Mapping[int, _ItemList],
    ground_truth: Mapping[int, Iterable[int]],
    *,
    k: int = 10,
    n_items: int | None = None,
    item_popularity: Mapping[int, float] | None = None,
    item_embeddings: np.ndarray | None = None,
    extras: Mapping[str, float] | None = None,
) -> EvalResult:
    """Score a recommender on every metric in the harness."""
    truth_sets: dict[int, set[int]] = {u: set(t) for u, t in ground_truth.items() if t}
    ndcgs: list[float] = []
    recalls: list[float] = []
    aps: list[float] = []
    hits: list[float] = []
    scored = 0
    for user, items in predictions.items():
        truth = truth_sets.get(int(user))
        if not truth:
            continue
        scored += 1
        for metric_list, fn in (
            (ndcgs, _ndcg_for_user),
            (recalls, _recall_for_user),
            (aps, _ap_for_user),
            (hits, _hit_for_user),
        ):
            value = fn(items, truth, k)
            if value is not None:
                metric_list.append(value)

    coverage = _coverage(predictions, n_items or 0, k)
    novelty = _novelty(predictions, item_popularity or {}, k)
    diversity = _diversity(predictions, item_embeddings, k)

    return EvalResult(
        k=k,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        map_score=float(np.mean(aps)) if aps else 0.0,
        hit_rate=float(np.mean(hits)) if hits else 0.0,
        coverage=float(coverage),
        novelty=float(novelty),
        diversity=float(diversity),
        n_users_scored=scored,
        extras=dict(extras or {}),
    )


def relative_lift(challenger: EvalResult, baseline: EvalResult) -> float:
    """Relative NDCG@K lift of ``challenger`` over ``baseline``."""
    if baseline.ndcg <= 0:
        return 0.0
    return (challenger.ndcg - baseline.ndcg) / baseline.ndcg
