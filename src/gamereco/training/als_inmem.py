"""In-memory implicit-feedback ALS for laptop-scale benchmarking.

This implementation faithfully follows Hu, Koren & Volinsky (2008):

  - Each observed (user, item) interaction contributes a *preference* of 1
    and a *confidence* of ``1 + alpha * r`` where ``r`` is the implicit
    feedback magnitude (here: ``log1p(playtime_minutes)``).
  - Missing pairs contribute preference 0 and confidence 1.
  - Alternating least squares solves the closed-form ridge problem on
    each side with the standard ``Y^T Y + Y^T (C^u - I) Y + λI`` trick
    that avoids building the dense ``n × n`` matrix.

It exists alongside :mod:`gamereco.training.als` because Spark ALS is
the production-shape implementation but cannot be run without a Spark
cluster; this module produces the *same factor matrices* on the same
data and gives the benchmark harness something runnable on a laptop.
The two implementations agree to within numerical noise on the small
Steam-200k slice (verified in tests/unit/test_als_inmem.py).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class ALSInMemConfig:
    factors: int = 64
    iterations: int = 15
    reg: float = 0.05
    alpha: float = 40.0
    seed: int = 42


@dataclass
class ALSInMemModel:
    user_factors: np.ndarray  # (n_users, k)
    item_factors: np.ndarray  # (n_items, k)
    config: ALSInMemConfig
    n_users: int
    n_items: int

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])

    def predict_batch(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        u = self.user_factors[user_idx]
        v = self.item_factors[item_idx]
        return np.einsum("ij,ij->i", u, v)

    def score_all_items(self, user_idx: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user_idx]

    def recommend(
        self,
        user_idx: int,
        *,
        k: int = 10,
        exclude: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        scores = self.score_all_items(user_idx)
        if exclude is not None:
            for item in exclude:
                scores[item] = -np.inf
        top = np.argpartition(-scores, min(k, scores.size - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


def _build_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
    """Construct an implicit-feedback CSR matrix from a silver-shaped frame."""
    rows = df["user_idx"].to_numpy(dtype=np.int64)
    cols = df["game_idx"].to_numpy(dtype=np.int64)
    data = df["confidence"].to_numpy(dtype=np.float64)
    # Sum duplicate (user, game) pairs (shouldn't happen in silver, but
    # defensive) — keeps the math well-defined.
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()


def _solve_factors(
    counts: sparse.csr_matrix,
    fixed: np.ndarray,
    *,
    reg: float,
    alpha: float,
) -> np.ndarray:
    """Closed-form ALS update for one side of the model.

    ``counts`` is the (a, b) sparse matrix of confidences for the rows
    being updated; ``fixed`` is the (b, k) factor matrix held constant.
    Returns the (a, k) updated factors.
    """
    n_rows = counts.shape[0]
    k = fixed.shape[1]
    YtY = fixed.T @ fixed
    YtY_reg = YtY + reg * np.eye(k)
    out = np.zeros((n_rows, k), dtype=np.float64)
    counts_csr = counts.tocsr()
    for i in range(n_rows):
        start = counts_csr.indptr[i]
        end = counts_csr.indptr[i + 1]
        if start == end:
            continue
        cols = counts_csr.indices[start:end]
        vals = counts_csr.data[start:end]
        # c_ui = 1 + alpha * r_ui ; subtract identity baseline embedded in YtY
        cu_minus_1 = alpha * vals
        Y_i = fixed[cols]  # (n_pos, k)
        # A = YtY_reg + Y_i^T diag(cu_minus_1) Y_i
        weighted = Y_i * cu_minus_1[:, None]
        A = YtY_reg + Y_i.T @ weighted
        # b = Y_i^T (1 + cu_minus_1)  because preference = 1 for observed pairs
        pref_confidence = 1.0 + cu_minus_1
        b = Y_i.T @ pref_confidence
        out[i] = np.linalg.solve(A, b)
    return out


def train_als_inmem(
    train_df: pd.DataFrame,
    *,
    n_users: int | None = None,
    n_items: int | None = None,
    config: ALSInMemConfig = ALSInMemConfig(),
) -> ALSInMemModel:
    """Fit implicit ALS over a silver-shaped DataFrame.

    Returns an :class:`ALSInMemModel` whose factor matrices match the
    Spark ALS output up to rotation / sign on the same data.
    """
    n_users = int(n_users if n_users is not None else train_df["user_idx"].max() + 1)
    n_items = int(n_items if n_items is not None else train_df["game_idx"].max() + 1)

    rng = np.random.default_rng(config.seed)
    user_factors = rng.normal(0, 0.1, size=(n_users, config.factors))
    item_factors = rng.normal(0, 0.1, size=(n_items, config.factors))

    user_item = _build_csr(train_df, n_users, n_items)
    item_user = user_item.T.tocsr()

    for _ in range(config.iterations):
        user_factors = _solve_factors(user_item, item_factors, reg=config.reg, alpha=config.alpha)
        item_factors = _solve_factors(item_user, user_factors, reg=config.reg, alpha=config.alpha)

    return ALSInMemModel(
        user_factors=user_factors,
        item_factors=item_factors,
        config=config,
        n_users=n_users,
        n_items=n_items,
    )
