"""Cold-start fallback chain.

A read for a user we have *no* personalised recommendations for should
never 404 if there's anything sensible we can serve. The cascade:

    personal  ──►  cohort  ──►  global

`personal` is the per-user table written after training.
`cohort` is the K-Means user cohort top-K — useful for users we've
seen but who don't have enough history for the personal table to be
populated (e.g. they just signed up but already told us a few liked
games).
`global` is the catalog-wide top-K, the same list `/global` returns.

Each layer reports its own ``served_from`` value so observability can
attribute hit rates to the layer that actually answered the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gamereco.common.schemas import RecommendationItem


class _ColdstartStore(Protocol):  # pragma: no cover - structural typing only
    def fetch_user_recommendations(
        self, user_id: str, *, limit: int
    ) -> list[RecommendationItem]: ...

    def user_cohort(self, user_id: str) -> int | None: ...

    def cohort_top(self, cohort_id: int, *, limit: int) -> list[RecommendationItem]: ...

    def global_top(self, limit: int) -> list[RecommendationItem]: ...


@dataclass(frozen=True)
class ResolvedRecommendations:
    items: list[RecommendationItem]
    served_from: str
    cohort_id: int | None


def resolve(
    store: _ColdstartStore,
    user_id: str,
    *,
    limit: int,
) -> ResolvedRecommendations:
    """Walk the personal → cohort → global cascade.

    Returns the first layer that produced any items, along with the
    label the API reports back to the client.
    """
    personal = store.fetch_user_recommendations(user_id, limit=limit)
    if personal:
        return ResolvedRecommendations(items=personal, served_from="personal", cohort_id=None)

    cohort_id = store.user_cohort(user_id)
    if cohort_id is not None:
        cohort_items = store.cohort_top(cohort_id, limit=limit)
        if cohort_items:
            return ResolvedRecommendations(
                items=cohort_items, served_from="cohort", cohort_id=cohort_id
            )

    fallback = store.global_top(limit)
    return ResolvedRecommendations(
        items=fallback,
        served_from="global_fallback",
        cohort_id=cohort_id,
    )
