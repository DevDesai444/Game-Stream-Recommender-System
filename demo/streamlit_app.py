"""Streamlit demo for the Game Stream Recommender.

A single-page app that hits the FastAPI service and renders three
panels side-by-side: personalised recommendations for a given user
(with the cold-start cascade in play), "more like this" via pgvector,
and the cohort top / global fallback.

Run locally:
    docker compose up -d api postgres redis
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


API_BASE = os.environ.get("GAMERECO_API_BASE", "http://localhost:8000")


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=5.0)


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.8em;">{text}</span>'
    )


SERVED_FROM_COLOR = {
    "cache": "#0d9488",
    "personal": "#1d4ed8",
    "cohort": "#9333ea",
    "global_fallback": "#f59e0b",
    "onboarding_pgvector": "#db2777",
    "pgvector": "#0891b2",
}


def render_items(payload: dict[str, Any]) -> None:
    served = payload.get("served_from", "n/a")
    color = SERVED_FROM_COLOR.get(served, "#6b7280")
    latency_label = f"latency {payload['latency_ms']} ms"
    st.markdown(
        f"{_badge(served, color)} {_badge(latency_label, '#374151')}",
        unsafe_allow_html=True,
    )
    items = payload.get("items", [])
    if not items:
        st.info("no items returned")
        return
    for rank, item in enumerate(items, start=1):
        cols = st.columns([1, 4, 2])
        with cols[0]:
            if item.get("header_image"):
                st.image(item["header_image"], width=120)
        with cols[1]:
            st.markdown(f"**#{rank} · {item['name']}**")
            st.caption(f"steam_appid {item['steam_appid']}")
        with cols[2]:
            st.metric("score", f"{item['score']:.3f}")


def fetch_recommendations(user_id: str, limit: int) -> dict[str, Any] | None:
    try:
        resp = _client().get(
            f"/recommendations/{user_id}", params={"limit": limit}
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"failed to call /recommendations: {exc}")
        return None


def fetch_similar(appid: int, limit: int) -> dict[str, Any] | None:
    try:
        resp = _client().get(f"/similar/{appid}", params={"limit": limit})
        if resp.status_code == 404:
            return {"served_from": "n/a", "latency_ms": 0, "items": []}
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"failed to call /similar: {exc}")
        return None


def fetch_global(limit: int) -> dict[str, Any] | None:
    try:
        resp = _client().get("/global", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"failed to call /global: {exc}")
        return None


def onboard(user_id: str, liked: list[int], limit: int) -> dict[str, Any] | None:
    try:
        resp = _client().post(
            "/onboard",
            params={"limit": limit},
            json={"user_id": user_id, "liked_steam_appids": liked},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"failed to call /onboard: {exc}")
        return None


def fetch_health() -> dict[str, Any] | None:
    try:
        resp = _client().get("/health")
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


st.set_page_config(page_title="Game Stream Recommender", layout="wide")
st.title("Game Stream Recommender — live demo")
st.caption(
    "Hybrid ALS + NCF + KMeans + XGBoost ensemble served by FastAPI, "
    "pgvector, and Redis. The cold-start cascade keeps unknown users "
    "from 404-ing."
)

with st.sidebar:
    st.header("Settings")
    st.text_input("API base URL", value=API_BASE, key="api_base", disabled=True)
    limit = st.slider("Items per panel", 3, 25, 10)
    health = fetch_health()
    if health:
        st.success(f"API healthy · redis {health.get('redis', 'n/a')}")
    else:
        st.error("API unreachable — check GAMERECO_API_BASE")
    st.divider()
    st.markdown(
        "**Layers**:  \n"
        + f"{_badge('cache', SERVED_FROM_COLOR['cache'])} "
        + f"{_badge('personal', SERVED_FROM_COLOR['personal'])} "
        + f"{_badge('cohort', SERVED_FROM_COLOR['cohort'])} "
        + f"{_badge('global_fallback', SERVED_FROM_COLOR['global_fallback'])} "
        + f"{_badge('pgvector', SERVED_FROM_COLOR['pgvector'])} "
        + f"{_badge('onboarding_pgvector', SERVED_FROM_COLOR['onboarding_pgvector'])}",
        unsafe_allow_html=True,
    )

tab_personal, tab_similar, tab_onboard, tab_global = st.tabs(
    ["Personal", "More like this", "Onboard new user", "Global"]
)

with tab_personal:
    st.subheader("Personalised top-K")
    user_id = st.text_input("Steam user_id", value="76561198000000000")
    if st.button("Fetch", key="personal_btn"):
        payload = fetch_recommendations(user_id, limit)
        if payload is not None:
            render_items(payload)

with tab_similar:
    st.subheader("More like this (pgvector cosine search)")
    appid = st.number_input(
        "steam_appid", min_value=1, value=440, step=1, format="%d"
    )
    if st.button("Fetch", key="similar_btn"):
        payload = fetch_similar(int(appid), limit)
        if payload is not None:
            render_items(payload)

with tab_onboard:
    st.subheader("Brand-new user onboarding")
    new_user = st.text_input("New user_id", value="u_signup_demo")
    liked = st.text_input(
        "Liked appids (comma-separated)", value="440, 730, 570"
    )
    if st.button("Onboard", key="onboard_btn"):
        try:
            ids = [int(x.strip()) for x in liked.split(",") if x.strip()]
        except ValueError:
            st.error("expected a comma-separated list of integers")
            ids = []
        if ids:
            payload = onboard(new_user, ids, limit)
            if payload is not None:
                render_items(payload)

with tab_global:
    st.subheader("Global top — non-personalised baseline")
    if st.button("Fetch", key="global_btn"):
        payload = fetch_global(limit)
        if payload is not None:
            render_items(payload)
