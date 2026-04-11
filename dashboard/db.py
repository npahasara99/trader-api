"""Read-only Supabase reporting DB helpers for the Streamlit dashboard."""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def _normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")


def get_supabase_db_url() -> str:
    db_url = os.getenv("SUPABASE_DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("SUPABASE_DATABASE_URL is not set.")
    return _normalize_db_url(db_url)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_supabase_db_url(), pool_pre_ping=True)
