"""Read-only Supabase reporting DB helpers for the Streamlit dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def _normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return db_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")


@lru_cache(maxsize=1)
def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_supabase_db_url() -> str:
    _load_repo_env()
    db_url = os.getenv("SUPABASE_DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("SUPABASE_DATABASE_URL is not set. Add it to your environment or repo .env for local dashboard use.")
    return _normalize_db_url(db_url)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_supabase_db_url(), pool_pre_ping=True)
