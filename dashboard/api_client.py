"""HTTP client helpers for triggering trader API workflows from Streamlit."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import requests


class TraderAPIError(RuntimeError):
    """Readable API error for the dashboard runner."""

    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


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


def get_api_base_url() -> str:
    _load_repo_env()
    base_url = os.getenv("TRADER_API_BASE_URL", "").strip()
    if not base_url:
        raise TraderAPIError(
            "TRADER_API_BASE_URL is not set. Add it to your environment or repo .env so the dashboard can call the trader API."
        )
    return base_url.rstrip("/")


def get_api_bearer_token() -> str | None:
    _load_repo_env()
    token = os.getenv("API_BEARER_TOKEN", "").strip()
    return token or None


def api_config_status() -> dict[str, Any]:
    try:
        base_url = get_api_base_url()
    except TraderAPIError:
        base_url = None
    token = get_api_bearer_token()
    return {
        "base_url": base_url,
        "has_bearer_token": bool(token),
    }


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = get_api_bearer_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_json(path: str, payload: dict[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    base_url = get_api_base_url()
    url = f"{base_url}{path}"
    try:
        response = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise TraderAPIError(f"Could not reach the trader API at {url}: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = None

    if not response.ok:
        detail = data.get("detail") if isinstance(data, dict) else data
        message = f"Trader API returned HTTP {response.status_code} for {path}."
        if detail:
            message = f"{message} {detail}"
        raise TraderAPIError(message, status_code=response.status_code, detail=detail)

    if not isinstance(data, dict):
        raise TraderAPIError(f"Trader API returned a malformed response for {path}.")

    return data


def run_sp100_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    return post_json("/workflow/sp100/top10-log", payload)


def run_single_stock_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    return post_json("/workflow/swing-plan-log", payload)


def run_manual_basket(payload: dict[str, Any]) -> dict[str, Any]:
    return post_json("/plan/swing", payload)
