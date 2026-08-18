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
    repo_root = Path(__file__).resolve().parents[1]
    file_values: dict[str, str] = {}
    for filename in (".env", ".env.bot"):
        env_path = repo_root / filename
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            file_values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in file_values.items():
        # Deployed service variables remain authoritative over repo-local files.
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


def get_api_timeout_seconds(default: int) -> int:
    _load_repo_env()
    raw = os.getenv("TRADER_API_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return default
    try:
        return max(30, int(raw))
    except ValueError:
        return default


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


def get_json(path: str, params: dict[str, Any] | None = None, *, timeout: int = 120) -> dict[str, Any]:
    base_url = get_api_base_url()
    url = f"{base_url}{path}"
    try:
        response = requests.get(url, params=params or {}, headers=_headers(), timeout=timeout)
    except requests.Timeout as exc:
        raise TraderAPIError(
            f"The trader API did not finish within {timeout} seconds for {path}. "
            f"This request may need more time or the API may be under load."
        ) from exc
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


def post_json(path: str, payload: dict[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    base_url = get_api_base_url()
    url = f"{base_url}{path}"
    try:
        response = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    except requests.Timeout as exc:
        raise TraderAPIError(
            f"The trader API did not finish within {timeout} seconds for {path}. "
            f"This usually means the workflow is still running and needs a longer timeout."
        ) from exc
    except requests.RequestException as exc:
        raise TraderAPIError(f"Could not reach the trader API at {url}: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = None

    if not response.ok:
        detail = data.get("detail") if isinstance(data, dict) else data
        message = f"Trader API returned HTTP {response.status_code} for {path}."
        if response.status_code in {404, 405}:
            message = (
                f"{message} This usually means TRADER_API_BASE_URL is pointing at the wrong service "
                f"(for example the Streamlit dashboard domain instead of the FastAPI API base URL)."
            )
        if detail:
            message = f"{message} {detail}"
        raise TraderAPIError(message, status_code=response.status_code, detail=detail)

    if not isinstance(data, dict):
        raise TraderAPIError(f"Trader API returned a malformed response for {path}.")

    return data


def put_json(path: str, payload: dict[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    base_url = get_api_base_url()
    url = f"{base_url}{path}"
    try:
        response = requests.put(url, json=payload, headers=_headers(), timeout=timeout)
    except requests.Timeout as exc:
        raise TraderAPIError(
            f"The trader API did not finish within {timeout} seconds for {path}."
        ) from exc
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
    return post_json(
        "/workflow/sp100/top10-log",
        payload,
        timeout=get_api_timeout_seconds(420),
    )


def run_sp500_daily_opportunities(payload: dict[str, Any]) -> dict[str, Any]:
    return post_json(
        "/workflow/sp500/daily-opportunities",
        payload,
        timeout=get_api_timeout_seconds(900),
    )


def run_single_stock_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    return post_json(
        "/workflow/swing-plan-log",
        payload,
        timeout=get_api_timeout_seconds(180),
    )


def run_manual_basket(payload: dict[str, Any]) -> dict[str, Any]:
    return post_json(
        "/plan/swing",
        payload,
        timeout=get_api_timeout_seconds(300),
    )


def fetch_earnings_calendar(
    *,
    days_ahead: int = 30,
    sector: str | None = None,
    industry: str | None = None,
    sp100_only: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "days_ahead": int(days_ahead),
        "sp100_only": str(bool(sp100_only)).lower(),
    }
    if sector:
        params["sector"] = sector
    if industry:
        params["industry"] = industry
    return get_json(
        "/calendar/earnings",
        params=params,
        timeout=get_api_timeout_seconds(120),
    )


def fetch_earnings_detail(ticker: str, *, days_ahead: int = 30) -> dict[str, Any]:
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        raise TraderAPIError("Ticker is required for earnings detail.")
    return get_json(
        f"/calendar/earnings/{normalized_ticker}",
        params={"days_ahead": int(days_ahead)},
        timeout=get_api_timeout_seconds(120),
    )



def fetch_live_quotes(tickers: list[str]) -> dict[str, Any]:
    normalized = [str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()]
    if not normalized:
        return {"as_of": None, "quote_count": 0, "available_count": 0, "unavailable_count": 0, "rows": []}
    return get_json(
        "/market/quotes",
        params={"tickers": ",".join(normalized)},
        timeout=get_api_timeout_seconds(60),
    )


def fetch_bot_status() -> dict[str, Any]:
    return get_json("/bot/status", timeout=get_api_timeout_seconds(60))


def fetch_bot_config() -> dict[str, Any]:
    return get_json("/bot/config", timeout=get_api_timeout_seconds(60))


def update_bot_config(payload: dict[str, Any]) -> dict[str, Any]:
    return put_json("/bot/config", {"config": payload}, timeout=get_api_timeout_seconds(120))


def bot_action(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return post_json(path, payload or {}, timeout=get_api_timeout_seconds(120))


def fetch_bot_rows(path: str) -> list[dict[str, Any]]:
    payload = get_json(path, timeout=get_api_timeout_seconds(120))
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else []


def fetch_bot_payload(path: str) -> dict[str, Any]:
    return get_json(path, timeout=get_api_timeout_seconds(120))
