"""Read-only Interactive Brokers connectivity and account-safety check."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bot.broker import IBKRBroker
from app.settings import settings


def _mask_account(account_id: str) -> str:
    return f"***{account_id[-4:]}" if account_id else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="Symbol used for the quote check")
    args = parser.parse_args()

    broker = IBKRBroker(
        host=settings.IBKR_HOST,
        port=settings.IBKR_PORT,
        client_id=settings.IBKR_CLIENT_ID,
        account_id=settings.IBKR_ACCOUNT_ID,
        read_only=True,
    )
    try:
        broker.connect()
        account = broker.account_summary()
        if settings.IBKR_REQUIRE_PAPER_ACCOUNT and not account.is_paper:
            raise RuntimeError("Connected account is not an IBKR paper account")
        quote = None
        quote_error = None
        try:
            quote = broker.current_quote(args.ticker)
        except Exception as exc:
            quote_error = str(exc)
        payload = {
            "connected": broker.health_check().get("connected", False),
            "account_id": _mask_account(account.account_id),
            "account_type": account.account_type,
            "is_paper": account.is_paper,
            "net_liquidation_value": account.net_liquidation_value,
            "buying_power": account.buying_power,
            "quote_available": quote is not None,
            "quote": asdict(quote) if quote is not None else None,
            "quote_error": quote_error,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"connected": False, "error": str(exc)}, indent=2))
        return 1
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
