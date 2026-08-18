from __future__ import annotations

from pathlib import Path


def test_dashboard_components_import_backend_app_package() -> None:
    from dashboard import components
    from app import live_plan_consistency

    backend_module = Path(live_plan_consistency.__file__).resolve()
    dashboard_script = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    assert backend_module != dashboard_script.resolve()
    assert backend_module.parent.name == "app"
    assert callable(components.format_run_history_display)


def test_safe_json_converts_pandas_series_to_mapping() -> None:
    import pandas as pd

    from dashboard.utils import safe_json

    result = safe_json(pd.Series({"ticker": "AMD", "final_action": "WAIT"}))

    assert result == {"ticker": "AMD", "final_action": "WAIT"}


def test_active_market_view_merges_live_quote_without_future_warning() -> None:
    import warnings

    import pandas as pd

    from dashboard.utils import build_active_market_view

    snapshots = pd.DataFrame(
        [
            {
                "ticker": "AMD",
                "current_price": 100.0,
                "current_price_asof": "2026-08-18T12:00:00Z",
                "preferred_entry": 99.0,
                "stop_loss": 95.0,
                "take_profit_1": 108.0,
            }
        ]
    )
    quotes = pd.DataFrame(
        [
            {
                "ticker": "AMD",
                "live_price": 101.5,
                "live_price_asof": "2026-08-18T13:00:00Z",
                "available": True,
                "status": "available",
                "price_source": "live_quote",
            }
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = build_active_market_view(snapshots, quotes)

    assert result.iloc[0]["live_price"] == 101.5
    assert str(result.iloc[0]["live_price_asof"]) == "2026-08-18 13:00:00+00:00"
