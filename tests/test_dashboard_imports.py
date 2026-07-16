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
