"""Persistent manual swing-monitoring subsystem.

This package is intentionally independent from ``app.bot``. It produces
advisory/manual order plans only and has no broker order submission imports.
"""

from .service import LiveMonitorService, get_live_monitor_service

__all__ = ["LiveMonitorService", "get_live_monitor_service"]
