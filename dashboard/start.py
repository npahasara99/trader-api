"""Start Streamlit with a validated Railway/local port."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _resolve_port() -> int:
    for candidate in (os.getenv("PORT"), os.getenv("STREAMLIT_SERVER_PORT"), "8501"):
        try:
            port = int(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return 8501


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "dashboard" / "app.py"
    port = _resolve_port()

    # Override invalid values such as the literal string "$PORT" before
    # Streamlit parses its environment-backed command options.
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
