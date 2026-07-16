"""Start FastAPI/Uvicorn with a validated Railway or local port."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_repo_env() -> None:
    file_values: dict[str, str] = {}
    for filename in (".env", ".env.bot"):
        path = PROJECT_ROOT / filename
        if not path.exists():
            continue
        file_values.update({key: value for key, value in dotenv_values(path).items() if value is not None})
    for key, value in file_values.items():
        # Railway/service variables remain authoritative over local files.
        os.environ.setdefault(key, value)


_load_repo_env()


def _resolve_port() -> int:
    for candidate in (os.getenv("PORT"), os.getenv("TRADER_API_PORT"), os.getenv("UVICORN_PORT"), "8080"):
        try:
            port = int(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return 8080


def main() -> None:
    os.chdir(PROJECT_ROOT)
    port = _resolve_port()

    os.environ["UVICORN_PORT"] = str(port)
    # The in-process bot singleton and IBKR client require one API worker.
    os.environ["WEB_CONCURRENCY"] = "1"

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
