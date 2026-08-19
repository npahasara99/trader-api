from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from app.db import Base, engine  # noqa: E402
from app import models  # noqa: F401,E402
from sqlalchemy import text  # noqa: E402


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text(
                    "ALTER TABLE swing_decisions "
                    "ALTER COLUMN mode TYPE VARCHAR(80)"
                )
            )
    print("Schema sync complete.")


if __name__ == "__main__":
    main()
