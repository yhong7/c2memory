from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(name: str = "service") -> None:
    logger.remove()
    safe_name = name.strip().replace(" ", "_")[:80]
    log_path = Path("logs") / safe_name / "{time:YYYY-MM-DD}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stderr, level="INFO")
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="20 MB",
        retention="14 days",
        encoding="utf-8",
    )

