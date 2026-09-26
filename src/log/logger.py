"""Loguru-based logger configuration with task-based file routing.

Call ``setup(task_name)`` once at startup to configure console + file sinks.
All other modules simply ``from loguru import logger`` and use it directly.
"""

import sys
from pathlib import Path

from loguru import logger

LOGS_DIR = Path("logs")

# Remove the default stderr handler added by loguru's import-time behaviour
logger.remove()


def setup(task_name: str = "default") -> None:
    """Configure loguru sinks.

    Parameters
    ----------
    task_name:
        Used as the sub-directory name inside ``logs/`` so that each task
        writes to its own folder (e.g. ``logs/supervisor_fanout/``).
    """
    # ── console sink (colourised, INFO and above) ──────────────────
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # ── file sink (structured, DEBUG and above, per-task folder) ──
    # sanitise task name for use as a directory name
    safe_name = task_name.strip().replace(" ", "_")[:80]
    log_path = LOGS_DIR / safe_name / "{time:YYYY-MM-DD}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )
