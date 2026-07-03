"""
Centralized logging setup for the whole project. Every module imports
`get_logger(__name__)` from here instead of calling `logging.basicConfig`
or `logging.getLogger` directly, so the format + destinations stay
identical everywhere

Usage
-----
Call `setup_logging()` ONCE, as early as possible in the process (top of
whatever your real entrypoint is -- backend/main.py, or the `run()` /
`__main__` block of a standalone script):

    from src.core.log import setup_logging, get_logger
    setup_logging(run_id="run_name")   # run_id is optional
    logger = get_logger(__name__)

Every other module then just does:

    from src.core.log import get_logger
    logger = get_logger(__name__)

If some module gets imported/used before setup_logging() runs (import
order accidents happen), get_logger() will lazily call setup_logging()
with defaults so you still get sane output instead of silence.
"""

from __future__ import annotations

import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Optional

# Path config lives in one place: src/config.py
from src.config import LOG_DIR, LOG_LEVEL as _DEFAULT_LEVEL_NAME

DEFAULT_LEVEL = getattr(logging, _DEFAULT_LEVEL_NAME.upper(), logging.INFO)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _build_log_path(run_id: Optional[str] = None) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{run_id}.log" if run_id else f"{stamp}.log"
    return LOG_DIR / filename


def setup_logging(
        level: int = DEFAULT_LEVEL,
        run_id: Optional[str] = None,
        console: bool = True,
        file: bool = True,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
) -> Optional[Path]:
    """
    Configures the ROOT logger once. Idempotent -- a second call is a no-op,
    so accidentally calling it from more than one entrypoint (e.g. both a
    test script and an imported module) will never duplicate handlers or
    duplicate every log line.

    level         -- root log level (module-specific loggers can still be
                      set louder/quieter individually if you ever need that)
    run_id        -- optional tag folded into the log filename, e.g. a
                      persona name or simulation id, so a run's logs are
                      easy to find in output/logs/
    console       -- also stream logs to stdout
    file          -- also write logs to output/logs/<timestamp>[_<run_id>].log
    max_bytes /
    backup_count  -- rotation settings for the file handler

    Returns the Path of the log file created, or None if file=False.
    """
    global _configured
    if _configured:
        return None

    handlers: dict = {}
    log_path: Optional[Path] = None

    if console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
            "level": level,
        }

    if file:
        log_path = _build_log_path(run_id)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "filename": str(log_path),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
            "level": level,
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": DATE_FORMAT,
            },
        },
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": list(handlers.keys()),
        },
        # Quiet down noisy third-party loggers so your own log lines don't
        # get buried -- add to this as you bring in more SDKs.
        "loggers": {
            "google_genai": {"level": "WARNING", "propagate": True},
            "httpx": {"level": "WARNING", "propagate": True},
            "httpcore": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(config)
    _configured = True

    if log_path:
        logging.getLogger(__name__).info("[log] logging initialized -> %s", log_path)

    return log_path


def get_logger(name: str) -> logging.Logger:
    """
    Thin wrapper so call sites read `from src.core.log import get_logger`
    instead of raw `logging.getLogger`. Lazily calls setup_logging() with
    defaults if nothing has configured logging yet.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)