import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from src import config


class JsonFormatter(logging.Formatter):
    """One JSON object per line, easy to search or ship to a log tool."""

    def format(self, record):
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "ctx"):
            payload.update(record.ctx)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(config.LOG_LEVEL)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File: structured JSON, rotates at 1 MB, keeps 3 backups
    file_handler = RotatingFileHandler(
        config.LOG_DIR / "pipeline.log", maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(JsonFormatter())

    # Console: human readable
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger


@contextmanager
def log_step(logger, step):
    """Logs start, end, duration and failures of a pipeline step."""
    start = time.perf_counter()
    logger.info(f"START {step}", extra={"ctx": {"step": step, "event": "start"}})
    try:
        yield
    except Exception:
        logger.exception(f"FAILED {step}", extra={"ctx": {"step": step, "event": "failed"}})
        raise
    secs = round(time.perf_counter() - start, 2)
    logger.info(
        f"END {step} ({secs}s)",
        extra={"ctx": {"step": step, "event": "end", "duration_s": secs}},
    )
