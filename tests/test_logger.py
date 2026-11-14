import json
import logging
import sys

import pytest

from src.logger import JsonFormatter, log_step


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured():
    logger = logging.getLogger("test_logger_capture")
    logger.handlers.clear()
    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, handler


def make_record(msg="hello", level=logging.INFO, **extra):
    rec = logging.makeLogRecord({"name": "t", "levelno": level,
                                 "levelname": logging.getLevelName(level), "msg": msg})
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_formatter_outputs_valid_json():
    out = json.loads(JsonFormatter().format(make_record()))
    assert out["msg"] == "hello"
    assert out["level"] == "INFO"
    assert "ts" in out


def test_formatter_includes_context_fields():
    out = json.loads(JsonFormatter().format(make_record(ctx={"batch": "b1", "rows": 10})))
    assert out["batch"] == "b1"
    assert out["rows"] == 10


def test_formatter_handles_non_serializable_values():
    out = json.loads(JsonFormatter().format(make_record(ctx={"obj": object()})))
    assert "obj" in out  # converted to string instead of crashing


def test_formatter_includes_exception():
    try:
        raise ValueError("bad value")
    except ValueError:
        rec = make_record(level=logging.ERROR, exc_info=sys.exc_info())
    out = json.loads(JsonFormatter().format(rec))
    assert "ValueError: bad value" in out["exception"]


def test_log_step_logs_start_and_end(captured):
    logger, handler = captured
    with log_step(logger, "demo"):
        pass
    events = [r.ctx["event"] for r in handler.records]
    assert events == ["start", "end"]
    assert handler.records[-1].ctx["duration_s"] >= 0


def test_log_step_reraises_and_logs_failure(captured):
    logger, handler = captured
    with pytest.raises(RuntimeError):
        with log_step(logger, "demo"):
            raise RuntimeError("boom")
    last = handler.records[-1]
    assert last.ctx["event"] == "failed"
    assert last.levelno == logging.ERROR
