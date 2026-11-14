import json

import pytest

from src import alert, config


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ALERTS_DIR", tmp_path)
    monkeypatch.setattr(config, "ALERT_THRESHOLD", 2)
    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "")
    return tmp_path


def anomalies(n):
    return [{"feature": f"F{i}"} for i in range(n)]


def read_alerts(path):
    file = path / "alerts.jsonl"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []


def test_no_anomalies_no_alert(isolate):
    assert alert.check("b", []) is False
    assert read_alerts(isolate) == []


def test_below_threshold_no_alert(isolate):
    assert alert.check("b", anomalies(1)) is False
    assert read_alerts(isolate) == []


def test_at_threshold_alerts(isolate):
    assert alert.check("b", anomalies(2)) is True
    rec = read_alerts(isolate)[0]
    assert rec["batch"] == "b"
    assert rec["anomaly_count"] == 2
    assert rec["features"] == ["F0", "F1"]


def test_above_threshold_alerts(isolate):
    assert alert.check("b", anomalies(5)) is True


def test_alerts_append_not_overwrite(isolate):
    alert.check("b1", anomalies(3))
    alert.check("b2", anomalies(3))
    assert [a["batch"] for a in read_alerts(isolate)] == ["b1", "b2"]


def test_slack_not_called_without_url(monkeypatch):
    called = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append(1))
    alert.check("b", anomalies(3))
    assert called == []


def test_slack_called_with_url(monkeypatch):
    called = []
    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.example.com/x")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append(1))
    alert.check("b", anomalies(3))
    assert called == [1]


def test_slack_failure_does_not_crash(monkeypatch):
    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.example.com/x")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert alert.check("b", anomalies(3)) is True  # alert still recorded
