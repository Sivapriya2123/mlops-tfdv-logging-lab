import json
import urllib.request
from datetime import datetime, timezone

from src import config
from src.logger import get_logger

log = get_logger("alert")


def send_slack(alert):
    if not config.SLACK_WEBHOOK_URL:
        return
    text = f":rotating_light: {alert['batch']}: {alert['anomaly_count']} anomalies in {', '.join(alert['features'])}"
    req = urllib.request.Request(
        config.SLACK_WEBHOOK_URL,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        log.info("Slack alert sent")
    except Exception as e:
        log.warning(f"Slack alert failed: {e}")


def check(batch, anomalies):
    """Raises an alert when anomaly count crosses ALERT_THRESHOLD."""
    if len(anomalies) < config.ALERT_THRESHOLD:
        return False

    alert = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "batch": batch,
        "anomaly_count": len(anomalies),
        "threshold": config.ALERT_THRESHOLD,
        "features": [a["feature"] for a in anomalies],
    }
    config.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ALERTS_DIR / "alerts.jsonl", "a") as f:
        f.write(json.dumps(alert) + "\n")

    log.error(f"ALERT {batch}: {len(anomalies)} anomalies", extra={"ctx": alert})
    send_slack(alert)
    return True
