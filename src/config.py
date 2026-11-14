import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SCHEMA_DIR = ROOT / "schema"
REPORTS_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
ALERTS_DIR = ROOT / "alerts"

TRAIN_CSV = DATA_DIR / "train.csv"

# name -> (file, schema environment)
BATCHES = {
    "eval": (DATA_DIR / "eval.csv", "TRAINING"),
    "serving": (DATA_DIR / "serving.csv", "SERVING"),     # label dropped
    "batch_corrupt": (DATA_DIR / "batch_corrupt.csv", "TRAINING"),
    "batch_drift": (DATA_DIR / "batch_drift.csv", "TRAINING"),
}

TRAIN_STATS = SCHEMA_DIR / "train_stats.pbtxt"
SCHEMA_FILE = SCHEMA_DIR / "schema.pbtxt"

LABEL = "DEFAULT"
DROP_FEATURES = {"ID"}                 # internal tracking column, no signal
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
STRING_COLS = CATEGORICAL + PAY_COLS

# Documented codes from the UCI dataset description
VALID_CODES = {
    "SEX": ["1", "2"],
    "EDUCATION": ["1", "2", "3", "4"],
    "MARRIAGE": ["1", "2", "3"],
}
PAY_STATUS_CODES = [str(v) for v in range(-2, 9)]   # shared domain for PAY_0..PAY_6

# Raw data has a small share of undocumented codes, so allow up to 3%
RELAXED_FEATURES = {"EDUCATION": 0.97, "MARRIAGE": 0.97}

SLICE_FEATURES = ["SEX", "EDUCATION"]

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.1"))
SKEW_THRESHOLD = float(os.getenv("SKEW_THRESHOLD", "0.03"))
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "2"))
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SEED = 42
