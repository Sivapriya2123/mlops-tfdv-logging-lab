import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def sample_df():
    """Small synthetic credit dataset with valid values only."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "SEX": rng.choice(["1", "2"], n),
        "EDUCATION": rng.choice(["1", "2", "3", "4"], n),
        "MARRIAGE": rng.choice(["1", "2", "3"], n),
        "AGE": rng.integers(21, 70, n),
    })
    for col in config.PAY_COLS:
        df[col] = rng.choice(["-1", "0", "1", "2"], n)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.integers(0, 100_000, n)
        df[f"PAY_AMT{i}"] = rng.integers(0, 10_000, n)
    df[config.LABEL] = (rng.random(n) < 0.25).astype(int)
    return df
