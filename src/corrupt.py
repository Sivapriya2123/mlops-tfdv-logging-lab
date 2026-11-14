import numpy as np
import pandas as pd

from src import config


def corrupt(df):
    """Simulates a broken upstream feed (data quality problems)."""
    df = df.copy()
    rng = np.random.default_rng(config.SEED)
    n = len(df)

    def pick(frac):
        return df.index[rng.choice(n, size=int(frac * n), replace=False)]

    df.loc[pick(0.15), "BILL_AMT1"] = np.nan   # missing values
    df.loc[pick(0.03), "AGE"] = 150            # out of range
    df.loc[pick(0.10), "EDUCATION"] = "9"      # unseen category above the 3% tolerance
    df.loc[pick(0.02), "SEX"] = "unknown"      # invalid category
    df.loc[pick(0.03), "PAY_0"] = "12"         # outside the shared PAY_STATUS domain
    return df.drop(columns=["PAY_AMT6"])       # column dropped upstream


def drift(df):
    """Simulates a real world shift (valid data, different distribution)."""
    df = df.copy()
    rng = np.random.default_rng(config.SEED)

    df["LIMIT_BAL"] = (df["LIMIT_BAL"] * 1.8).round().astype("int64")  # credit limits raised
    df.loc[rng.random(len(df)) < 0.4, "EDUCATION"] = "3"                # customer mix shifts
    defaulters = df[df[config.LABEL] == 1].sample(frac=1.5, replace=True, random_state=config.SEED)
    return pd.concat([df, defaulters], ignore_index=True)                # more defaulters
