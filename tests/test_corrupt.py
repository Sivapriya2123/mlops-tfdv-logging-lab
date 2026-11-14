import pandas as pd

from src import config
from src.corrupt import corrupt, drift


# ---------- corrupt() ----------

def test_corrupt_drops_pay_amt6(sample_df):
    assert "PAY_AMT6" not in corrupt(sample_df).columns


def test_corrupt_keeps_row_count(sample_df):
    assert len(corrupt(sample_df)) == len(sample_df)


def test_corrupt_adds_missing_bill_amt1(sample_df):
    out = corrupt(sample_df)
    assert out["BILL_AMT1"].isna().sum() == int(0.15 * len(sample_df))


def test_corrupt_adds_out_of_range_age(sample_df):
    out = corrupt(sample_df)
    assert (out["AGE"] == 150).sum() == int(0.03 * len(sample_df))
    assert out["AGE"].max() > 100


def test_corrupt_adds_invalid_categories(sample_df):
    out = corrupt(sample_df)
    assert (out["EDUCATION"] == "9").sum() == int(0.10 * len(sample_df))
    assert (out["SEX"] == "unknown").sum() == int(0.02 * len(sample_df))
    assert (out["PAY_0"] == "12").sum() == int(0.03 * len(sample_df))


def test_corrupt_does_not_modify_input(sample_df):
    before = sample_df.copy()
    corrupt(sample_df)
    pd.testing.assert_frame_equal(sample_df, before)


def test_corrupt_is_reproducible(sample_df):
    pd.testing.assert_frame_equal(corrupt(sample_df), corrupt(sample_df))


# ---------- drift() ----------

def test_drift_scales_limit_bal(sample_df):
    out = drift(sample_df)
    n = len(sample_df)
    expected = (sample_df["LIMIT_BAL"] * 1.8).round().astype("int64").reset_index(drop=True)
    pd.testing.assert_series_equal(out["LIMIT_BAL"].iloc[:n], expected, check_names=False)


def test_drift_increases_default_rate(sample_df):
    out = drift(sample_df)
    assert out[config.LABEL].mean() > sample_df[config.LABEL].mean()


def test_drift_adds_rows(sample_df):
    defaulters = (sample_df[config.LABEL] == 1).sum()
    assert len(drift(sample_df)) == len(sample_df) + round(1.5 * defaulters)


def test_drift_shifts_education_toward_3(sample_df):
    out = drift(sample_df)
    assert (out["EDUCATION"] == "3").mean() > (sample_df["EDUCATION"] == "3").mean()


def test_drift_keeps_all_columns(sample_df):
    assert list(drift(sample_df).columns) == list(sample_df.columns)


def test_drift_with_no_defaulters(sample_df):
    df = sample_df.assign(**{config.LABEL: 0})
    out = drift(df)
    assert len(out) == len(df)  # nothing to oversample, no crash
