"""Checks that the curated TFDV schema accepts good data and flags bad data."""
import pytest

tfdv = pytest.importorskip("tensorflow_data_validation")

from src import config  # noqa: E402
from src.baseline import curate_schema, stats_options  # noqa: E402


def stats(df, schema=None):
    cols = [c for c in df.columns if c not in config.DROP_FEATURES]
    return tfdv.generate_statistics_from_dataframe(df, stats_options=stats_options(cols, schema))


@pytest.fixture
def schema(sample_df):
    return curate_schema(tfdv.infer_schema(stats(sample_df)))


def anomaly_features(df, schema, env="TRAINING"):
    result = tfdv.validate_statistics(stats(df, schema), schema, environment=env)
    return set(result.anomaly_info.keys())


def test_valid_data_passes(sample_df, schema):
    assert anomaly_features(sample_df, schema) == set()


def test_unseen_education_above_tolerance_flagged(sample_df, schema):
    df = sample_df.copy()
    df.loc[df.index[:20], "EDUCATION"] = "9"  # 10%, above the 3% tolerance
    assert "EDUCATION" in anomaly_features(df, schema)


def test_unseen_education_within_tolerance_passes(sample_df, schema):
    df = sample_df.copy()
    df.loc[df.index[:2], "EDUCATION"] = "5"  # 1%, within the 3% tolerance
    assert "EDUCATION" not in anomaly_features(df, schema)


def test_invalid_sex_flagged(sample_df, schema):
    df = sample_df.copy()
    df.loc[df.index[:1], "SEX"] = "unknown"  # no tolerance for SEX
    assert "SEX" in anomaly_features(df, schema)


def test_age_out_of_range_flagged(sample_df, schema):
    df = sample_df.copy()
    df.loc[df.index[:1], "AGE"] = 150
    assert "AGE" in anomaly_features(df, schema)


def test_negative_limit_bal_flagged(sample_df, schema):
    df = sample_df.copy()
    df.loc[df.index[:1], "LIMIT_BAL"] = -5
    assert "LIMIT_BAL" in anomaly_features(df, schema)


def test_shared_pay_domain_flags_any_pay_column(sample_df, schema):
    df = sample_df.copy()
    df.loc[df.index[:5], "PAY_6"] = "12"
    assert "PAY_6" in anomaly_features(df, schema)


def test_missing_label_ok_in_serving(sample_df, schema):
    serving = sample_df.drop(columns=[config.LABEL])
    assert config.LABEL not in anomaly_features(serving, schema, env="SERVING")


def test_missing_label_flagged_in_training(sample_df, schema):
    serving = sample_df.drop(columns=[config.LABEL])
    assert config.LABEL in anomaly_features(serving, schema, env="TRAINING")
