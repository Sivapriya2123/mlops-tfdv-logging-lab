import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

from src import config
from src.logger import get_logger

log = get_logger("baseline")


def read_csv(path):
    cols = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: str for c in config.STRING_COLS if c in cols})


def allowlist(df):
    """Features used for stats (drops internal ID columns)."""
    return [c for c in df.columns if c not in config.DROP_FEATURES]


def stats_options(cols, schema=None):
    return tfdv.StatsOptions(
        schema=schema,
        infer_type_from_schema=schema is not None,
        feature_allowlist=cols,
    )


def save_html(lhs, path, rhs=None, lhs_name="train", rhs_name="batch"):
    """Saves a Facets stats view you can open in a browser."""
    try:
        from tensorflow_data_validation.utils.display_util import get_statistics_html

        html = get_statistics_html(lhs, rhs_statistics=rhs, lhs_name=lhs_name, rhs_name=rhs_name)
        path.write_text(html)
        log.info("Saved stats HTML", extra={"ctx": {"file": str(path)}})
    except Exception as e:
        log.warning(f"Could not save HTML: {e}")


def curate_schema(schema):
    # 1. Only documented category codes
    for feat, codes in config.VALID_CODES.items():
        tfdv.set_domain(schema, feat, schema_pb2.StringDomain(name=feat, value=codes))

    # 2. Relax domain mass where the raw data has a few undocumented codes
    for feat, mass in config.RELAXED_FEATURES.items():
        tfdv.get_feature(schema, feat).distribution_constraints.min_domain_mass = mass

    # 3. Shared domain for all repayment status columns
    pay = schema.string_domain.add()
    pay.name = "PAY_STATUS"
    pay.value.extend(config.PAY_STATUS_CODES)
    for feat in config.PAY_COLS:
        tfdv.set_domain(schema, feat, "PAY_STATUS")

    # 4. Business rules for numeric ranges
    tfdv.set_domain(schema, "AGE", schema_pb2.IntDomain(name="AGE", min=18, max=100))
    tfdv.set_domain(schema, "LIMIT_BAL", schema_pb2.IntDomain(name="LIMIT_BAL", min=0))

    # 5. Environments: label is not expected at serving time
    schema.default_environment.extend(["TRAINING", "SERVING"])
    tfdv.get_feature(schema, config.LABEL).not_in_environment.append("SERVING")

    # 6. Skew (train vs serving) and drift (train vs new batch)
    for feat in config.CATEGORICAL + ["PAY_0"]:
        f = tfdv.get_feature(schema, feat)
        f.skew_comparator.infinity_norm.threshold = config.SKEW_THRESHOLD
        f.drift_comparator.infinity_norm.threshold = config.DRIFT_THRESHOLD
    for feat in ["LIMIT_BAL", "AGE"]:
        try:
            comp = tfdv.get_feature(schema, feat).drift_comparator
            comp.jensen_shannon_divergence.threshold = config.DRIFT_THRESHOLD
        except AttributeError:
            log.warning(f"JSD drift not supported in this TFDV version for {feat}")
    return schema


def run():
    config.SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = read_csv(config.TRAIN_CSV)
    cols = allowlist(train)
    stats = tfdv.generate_statistics_from_dataframe(train, stats_options=stats_options(cols))
    tfdv.write_stats_text(stats, str(config.TRAIN_STATS))

    schema = curate_schema(tfdv.infer_schema(stats))
    tfdv.write_schema_text(schema, str(config.SCHEMA_FILE))  # frozen schema
    log.info("Schema saved", extra={"ctx": {"features": len(schema.feature), "file": str(config.SCHEMA_FILE)}})

    # Sanity check: curated schema should accept the training data
    train_anoms = tfdv.validate_statistics(stats, schema, environment="TRAINING")
    if not train_anoms.anomaly_info:
        log.info("Training data passes the curated schema")
    for feat, info in train_anoms.anomaly_info.items():
        log.warning(f"Training data issue in {feat}: {info.short_description}",
                    extra={"ctx": {"feature": feat, "description": info.description}})

    save_html(stats, config.REPORTS_DIR / "train_stats.html")
    return schema, stats, cols


if __name__ == "__main__":
    run()
