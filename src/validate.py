import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2

from src import alert, config
from src.baseline import read_csv, save_html, stats_options
from src.logger import get_logger

log = get_logger("validate")


def to_list(anomalies, batch, kind):
    items = []
    for feat, info in anomalies.anomaly_info.items():
        items.append({
            "batch": batch,
            "check": kind,
            "feature": feat,
            "severity": anomalies_pb2.AnomalyInfo.Severity.Name(info.severity),
            "short": info.short_description,
            "description": info.description,
        })
    return items


def validate_batch(name, path, env, schema, train_stats, cols):
    df = read_csv(path)
    stats = tfdv.generate_statistics_from_dataframe(df, stats_options=stats_options(cols, schema))
    save_html(stats, config.REPORTS_DIR / f"{name}_vs_train.html", rhs=train_stats,
              lhs_name=name, rhs_name="train")

    if env == "SERVING":
        # Schema check with the SERVING environment (label not required)
        anomalies = tfdv.validate_statistics(stats, schema, environment="SERVING")
        found = to_list(anomalies, name, "schema")
        # Skew check: train vs serving
        skew = tfdv.validate_statistics(train_stats, schema, serving_statistics=stats, environment="TRAINING")
        found += to_list(skew, name, "skew")
    else:
        # Schema + drift check against the training baseline
        anomalies = tfdv.validate_statistics(stats, schema, previous_statistics=train_stats,
                                             environment="TRAINING")
        found = to_list(anomalies, name, "schema+drift")

    tfdv.write_anomalies_text(anomalies, str(config.REPORTS_DIR / f"{name}_anomalies.pbtxt"))

    for item in found:
        log.warning(f"{name} | {item['check']} | {item['feature']} | {item['short']}", extra={"ctx": item})
    if not found:
        log.info(f"{name} passed validation", extra={"ctx": {"batch": name, "rows": len(df)}})

    alerted = alert.check(name, found)
    return {"batch": name, "env": env, "rows": len(df), "anomalies": len(found), "alerted": alerted}


def run(schema=None, train_stats=None, cols=None):
    if schema is None:
        schema = tfdv.load_schema_text(str(config.SCHEMA_FILE))
    if train_stats is None:
        train_stats = tfdv.load_stats_text(str(config.TRAIN_STATS))
    if cols is None:
        cols = [c for c in read_csv(config.TRAIN_CSV).columns if c not in config.DROP_FEATURES]
    return [validate_batch(n, p, env, schema, train_stats, cols) for n, (p, env) in config.BATCHES.items()]


if __name__ == "__main__":
    run()
