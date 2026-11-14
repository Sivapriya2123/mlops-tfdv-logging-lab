from src import baseline, prepare_data, slicing, validate
from src.logger import get_logger, log_step

log = get_logger("pipeline")


def main():
    with log_step(log, "prepare_data"):
        prepare_data.run()
    with log_step(log, "baseline"):
        schema, stats, cols = baseline.run()
    try:
        with log_step(log, "slicing"):
            slicing.run(schema, cols)
    except Exception:
        log.error("Slicing failed, continuing with validation")
    with log_step(log, "validate"):
        results = validate.run(schema, stats, cols)

    log.info("===== SUMMARY =====")
    for r in results:
        status = "ALERT" if r["alerted"] else ("WARN" if r["anomalies"] else "OK")
        log.info(f"{r['batch']:<14} env={r['env']:<8} rows={r['rows']:<6} "
                 f"anomalies={r['anomalies']:<3} status={status}", extra={"ctx": r})


if __name__ == "__main__":
    main()
