import pandas as pd
from ucimlrepo import fetch_ucirepo

from src import config
from src.corrupt import corrupt, drift
from src.logger import get_logger

log = get_logger("prepare_data")

RENAME = {"X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE"}
RENAME.update({f"X{i}": n for i, n in zip(range(6, 12), config.PAY_COLS)})
RENAME.update({f"X{i}": f"BILL_AMT{i - 11}" for i in range(12, 18)})
RENAME.update({f"X{i}": f"PAY_AMT{i - 17}" for i in range(18, 24)})
RENAME["Y"] = config.LABEL


def load_dataset():
    ds = fetch_ucirepo(id=350)  # Default of Credit Card Clients
    parts = [ds.data.features, ds.data.targets]
    if getattr(ds.data, "ids", None) is not None:
        parts.insert(0, ds.data.ids)
    df = pd.concat(parts, axis=1).rename(columns=RENAME)
    df.columns = [c.upper().replace(" ", "_") for c in df.columns]
    if "DEFAULT_PAYMENT_NEXT_MONTH" in df.columns:
        df = df.rename(columns={"DEFAULT_PAYMENT_NEXT_MONTH": config.LABEL})
    for col in config.STRING_COLS:
        df[col] = df[col].astype(str)
    return df


def run():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    log.info("Dataset loaded", extra={"ctx": {"rows": len(df), "cols": df.shape[1]}})

    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    n = len(df)
    cuts = [int(n * f) for f in (0.70, 0.80, 0.90, 0.95)]
    train = df.iloc[: cuts[0]]
    eval_df = df.iloc[cuts[0]: cuts[1]]
    serving = df.iloc[cuts[1]: cuts[2]].drop(columns=[config.LABEL])  # no label at serving time
    b_corrupt = corrupt(df.iloc[cuts[2]: cuts[3]])
    b_drift = drift(df.iloc[cuts[3]:])

    train.to_csv(config.TRAIN_CSV, index=False)
    for name, frame in [("eval", eval_df), ("serving", serving),
                        ("batch_corrupt", b_corrupt), ("batch_drift", b_drift)]:
        frame.to_csv(config.BATCHES[name][0], index=False)

    log.info("Splits written", extra={"ctx": {
        "train": len(train), "eval": len(eval_df), "serving": len(serving),
        "batch_corrupt": len(b_corrupt), "batch_drift": len(b_drift)}})


if __name__ == "__main__":
    run()
