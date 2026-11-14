import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList

from src import config
from src.baseline import read_csv, save_html
from src.logger import get_logger

log = get_logger("slicing")


def split_datasets(stats_list):
    out = []
    for ds in stats_list.datasets:
        proto = DatasetFeatureStatisticsList()
        proto.datasets.extend([ds])
        out.append(proto)
    return out


def slice_options(schema, cols, slice_fns):
    base = dict(schema=schema, infer_type_from_schema=True, feature_allowlist=cols)
    try:
        return tfdv.StatsOptions(experimental_slice_functions=slice_fns, **base)
    except TypeError:
        return tfdv.StatsOptions(slice_functions=slice_fns, **base)


def run(schema, cols):
    slice_fns = [slicing_util.get_feature_value_slicer(features={f: None}) for f in config.SLICE_FEATURES]
    sliced = tfdv.generate_statistics_from_csv(
        str(config.TRAIN_CSV), stats_options=slice_options(schema, cols, slice_fns)
    )
    slices = {d.datasets[0].name: d for d in split_datasets(sliced)}
    for name, d in slices.items():
        log.info(f"Slice {name}: {d.datasets[0].num_examples} rows",
                 extra={"ctx": {"slice": name, "rows": d.datasets[0].num_examples}})

    if "SEX_1" in slices and "SEX_2" in slices:
        save_html(slices["SEX_1"], config.REPORTS_DIR / "slice_sex_1_vs_2.html",
                  rhs=slices["SEX_2"], lhs_name="SEX_1 (male)", rhs_name="SEX_2 (female)")

    # Default rate per group, useful for spotting bias across slices
    train = read_csv(config.TRAIN_CSV)
    for feat in config.SLICE_FEATURES:
        rates = train.groupby(feat)[config.LABEL].mean().round(3).to_dict()
        log.info(f"Default rate by {feat}: {rates}", extra={"ctx": {"feature": feat, "default_rate": rates}})
    return slices


if __name__ == "__main__":
    s = tfdv.load_schema_text(str(config.SCHEMA_FILE))
    run(s, [c for c in read_csv(config.TRAIN_CSV).columns if c not in config.DROP_FEATURES])
