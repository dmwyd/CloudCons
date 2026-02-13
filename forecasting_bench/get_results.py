import os
import pandas as pd

BASE_DIR = "outputs/forecasting"
models = [
    "autoarima",
    "autoets",
    "autotheta",
    "d_linear",
    "deepar",
    "patch_tst",
    "tft",
    "moirai2",
    "chronos2",
    "timesfm25",
    "sundial",
    "toto",
    "flowstate-9.1m",
    "kairos_50m",
]
dataset_name = [
    "azure2019",
    "huawei2025",
    "borg2019_d_trace",
    "borg2019_e_trace",
]
freq = ["5T", "30T", "1H"]


def get_seasonal_naive_results():
    seasonal_naive_results = pd.DataFrame(
        columns=["dataset", "freq", "prediction_length", "MASE", "CRPS"]
    )
    for dataset in dataset_name:
        for f in freq:
            dir_path = f"{BASE_DIR}/seasonalnaive/{f}/{dataset}"
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith(".csv"):
                            csv_path = os.path.join(subdir_path, file)
                            df = pd.read_csv(csv_path)
                            mase = (
                                df["eval_metrics/MASE[0.5]"].values[0]
                                if "eval_metrics/MASE[0.5]" in df.columns
                                else df["MASE[0.5]"].values[0]
                            )
                            crps = (
                                df["eval_metrics/mean_weighted_sum_quantile_loss"].values[0]
                                if "eval_metrics/mean_weighted_sum_quantile_loss"
                                in df.columns
                                else df["mean_weighted_sum_quantile_loss"].values[0]
                            )
                            new_row = pd.DataFrame(
                                {
                                    "dataset": [dataset],
                                    "freq": [f],
                                    "prediction_length": [subdir],
                                    "MASE": [mase],
                                    "CRPS": [crps],
                                }
                            )
                            seasonal_naive_results = pd.concat(
                                [seasonal_naive_results, new_row], ignore_index=True
                            )
    return seasonal_naive_results


def main():
    seasonal_naive_results = get_seasonal_naive_results()
    results_all = pd.DataFrame()

    for model in models:
        for dataset in dataset_name:
            results = pd.DataFrame(columns=["MASE", "CRPS"])
            for f in freq:
                dir_path = f"{BASE_DIR}/{model}/{f}/{dataset}"
                for subdir in os.listdir(dir_path):
                    subdir_path = os.path.join(dir_path, subdir)
                    if os.path.isdir(subdir_path):
                        for file in os.listdir(subdir_path):
                            if file.endswith(".csv"):
                                csv_path = os.path.join(subdir_path, file)
                                df = pd.read_csv(csv_path)
                                mase = (
                                    df["eval_metrics/MASE[0.5]"].values[0]
                                    if "eval_metrics/MASE[0.5]" in df.columns
                                    else df["MASE[0.5]"].values[0]
                                )
                                crps = (
                                    df[
                                        "eval_metrics/mean_weighted_sum_quantile_loss"
                                    ].values[0]
                                    if "eval_metrics/mean_weighted_sum_quantile_loss"
                                    in df.columns
                                    else df["mean_weighted_sum_quantile_loss"].values[0]
                                )
                                new_row = pd.DataFrame(
                                    {
                                        "MASE": [mase],
                                        "CRPS": [crps],
                                        "MASE_norm": [
                                            mase
                                            / seasonal_naive_results[
                                                (seasonal_naive_results["dataset"] == dataset)
                                                & (seasonal_naive_results["freq"] == f)
                                                & (
                                                    seasonal_naive_results[
                                                        "prediction_length"
                                                    ]
                                                    == subdir
                                                )
                                            ]["MASE"].values[0]
                                        ],
                                        "CRPS_norm": [
                                            crps
                                            / seasonal_naive_results[
                                                (seasonal_naive_results["dataset"] == dataset)
                                                & (seasonal_naive_results["freq"] == f)
                                                & (
                                                    seasonal_naive_results[
                                                        "prediction_length"
                                                    ]
                                                    == subdir
                                                )
                                            ]["CRPS"].values[0]
                                        ],
                                    }
                                )
                                results = pd.concat([results, new_row], ignore_index=True)

            avg_results = pd.DataFrame(
                {
                    "dataset": [dataset],
                    "model": [model],
                    "MASE": [(results["MASE"].prod()) ** (1 / len(results))],
                    "CRPS": [(results["CRPS"].prod()) ** (1 / len(results))],
                    "MASE_norm": [(results["MASE_norm"].prod()) ** (1 / len(results))],
                    "CRPS_norm": [(results["CRPS_norm"].prod()) ** (1 / len(results))],
                }
            )
            results_all = pd.concat([results_all, avg_results], ignore_index=True)

    results_all.to_csv(
        f"{BASE_DIR}/results.csv", index=False
    )


if __name__ == "__main__":
    main()