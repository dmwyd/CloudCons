from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
import torch
import numpy as np
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
import logging
import csv
import os
from data import Dataset,Granularity
from gluonts.model import evaluate_model, Forecast
from gluonts.time_feature import get_seasonality
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Kairos"))

from tsfm.model.kairos import AutoModel
from typing import List
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

OUTPUT_DIR="outputs/forecasting/kairos_50m"


eval_config=[
    {
      "dataset_name": "huawei2025",
      "to_univariate": True,
   },
   {
      "dataset_name": "borg2019_d_trace",
      "to_univariate": True,
   },
   {
      "dataset_name": "borg2019_e_trace",
      "to_univariate": True,
   },
   {
      "dataset_name": "azure2019",
      "to_univariate": False,
   }
]


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()

def pad_or_truncate(sequence, max_length=2048, pad_value=np.nan):
    """
    Pads or truncates a sequence on the left to a specified max_length.

    Args:
        sequence (list or np.ndarray): The input sequence.
        max_length (int): The target length.
        pad_value (int or float): The value to use for padding, defaults to np.nan.

    Returns:
        np.ndarray: A NumPy array of length max_length.
    """
    seq_np = np.array(sequence)
    current_length = len(seq_np)

    if current_length < max_length:
        # If the current length is less than the target, calculate the required padding
        padding_size = max_length - current_length
        # Use np.pad to add padding to the left
        # (padding_size, 0) means pad `padding_size` elements at the beginning of the first (and only) axis
        return np.pad(seq_np, (padding_size, 0), 'constant', constant_values=pad_value)
    else:
        # If the current length is greater than or equal to the target, truncate to the last max_length elements
        return seq_np[-max_length:]

class KairosPredictor:
    def __init__(
        self,
        model_path,
        prediction_length: int,
        context_length: int,
        *args,
        **kwargs,
    ):
        self.prediction_length = prediction_length
        self.context_length = context_length

        # 1. Check for CUDA availability and set the primary device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load the model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # 3. Move the model to the primary device
        self.model.to(self.device)

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        self.model.eval()
        model = self.model
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                with torch.no_grad():
                    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                        context = [torch.tensor(pad_or_truncate(entry["target"], max_length=self.context_length)) for entry in batch]
                        forecast_outputs.append(
                            model(
                                past_target=torch.stack(context).to(self.device),
                                prediction_length=self.prediction_length,
                                generation=True,
                                infer_is_positive=True,
                                force_flip_invariance=True,
                            )["prediction_outputs"].detach().cpu().numpy()
                        )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts

def predict(granularity,dataset_name,prediction_length,context_length,to_univariate=True):
    hf_dataset=Dataset(granularity=granularity,dataset_name=dataset_name, prediction_length=prediction_length,to_univariate=to_univariate)
    season_length = get_seasonality(hf_dataset.freq)

    output_dir=os.path.join(
        OUTPUT_DIR,
        granularity.value,
        hf_dataset.dataset_name,
        str(hf_dataset.prediction_length),
    )
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "metrics.csv")

    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(
            [
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
            ]
        )

    #TODO
    predictor = KairosPredictor(
        model_path="mldi-lab/Kairos_50m",
        prediction_length=hf_dataset.prediction_length,
        context_length=context_length,
    )

    # Measure the time taken for evaluation
    res = evaluate_model(
        predictor,
        test_data=hf_dataset.test_data,
        metrics=metrics,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Kairos_50m",
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
            ]
        )

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(
        WarningFilter("The mean prediction is not stored in the forecast data")
    )

    for config in eval_config:
        for pl in [6,8,10,12]:
            predict(granularity=Granularity.HOUR,dataset_name=config["dataset_name"], prediction_length=pl,context_length=36,to_univariate=config["to_univariate"])

    




