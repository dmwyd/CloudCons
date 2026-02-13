from typing import Optional
import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model import evaluate_model
import csv
import os
import time
from gluonts.time_feature import get_seasonality

from data import Dataset, Granularity
from gluonts.model.forecast import QuantileForecast
import json
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import logging

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


OUTPUT_DIR="outputs/forecasting/moirai2"

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

def get_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    return device

class MoiraiQuantilePredictor:
    def __init__(
            self,
            model_path: str,
            prediction_length: int = 100,
            context_length: int = 4000,
            target_dim: int =1,
            feat_dynamic_real_dim: int =0,
            past_feat_dynamic_real_dim: int =0,
            device: str = 'auto',
            batch_size: int = 2048,
            quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        ):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = get_device(device)
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_path),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        ).to(self.device)

    def predict(self, test_data_input):
        while True:
            try:
                print("Model - MoiraiQuantile loaded with batch_size:", self.batch_size)
                # Generate forecast samples
                forecast_quantiles = []
                for batch in (batcher(test_data_input, batch_size=self.batch_size)):
                    past_target = [entry["target"] for entry in batch]
                    forecasts = self.model.predict(past_target) # full_forecasts shape: (batch num_quantiles future_time #tgt)
                    forecast_quantiles.append(forecasts)
                forecast_quantiles = np.concatenate(forecast_quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {self.batch_size}, reducing to {self.batch_size // 2}"
                )
                self.batch_size //= 2

        # Convert forecast samples into gluonts QuantileForecast objects
        quantile_forecasts = []
        for item, ts in zip(forecast_quantiles, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            quantile_forecasts.append(
                QuantileForecast(
                item_id = ts["item_id"],
                forecast_arrays=item,
                start_date=forecast_start_date,
                forecast_keys=list(map(str, self.quantile_levels)))
            )
        return quantile_forecasts


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
    predictor = MoiraiQuantilePredictor(
        model_path=f"Salesforce/moirai-2.0-R-small",
        prediction_length=hf_dataset.prediction_length,
        context_length=context_length,
        target_dim=1,
        past_feat_dynamic_real_dim=hf_dataset.past_feat_dynamic_real_dim,
        batch_size=512,
        device="cuda",
    )

    res = evaluate_model(
        predictor,
        test_data=hf_dataset.test_data,
        metrics=metrics,
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Moirai2-small",
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
    
    for config in eval_config:
        for pl in [6,8,10,12]:
            predict(granularity=Granularity.HOUR,dataset_name=config["dataset_name"], prediction_length=pl,context_length=36,to_univariate=config["to_univariate"])