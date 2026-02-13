import os
from typing import List
import numpy as np
from tqdm.auto import tqdm
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast,SampleForecast
import logging
from gluonts.model import evaluate_model
import csv
import os
from gluonts.time_feature import get_seasonality
from data import Dataset,Granularity
from chronos import BaseChronosPipeline, Chronos2Pipeline
from dataclasses import dataclass, field
from typing import Optional,List
import torch
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
    WeightedSumQuantileLoss
)
import itertools
import pandas as pd

from gluonts.model import evaluate_forecasts


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
    )
]

OUTPUT_DIR="outputs/forecasting/chronos2"

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

logger = logging.getLogger("Chronos-2 Predictor")
logger.setLevel(logging.INFO)

class Chronos2Predictor:
    def __init__(
        self,
        model_name: str,
        prediction_length: int,
        context_length: int,
        batch_size: int,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        predict_batches_jointly: bool = False,
        **kwargs
    ):
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            **kwargs,
        )
        assert isinstance(self.pipeline, Chronos2Pipeline), "This is Predictor is for Chronos-2, see other notebook for Chronos and Chronos-Bolt"
        self.prediction_length = prediction_length
        self.context_length=context_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.predict_batches_jointly = predict_batches_jointly


    def _pack_model_items(self, items):
        for item in items:
            model_input = {
                "target": item["target"],
            }
            if model_input["target"].ndim==1:
                model_input["target"]=model_input["target"][-self.context_length:]
            else:
                model_input["target"]=model_input["target"][:,-self.context_length:]
            yield model_input


    def predict(self, test_data_input) -> List[Forecast]:
        pipeline = self.pipeline
        model_batch_size = self.batch_size
        if self.predict_batches_jointly:
            logger.info("Note: Using cross learning mode. Please ensure that different rolling windows of the same time series are not in `test_data_input` to avoid any potential leakage due to in-context learning.")

        # Generate forecasts
        forecast_outputs = []
        input_data = list(self._pack_model_items(test_data_input))
        is_univariate_data = input_data[0]["target"].ndim == 1  # homogenous across all intputs
        while True:
            try:
                quantiles, _ = pipeline.predict_quantiles(
                        inputs=input_data,
                        prediction_length=self.prediction_length,
                        batch_size=model_batch_size,
                        quantile_levels=self.quantile_levels,
                        predict_batches_jointly=self.predict_batches_jointly,
                )
                quantiles = torch.stack(quantiles)
                # quantiles [batch, variates, seq_len, quantiles]
                quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
                # forecast_outputs [batch, quantiles, seq_len, variates]
                if is_univariate_data:
                    quantiles = quantiles.squeeze(-1) # squeeze variate to avoid error in eval due to broadcasting
                assert quantiles.shape[1] == len(self.quantile_levels)
                assert quantiles.shape[2] == self.prediction_length
                forecast_outputs.append(quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OutOfMemoryError at model_batch_size {model_batch_size}, reducing to {model_batch_size // 2}")
                model_batch_size //= 2

        # Convert forecasts into gluonts Forecast objects
        forecast_outputs = np.concatenate(forecast_outputs, axis=0)
        assert len(forecast_outputs) == len(input_data)
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecast = QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantile_levels)),
                start_date=forecast_start_date,
            )
            forecasts.append(forecast)
        return forecasts


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


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
                "eval_metrics/mean_weighted_sum_quantile_loss"
                
            ]
        )

    # TODO
    predictor = Chronos2Predictor(
        model_name="s3://autogluon/chronos-2",
        prediction_length=hf_dataset.prediction_length,
        context_length=context_length,
        batch_size=256,
        predict_batches_jointly=True,
        device_map="cuda",
        torch_dtype="float32",
    )
    
    # Avoid cross batch leakage of rolling evalution by prediction of windows individually.
    forecast_windows = []
    n_windows = hf_dataset.test_data.windows
    for window_idx in range(n_windows):
        entries_window_k = list(itertools.islice(hf_dataset.test_data.input, window_idx, None, n_windows))
        forecasts_window_k = list(predictor.predict(entries_window_k))
        forecast_windows.append(forecasts_window_k)        

    forecasts = [item for items in zip(*forecast_windows) for item in items]

    res=evaluate_forecasts(
        forecasts,
        test_data=hf_dataset.test_data,
        metrics=metrics,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    ) 

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Chronos2",
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
                res["mean_weighted_sum_quantile_loss"][0]
            ]
        )
  


if __name__ == "__main__":
    for config in eval_config:
        for pl in [6,8,10,12]:
            predict(granularity=Granularity.HOUR,dataset_name=config["dataset_name"], prediction_length=pl,context_length=36,to_univariate=config["to_univariate"])
