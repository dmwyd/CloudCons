import os
from typing import List
import numpy as np
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast,SampleForecast
import os

from chronos import BaseChronosPipeline, Chronos2Pipeline
from typing import Optional,List
import torch

from .data import Dataset
import logging 
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



