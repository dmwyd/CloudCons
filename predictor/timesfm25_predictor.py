import os
from typing import List
import numpy as np
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from tqdm import tqdm
from typing import Optional,List
import timesfm
import logging 
logger = logging.getLogger("TimesFM-2.5-200M Predictor")
logger.setLevel(logging.INFO)

class TimesFmPredictor:
    def __init__(
        self,
        tfm,
        prediction_length: int,
        context_length: int,
        *args,
        **kwargs,
    ):
        self.tfm = tfm
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.quantiles = list(np.arange(1, 10) / 10.0)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = []
            max_context = 0
            for entry in batch:
                arr = np.array(entry["target"][-self.context_length:])
                if max_context < arr.shape[0]:
                    max_context = arr.shape[0]
                context.append(arr)
            max_context = (
                (max_context + self.tfm.model.p - 1) // self.tfm.model.p
            ) * self.tfm.model.p
            self.tfm.compile(
                forecast_config=timesfm.ForecastConfig(
                    max_context=min(15360, max_context),
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=128,
                ),
            )
            _, full_preds = self.tfm.forecast(
                horizon=self.prediction_length,
                inputs=context,
            )
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.quantiles)),
                    start_date=forecast_start_date,
                )
            )

        return forecasts

