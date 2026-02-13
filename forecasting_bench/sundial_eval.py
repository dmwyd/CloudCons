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
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast
import logging
import csv
import os
from data import Dataset,Granularity
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality


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

OUTPUT_DIR="outputs/forecasting/sundial"

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

class SundialPredictor:
    def __init__(
        self,
        num_samples: int,
        prediction_length: int,
        device_map,
        context_length: int=512,
        batch_size: int = 512,
    ):
        self.device = device_map
        self.model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)  
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.context_length=context_length

    def left_pad_and_stack_1D(self, tensors):
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 1
            padding = torch.full(
                size=(max_len - len(c),), fill_value=torch.nan, device=c.device
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def prepare_and_validate_context(self, context):
        if isinstance(context, list):
            context = self.left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def predict(
        self,
        test_data_input,
    ):
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
            context = [torch.tensor(entry["target"]) for entry in batch]
            batch_x = self.prepare_and_validate_context(context)
            if batch_x.shape[-1] > self.context_length:
                batch_x = batch_x[..., -self.context_length:]
            if torch.isnan(batch_x).any():
                batch_x = np.array(batch_x)
                imputed_rows = []
                for i in range(batch_x.shape[0]):
                    row = batch_x[i]
                    imputed_row = LastValueImputation()(row)
                    imputed_rows.append(imputed_row)
                batch_x = np.vstack(imputed_rows)
                batch_x = torch.tensor(batch_x)
            batch_x = batch_x.to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model.generate(batch_x, max_new_tokens=self.prediction_length, revin=True, num_samples=self.num_samples)
            forecast_outputs.append(
                outputs.detach().cpu().numpy()
            )
        forecast_outputs = np.concatenate(forecast_outputs)

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
    predictor = SundialPredictor(
            num_samples=100,
            prediction_length=hf_dataset.prediction_length,
            device_map="cuda:1",
            context_length=context_length,
            batch_size=512,  
        )
    
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
                "Sundial",
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
    




