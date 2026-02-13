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
import logging
import csv
import os
import random
from data import Dataset,Granularity
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
import sys

granite_tsfm_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "granite-tsfm",
    )
)
sys.path.append(granite_tsfm_path)

from tsfm_public import FlowStateForPrediction
from notebooks.hfdemo.flowstate.gift_wrapper import FlowState_Gift_Wrapper

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

OUTPUT_DIR="outputs/forecasting/flowstate-9.1m"

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()

def LoadFlowState(model_name, prediction_length,context_length, target_dim, freq, batch_size,device='cuda', domain=None, nd=False):
    flowstate = FlowStateForPrediction.from_pretrained(model_name).to(device)

    config = flowstate.config
    config.min_context = 0
    config.device = device
    flowstate = FlowState_Gift_Wrapper(flowstate, prediction_length, context_length=context_length, n_ch=target_dim, batch_size=batch_size, 
                                 f=freq, device=device, domain=domain, no_daily=nd)
    return flowstate


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
    flowstate = LoadFlowState(model_name="ibm-research/FlowState",
                                prediction_length=prediction_length,
                                context_length=context_length,
                                target_dim=1,        
                                freq=hf_dataset.freq if hf_dataset.freq[-1] != "H" else "H",      
                                device="cuda",
                                batch_size=64,
                                )
    
    res = evaluate_model(
        flowstate,
        test_data=hf_dataset.test_data,
        metrics=metrics,
        axis=None,
        batch_size=64,
        mask_invalid_label=True,
        allow_nan_forecast=False,
    )

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "FlowState-9.1M",
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

    




