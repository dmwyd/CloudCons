import os
import gc

# Third-party imports
from dataclasses import dataclass

import numpy as np
import torch
import csv

# Local imports
from gluonts.dataset.split import split
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
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
import logging

from data import Dataset,Granularity

from toto.model.toto import Toto
from toto.inference.gluonts_predictor import Multivariate, TotoPredictor

#TODO
NUM_SAMPLES=50
USE_KV_CACHE=True


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

eval_config=[
    {
      "dataset_name": "huawei2025",
      "to_univariate": True,
   },
#     {
#       "dataset_name": "alibaba2018",
#       "to_univariate": True,
#    },
#     {
#       "dataset_name": "borg2011",
#       "to_univariate": True,
#    },
   {
      "dataset_name": "borg2019_d_trace",
      "to_univariate": True,
   },
   {
      "dataset_name": "borg2019_e_trace",
      "to_univariate": True,
   },
#    {
#       "dataset_name": "azure2017",
#       "to_univariate": False,
#    },
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

def get_total_gpu_memory():
    """Get total GPU VRAM capacity in MB."""
    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)


def calculate_optimal_batch_size(
    model,
    target_dim,
    prediction_length,
    context_length,
    use_kv_cache,
    num_samples,
    safety_factor=0.2,
):
    """
    Calculate the optimal batch size based on available GPU memory and model requirements.

    Args:
        model: Pre-loaded TOTO model
        target_dim: Target dimensionality (number of variates)
        prediction_length: Length of prediction horizon
        context_length: Context window length
        use_kv_cache: Whether KV cache is used
        num_samples: Number of samples to generate
        safety_factor: Safety factor to apply when calculating available memory (default=0.01)

    Returns:
        Suggested batch size for prediction
    """

    try:
        # Extract model size information
        model_width = model.model.embed_dim  # Feature dimension
        model_depth = model.model.num_layers  # Number of transformer layers

        # Calculate model's parameter memory footprint in MB
        model_param_memory_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        # Base memory per sample in MB (parameters + activations + gradients)
        base_memory_per_sample = (model_width * model_depth * 4) / (1024 * 1024)

        # Memory for input/output tensors
        io_memory = (target_dim * (context_length + prediction_length) * 4) / (
            1024 * 1024
        )

        # KV cache memory (if used)
        kv_memory = 0
        if use_kv_cache:
            kv_memory = (model_depth * model_width * 2 * context_length * 4) / (
                1024 * 1024
            )

        # Total memory per sample
        mem_per_sample_mb = base_memory_per_sample + io_memory + kv_memory

        # Factor in target dimensions and samples directly
        # Each dimension and sample has a direct multiplicative effect on memory
        mem_per_batch_mb = (
            mem_per_sample_mb * target_dim * num_samples
        )  # Total memory for a batch with num_samples samples

        # Get total GPU VRAM capacity and subtract model parameter memory
        gpu_mem = get_total_gpu_memory()  # in MB
        cuda_reserved_mb = 1024  # Reserve 1GB for CUDA runtime and other overhead

        # Available memory = (Total VRAM - Model parameters - CUDA reserved) * safety factor
        available_memory = (
            gpu_mem - model_param_memory_mb - cuda_reserved_mb
        ) * safety_factor

        # Calculate max batch size based on available memory
        max_batch_size = max(
            1, int(available_memory / (mem_per_batch_mb / num_samples))
        )

        max_batch_size = min(16, max_batch_size)
        return max_batch_size
    except RuntimeError as e:
        print(f"Error calculating optimal batch size: {e}")
        return 1

class TOTOModelPredictorWrapper:
    """Wrapper for TOTOPredictor that handles OOM errors by adjusting batch size."""

    def __init__(
        self,
        model,
        prediction_length,
        context_length,
        mode,
        num_samples=128,
        use_kv_cache=True,
    ):
        """
        Initialize the predictor wrapper with specified parameters.

        Args:
            model: The loaded TOTO model instance to use for predictions
            prediction_length: The length of the prediction horizon.
            context_length: The length of the context window.
            mode: Mode of prediction (e.g., "forecast").
            num_samples: Total number of samples to generate.
            use_kv_cache: Whether to use key-value caching.
        """

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.mode = mode
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.samples_per_batch = (
            num_samples  # Start with full batch size and adjust if needed
        )
        self.model = model
        self._adjusted = False  # Tracks whether adjustment has been done

        self._initialize_predictor()

    def _initialize_predictor(self):
        """
        Initialize the TOTOPredictor with the current samples_per_batch.
        """
        self.predictor = TotoPredictor.create_for_eval(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            mode=self.mode,
            samples_per_batch=self.samples_per_batch,
        )

    def predict(self, gluonts_test_data: tuple):
        """
        Perform prediction while adjusting num_samples, samples_per_batch, and context_length if OOM errors occur.
        """
        predictions = None

        # Adjust only on the first call.
        if not self._adjusted:

            print(
                "Initializing predictor with samples_per_batch =",
                self.samples_per_batch,
            )
            while self.samples_per_batch >= 1:
                try:
                    print(
                        f"Attempting prediction with samples_per_batch = {self.samples_per_batch} and context_length = {self.context_length}"
                    )
                    # Attempt prediction (consume the generator to catch any OOM)
                    predictions = list(
                        self.predictor.predict(
                            gluonts_test_data,
                            use_kv_cache=self.use_kv_cache,
                            num_samples=self.num_samples,
                        )
                    )
                    self._adjusted = True
                    return predictions  # Prediction succeeded

                except RuntimeError as e:
                    print("Caught exception during prediction:", e)
                    if "CUDA out of memory" in str(e):
                        # First, try reducing the batch size if possible.
                        if self.samples_per_batch > 1:
                            print(
                                f"Out of memory with samples_per_batch = {self.samples_per_batch}. Reducing batch size."
                            )
                            self.samples_per_batch = self.samples_per_batch // 2
                            # Clean up GPU memory before trying with smaller batch size
                            torch.cuda.empty_cache()
                        else:
                            # Cannot reduce batch size further, so we fail
                            print(
                                f"OOM at minimal batch size. Cannot proceed with this context length and sample count."
                            )
                            raise e
                        # Reinitialize the predictor with the new settings.
                        self._initialize_predictor()
                    else:
                        raise e  # Re-raise unexpected exceptions

        # For subsequent calls, simply return the generator.
        return self.predictor.predict(
            gluonts_test_data,
            use_kv_cache=self.use_kv_cache,
            num_samples=self.num_samples,
        )

def try_prediction_with_config(
    model,
    prediction_length,
    context_length,
    mode,
    num_samples,
    test_data,
    season_length,
    use_kv_cache,
    min_context_length=None,
):
    """
    Attempt prediction with a specific configuration, handling OOM errors.

    Args:
        model: The loaded model instance to use
        prediction_length: Prediction horizon length
        context_length: Context window length
        mode: Prediction mode
        num_samples: Number of samples to generate (fixed for evaluation)
        test_data: data to evaluate on
        use_kv_cache: Whether to use key-value caching
        min_context_length: Minimum allowed context length

    Returns:
        Metrics result if successful, None if OOM occurs and can't be resolved
    """
    # Get patch size if min_context_length not provided
    if min_context_length is None:
        min_context_length = model.model.patch_embed.stride

    # Ensure context_length is not smaller than the minimum
    context_length = max(context_length, min_context_length)

    # Use the TOTOModelPredictorWrapper
    predictor_wrapper = TOTOModelPredictorWrapper(
        model=model,
        prediction_length=prediction_length,
        context_length=context_length,
        mode=mode,
        num_samples=num_samples,
        use_kv_cache=use_kv_cache,
    )

    try:
        # Attempt prediction and evaluation
        res = evaluate_model(
            predictor_wrapper,
            test_data=test_data,
            metrics=metrics,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
        return res
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def predict(granularity,dataset_name,context_length,prediction_length,to_univariate=True):
    torch.set_float32_matmul_precision("high")
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

    model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = torch.compile(model)
    min_context_length = model.model.patch_embed.stride
    print(f"Model min context length (patch size): {min_context_length}")

    #TODO
    # Calculate optimal batch size based on available GPU memory, not used for prediction
    suggested_batch_size = calculate_optimal_batch_size(
        model=model,
        target_dim=1,
        prediction_length=hf_dataset.prediction_length,
        context_length=context_length,
        use_kv_cache=USE_KV_CACHE,
        num_samples=NUM_SAMPLES,
    )

    # Try prediction with the optimal parameters - pass loaded model directly
    res = try_prediction_with_config(
        model=model,
        prediction_length=hf_dataset.prediction_length,
        context_length=context_length,
        mode=Multivariate(batch_size=suggested_batch_size),
        num_samples=NUM_SAMPLES,
        test_data=hf_dataset.test_data,
        season_length=season_length,
        use_kv_cache=USE_KV_CACHE,
        min_context_length=min_context_length,
    )

    # Cleanup model and memory only after completing all tasks
    del model
    torch.cuda.empty_cache()
    gc.collect()

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Toto",
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
    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(
        WarningFilter("The mean prediction is not stored in the forecast data")
    )

    for config in eval_config:
        for pl in [6,8,10,12]:
            predict(granularity=Granularity.HOUR,dataset_name=config["dataset_name"], prediction_length=pl,context_length=36,to_univariate=config["to_univariate"])
