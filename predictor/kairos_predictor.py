import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast
from gluonts.model import evaluate_model, Forecast

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting_bench", "Kairos")))

from tsfm.model.kairos import AutoModel
from typing import List


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




    




