import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast

class SundialPredictor:
    def __init__(
        self,
        model_path: str,
        num_samples: int,
        prediction_length: int,
        device_map: str,
        context_length: int=512,
        batch_size: int = 512,
    ):
        self.device = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,device_map=self.device) 
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


    




    




