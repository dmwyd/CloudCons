
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting_bench", "toto")))
from toto.model.toto import Toto
from toto.inference.gluonts_predictor import Multivariate, TotoPredictor



class TOTOModelPredictorWrapper:
    """Wrapper for TOTOPredictor that handles OOM errors by adjusting batch size."""

    def __init__(
        self,
        model_path,
        prediction_length,
        context_length,
        batch_size=256,
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
        self.mode =Multivariate(batch_size=batch_size)
        self.model=Toto.from_pretrained(model_path)
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.samples_per_batch = (
            num_samples  # Start with full batch size and adjust if needed
        )
        self._adjusted = False  # Tracks whether adjustment has been done

        self._initialize_predictor()

    def _initialize_predictor(self):
        """
        Initialize the TOTOPredictor with the current samples_per_batch.
        """
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.eval()
        self.model = torch.compile(self.model)
        min_context_length = self.model.model.patch_embed.stride
        self.context_length=max(self.context_length,min_context_length)
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






