"""Zero-shot Amazon Chronos-2 adapter using the official Chronos pipeline."""
from __future__ import annotations

import numpy as np

from dual_evidence.forecasting.windows import WindowSet


class Chronos2Forecaster:
    def __init__(self, model_id: str = "amazon/chronos-2", device_map: str = "cpu",
                 context_length: int = 336, batch_size: int = 32):
        self.model_id = model_id
        self.device_map = device_map
        self.context_length = context_length
        self.batch_size = batch_size
        self.pipeline = None

    def load(self):
        try:
            from chronos import Chronos2Pipeline
            import torch
        except ImportError as exc:
            raise RuntimeError("Chronos-2 requires `pip install -e '.[chronos]'`") from exc
        # Intel macOS ships two OpenMP runtimes through NumPy and PyTorch. A
        # single Torch intra-op thread avoids a native barrier stall in inference.
        torch.set_num_threads(1)
        self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device_map)
        return self

    def predict(self, windows: WindowSet) -> np.ndarray:
        if self.pipeline is None:
            self.load()
        if not len(windows):
            return np.empty((0, 48), dtype=np.float32)
        import torch
        outputs = []
        for start in range(0, len(windows), self.batch_size):
            batch = windows.history[start:start+self.batch_size]
            contexts = [torch.as_tensor(row[-self.context_length:], dtype=torch.float32) for row in batch]
            forecasts = self.pipeline.predict(contexts, prediction_length=48,
                                               batch_size=self.batch_size)
            quantiles = np.asarray(getattr(self.pipeline, "quantiles", [0.1, 0.5, 0.9]))
            median_index = int(np.argmin(np.abs(quantiles - 0.5)))
            for forecast in forecasts:
                # Official Chronos-2 API returns one [variates, quantiles, horizon]
                # tensor per input series; ours are univariate.
                outputs.append(forecast[0, median_index, :].detach().cpu().numpy()[None, :])
        return np.concatenate(outputs).astype(np.float32)
