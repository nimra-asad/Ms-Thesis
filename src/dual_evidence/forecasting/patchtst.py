"""Small CPU PatchTST forecaster for pooled household day-ahead windows."""
from __future__ import annotations

import numpy as np

from dual_evidence.forecasting.windows import WindowSet


class PatchTSTForecaster:
    def __init__(self, patch_length: int = 16, d_model: int = 64, epochs: int = 8,
                 batch_size: int = 128, learning_rate: float = 1e-3,
                 seed: int = 42, device: str = "cpu"):
        self.patch_length = patch_length
        self.d_model = d_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device
        self.network = None

    def _build(self):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PatchTST requires optional dependencies: pip install -e '.[patchtst]'") from exc
        class PatchTST(nn.Module):
            def __init__(self, patch_len, d_model):
                super().__init__()
                self.patch_len = patch_len
                patch_count = 336 // patch_len
                self.patch = nn.Linear(patch_len, d_model)
                layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                    dropout=.1, batch_first=True, norm_first=True, activation="gelu")
                self.encoder = nn.TransformerEncoder(layers, num_layers=2)
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(patch_count * d_model, 48)
                self.position = nn.Parameter(torch.zeros(1, patch_count, d_model))
            def forward(self, x):
                mean = x.mean(dim=1, keepdim=True)
                scale = x.std(dim=1, keepdim=True).clamp_min(1e-3)
                x = (x - mean) / scale
                x = x.reshape(x.shape[0], 336 // self.patch_len, self.patch_len)
                x = self.patch(x) + self.position
                x = self.norm(self.encoder(x)).flatten(start_dim=1)
                return self.head(x) * scale + mean
        return torch, PatchTST(self.patch_length, self.d_model).to(self.device)

    def fit(self, train: WindowSet, validation: WindowSet | None = None):
        import torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available() and self.device == "mps":
            torch.mps.manual_seed(self.seed)
        self.network = self._build()[1]
        x = torch.as_tensor(train.history, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(train.y, dtype=torch.float32, device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        best_loss, best_state = float("inf"), None
        for _ in range(self.epochs):
            self.network.train()
            order = torch.randperm(len(x), generator=generator)
            for idx in order.split(self.batch_size):
                idx = idx.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.l1_loss(self.network(x[idx]), y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
            if validation is not None and len(validation):
                self.network.eval()
                with torch.no_grad():
                    losses = []
                    for start in range(0, len(validation), self.batch_size):
                        vx = torch.as_tensor(validation.history[start:start+self.batch_size],
                                             dtype=torch.float32, device=self.device)
                        vy = torch.as_tensor(validation.y[start:start+self.batch_size],
                                             dtype=torch.float32, device=self.device)
                        losses.append(torch.nn.functional.l1_loss(self.network(vx), vy).item())
                    val_loss = float(np.mean(losses))
                if val_loss < best_loss:
                    best_loss, best_state = val_loss, {k: v.detach().cpu().clone()
                                                       for k, v in self.network.state_dict().items()}
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def predict(self, windows: WindowSet) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("fit must be called before predict")
        import torch
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                x = torch.as_tensor(windows.history[start:start+self.batch_size],
                                    dtype=torch.float32, device=self.device)
                outputs.append(self.network(x).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32) if outputs else np.empty((0, 48), np.float32)
