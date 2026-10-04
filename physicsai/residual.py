"""Fixed PCA-128 residual head; no search, backbone fine-tuning or joint fitting."""
import numpy as np
import torch
import torch.nn as nn

class VisualResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(0.20), nn.Linear(hidden_dim, 32), nn.SiLU(), nn.Linear(32, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)

def train_residual(x: np.ndarray, residual: np.ndarray, weight_decay: float,
                   device: torch.device, seed: int) -> VisualResidualMLP:
    torch.manual_seed(seed)
    model = VisualResidualMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160, eta_min=1e-5)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(residual, dtype=torch.float32, device=device).unsqueeze(1)
    for _ in range(160):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.eval()

def residual_predict(model: VisualResidualMLP, x: np.ndarray,
                     device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        return model(tensor).cpu().numpy().ravel()
