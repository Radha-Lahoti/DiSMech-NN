import torch
import torch.nn as nn

class EnergyNN(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=dtype),  # scalar energy
        )
        # (Optional) init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, q_free: torch.Tensor) -> torch.Tensor:
        """
        q_free shape: (..., n_free)
        returns energy with shape (..., 1)
        """
        return self.net(q_free)
