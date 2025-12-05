import torch
import torch.nn as nn

class EnergyNN(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, dtype=torch.float32):
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
    


class LinearElasticEnergyNN(nn.Module):
    def __init__(self, n_strain: int, dtype=torch.float32):
        super().__init__()
        # k_i: stiffness for each strain component
        self.k = nn.Parameter(torch.ones(n_strain, dtype=dtype))

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """
        strain: (..., n_strain)
        returns scalar energy: (..., 1)
        """
        eps2 = strain**2  # assume input is strain, not squared yet
        # elementwise k_i * eps_i^2, then sum and multiply by 1/2
        E = 0.5 * (self.k * eps2).sum(dim=-1, keepdim=True)
        return E


class LinearEnergyLayer(nn.Module):
    def __init__(self, n_strain: int, dtype=torch.float32):
        super().__init__()
        # One neuron, no bias; its weights are the stiffnesses
        self.layer = nn.Linear(n_strain, 1, bias=False, dtype=dtype)

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """
        strain: (..., n_strain)
        returns scalar energy: (..., 1)
        """
        eps2 = strain**2
        E = 0.5 * self.layer(eps2)  # weights of layer = k_i
        return E

