import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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


# class LinearEnergyLayer(nn.Module):
#     def __init__(self, n_strain: int, dtype=torch.float32):
#         super().__init__()
#         # One neuron, no bias; its weights are the stiffnesses
#         self.layer = nn.Linear(n_strain, 1, bias=False, dtype=dtype)

#     def forward(self, strain: torch.Tensor) -> torch.Tensor:
#         """
#         strain: (..., n_strain)
#         returns scalar energy: (..., 1)
#         """
#         eps2 = strain**2
#         E = 0.5 * self.layer(eps2)  # weights of layer = k_i
#         return E



class LinearEnergyLayer(nn.Module):
    def __init__(
        self,
        n_strain: int,
        dtype: torch.dtype = torch.float32,
        weights: Optional[torch.Tensor] = None,  # physical stiffness guess
        eps: float = 1e-8,  # numerical safety
    ):
        super().__init__()

        self.n_strain = n_strain
        self.eps = eps

        # Raw (unconstrained) stiffness parameters
        self.raw_k = nn.Parameter(torch.zeros(n_strain, dtype=dtype))

        if weights is not None:
            assert weights.shape == (n_strain,) or weights.shape == (1, n_strain)

            # flatten to (n_strain,)
            weights = weights.view(-1).to(dtype=dtype)

            # inverse softplus so that softplus(raw_k) ≈ weights
            with torch.no_grad():
                w = weights.view(-1).to(dtype=dtype)
                w = torch.clamp(w, min=1e-8)  # important
                self.raw_k.copy_(self.inv_softplus(w))


    @staticmethod
    def inv_softplus(y, eps=1e-12):
        # require y > 0
        y = torch.clamp(y, min=eps)
        # stable inverse: y + log(1 - exp(-y))
        return y + torch.log(-torch.expm1(-y))

    def stiffness(self):
        """Physical (positive) stiffness"""
        return F.softplus(self.raw_k) + self.eps

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """
        strain: (..., n_strain)
        returns energy: (..., 1)
        """
        eps2 = strain ** 2                      # (..., n_strain)
        k = self.stiffness()                    # (n_strain,)
        E = 0.5 * torch.sum(k * eps2, dim=-1, keepdim=True)
        return E



