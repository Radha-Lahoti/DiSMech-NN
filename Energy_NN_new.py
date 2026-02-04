import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

class Shared3LinearEnergy(nn.Module):
    """
    Learns 3 positive stiffnesses shared across all springs:
      k_eps, k_kappa1, k_kappa2
    """
    def __init__(self, dtype: torch.dtype = torch.float32, weights: Optional[torch.Tensor] = None, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.raw_k = nn.Parameter(torch.zeros(3, dtype=dtype))  # ONLY 3 params

        if weights is not None:
            assert weights.numel() == 3
            w = weights.view(-1).to(dtype=dtype)
            with torch.no_grad():
                w = torch.clamp(w, min=1e-8)
                self.raw_k.copy_(self.inv_softplus(w))

    @staticmethod
    def inv_softplus(y, eps=1e-12):
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    def stiffness(self):
        return F.softplus(self.raw_k) + self.eps  # (3,)

    def forward(self, eps: torch.Tensor, kappa1: torch.Tensor, kappa2: torch.Tensor) -> torch.Tensor:
        """
        eps:    (..., S)
        kappa1: (..., Sb)
        kappa2: (..., Sb)

        returns:
            E_total: (...)  (summed over all springs)
        """
        k = self.stiffness()
        k_eps, k_k1, k_k2 = k[0], k[1], k[2]

        E_stretch = 0.5 * k_eps * (eps ** 2).sum(dim=-1)              # (...)
        E_bend    = 0.5 * k_k1 * (kappa1 ** 2).sum(dim=-1) + \
                    0.5 * k_k2 * (kappa2 ** 2).sum(dim=-1)            # (...)

        return E_stretch + E_bend
