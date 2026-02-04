import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Sequence, List

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


class SharedLinearEnergy(nn.Module):
    def __init__(
        self,
        group_sizes: Sequence[int],
        dtype: torch.dtype = torch.float32,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = float(eps)

        gs = torch.as_tensor(group_sizes, dtype=torch.long)
        if gs.ndim != 1 or gs.numel() == 0:
            raise ValueError("group_sizes must be a 1D non-empty sequence of ints.")
        if (gs < 0).any():
            raise ValueError("group_sizes must be nonnegative.")

        # ✅ rename buffer to avoid collisions, and annotate
        self._group_sizes: torch.Tensor
        self.register_buffer("_group_sizes", gs)

        n_groups = int(gs.numel())
        self.raw_k = nn.Parameter(torch.zeros(n_groups, dtype=dtype))

        if weights is not None:
            w = torch.as_tensor(weights, dtype=dtype).view(-1)
            if w.numel() != n_groups:
                raise ValueError(f"weights must have numel() == {n_groups}, got {w.numel()}.")
            with torch.no_grad():
                w = torch.clamp(w, min=1e-8)
                self.raw_k.copy_(self.inv_softplus(w))

        self._group_sizes_list: List[int] = [int(x) for x in gs.tolist()]

    @staticmethod
    def inv_softplus(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    def stiffness(self) -> torch.Tensor:
        return F.softplus(self.raw_k) + self.eps

    def forward(self, strains: torch.Tensor) -> torch.Tensor:
        if strains.ndim < 1:
            raise ValueError("strains must have at least 1 dimension (last dim is features).")

        # ✅ type checker now knows this is a Tensor
        N_total_expected = int(self._group_sizes.sum().item())
        if strains.shape[-1] != N_total_expected:
            raise ValueError(
                f"strains.shape[-1] must be {N_total_expected}, got {strains.shape[-1]}."
            )

        k = self.stiffness()
        groups = torch.split(strains, self._group_sizes_list, dim=-1)

        E = strains.new_zeros(strains.shape[:-1])
        for g, kg in zip(groups, k):
            if g.numel() == 0:
                continue
            E = E + 0.5 * kg * (g ** 2).sum(dim=-1)
        return E
