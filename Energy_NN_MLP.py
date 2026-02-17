import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SharedDense3x3QuadraticEnergy(nn.Module):
    """
    Shared PSD 3x3 quadratic energy per (bending) element:
      x_s = [e_s, k1_s, k2_s]
      E_total = sum_s 0.5 * x_s^T Theta x_s
    Theta = L L^T + eps I, with L lower-triangular and diag(L)>0.

    Optional init_theta_diag initializes diag(Theta) approximately to [EA, EI1, EI2].
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        init_theta_diag: Optional[torch.Tensor] = None,  # (3,)
        init_offdiag_zero: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype

        # raw_tril = [L00, L10, L11, L20, L21, L22]
        self.raw_tril = nn.Parameter(torch.zeros(6, dtype=dtype))

        if init_theta_diag is not None:
            self.initialize_from_theta_diag(init_theta_diag, zero_offdiag=init_offdiag_zero)

    @staticmethod
    def inv_softplus(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    def initialize_from_theta_diag(self, theta_diag: torch.Tensor, zero_offdiag: bool = True):
        with torch.no_grad():
            td = theta_diag.to(device=self.raw_tril.device, dtype=self.raw_tril.dtype).view(3)
            td = torch.clamp(td, min=1e-12)

            Ldiag_desired = torch.sqrt(td)                            # want L_ii ~ sqrt(theta_ii)
            raw_diag = self.inv_softplus(torch.clamp(Ldiag_desired - self.eps, min=1e-12))

            if zero_offdiag:
                self.raw_tril.zero_()

            self.raw_tril[0] = raw_diag[0]  # L00
            self.raw_tril[2] = raw_diag[1]  # L11
            self.raw_tril[5] = raw_diag[2]  # L22

    def _L(self) -> torch.Tensor:
        """
        Build lower-triangular L with positive diagonal, WITHOUT in-place ops on views.
        """
        r = self.raw_tril
        L00 = F.softplus(r[0]) + self.eps
        L10 = r[1]
        L11 = F.softplus(r[2]) + self.eps
        L20 = r[3]
        L21 = r[4]
        L22 = F.softplus(r[5]) + self.eps

        L = torch.stack([
            torch.stack([L00, torch.zeros_like(L00), torch.zeros_like(L00)]),
            torch.stack([L10, L11, torch.zeros_like(L00)]),
            torch.stack([L20, L21, L22]),
        ])
        return L  # (3,3)

    def theta_matrix(self) -> torch.Tensor:
        L = self._L()
        return L @ L.T + self.eps * torch.eye(3, device=L.device, dtype=L.dtype)

    def _align_eps_to_kappa(self, eps: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        Map eps (..., Se) to eps_b (..., Sb) where Sb = kappa.shape[-1]
        """
        Se = eps.shape[-1]
        Sb = kappa.shape[-1]
        if Se == Sb:
            return eps
        if Se == Sb + 1:
            return 0.5 * (eps[..., :-1] + eps[..., 1:])
        if Se == Sb + 2:
            return eps[..., 1:-1]
        raise ValueError(f"Cannot align eps length Se={Se} with kappa length Sb={Sb}.")

    def forward(self, eps: torch.Tensor, kappa1: torch.Tensor, kappa2: torch.Tensor) -> torch.Tensor:
        # align sizes (since eps and kappa often live on different elements)
        eps_b = self._align_eps_to_kappa(eps, kappa1)

        x = torch.stack([eps_b, kappa1, kappa2], dim=-1)  # (..., Sb, 3)
        L = self._L()                                     # (3,3)
        y = x @ L                                         # (..., Sb, 3)
        E_per = 0.5 * (y * y).sum(dim=-1)                  # (..., Sb)
        return E_per.sum(dim=-1)                           # (...)
