import torch
import torch.nn as nn

class StiffnessNN(nn.Module):
    def __init__(self, n_strain: int, dtype=torch.float64):
        super().__init__()

        # Linear layer: k = W * (strain^2) + b
        self.layer = nn.Linear(n_strain, n_strain, bias=True, dtype=dtype)

        # Initialize as constant stiffness: weights = 0, biases = 1.0
        nn.init.zeros_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 1.0)

    def forward(self, strain: torch.Tensor, already_squared=False):
        """
        strain: (..., n_strain)
        Returns:
            K: (..., n_strain, n_strain) diagonal stiffness matrices
        """
        x = strain if already_squared else strain**2

        # Predict stiffness vector k_i
        k_vec = self.layer(x)  # shape (..., n_strain)

        # Convert to diagonal matrix
        # torch.diag_embed handles batches automatically
        K = torch.diag_embed(k_vec)  # shape (..., n_strain, n_strain)

        return K
