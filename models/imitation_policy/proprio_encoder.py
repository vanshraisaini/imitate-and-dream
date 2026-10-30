import torch.nn as nn


class ProprioEncoder(nn.Module):
    """MLP that maps the proprioception vector to a single token.

    Input : (B, proprio_dim)  (already normalized)
    Output: (B, 1, dim)
    """

    def __init__(self, proprio_dim=9, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, proprio):
        return self.mlp(proprio).unsqueeze(1)