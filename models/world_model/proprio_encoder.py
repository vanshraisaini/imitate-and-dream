import torch.nn as nn


class ProprioEncoder(nn.Module):
    """
    MLP mapping proprioception (B, proprio_dim) to a single token (B, d_model).
    Expects already-normalized proprio (the dataset normalizes it).
    """

    def __init__(self, proprio_dim=9, d_model=256, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, proprio):
        return self.mlp(proprio)