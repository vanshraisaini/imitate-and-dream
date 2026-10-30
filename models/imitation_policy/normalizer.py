import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Per-dimension min/max normalization to [-1, 1].

    Stats are buffers, so they are saved/loaded with the policy state_dict.
    """

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("lo", -torch.ones(dim))
        self.register_buffer("hi", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        self.lo.copy_(x.min(0).values)
        self.hi.copy_(x.max(0).values)

    def _range(self):
        rng = self.hi - self.lo
        return torch.where(rng < 1e-6, torch.ones_like(rng), rng)  # constant dims

    def normalize(self, x):
        return 2.0 * (x - self.lo) / self._range() - 1.0

    def unnormalize(self, x):
        return (x + 1.0) / 2.0 * self._range() + self.lo