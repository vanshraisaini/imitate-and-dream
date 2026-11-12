import torch
import torch.nn.functional as F


def random_shift(x, pad=4):
    """Random-shift augmentation (pad + random crop back to size), per sample.

    x: (B, C, H, W) with H == W.
    """
    n, _, h, w = x.shape
    assert h == w, "random_shift expects square images"
    x = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    grid = base_grid + shift * (2.0 / (h + 2 * pad))
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)