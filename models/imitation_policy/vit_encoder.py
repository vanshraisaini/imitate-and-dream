import torch
import torch.nn as nn


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTEncoder(nn.Module):
    """Small ViT trained from scratch.

    Input : image (B, 3, H, W), float in [0, 1]
    Output: patch tokens (B, N, dim) with N = (H / patch_size) * (W / patch_size)
            For 84x84 with patch_size=12 -> 7x7 = 49 tokens.
    No CLS token: all spatial tokens are kept for the policy to attend to.
    """

    def __init__(self, img_size=84, patch_size=12, in_chans=3, dim=256, depth=4, heads=8):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.num_tokens = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        x = (img - 0.5) / 0.5                                   # [0,1] -> [-1,1]
        x = self.patch_embed(x)                                 # (B, dim, H/patch_size, W/patch_size)
        x = x.flatten(2)                                        # (B, dim, N) N = (H/patch_size)*(W/patch_size)
        x = x.transpose(1, 2)                                   # (B, N, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


if __name__ == "__main__":
    # Test the ViTEncoder with a random image
    img = torch.rand(2, 3, 84, 84)  # Batch of 2 images
    encoder = ViTEncoder()
    tokens = encoder(img)
    print("Output shape:", tokens.shape)  # Should be (2, 49, 256)