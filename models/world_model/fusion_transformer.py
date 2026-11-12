import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    """
    Fuses vision patch tokens (from several cameras), a proprio token and a
    learned CLS token with self-attention. Returns the updated CLS token as
    the latent z.

    vision_tokens: (B, n_cams, N, d_vis)
    proprio_token: (B, d_model)
    returns z    : (B, d_model)
    """

    def __init__(
        self,
        d_vis=384,
        d_model=256,
        num_cameras=2,
        num_patches=36,
        num_layers=3,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()
        self.vis_proj = nn.Linear(d_vis, d_model)

        # Learned embeddings: which camera, and where in the patch grid.
        self.camera_embed = nn.Parameter(torch.zeros(num_cameras, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, d_model))
        self.proprio_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        for p in (self.camera_embed, self.pos_embed, self.proprio_type_embed, self.cls_token):
            nn.init.trunc_normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

        # LayerNorm keeps z bounded, which matters for multi-step latent rollouts.
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, vision_tokens, proprio_token):
        B, C, N, _ = vision_tokens.shape

        v = self.vis_proj(vision_tokens)                       # (B, C, N, d)
        v = v + self.camera_embed.unsqueeze(0) + self.pos_embed  # broadcast
        v = v.reshape(B, C * N, -1)                            # (B, C*N, d)

        p = (proprio_token.unsqueeze(1) + self.proprio_type_embed)  # (B, 1, d)
        cls = self.cls_token.expand(B, -1, -1)                      # (B, 1, d)

        tokens = torch.cat([cls, p, v], dim=1)                 # CLS first
        tokens = self.transformer(tokens)
        return self.out_norm(tokens[:, 0])                     # (B, d)