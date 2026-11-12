from dataclasses import dataclass

import torch
import torch.nn as nn

from models.common.dinov2_encoder import DINOv2Tokenizer
from models.world_model.fusion_transformer import FusionTransformer
from models.world_model.proprio_encoder import ProprioEncoder


@dataclass
class LatentEncoderConfig:
    # vision
    dino_model: str = "dinov2_vits14"
    image_size: int = 84          # 84 -> 6x6 grid, 112 -> 8x8 grid
    # proprio
    proprio_dim: int = 9
    # fusion
    latent_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0


class LatentEncoder(nn.Module):
    """
    obs -> z

      agent image  --\
                      DINOv2 (frozen) -> patch tokens --\
      wrist image  --/                                    \
                                                           FusionTransformer -> CLS -> z
      proprio -> MLP -> 1 token ------------------------- /
                                  learned CLS token ------/

    Two entry points:
      forward(obs)                         full pipeline from raw images
      encode_from_tokens(tokens, proprio)  skip DINOv2 (use cached tokens)
    """

    def __init__(self, cfg: LatentEncoderConfig = LatentEncoderConfig(), tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer or DINOv2Tokenizer(cfg.dino_model, cfg.image_size)
        self.proprio_encoder = ProprioEncoder(cfg.proprio_dim, cfg.latent_dim)
        self.fusion = FusionTransformer(
            d_vis=self.tokenizer.embed_dim,
            d_model=cfg.latent_dim,
            num_cameras=2,
            num_patches=self.tokenizer.num_patches,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )

    @torch.no_grad()
    def tokenize(self, agent_image, wrist_image):
        """Images (B,3,H,W) x2 -> cached-able tokens (B, 2, N, D_vis)."""
        B = agent_image.shape[0]
        both = torch.cat([agent_image, wrist_image], dim=0)  # one DINOv2 pass
        tokens = self.tokenizer(both)                        # (2B, N, D)
        return torch.stack([tokens[:B], tokens[B:]], dim=1)  # (B, 2, N, D)

    def encode_from_tokens(self, vision_tokens, proprio):
        return self.fusion(vision_tokens, self.proprio_encoder(proprio))

    def forward(self, obs):
        vision_tokens = self.tokenize(obs["agent_image"], obs["wrist_image"])
        return self.encode_from_tokens(vision_tokens, obs["proprio"])

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.train(False)  # DINOv2 always frozen / eval
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]