import torch
import torch.nn as nn

from models.common.augment import random_shift
from models.common.dinov2_encoder import DINOv2Tokenizer
from models.common.proprio_encoder import ProprioEncoder
from models.imitation_policy.action_transformer import ActionTransformer
from models.imitation_policy.vit_encoder import ViTEncoder

VISION_ENCODERS = ("dinov2", "vit")


def _to_float_image(img):
    """uint8 [0, 255] -> float [0, 1]. Float images are assumed to be in [0, 1] already."""
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    return img.float()


class FlowMatchingPolicy(nn.Module):
    """Flow matching policy: observation -> action chunk.

    Batch format (matches RoboMimicDataset):
        batch["observation"]["agent_image"]  (B, 3, 84, 84)
        batch["observation"]["wrist_image"]  (B, 3, 84, 84)
        batch["observation"]["proprio"]      (B, 9)
        batch["action_chunk"]                (B, 16, 7), already in [-1, 1]
        batch["action_mask"]                 (B, 16), True = real action, False = padding

    No normalization is applied: actions are used as they come from the dataset.

    Vision encoder (vision_encoder=...):
        "dinov2": one pretrained DINOv2 shared by both cameras (frozen unless freeze_dino=False),
                  followed by a per-camera linear projection to `dim`.
        "vit":    one small ViT per camera, trained from scratch (the original setup).

    Flow convention: t = 0 is noise, t = 1 is data.
        A_t = t * A_data + (1 - t) * noise,   target velocity = A_data - noise
    """

    def __init__(
        self,
        img_size=84,
        proprio_dim=9,
        action_dim=7,
        horizon=16,
        dim=256,
        depth=6,
        heads=8,
        shift_pad=4,
        vision_encoder="dinov2",
        # vision_encoder="dinov2"
        dino_model="dinov2_vits14",
        dino_image_size=84,
        freeze_dino=True,
        # vision_encoder="vit"
        patch_size=12,
        vit_depth=4,
        vit_heads=8,
    ):
        super().__init__()
        if vision_encoder not in VISION_ENCODERS:
            raise ValueError(f"unknown vision_encoder {vision_encoder!r}, expected one of {VISION_ENCODERS}")
        self.horizon, self.action_dim, self.shift_pad = horizon, action_dim, shift_pad
        self.vision_encoder = vision_encoder

        # observation encoders
        if vision_encoder == "dinov2":
            self.dino = DINOv2Tokenizer(dino_model, dino_image_size, freeze=freeze_dino)
            self.agent_proj = nn.Linear(self.dino.embed_dim, dim)
            self.wrist_proj = nn.Linear(self.dino.embed_dim, dim)
        else:
            self.agent_enc = ViTEncoder(img_size, patch_size, 3, dim, vit_depth, vit_heads)
            self.wrist_enc = ViTEncoder(img_size, patch_size, 3, dim, vit_depth, vit_heads)
        self.proprio_enc = ProprioEncoder(proprio_dim, dim, hidden_dim=dim)
        self.src_emb = nn.Parameter(torch.zeros(3, dim))  # agent / wrist / proprio
        nn.init.normal_(self.src_emb, std=0.02)
        self.mem_norm = nn.LayerNorm(dim)

        # flow matching velocity network
        self.net = ActionTransformer(action_dim, horizon, dim, depth, heads)

    @classmethod
    def from_checkpoint(cls, ckpt, map_location=None, weights="ema"):
        """Rebuild a policy from a checkpoint saved by train_policy.py (path or loaded dict).

        Checkpoints from before the DINOv2 option have no "vision_encoder" in their config;
        they were trained with the from-scratch ViT.
        """
        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt, map_location=map_location, weights_only=False)
        config = {"vision_encoder": "vit", **ckpt["config"]}
        model = cls(**config)
        model.load_state_dict(ckpt[weights])
        return model

    def _encode_images(self, agent, wrist):
        """(B, 3, H, W) x2 in [0, 1] -> (B, N, dim) x2."""
        if self.vision_encoder == "dinov2":
            B = agent.shape[0]
            tokens = self.dino(torch.cat([agent, wrist], dim=0))  # one DINOv2 pass for both cameras
            return self.agent_proj(tokens[:B]), self.wrist_proj(tokens[B:])
        return self.agent_enc(agent), self.wrist_enc(wrist)

    def encode_obs(self, obs, augment=False):
        """obs: dict with agent_image, wrist_image, proprio. Returns (B, 2N + 1, dim) tokens."""
        agent = _to_float_image(obs["agent_image"])
        wrist = _to_float_image(obs["wrist_image"])
        if augment and self.shift_pad > 0:
            agent = random_shift(agent, self.shift_pad)
            wrist = random_shift(wrist, self.shift_pad)
        agent_tokens, wrist_tokens = self._encode_images(agent, wrist)
        tokens = torch.cat(
            [
                agent_tokens + self.src_emb[0],
                wrist_tokens + self.src_emb[1],
                self.proprio_enc(obs["proprio"].float()).unsqueeze(1) + self.src_emb[2],
            ],
            dim=1,
        )
        return self.mem_norm(tokens)

    def loss(self, batch):
        mem = self.encode_obs(batch["observation"], augment=self.training)

        a1 = batch["action_chunk"].float()               # (B, H, A) clean data
        mask = batch["action_mask"].float()              # (B, H), 0 on padded steps
        noise = torch.randn_like(a1)
        t = torch.rand(a1.shape[0], device=a1.device)
        t_ = t[:, None, None]
        a_t = t_ * a1 + (1.0 - t_) * noise

        v = self.net(a_t, t, mem)
        se = (v - (a1 - noise)) ** 2                     # (B, H, A)
        denom = (mask.sum() * a1.shape[-1]).clamp(min=1.0)
        return (se * mask[..., None]).sum() / denom      # padded steps excluded

    @torch.no_grad()
    def predict(self, obs, num_steps=10):
        """obs: same dict as batch["observation"]. Returns an action chunk (B, horizon, action_dim)."""
        mem = self.encode_obs(obs, augment=False)
        B = mem.shape[0]
        a = torch.randn(B, self.horizon, self.action_dim, device=mem.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):  # Euler integration from t=0 (noise) to t=1 (data)
            t = torch.full((B,), i * dt, device=a.device)
            a = a + dt * self.net(a, t, mem)
        return a.clamp(-1.0, 1.0)  # dataset actions live in [-1, 1]