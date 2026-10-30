import torch
import torch.nn as nn

from models.imitation_policy.action_transformer import ActionTransformer
from models.imitation_policy.augment import random_shift
from models.imitation_policy.proprio_encoder import ProprioEncoder
from models.imitation_policy.vit_encoder import ViTEncoder


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

    Flow convention: t = 0 is noise, t = 1 is data.
        A_t = t * A_data + (1 - t) * noise,   target velocity = A_data - noise
    """

    def __init__(
        self,
        img_size=84,
        patch_size=12,
        proprio_dim=9,
        action_dim=7,
        horizon=16,
        dim=256,
        vit_depth=4,
        vit_heads=8,
        depth=6,
        heads=8,
        shift_pad=4,
    ):
        super().__init__()
        self.horizon, self.action_dim, self.shift_pad = horizon, action_dim, shift_pad

        # observation encoders (one ViT per camera, trained from scratch)
        self.agent_enc = ViTEncoder(img_size, patch_size, 3, dim, vit_depth, vit_heads)
        self.wrist_enc = ViTEncoder(img_size, patch_size, 3, dim, vit_depth, vit_heads)
        self.proprio_enc = ProprioEncoder(proprio_dim, dim)
        self.src_emb = nn.Parameter(torch.zeros(3, dim))  # agent / wrist / proprio
        nn.init.normal_(self.src_emb, std=0.02)
        self.mem_norm = nn.LayerNorm(dim)

        # flow matching velocity network
        self.net = ActionTransformer(action_dim, horizon, dim, depth, heads)

    def encode_obs(self, obs, augment=False):
        """obs: dict with agent_image, wrist_image, proprio. Returns (B, 2N + 1, dim) tokens."""
        agent = _to_float_image(obs["agent_image"])
        wrist = _to_float_image(obs["wrist_image"])
        if augment and self.shift_pad > 0:
            agent = random_shift(agent, self.shift_pad)
            wrist = random_shift(wrist, self.shift_pad)
        tokens = torch.cat(
            [
                self.agent_enc(agent) + self.src_emb[0],
                self.wrist_enc(wrist) + self.src_emb[1],
                self.proprio_enc(obs["proprio"].float()) + self.src_emb[2],
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