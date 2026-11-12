"""
Auxiliary state head: z -> 21-D privileged state (nut pose, nut pose in eef frame, eef pose).

Target layout (21-D), built by build_aux_targets:
    0:3    nut pos (world)
    3:7    nut quat (xyzw)                 <- quaternion
    7:10   nut pos in eef frame
    10:14  nut quat in eef frame (xyzw)    <- quaternion
    14:17  eef pos
    17:21  eef quat (xyzw)                 <- quaternion

Quaternions are compared with a sign-invariant loss (q and -q are the same rotation).
eef pose is also in the proprio input, so its errors only show the head can copy it.
"""
import numpy as np
import torch
import torch.nn as nn

TARGET_KEYS = ("object", "robot0_eef_pos", "robot0_eef_quat")

# name -> (start, end, is_quaternion)
GROUPS = {
    "nut_pos":      (0, 3, False),
    "nut_quat":     (3, 7, True),
    "nut_rel_pos":  (7, 10, False),
    "nut_rel_quat": (10, 14, True),
    "eef_pos":      (14, 17, False),
    "eef_quat":     (17, 21, True),
}
TARGET_DIM = 21


def _quat_to_mat(q):
    """(N, 4) xyzw -> (N, 3, 3)."""
    x, y, z, w = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def _quat_mul(a, b):
    """Hamilton product of (N, 4) xyzw quaternions."""
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def build_aux_targets(obj, eef_pos, eef_quat):
    """
    Raw HDF5 arrays for one episode -> targets (T, 21) float32.

    obj is robosuite's 14-D Square "object-state" (saved as "object"):
        0:7   nut pose in eef frame, but computed from the PREVIOUS step's nut pose
              (robosuite 1.5 sensor-cache order) and garbage at t=0 -> not used
        7:14  nut pose in world frame, current step
    The relative pose is recomputed here from the current nut and eef poses instead.
    """
    obj, eef_pos, eef_quat = (np.asarray(a, dtype=np.float64) for a in (obj, eef_pos, eef_quat))
    if obj.shape[-1] != 14:
        raise ValueError(f"expected 14-D Square object state, got {obj.shape}")
    nut_pos, nut_quat = obj[:, 7:10], obj[:, 10:14]
    rel_pos = np.einsum("nji,nj->ni", _quat_to_mat(eef_quat), nut_pos - eef_pos)  # R_eef^T (p_nut - p_eef)
    conj = eef_quat * np.array([-1.0, -1.0, -1.0, 1.0])
    rel_quat = _quat_mul(conj, nut_quat)
    return np.concatenate([nut_pos, nut_quat, rel_pos, rel_quat, eef_pos, eef_quat], axis=-1).astype(np.float32)


class AuxStateHead(nn.Module):
    """
    MLP head on the latent z. Optional per-dim target z-scoring (position dims only;
    quaternion dims are never normalized). Call fit_stats(targets) once if you want it,
    otherwise targets are used as-is.
    """

    def __init__(self, latent_dim=256, hidden_dim=256, detach_input=False):
        super().__init__()
        self.detach_input = detach_input  # True -> pure linear-probe-style test, no grad to encoder
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, TARGET_DIM),
        )
        self.register_buffer("mean", torch.zeros(TARGET_DIM))
        self.register_buffer("std", torch.ones(TARGET_DIM))

    @torch.no_grad()
    def fit_stats(self, targets, eps=1e-6):
        """targets: (N, 21) over the training set."""
        mean, std = targets.mean(0), targets.std(0).clamp_min(eps)
        for s, e, is_quat in GROUPS.values():
            if is_quat:
                mean[s:e], std[s:e] = 0.0, 1.0
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, z):
        """Returns predictions in normalized target space (B, 21)."""
        if self.detach_input:
            z = z.detach()
        return self.net(z)

    def normalize(self, targets):
        return (targets - self.mean) / self.std

    def denormalize(self, pred):
        return pred * self.std + self.mean


def _quat_loss(pred, target):
    # sign-invariant squared error, per sample: min(|p - t|^2, |p + t|^2)
    d_pos = ((pred - target) ** 2).sum(-1)
    d_neg = ((pred + target) ** 2).sum(-1)
    return torch.minimum(d_pos, d_neg)


def aux_loss(head, z, targets, weights=None):
    """
    Returns (total_loss, per_group dict of detached mean losses).
    Per-group loss is mean over dims for positions, summed-then-averaged over the 4
    components for quaternions. `weights` optionally maps group name -> float.
    """
    pred = head(z)
    tgt = head.normalize(targets)
    total, logs = 0.0, {}
    for name, (s, e, is_quat) in GROUPS.items():
        if is_quat:
            l = _quat_loss(pred[:, s:e], tgt[:, s:e]).mean() / (e - s)
        else:
            l = ((pred[:, s:e] - tgt[:, s:e]) ** 2).mean()
        total = total + (weights.get(name, 1.0) if weights else 1.0) * l
        logs[name] = l.detach()
    return total, logs


@torch.no_grad()
def eval_metrics(head, z, targets):
    """Per-group error in ORIGINAL units: pos RMSE in meters, quat angle error in degrees,
    plus R^2 for position groups."""
    pred = head.denormalize(head(z))
    out = {}
    for name, (s, e, is_quat) in GROUPS.items():
        p, t = pred[:, s:e], targets[:, s:e]
        if is_quat:
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            dot = (p * t).sum(-1).abs().clamp(max=1.0)
            out[f"{name}_deg"] = (2 * torch.acos(dot)).rad2deg().mean().item()
        else:
            out[f"{name}_rmse_m"] = ((p - t) ** 2).mean().sqrt().item()
            ss_res = ((p - t) ** 2).sum(0)
            ss_tot = ((t - t.mean(0)) ** 2).sum(0).clamp_min(1e-12)
            out[f"{name}_r2"] = (1 - ss_res / ss_tot).mean().item()
    return out