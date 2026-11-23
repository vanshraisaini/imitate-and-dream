"""
Progress head: z -> g(z) in [0, 1], an estimate of how far along the task a state is.

Labels (see progress_labels):
    demonstration with T frames:          rho_t = t / (T - 1)               (0 at the first frame, 1 at the last)
    successful policy rollout:            rho_t = min(t / t_success, 1)     (1 from the success step on)
    failed policy rollout:                rho_t = 0
    random-action episode:                rho_t = 0

Early frames of successful and failed rollouts look alike, so with a squared loss the head
regresses towards P(success | state) * progress: a value-like score, which is what ranking needs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPISODE_KINDS = ("demo", "random", "policy")


def progress_labels(length, kind, success_step=None):
    """
    Per-step labels (length,) float32.

    kind="demo":    t / (length - 1)
    kind="random":  0
    kind="policy":  min(t / success_step, 1) if the rollout succeeded (success_step >= 0), else 0.
                    success_step is the index of the frame whose action completed the task.
    """
    if kind not in EPISODE_KINDS:
        raise ValueError(f"unknown episode kind {kind!r}, expected one of {EPISODE_KINDS}")
    t = np.arange(length, dtype=np.float64)
    if kind == "demo":
        return (t / max(length - 1, 1)).astype(np.float32)
    if kind == "policy" and success_step is not None and success_step >= 0:
        return np.minimum(t / max(success_step, 1), 1.0).astype(np.float32)
    return np.zeros(length, dtype=np.float32)


class ProgressHead(nn.Module):
    """MLP on the latent z. forward returns logits; predict returns g(z) in [0, 1]."""

    def __init__(self, latent_dim=256, hidden_dim=256, detach_input=False):
        super().__init__()
        self.detach_input = detach_input  # True -> head trains, encoder gets no gradient from it
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        """z: (..., d) -> logits (...)."""
        if self.detach_input:
            z = z.detach()
        return self.net(z).squeeze(-1)

    def predict(self, z):
        return torch.sigmoid(self(z))


def progress_loss(head, z, targets, mask=None, kind="mse"):
    """
    z: (..., d), targets: (...) in [0, 1], mask: optional (...) bool, True = use this frame.

    kind="mse": squared error on sigmoid(logits), as in report Eq. 4.
    kind="bce": BCE with soft targets. Same optimum, but it does not saturate when a
                prediction is confidently wrong, so it usually trains faster.
    Returns (loss, logs) with logs holding detached scalars.
    """
    logits = head(z)
    targets = targets.to(logits.dtype)
    if kind == "mse":
        per = (torch.sigmoid(logits) - targets) ** 2
    elif kind == "bce":
        per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    else:
        raise ValueError(f"unknown progress loss {kind!r}")

    if mask is None:
        loss = per.mean()
    else:
        mask = mask.to(per.dtype)
        loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {"progress": loss.detach()}


def _rank(x):
    # Average ranks are not needed here: labels from failed episodes tie at 0, and
    # ordinal ranks only bias Spearman slightly. Kept dependency-free on purpose.
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return r


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)).item()


@torch.no_grad()
def eval_metrics(head, z, targets, success=None):
    """
    z: (N, d), targets: (N,), success: optional (N,) bool, the episode outcome per frame.

    rmse / mae / pearson / spearman over all frames. If success is given and both outcomes
    are present, also the mean prediction on frames from successful vs failed episodes, and
    the AUROC of g for telling them apart (the separation ranking actually relies on).
    """
    pred = head.predict(z).float().flatten()
    t = targets.float().flatten()
    out = {
        "rmse": ((pred - t) ** 2).mean().sqrt().item(),
        "mae": (pred - t).abs().mean().item(),
        "pearson": _pearson(pred, t),
        "spearman": _pearson(_rank(pred), _rank(t)),
    }
    if success is not None:
        success = success.bool().flatten()
        n_pos, n_neg = int(success.sum()), int((~success).sum())
        if n_pos and n_neg:
            out["pred_mean_success"] = pred[success].mean().item()
            out["pred_mean_failure"] = pred[~success].mean().item()
            # Mann-Whitney U: P(g on a success frame > g on a failure frame)
            r = _rank(pred)
            out["auroc"] = ((r[success].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).item()
    return out
