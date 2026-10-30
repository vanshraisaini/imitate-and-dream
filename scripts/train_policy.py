import argparse
import copy
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from scripts.dataset import RoboMimicDataset  # your dataset.py
from models.imitation_policy.policy import FlowMatchingPolicy


def to_device(x, device):
    """Moves a (possibly nested) dict of tensors to the device."""
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x.to(device, non_blocking=True)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1.0 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


def save(path, policy, ema):
    torch.save({"policy": policy.state_dict(), "ema": ema.state_dict()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="./datasets/lift_image_v15.hdf5")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--workers", type=int, default=0)  # raise only if your dataset is worker-safe (no open h5py handle)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--out", type=str, default="policy.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RoboMimicDataset(args.dataset)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )

    # infer sizes from one sample so the model always matches the data
    s = ds[0]
    obs = s["observation"]
    policy = FlowMatchingPolicy(
        img_size=obs["agent_image"].shape[-1],
        proprio_dim=obs["proprio"].shape[-1],
        action_dim=s["action_chunk"].shape[-1],
        horizon=s["action_chunk"].shape[0],
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"{len(ds)} samples, {n_params / 1e6:.2f}M parameters, device={device}")

    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    opt = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total = args.epochs * len(loader)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for epoch in range(args.epochs):
        policy.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            batch = to_device(batch, device)
            loss = policy.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            ema_update(ema, policy, args.ema_decay)
            running += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss {running / len(loader):.4f}")
        if (epoch + 1) % args.save_every == 0:
            save(args.out, policy, ema)

    save(args.out, policy, ema)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()