import argparse
import copy
import glob
import inspect
import math
import os

import torch
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.datasets.robomimic_dataset import RoboMimicDataset  # your dataset.py
from models.imitation_policy.policy import FlowMatchingPolicy


# ------------------------------------------------------------------ helpers
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


@torch.no_grad()
def evaluate(model, loader, device, action_dim):
    """Offline evaluation on held-out demos (run on the EMA model).

    val/flow_loss  : masked flow matching loss (same objective as training, no augmentation)
    val/action_mse : MSE between one sampled action chunk (10 Euler steps) and the
                     ground-truth chunk, masked, averaged over all dims (+ per dim)
    """
    model.eval()
    flow_sum, n = 0.0, 0
    se_dim, count = torch.zeros(action_dim), 0.0
    for batch in loader:
        batch = to_device(batch, device)
        bs = batch["action_chunk"].shape[0]
        flow_sum += model.loss(batch).item() * bs
        n += bs

        pred = model.predict(batch["observation"])
        mask = batch["action_mask"].float()[..., None]  # (B, H, 1)
        se = ((pred - batch["action_chunk"].float()) ** 2) * mask
        se_dim += se.sum(dim=(0, 1)).cpu()
        count += mask.sum().item()

    mse_dim = se_dim / max(count, 1.0)
    metrics = {
        "val/flow_loss": flow_sum / max(n, 1),
        "val/action_mse": mse_dim.mean().item(),
    }
    metrics.update({f"val/action_mse_dim{i}": v.item() for i, v in enumerate(mse_dim)})
    return metrics


# ------------------------------------------------------------------ setup
def parse_args():
    ap = argparse.ArgumentParser()
    # data / optimization
    ap.add_argument("--dataset", type=str, default="./datasets/lift_image_v15.hdf5")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--ema_decay", type=float, default=0.995)
    ap.add_argument("--workers", type=int, default=0, help="raise only if your dataset is worker-safe (no open h5py handle)")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--out", type=str, default="policy/policy.pt")
    # evaluation
    ap.add_argument("--val_frac", type=float, default=0.05, help="fraction of samples held out for eval (0 disables)")
    ap.add_argument("--eval_every", type=int, default=5, help="evaluate every N epochs")
    # wandb
    ap.add_argument("--wandb_project", type=str, default="flow-matching-policy")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--log_every", type=int, default=10, help="log training metrics every N steps")
    return ap.parse_args()


def build_dataloaders(args, device):
    """Returns (dataset, train_loader, val_loader or None).

    Held-out split = the LAST val_frac of samples. This is a split by demo only if the
    dataset is ordered by demo.
    """
    ds = RoboMimicDataset(args.dataset)
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds = Subset(ds, range(n_train)) if n_val else ds
    val_ds = Subset(ds, range(n_train, len(ds))) if n_val else None

    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs) if val_ds is not None else None
    return ds, train_loader, val_loader


def model_config(sample):
    """Infer model sizes from one dataset sample so the model always matches the data."""
    return dict(
        img_size=sample["observation"]["agent_image"].shape[-1],
        proprio_dim=sample["observation"]["proprio"].shape[-1],
        action_dim=sample["action_chunk"].shape[-1],
        horizon=sample["action_chunk"].shape[0],
    )


def make_scheduler(opt, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to 0."""

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def init_wandb(args, config):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={**vars(args), **config},
    )
    # upload the exact code: policy.py + the model files next to it + this script
    code = wandb.Artifact("policy-code", type="code")
    policy_dir = os.path.dirname(inspect.getsourcefile(FlowMatchingPolicy))
    for path in sorted(glob.glob(os.path.join(policy_dir, "*.py"))):
        code.add_file(path, name=os.path.basename(path))
    code.add_file(os.path.abspath(__file__), name="train.py")
    run.log_artifact(code)
    return run


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, args, policy, cfg, train_loader, val_loader, device, run):
        self.args, self.cfg, self.device, self.run = args, cfg, device, run
        self.policy = policy
        self.train_loader, self.val_loader = train_loader, val_loader

        self.ema = copy.deepcopy(policy).eval().requires_grad_(False)
        self.opt = torch.optim.AdamW(
            policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
        self.sched = make_scheduler(self.opt, args.warmup_steps, args.epochs * len(train_loader))

        root, ext = os.path.splitext(args.out)
        self.last_path, self.best_path = args.out, f"{root}_best{ext}"
        self.step = 0
        self.best_mse, self.best_epoch = float("inf"), None

    # ---- one training epoch
    def train_epoch(self, epoch):
        args = self.args
        self.policy.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = to_device(batch, self.device)
            loss = self.policy.loss(batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            self.sched.step()
            ema_update(self.ema, self.policy, args.ema_decay)

            self.step += 1
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if self.step % args.log_every == 0:
                self.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": self.sched.get_last_lr()[0],
                    "epoch": epoch,
                })
        return total_loss / len(self.train_loader)

    # ---- evaluation on held-out demos with the EMA weights
    def eval_epoch(self, epoch):
        metrics = evaluate(self.ema, self.val_loader, self.device, self.cfg["action_dim"])
        print(f"  val flow_loss {metrics['val/flow_loss']:.4f}  action_mse {metrics['val/action_mse']:.4f}")
        self.log({**metrics, "epoch": epoch})
        if metrics["val/action_mse"] < self.best_mse:
            self.best_mse, self.best_epoch = metrics["val/action_mse"], epoch
            self.save(self.best_path, epoch)
            self.run.summary["best/val_action_mse"] = self.best_mse
            self.run.summary["best/epoch"] = self.best_epoch

    # ---- main loop
    def fit(self):
        args = self.args
        for epoch in range(1, args.epochs + 1):
            epoch_loss = self.train_epoch(epoch)
            print(f"epoch {epoch}/{args.epochs}  loss {epoch_loss:.4f}")
            self.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

            if self.val_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
                self.eval_epoch(epoch)
            if epoch % args.save_every == 0:
                self.save(self.last_path, epoch)

        self.save(self.last_path, args.epochs)
        print(f"saved to {self.last_path}")
        self.upload_model()

    # ---- bookkeeping
    def log(self, metrics):
        wandb.log(metrics, step=self.step)

    def save(self, path, epoch):
        """Config is stored so a rollout script can rebuild the policy."""
        torch.save(
            {"policy": self.policy.state_dict(), "ema": self.ema.state_dict(),
             "config": self.cfg, "epoch": epoch},
            path,
        )

    def upload_model(self):
        has_best = self.best_epoch is not None
        artifact = wandb.Artifact(
            "policy-model",
            type="model",
            metadata={**self.cfg, "final_epoch": self.args.epochs, "best_epoch": self.best_epoch,
                      "best_val_action_mse": self.best_mse if has_best else None},
        )
        artifact.add_file(self.last_path, name="policy.pt")          # final (policy + ema)
        if has_best:
            artifact.add_file(self.best_path, name="policy_best.pt")  # best by val/action_mse
        self.run.log_artifact(artifact, aliases=["latest"])


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ds, train_loader, val_loader = build_dataloaders(args, device)
    cfg = model_config(ds[0])
    policy = FlowMatchingPolicy(**cfg).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    n_val = len(val_loader.dataset) if val_loader is not None else 0
    n_train = len(train_loader.dataset)
    print(f"{n_train} train / {n_val} val samples, {n_params / 1e6:.2f}M parameters, device={device}")

    run = init_wandb(args, {**cfg, "n_params": n_params, "train_samples": n_train, "val_samples": n_val})
    Trainer(args, policy, cfg, train_loader, val_loader, device, run).fit()
    wandb.finish()


if __name__ == "__main__":
    main()