"""
Train the world-model encoder (fusion transformer + proprio MLP; DINOv2 frozen) with an
auxiliary state head and a progress head, on demonstrations and random-action data together.

  z    = LatentEncoder(obs)                                   (B, 256)
  loss = lambda_aux * aux_loss(AuxHead(z), state) + lambda_prog * progress_loss(ProgressHead(z), rho)

  rho = t / (T - 1) on demonstrations, 0 on random-action episodes.

Batches are balanced so that a fraction --demo_frac of frames are demonstrations.
Validation holds out --val_frac of the episodes of each source.

Example:
  python -m scripts.train_wm_encoder \
      --demo_datasets datasets/square_image_v15.hdf5 \
      --random_datasets datasets/random/square_random.hdf5 --epochs 30 --amp
"""
import argparse
import glob
import inspect
import math
import os

import torch
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.imitation_policy.augment import random_shift
from models.world_model.aux_head import AuxStateHead, aux_loss
from models.world_model.aux_head import eval_metrics as aux_metrics
from models.world_model.latent_encoder import LatentEncoder, LatentEncoderConfig
from models.world_model.progress_head import ProgressHead, progress_loss
from models.world_model.progress_head import eval_metrics as progress_metrics
from scripts.encoder_wm_dataset import EncoderWMDataset, Source, split_episodes


# ------------------------------------------------------------------ helpers
def to_device(x, device):
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x.to(device, non_blocking=True)


def prepare_obs(obs, shift_pad=0):
    """uint8 images -> float [0, 1], with optional random-shift augmentation (same as the policy)."""
    out = dict(obs)
    for k in ("agent_image", "wrist_image"):
        img = obs[k].float() / 255.0
        out[k] = random_shift(img, shift_pad) if shift_pad > 0 else img
    out["proprio"] = obs["proprio"].float()
    return out


def make_scheduler(opt, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to 0."""

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def encode(encoder, obs, amp):
    """Encoder forward, optionally in bf16. z is returned in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        z = encoder(obs)
    return z.float()


def compute_losses(args, aux_head, prog_head, z, batch):
    l_aux, aux_logs = aux_loss(aux_head, z, batch["aux_target"])
    l_prog, _ = progress_loss(prog_head, z, batch["progress"], kind=args.progress_loss)
    total = args.lambda_aux * l_aux + args.lambda_prog * l_prog
    logs = {"loss_aux": l_aux.detach(), "loss_progress": l_prog.detach(),
            **{f"loss_aux_{k}": v for k, v in aux_logs.items()}}
    return total, logs


@torch.no_grad()
def evaluate(args, encoder, aux_head, prog_head, loader, device, amp):
    """Held-out losses + aux errors (m / deg / R^2) and progress metrics, overall and per source."""
    encoder.eval()
    aux_head.eval()
    prog_head.eval()
    zs, aux_t, prog_t, demo = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        zs.append(encode(encoder, prepare_obs(batch["observation"]), amp))
        aux_t.append(batch["aux_target"])
        prog_t.append(batch["progress"])
        demo.append(batch["is_demo"])
    z, aux_t, prog_t, demo = torch.cat(zs), torch.cat(aux_t), torch.cat(prog_t), torch.cat(demo)

    total, logs = compute_losses(args, aux_head, prog_head, z, {"aux_target": aux_t, "progress": prog_t})
    m = {"val/loss": total.item(), **{f"val/{k}": v.item() for k, v in logs.items()}}
    m.update({f"val/{k}": v for k, v in aux_metrics(aux_head, z, aux_t).items()})
    # demo-vs-random separation (AUROC) uses all frames
    m.update({f"val/progress_{k}": v for k, v in progress_metrics(prog_head, z, prog_t, demo).items()})
    # how well g tracks progress WITHIN demonstrations (what ranking candidates relies on)
    if demo.any():
        m.update({f"val_demo/progress_{k}": v
                  for k, v in progress_metrics(prog_head, z[demo], prog_t[demo]).items()})
        m.update({f"val_demo/{k}": v for k, v in aux_metrics(aux_head, z[demo], aux_t[demo]).items()})
    if (~demo).any():
        m.update({f"val_random/{k}": v for k, v in aux_metrics(aux_head, z[~demo], aux_t[~demo]).items()})
        m["val_random/progress_pred_mean"] = prog_head.predict(z[~demo]).mean().item()
    return m


# ------------------------------------------------------------------ setup
def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--demo_datasets", nargs="+", default=["./datasets/square_image_v15.hdf5"])
    ap.add_argument("--random_datasets", nargs="*", default=["./datasets/random/square_random.hdf5"])
    ap.add_argument("--demo_frac", type=float, default=0.5,
                    help="fraction of each epoch's frames drawn from demonstrations")
    ap.add_argument("--samples_per_epoch", type=int, default=None,
                    help="frames drawn per epoch (default: number of training frames)")
    ap.add_argument("--val_frac", type=float, default=0.05, help="fraction of episodes held out per source")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    # optimization
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the encoder (CUDA only)")
    ap.add_argument("--shift_pad", type=int, default=4, help="random-shift augmentation pad (0 = off)")
    # losses
    ap.add_argument("--lambda_aux", type=float, default=1.0)
    ap.add_argument("--lambda_prog", type=float, default=1.0)
    ap.add_argument("--progress_loss", choices=["mse", "bce"], default="mse")
    # model
    ap.add_argument("--dino_model", type=str, default="dinov2_vits14")
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--head_hidden", type=int, default=256)
    # checkpoints / logging
    ap.add_argument("--out", type=str, default="encoder/wm_encoder.pt")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--wandb_project", type=str, default="world-model-encoder")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--log_every", type=int, default=10)
    return ap.parse_args()


def build_data(args, device):
    sources = [Source(p, True) for p in args.demo_datasets] + [Source(p, False) for p in args.random_datasets]
    train_eps, val_eps = split_episodes(sources, args.val_frac, args.split_seed)
    train_ds = EncoderWMDataset(sources, train_eps)
    val_ds = EncoderWMDataset(sources, val_eps) if args.val_frac > 0 else None

    n_samples = args.samples_per_epoch or len(train_ds)
    sampler = WeightedRandomSampler(train_ds.sample_weights(args.demo_frac), n_samples, replacement=True)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda",
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs) if val_ds is not None and len(val_ds) else None
    return sources, train_eps, val_eps, train_ds, val_ds, train_loader, val_loader


def init_wandb(args, config):
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                     mode=args.wandb_mode, config={**vars(args), **config})
    code = wandb.Artifact("wm-encoder-code", type="code")
    src_dir = os.path.dirname(inspect.getsourcefile(LatentEncoder))
    for path in sorted(glob.glob(os.path.join(src_dir, "*.py"))):
        code.add_file(path, name=f"world_model/{os.path.basename(path)}")
    for path in (os.path.abspath(__file__), inspect.getsourcefile(EncoderWMDataset)):
        code.add_file(path, name=os.path.basename(path))
    run.log_artifact(code)
    return run


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, args, encoder, aux_head, prog_head, enc_cfg, train_loader, val_loader,
                 device, run, split):
        self.args, self.enc_cfg, self.device, self.run, self.split = args, enc_cfg, device, run, split
        self.encoder, self.aux_head, self.prog_head = encoder, aux_head, prog_head
        self.train_loader, self.val_loader = train_loader, val_loader
        self.amp = args.amp and device == "cuda"

        self.params = (encoder.trainable_parameters() + list(aux_head.parameters())
                       + list(prog_head.parameters()))
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, betas=(0.9, 0.95),
                                     weight_decay=args.weight_decay)
        self.sched = make_scheduler(self.opt, args.warmup_steps, args.epochs * len(train_loader))

        root, ext = os.path.splitext(args.out)
        self.last_path, self.best_path = args.out, f"{root}_best{ext}"
        self.step = 0
        self.best_loss, self.best_epoch = float("inf"), None

    def train_epoch(self, epoch):
        args = self.args
        for m in (self.encoder, self.aux_head, self.prog_head):
            m.train()
        total = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = to_device(batch, self.device)
            z = encode(self.encoder, prepare_obs(batch["observation"], args.shift_pad), self.amp)
            loss, logs = compute_losses(args, self.aux_head, self.prog_head, z, batch)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.sched.step()

            self.step += 1
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", aux=f"{logs['loss_aux'].item():.4f}",
                             prog=f"{logs['loss_progress'].item():.4f}")
            if self.step % args.log_every == 0:
                self.log({
                    "train/loss": loss.item(),
                    **{f"train/{k}": v.item() for k, v in logs.items()},
                    "train/demo_frac_batch": batch["is_demo"].float().mean().item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": self.sched.get_last_lr()[0],
                    "epoch": epoch,
                })
        return total / len(self.train_loader)

    def eval_epoch(self, epoch):
        m = evaluate(self.args, self.encoder, self.aux_head, self.prog_head, self.val_loader,
                     self.device, self.amp)
        line = (f"  val loss {m['val/loss']:.4f} | nut_pos {m['val/nut_pos_rmse_m'] * 100:.2f} cm "
                f"(R2 {m['val/nut_pos_r2']:.3f}) | nut_quat {m['val/nut_quat_deg']:.1f} deg | "
                f"progress rmse {m['val/progress_rmse']:.3f}")
        if "val_demo/progress_spearman" in m:
            line += f" | demo spearman {m['val_demo/progress_spearman']:.3f}"
        if "val/progress_auroc" in m:
            line += f" | demo-vs-random AUROC {m['val/progress_auroc']:.3f}"
        print(line)
        self.log({**m, "epoch": epoch})
        if m["val/loss"] < self.best_loss:
            self.best_loss, self.best_epoch = m["val/loss"], epoch
            self.save(self.best_path, epoch)
            self.run.summary["best/val_loss"] = self.best_loss
            self.run.summary["best/epoch"] = self.best_epoch

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

    def log(self, metrics):
        wandb.log(metrics, step=self.step)

    def save(self, path, epoch):
        """Frozen DINOv2 weights are NOT saved (they reload from torch.hub), keeping files small.
        Restore with: enc = LatentEncoder(LatentEncoderConfig(**ckpt['encoder_config']));
                      enc.load_state_dict(ckpt['encoder'], strict=False)"""
        enc_state = {k: v for k, v in self.encoder.state_dict().items() if not k.startswith("tokenizer.")}
        torch.save({
            "encoder": enc_state,
            "aux_head": self.aux_head.state_dict(),
            "progress_head": self.prog_head.state_dict(),
            "encoder_config": vars(self.enc_cfg),
            "head_hidden": self.args.head_hidden,
            "split": self.split,  # sources and train/val episode names
            "epoch": epoch,
        }, path)

    def upload_model(self):
        has_best = self.best_epoch is not None
        artifact = wandb.Artifact(
            "wm-encoder-model", type="model",
            metadata={**vars(self.enc_cfg), "final_epoch": self.args.epochs, "best_epoch": self.best_epoch,
                      "best_val_loss": self.best_loss if has_best else None},
        )
        artifact.add_file(self.last_path, name=os.path.basename(self.last_path))
        if has_best:
            artifact.add_file(self.best_path, name=os.path.basename(self.best_path))
        self.run.log_artifact(artifact, aliases=["latest"])


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    sources, train_eps, val_eps, train_ds, val_ds, train_loader, val_loader = build_data(args, device)
    sample = train_ds[0]["observation"]
    enc_cfg = LatentEncoderConfig(
        dino_model=args.dino_model,
        image_size=sample["agent_image"].shape[-1],
        proprio_dim=sample["proprio"].shape[-1],
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    encoder = LatentEncoder(enc_cfg).to(device)
    aux_head = AuxStateHead(args.latent_dim, args.head_hidden).to(device)
    prog_head = ProgressHead(args.latent_dim, args.head_hidden).to(device)

    # z-score aux position targets using TRAIN frames only (quaternion dims are left alone)
    aux_head.fit_stats(train_ds.aux_targets.to(device))

    n_train_demo = int(train_ds.is_demo.sum())
    n_val = len(val_ds) if val_ds is not None else 0
    n_enc = sum(p.numel() for p in encoder.trainable_parameters())
    n_heads = sum(p.numel() for m in (aux_head, prog_head) for p in m.parameters())
    print(f"train: {len(train_ds)} frames ({n_train_demo} demo / {len(train_ds) - n_train_demo} random), "
          f"val: {n_val} frames | {len(train_loader)} steps/epoch at demo_frac={args.demo_frac} | "
          f"trainable: encoder {n_enc / 1e6:.2f}M, heads {n_heads / 1e6:.2f}M | device={device}")

    split = {"sources": [vars(s) for s in sources], "train": train_eps, "val": val_eps}
    run = init_wandb(args, {**vars(enc_cfg), "n_trainable_encoder": n_enc, "n_heads": n_heads,
                            "train_frames": len(train_ds), "train_demo_frames": n_train_demo,
                            "val_frames": n_val})
    Trainer(args, encoder, aux_head, prog_head, enc_cfg, train_loader, val_loader, device, run,
            split).fit()
    wandb.finish()


if __name__ == "__main__":
    main()
