from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from microtaur_velocity.distill_reliable.dataset import load_dataset, summarize_tensors
from microtaur_velocity.distill_reliable.policy import StudentPolicy


def make_loss(name: str, huber_beta: float):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=float(huber_beta))
    raise ValueError(f"Unsupported loss {name!r}; use 'mse' or 'huber'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs="+", default=["logs/distill_reliable/bc"])
    parser.add_argument("--out", type=str, default="logs/distill_reliable/students/student_128x64_bc.pt")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--obs-noise-std", type=float, default=0.0, help="Optional normalized-observation noise during training.")
    parser.add_argument("--require-obs-dim", type=int, default=None)
    parser.add_argument("--require-action-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs, actions, files = load_dataset(
        args.dataset_dirs,
        max_samples=args.max_samples,
        seed=args.seed,
        require_obs_dim=args.require_obs_dim,
        require_action_dim=args.require_action_dim,
    )
    obs_dim = int(obs.shape[-1])
    action_dim = int(actions.shape[-1])

    summary = summarize_tensors(obs, actions)
    print("Loaded aggregated distillation dataset:")
    print(f"  dirs: {args.dataset_dirs}")
    print(f"  usable files: {len(files)}")
    print(f"  samples: {summary['samples']}")
    print(f"  obs_dim: {summary['obs_dim']}")
    print(f"  action_dim: {summary['action_dim']}")
    print(f"  hidden_dims: {tuple(args.hidden_dims)}")
    print(f"  action_abs_mean: {summary['action_abs_mean']:.4f}")
    print(f"  action_std_mean: {summary['action_std_mean']:.4f}")

    if summary["action_abs_mean"] < 0.03:
        print("[WARN] teacher actions are very small; student may learn to stand/freeze.")
    if obs_dim not in (35, 38):
        print(f"[WARN] expected current Microtaur student obs_dim 35 or 38, got {obs_dim}.")

    obs_mean = obs.mean(dim=0)
    obs_std = obs.std(dim=0).clamp_min(1e-4)
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0).clamp_min(1e-4)

    dataset = TensorDataset(obs, actions)
    val_size = max(1, int(len(dataset) * float(args.val_frac)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    student = StudentPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        output_tanh=True,
    ).to(device)
    student.set_normalization(obs_mean.to(device), obs_std.to(device))

    print(f"  student parameters: {student.num_parameters()}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.loss, args.huber_beta)

    best_val = math.inf
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss_sum = 0.0
        train_count = 0
        pred_abs_sum = 0.0

        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)

            if args.obs_noise_std > 0.0:
                # Add noise in normalized space, then unnormalize back.
                norm_obs = (batch_obs - student.obs_mean) / student.obs_std
                norm_obs = norm_obs + args.obs_noise_std * torch.randn_like(norm_obs)
                batch_obs = norm_obs * student.obs_std + student.obs_mean

            pred = student(batch_obs)
            loss = loss_fn(pred, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            n = batch_obs.shape[0]
            train_loss_sum += float(loss.item()) * n
            pred_abs_sum += float(pred.detach().abs().mean().item()) * n
            train_count += n

        student.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_pred_abs_sum = 0.0
        per_dim_mse_sum = torch.zeros(action_dim, dtype=torch.float64)

        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                pred = student(batch_obs)
                loss = loss_fn(pred, batch_actions)

                n = batch_obs.shape[0]
                val_loss_sum += float(loss.item()) * n
                val_pred_abs_sum += float(pred.abs().mean().item()) * n
                per_dim_mse_sum += ((pred - batch_actions) ** 2).sum(dim=0).double().cpu()
                val_count += n

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        pred_abs = pred_abs_sum / max(train_count, 1)
        val_pred_abs = val_pred_abs_sum / max(val_count, 1)
        per_dim_mse = (per_dim_mse_sum / max(val_count, 1)).float()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "hidden_dims": tuple(args.hidden_dims),
                    "activation": args.activation,
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "action_abs_mean": actions.abs().mean(),
                    "action_std_mean": action_std.mean(),
                    "student_obs_layout": f"aligned robust deployable observation ({obs_dim}D)",
                    "dataset_dirs": list(args.dataset_dirs),
                    "num_dataset_files": len(files),
                    "num_samples": int(obs.shape[0]),
                    "loss": args.loss,
                    "huber_beta": args.huber_beta,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "per_dim_mse": per_dim_mse,
                },
                out_path,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d}/{args.epochs} | "
                f"train {args.loss} {train_loss:.6f} | "
                f"val {args.loss} {val_loss:.6f} | "
                f"pred_abs {pred_abs:.4f}/{val_pred_abs:.4f} | "
                f"best {best_val:.6f} @ {best_epoch}"
            )

    print(f"Saved best student policy to: {out_path}")


if __name__ == "__main__":
    main()
