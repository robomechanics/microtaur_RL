from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from microtaur_velocity.distill.student_policy import StudentPolicy


def load_dataset(dataset_dir: Path, max_samples: int | None = None, seed: int = 0):
    files = sorted(dataset_dir.glob("teacher_distill_chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"No teacher_distill_chunk_*.pt files found in {dataset_dir}")

    obs_list = []
    action_list = []
    for file in files:
        data = torch.load(file, map_location="cpu")
        obs_list.append(data["student_obs"].float())
        action_list.append(data["teacher_action"].float())

    obs = torch.cat(obs_list, dim=0)
    actions = torch.cat(action_list, dim=0)

    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    if max_samples is not None and max_samples > 0 and obs.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(obs.shape[0], generator=g)[:max_samples]
        obs = obs[idx]
        actions = actions[idx]

    return obs, actions, files


def make_loss(name: str):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=0.05)
    raise ValueError(f"Unsupported loss {name!r}; use 'mse' or 'huber'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="logs/distill")
    parser.add_argument("--out", type=str, default="logs/distill/student_policy_bc.pt")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs, actions, files = load_dataset(dataset_dir, max_samples=args.max_samples, seed=args.seed)
    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]

    obs_mean = obs.mean(dim=0)
    obs_std = obs.std(dim=0).clamp_min(1e-4)
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0).clamp_min(1e-4)
    action_abs_mean = actions.abs().mean()

    print("Loaded distillation dataset:")
    print(f"  files: {len(files)}")
    print(f"  obs: {tuple(obs.shape)}")
    print(f"  actions: {tuple(actions.shape)}")
    print(f"  hidden_dims: {tuple(args.hidden_dims)}")
    print(f"  obs abs mean: {float(obs.abs().mean()):.4f}")
    print(f"  action abs mean: {float(action_abs_mean):.4f}")
    print(f"  action std mean: {float(action_std.mean()):.4f}")
    if float(action_abs_mean) < 0.03:
        print("[WARN] teacher actions are very small; student may learn to stand/freeze.")

    dataset = TensorDataset(obs, actions)
    val_size = max(1, int(len(dataset) * args.val_frac))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)
    student.set_normalization(obs_mean.to(device), obs_std.to(device))

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.loss)

    best_val = math.inf
    for epoch in range(args.epochs):
        student.train()
        train_loss = 0.0
        train_batches = 0
        pred_abs_running = 0.0

        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)

            pred = student(batch_obs)
            loss = loss_fn(pred, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())
            pred_abs_running += float(pred.detach().abs().mean().item())
            train_batches += 1

        student.eval()
        val_loss = 0.0
        val_batches = 0
        val_pred_abs = 0.0
        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                pred = student(batch_obs)
                loss = loss_fn(pred, batch_actions)
                val_loss += float(loss.item())
                val_pred_abs += float(pred.abs().mean().item())
                val_batches += 1

        train_loss /= max(train_batches, 1)
        val_loss /= max(val_batches, 1)
        pred_abs = pred_abs_running / max(train_batches, 1)
        val_pred_abs /= max(val_batches, 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "hidden_dims": tuple(args.hidden_dims),
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "action_abs_mean": action_abs_mean,
                    "student_obs_layout": "base35 + older_actions_[t-2,t-3,t-4] = 59D",
                    "timing": "obs_before_action__push_action_after_env_step",
                    "best_val_loss": best_val,
                    "loss": args.loss,
                },
                out_path,
            )

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train {args.loss} {train_loss:.6f} | val {args.loss} {val_loss:.6f} | "
            f"pred_abs {pred_abs:.4f}/{val_pred_abs:.4f} | best {best_val:.6f}"
        )

    print(f"Saved best student policy to: {out_path}")


if __name__ == "__main__":
    main()
