"""Convert a flat-terrain Microtaur checkpoint into a rough-terrain warm start.

The rough task inserts a 63-ray `height_scan` term right after `command` in both
the actor and critic observation groups; everything else (terms, order, network
sizes, rewards) matches the flat task. The first-layer weights get 63 zero
columns at that position, so the warm-started policy initially ignores the scan
and behaves exactly like the flat policy. Adam moments for those weights get
zero columns too. Iteration is reset to 0 (4500 rough iterations end at
model_4499.pt); common_step_counter is kept, so the command curriculum resumes
at the stage the flat run finished in.

Usage: python make_flat_warmstart.py <flat model.pt> <out model.pt>
"""

import sys

import torch

HEIGHT_SCAN_DIM = 63
# obs dims before height_scan: lin_vel 3, ang_vel 3, gravity 3, joint_pos 8,
# joint_vel 8, actions (8 legs + 1 spine if active), command 3
NUM_LEG_ACTIONS = 8


def pad_columns(w: torch.Tensor, insert_at: int) -> torch.Tensor:
  zeros = w.new_zeros(w.shape[0], HEIGHT_SCAN_DIM)
  return torch.cat([w[:, :insert_at], zeros, w[:, insert_at:]], dim=1)


def main(src: str, dst: str) -> None:
  ckpt = torch.load(src, map_location="cpu", weights_only=False)
  actor, critic = ckpt["actor_state_dict"], ckpt["critic_state_dict"]
  num_actions = actor["distribution.std_param"].shape[0]
  insert_at = 3 + 3 + 3 + NUM_LEG_ACTIONS * 2 + num_actions + 3

  old_shapes = {tuple(actor["mlp.0.weight"].shape), tuple(critic["mlp.0.weight"].shape)}
  actor["mlp.0.weight"] = pad_columns(actor["mlp.0.weight"], insert_at)
  critic["mlp.0.weight"] = pad_columns(critic["mlp.0.weight"], insert_at)

  padded = 0
  for state in ckpt["optimizer_state_dict"]["state"].values():
    if tuple(state["exp_avg"].shape) in old_shapes:
      state["exp_avg"] = pad_columns(state["exp_avg"], insert_at)
      state["exp_avg_sq"] = pad_columns(state["exp_avg_sq"], insert_at)
      padded += 1
  if padded != 2:
    raise RuntimeError(f"expected 2 first-layer Adam states, padded {padded}")

  ckpt["iter"] = 0
  ckpt["infos"] = {**(ckpt.get("infos") or {}), "flat_warmstart_source": src}
  torch.save(ckpt, dst)
  print(
    f"{src} -> {dst}: actions={num_actions}, height_scan at obs[{insert_at}], "
    f"actor {tuple(actor['mlp.0.weight'].shape)}, "
    f"critic {tuple(critic['mlp.0.weight'].shape)}, "
    f"common_step_counter={ckpt['infos'].get('env_state')}"
  )


if __name__ == "__main__":
  main(*sys.argv[1:3])
