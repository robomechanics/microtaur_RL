"""Training curves, final return and time to convergence for the Microtaur policies.

Stages:
  flat   - flat-terrain runs trained from scratch (old_distilling/, trained on the Windows PC)
  rough  - rough-terrain runs warm-started from the flat model_4499 (*_rough_from_flat, this PC)

Definitions (all from TensorBoard `Train/mean_reward`, the mean undiscounted return of the
episodes that ended in each PPO iteration; max episode length 1000 steps = 20 s):
  smoothed return   trailing mean over 100 iterations
  final return      mean +- std of the raw return over the last 100 iterations (4400-4499)
  first hit (T95 first)  first iteration where the smoothed return reaches 95 % of the final return
  convergence (T95)      first iteration after which the smoothed return stays >= 95 % of the
                         final return for the rest of training. The flat runs use a 5-stage command
                         curriculum (stage changes at iterations 781, 1562, 2187, 2812) and the rough
                         runs a terrain curriculum, so the return is not stationary; dips after a stage
                         change push T95 later than the first hit.
  env steps         iterations x 32 steps x 2048 envs = 65,536 samples per iteration
  T90               same as T95 with a 90 % threshold (sensitivity check)
  wall-clock        cumulative Perf/collection_time + Perf/learning_time up to T95

Run: python make_training_results.py   (writes into this folder)
"""

import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent  # spines_help
SAMPLES_PER_ITER = 32 * 2048
WINDOW = 100
CONV_FRACTION = 0.95
CMD_STAGE_ITERS = [s // 32 for s in (25_000, 50_000, 70_000, 90_000)]  # command curriculum (flat runs)

# Fixed categorical order (reference dataviz palette, slots 1-4)
VARIANTS = (
  # key, label, color, flat run dir, rough folder
  ("rigid", "Rigid", "#2a78d6", "old_distilling/rigid_flat_fixed_m077", "rigid"),
  ("roll", "Active roll", "#eb6834", "old_distilling/active_twist_077_motors", "active_twist"),
  ("pitch", "Active pitch", "#1baf7a", "old_distilling/pitch_all_motors_fixed_v1", "active_pitch"),
  ("yaw", "Active yaw", "#eda100", "old_distilling/yaw_motros_fixed_flat", "active_yaw"),
)


def rough_dir(folder: str) -> str:
  pattern = f"microtaur_runs/{folder}/logs/rsl_rl/microtaur_minitaur_velocity/*_{folder}_rough_from_flat"
  (match,) = glob.glob(str(ROOT / pattern))
  return match


def load(run_dir: str) -> dict:
  ea = EventAccumulator(str(ROOT / run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  scalar = lambda tag: np.array([e.value for e in ea.Scalars(tag)])
  steps = np.array([e.step for e in ea.Scalars("Train/mean_reward")])
  compute_s = np.cumsum(scalar("Perf/collection_time") + scalar("Perf/learning_time"))
  return {
    "iter": steps,
    "return": scalar("Train/mean_reward"),
    "episode_length": scalar("Train/mean_episode_length"),
    "compute_s": compute_s,
  }


def summarize(run: dict) -> dict:
  r = run["return"]
  smooth = np.array([r[max(0, i - WINDOW + 1) : i + 1].mean() for i in range(len(r))])
  final_mean, final_std = r[-WINDOW:].mean(), r[-WINDOW:].std()
  below = np.nonzero(smooth < CONV_FRACTION * final_mean)[0]
  t95 = int(below[-1] + 1) if len(below) else 0
  t95_first = int(np.argmax(smooth >= CONV_FRACTION * final_mean))
  below90 = np.nonzero(smooth < 0.90 * final_mean)[0]
  t90 = int(below90[-1] + 1) if len(below90) else 0
  return {
    "smooth": smooth,
    "final_mean": final_mean,
    "final_std": final_std,
    "final_episode_length": run["episode_length"][-WINDOW:].mean(),
    "t95_first_iter": int(run["iter"][t95_first]),
    "t95_first_env_steps_M": run["iter"][t95_first] * SAMPLES_PER_ITER / 1e6,
    "t95_first_compute_min": run["compute_s"][t95_first] / 60,
    "t90_iter": int(run["iter"][t90]),
    "t95_iter": int(run["iter"][t95]),
    "t95_env_steps_M": run["iter"][t95] * SAMPLES_PER_ITER / 1e6,
    "t95_compute_min": run["compute_s"][t95] / 60,
    "total_compute_min": run["compute_s"][-1] / 60,
  }


def main() -> None:
  results = {}
  for key, label, color, flat_dir, rough_folder in VARIANTS:
    for stage, run_dir in (("flat", flat_dir), ("rough", rough_dir(rough_folder))):
      run = load(run_dir)
      results[(stage, key)] = {"run": run, "dir": run_dir, **summarize(run)}

  # ---- table -------------------------------------------------------------------------
  rows = []
  for stage in ("flat", "rough"):
    for key, label, *_ in VARIANTS:
      s = results[(stage, key)]
      rows.append({
        "terrain": "flat (from scratch)" if stage == "flat" else "rough (warm start from flat)",
        "variant": label,
        "final_return_mean": round(float(s["final_mean"]), 1),
        "final_return_std": round(float(s["final_std"]), 1),
        "final_episode_length": round(float(s["final_episode_length"]), 0),
        "t95_first_iteration": s["t95_first_iter"],
        "t95_first_env_steps_millions": round(float(s["t95_first_env_steps_M"]), 0),
        "t95_first_wallclock_min": round(float(s["t95_first_compute_min"]), 0),
        "t95_iteration": s["t95_iter"],
        "t95_env_steps_millions": round(float(s["t95_env_steps_M"]), 0),
        "t95_wallclock_min": round(float(s["t95_compute_min"]), 0),
        "t90_iteration": s["t90_iter"],
        "total_wallclock_min": round(float(s["total_compute_min"]), 0),
        "run_dir": str(Path(s["dir"]).relative_to(ROOT)) if Path(s["dir"]).is_absolute() else s["dir"],
      })
  with open(OUT / "results_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  # raw + smoothed curves for re-plotting elsewhere
  with open(OUT / "training_curves.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["terrain", "variant", "iteration", "return", "return_smoothed_100", "episode_length"])
    for (stage, key), s in results.items():
      for i, it in enumerate(s["run"]["iter"]):
        writer.writerow([stage, key, int(it), f"{s['run']['return'][i]:.3f}", f"{s['smooth'][i]:.3f}",
                         f"{s['run']['episode_length'][i]:.1f}"])

  # ---- figure ------------------------------------------------------------------------
  plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#8a8986", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
  })
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
  titles = {"flat": "Flat terrain (from scratch)", "rough": "Rough terrain (warm start from flat)"}
  for ax, stage in zip(axes, ("flat", "rough")):
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    if stage == "flat":
      for it in CMD_STAGE_ITERS:
        ax.axvline(it, color="#b5b4ae", linewidth=1, linestyle=(0, (3, 3)), zorder=0)
      ax.text(CMD_STAGE_ITERS[-1] + 40, 8, "dashed: command\ncurriculum stage changes", color="#52514e", fontsize=7.5)
    for key, label, color, *_ in VARIANTS:
      s = results[(stage, key)]
      ax.plot(s["run"]["iter"], s["smooth"], color=color, linewidth=1.6, label=label)
      ax.plot(s["t95_iter"], s["smooth"][s["t95_iter"]], "o", markersize=6, color=color,
              markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=5)
    ax.set_title(titles[stage], loc="left", color="#0b0b0b")
    ax.set_xlabel("PPO iteration  (1 iteration = 65,536 env steps)")
    ax.set_xlim(0, 4500)
  axes[0].set_ylabel("Mean episode return (100-iteration mean)")
  axes[0].set_ylim(0, 115)
  handles, labels = axes[0].get_legend_handles_labels()
  handles.append(plt.Line2D([], [], marker="o", linestyle="", color="#52514e", markersize=6))
  labels.append("T95 (converged)")
  axes[1].legend(handles, labels, loc="lower right", frameon=False, fontsize=8)
  for ext in ("png", "pdf"):
    fig.savefig(OUT / f"training_curves.{ext}", dpi=200, facecolor="#fcfcfb")

  for r in rows:
    print(r)


if __name__ == "__main__":
  main()
