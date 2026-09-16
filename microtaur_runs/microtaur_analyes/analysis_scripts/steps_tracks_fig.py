"""Top-down ground tracks on the step field over the exact tile height map.

    python scripts/steps_tracks_fig.py                    # open loop, peak speed
    python scripts/steps_tracks_fig.py --vx 0.14
    python scripts/steps_tracks_fig.py --waypoint         # lane-kept runs

One panel per variant: every seed's trunk track (first episode only -- a reset
teleports the robot to spawn and would draw a false straight line) laid over the
tile heights the simulator actually used. The heights are read from the compiled
terrain geometry, not re-generated, so they match the rollouts exactly.

Panel subtitle: share of time on the tiles, runs that left the side, furthest
point reached, and the mean number of risers (height changes) crossed.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from steps_onfield import rollouts, X0, X1, Y_HALF  # noqa: E402

TILE = 0.08
LEVELS_MM = [0.0, 2.5, 5.0, 7.5, 10.0]
CACHE = "rollouts/steps_tracks/heightmap.npz"

# Track colours are deepened (light) / lifted (dark) from the report's variant
# hues so they stay legible over every step of the grey height ramp.
VCOL = {"light": {"rigid": "#1f5f8b", "pitch": "#b85a0d", "yaw": "#1f7a4f", "roll": "#7a3aad"},
        "dark":  {"rigid": "#6fb3dc", "pitch": "#f0a45c", "yaw": "#5cc896", "roll": "#c39be4"}}
# Sequential, single (neutral) hue, light -> dark with height in the light theme
# and surface -> lighter in the dark theme. Neutral so the categorical track
# colours are the only hues on the panel.
TH = {"light": dict(bg="#ffffff", ink="#131d25", ink2="#586a76", grid="#d9e0e5", edge="#586a76",
                    ramp=["#f3f5f7", "#dde3e8", "#c3ccd3", "#a3afb9", "#7d8b96"]),
      "dark":  dict(bg="#141d21", ink="#e3ebef", ink2="#90a3ad", grid="#26343b", edge="#90a3ad",
                    ramp=["#1b252a", "#27343b", "#384850", "#4f6069", "#6c7f89"])}


def heightmap():
    """Tile heights in metres as a (ny, nx) grid: rows = lateral y, cols = forward x."""
    if os.path.exists(CACHE):
        return np.load(CACHE)["h"]
    import importlib.util
    import mujoco
    from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
    # Load the terrain module by FILE, not via the package: importing
    # `microtaur_velocity` runs its __init__, which builds env configs from
    # whatever env_cfgs.py a concurrent rollout has swapped in, and fails on a
    # variant mismatch. microtaur_terrains depends only on mjlab.
    spec = importlib.util.spec_from_file_location(
        "microtaur_terrains", "src/microtaur_velocity/microtaur_terrains.py")
    mt = importlib.util.module_from_spec(spec)
    sys.modules["microtaur_terrains"] = mt      # dataclasses resolve their module here
    spec.loader.exec_module(mt)
    steps_terrain = mt.steps_terrain
    ent = TerrainEntity(TerrainEntityCfg(terrain_type="generator",
                                         terrain_generator=steps_terrain(), num_envs=1),
                        device="cpu")
    m = ent.spec.compile()
    o = ent.env_origins[0].numpy()
    nx = int(round((X1 - X0) / TILE))
    ny = int(round(2 * Y_HALF / TILE))
    h = np.zeros((ny, nx))
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        sx, sy, sz = m.geom_size[g]
        if abs(sx - TILE / 2) > 1e-6 or abs(sy - TILE / 2) > 1e-6:
            continue                      # the base plane / border, not a tile
        px, py, pz = m.geom_pos[g]
        ix = int(round((px - o[0] - X0) / TILE - 0.5))
        iy = int(round((py - o[1] + Y_HALF) / TILE - 0.5))
        if 0 <= ix < nx and 0 <= iy < ny:
            h[iy, ix] = pz + sz
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, h=h)
    return h


def height_under(h, x, y):
    ix = np.floor((x - X0) / TILE).astype(int)
    iy = np.floor((y + Y_HALF) / TILE).astype(int)
    ok = (ix >= 0) & (ix < h.shape[1]) & (iy >= 0) & (iy < h.shape[0])
    out = np.full(len(x), np.nan)
    out[ok] = h[iy[ok], ix[ok]]
    return out


def first_episode(csv):
    df = pd.read_csv(csv)
    rs = df.index[(df.done > 0) & (df.index < len(df) - 1)]
    return (df.loc[: rs[0] - 1] if len(rs) else df), bool(len(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.20)
    ap.add_argument("--waypoint", action="store_true", help="plot lane-kept (waypoint) runs")
    ap.add_argument("--out", default="rollouts/steps_tracks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h = heightmap()
    tag = "waypoint" if a.waypoint else "openloop"
    stats = {}
    for mode, t in TH.items():
        plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink2"],
                             "xtick.color": t["ink2"], "ytick.color": t["ink2"],
                             "font.size": 9})
        cmap = ListedColormap(t["ramp"])
        norm = BoundaryNorm([-1.25, 1.25, 3.75, 6.25, 8.75, 11.25], cmap.N)
        fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.8), sharey=True, facecolor=t["bg"])
        im = None
        for ax, v in zip(axes, ["rigid", "pitch", "yaw", "roll"]):
            ax.set_facecolor(t["bg"])
            im = ax.imshow(h.T * 1000, origin="lower", cmap=cmap, norm=norm,
                           interpolation="nearest", extent=(-Y_HALF, Y_HALF, X0, X1), zorder=0)
            ax.add_patch(Rectangle((-Y_HALF, X0), 2 * Y_HALF, X1 - X0, fill=False,
                                   edgecolor=t["edge"], lw=1.2, zorder=1))
            ax.axvline(0, color=t["edge"], lw=0.7, ls=":", zorder=1)
            runs = {k: p for k, p in rollouts(v, lane_keep=a.waypoint).items()
                    if abs(k[0] - a.vx) < 1e-9}
            on_fr, left, reach, risers = [], 0, [], []
            for (_cmd, _seed), csv in sorted(runs.items()):
                ep, _ = first_episode(csv)
                ep = ep[ep.done == 0]
                x, y = ep.base_x.to_numpy(), ep.base_y.to_numpy()
                ss = ep.t.to_numpy() >= 1.0
                on = (x >= X0) & (x <= X1) & (np.abs(y) <= Y_HALF)
                on_fr.append(on[ss].mean())
                left += int((np.abs(y) > Y_HALF).any())
                reach.append(float(x.max()))
                hz = height_under(h, x, y)
                hz = hz[~np.isnan(hz)]
                risers.append(int((np.abs(np.diff(hz)) > 1e-4).sum()) if len(hz) > 1 else 0)
                ax.plot(y, x, lw=1.7, color=VCOL[mode][v], alpha=0.92,
                        solid_capstyle="round", zorder=3)
                ax.plot(y[-1], x[-1], "o", ms=4.5, color=VCOL[mode][v], zorder=4,
                        mec=t["bg"], mew=0.8)
            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-0.25, 3.9)
            ax.set_aspect("equal")
            ax.set_xlabel("lateral  y  [m]")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(t["grid"])
            if not runs:
                ax.set_title(f"{v}\n(no runs yet)", fontsize=10, color=t["ink"])
                continue
            stats[v] = dict(on=np.mean(on_fr), left=left, n=len(runs), reach=np.max(reach),
                            reach_mean=np.mean(reach), risers=np.mean(risers))
            ax.set_title(f"{v}\n{np.mean(on_fr) * 100:.0f}% on tiles  |  {left}/{len(runs)} left the side\n"
                         f"furthest {np.max(reach):.2f} m  |  mean {np.mean(reach):.2f} m  |  "
                         f"{np.mean(risers):.0f} risers crossed",
                         fontsize=9.3, color=t["ink"])
        axes[0].set_ylabel("forward  x  [m]   (spawn at 0)")
        cb = fig.colorbar(im, ax=list(axes), fraction=0.012, pad=0.012, ticks=LEVELS_MM)
        cb.set_label("tile height  [mm]", color=t["ink2"])
        cb.outline.set_edgecolor(t["grid"])
        cb.ax.tick_params(colors=t["ink2"])
        how = "waypoint-steered (lane-kept)" if a.waypoint else "open loop, no steering"
        fig.suptitle(f"Step field, cmd {a.vx:.2f} m/s, 8 seeds, {how}: tracks over the tile "
                     f"heights the simulator used (first episode only)",
                     fontsize=11, color=t["ink"])
        fig.subplots_adjust(left=0.05, right=0.92, top=0.79, bottom=0.10, wspace=0.08)
        p = f"{a.out}/tracks_{tag}_vx{a.vx:.2f}_{mode}.png"
        fig.savefig(p, dpi=130, facecolor=t["bg"])
        plt.close(fig)
        print("wrote", p)
    for v, s in stats.items():
        print(f"  {v:6s} on-tiles {s['on'] * 100:4.0f}%  left {s['left']}/{s['n']}  "
              f"furthest {s['reach']:.2f} m  mean {s['reach_mean']:.2f} m  "
              f"risers {s['risers']:.0f}")


if __name__ == "__main__":
    main()
