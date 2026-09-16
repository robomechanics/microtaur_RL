"""Assemble the 4-variant comparison HTML artifact (figures embedded)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

RO = Path("rollouts")
figs = json.loads((RO / "compare_figs" / "_b64.json").read_text())

FWD, YAWD = {}, {}
for v in ("rigid", "pitch", "yaw", "roll"):
  for d in sorted(glob.glob(f"rollouts/{v}_*")):
    cs = glob.glob(f"{d}/rollout_*.csv")
    n, sp = len(cs), len({c.split("rollout_vx")[-1][:4] for c in cs})
    if n >= 14 and sp >= 4:
      FWD[v] = d
    elif 6 <= n < 14:
      YAWD[v] = d

REV = {"rigid": "2026-09-05-rigid-aligned-baseline-v1",
       "pitch": "2026-09-01-active-pitch-final-xml-v2",
       "yaw": "2026-08-30-active-yaw-spine-residual-v1  (M077, reconstructed)",
       "roll": "2026-09-05-active-twist-emergent-v4-mean-bias-fix"}


def fwd_agg(d):
  rows = []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f); ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) == 0:
      ss = x
    rows.append(dict(cmd=float(x.cmd_vx.iloc[0]), v=ss.v_body_x.mean(),
                     terr=(ss.v_body_x - ss.cmd_vx).abs().mean(),
                     cotp=ss.CoT_pos.mean(), cota=ss.CoT_abs.mean(),
                     tilt=ss.tilt_deg.mean(), frozen=ss.v_below_eps.mean(),
                     mass=x.attrs.get("m", np.nan)))
  return pd.DataFrame(rows).groupby("cmd").mean().reset_index()


def mass_of(d):
  mj = sorted(glob.glob(f"{d}/rollout_*.meta.json"))
  return json.loads(Path(mj[0]).read_text())["mass_kg"] if mj else float("nan")


A = {v: fwd_agg(d) for v, d in FWD.items()}
MASS = {v: mass_of(d) for v, d in FWD.items()}

_LEGS = [f"P_leg{i}_{s}" for i in (1, 2, 3, 4) for s in ("a", "e")]


def spine_work(d):
  """Spine negative-work share + net power over the usable-speed rollouts."""
  neg_share, net_mW, spine_of_robot = [], [], []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f)
    ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) < 50 or "P_spine" not in ss or ss.v_below_eps.mean() > 0.5:
      continue
    P = ss["P_spine"].to_numpy()
    tot = np.abs(P).sum()
    if tot == 0:
      continue
    sneg = np.abs(np.clip(P, None, 0)).sum()
    lneg = np.abs(np.clip(ss[_LEGS].to_numpy(), None, 0)).sum()
    neg_share.append(sneg / tot)
    net_mW.append(P.mean() * 1e3)
    spine_of_robot.append(sneg / (sneg + lneg))
  if not neg_share:
    return None
  return dict(neg_share=float(np.mean(neg_share)), net_mW=float(np.mean(net_mW)),
             spine_of_robot=float(np.mean(spine_of_robot)))


SPW = {v: spine_work(d) for v, d in FWD.items()}
SPW = {v: s for v, s in SPW.items() if s}

# per-motor CoT and stability summary
NMOT = {}
for v, d in FWD.items():
  mj = sorted(glob.glob(f"{d}/rollout_*.meta.json"))
  NMOT[v] = json.loads(Path(mj[0]).read_text())["action_dim"] if mj else 8


def cot_per_motor(v, d):
  u_speeds = (0.12, 0.14, 0.16, 0.18, 0.20) if v == "roll" else None
  cp, lo = [], []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f); ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) == 0 or ss.v_below_eps.mean() > 0.5:
      continue
    if u_speeds and round(float(x.cmd_vx.iloc[0]), 2) not in u_speeds:
      continue
    cp.append(ss.CoT_pos.mean()); lo.append(ss.CoT_pos_legs_only.mean())
  return (float(np.mean(cp)) / NMOT[v], float(np.mean(lo)) / 8.0)


CPM = {v: cot_per_motor(v, d) for v, d in FWD.items()}

STAB = None
_sf = Path("rollouts/stability_figs/stability_summary.csv")
if _sf.exists():
  STAB = pd.read_csv(_sf).set_index("v")

# headline over usable speeds (>=0.12 for roll, all for others)
def usable(v, a):
  return a[a.cmd >= 0.12] if v == "roll" else a

hrows = []
for v, a in A.items():
  u = usable(v, a)
  yd = YAWD.get(v)
  yfrac = np.nan
  if yd:
    fr = []
    for f in glob.glob(f"{yd}/rollout_*.csv"):
      x = pd.read_csv(f); ss = x[(x.t >= 1.0) & (x.done == 0)]
      fr.append(ss.wz.mean() / ss.cmd_yaw.iloc[0])
    yfrac = float(np.mean(fr))
  hrows.append(dict(variant=v, mass=MASS[v],
                    cotp=u.cotp.mean(), cota=u.cota.mean(),
                    terr=u.terr.mean(), tilt=u.tilt.mean(),
                    yawfrac=yfrac,
                    envelope=("0.12–0.20" if v == "roll" else "0.08–0.20")))
H = pd.DataFrame(hrows)


def htable(df, cols, fmts, headers):
  th = "".join(f"<th>{h}</th>" for h in headers)
  body = ""
  for _, r in df.iterrows():
    tds = "".join(f"<td>{f(r[c])}</td>" for c, f in zip(cols, fmts))
    body += f"<tr><td class='v v-{r['variant']}'>{r['variant']}</td>{tds}</tr>"
  return f"<table><thead><tr><th>variant</th>{th}</tr></thead><tbody>{body}</tbody></table>"


head_tbl = htable(
  H, ["mass", "envelope", "cotp", "cota", "terr", "tilt", "yawfrac"],
  [lambda x: f"{x:.3f}", str, lambda x: f"{x:.2f}", lambda x: f"{x:.2f}",
   lambda x: f"{x:.3f}", lambda x: f"{x:.1f}", lambda x: "—" if pd.isna(x) else f"{x:.2f}"],
  ["mass kg", "speed envelope", "CoT⁺", "CoT_abs", "track err", "tilt°", "yaw-rate realised"],
)

# per-speed CoT+ mini table
sp_rows = sorted({c for a in A.values() for c in a.cmd})
sp_head = "".join(f"<th>{s:.2f}</th>" for s in sp_rows)
sp_body = ""
for v, a in A.items():
  cells = ""
  for s in sp_rows:
    m = a[np.isclose(a.cmd, s)]
    val = f"{m.cotp.iloc[0]:.2f}" if len(m) else "·"
    cls = " class='frozen'" if len(m) and m.frozen.iloc[0] > 0.5 else ""
    cells += f"<td{cls}>{val}</td>"
  sp_body += f"<tr><td class='v v-{v}'>{v}</td>{cells}</tr>"
sp_tbl = f"<table><thead><tr><th>CoT⁺ · cmd vₓ →</th>{sp_head}</tr></thead><tbody>{sp_body}</tbody></table>"

# spine-power table
_SPLABEL = {"pitch": "lossy spring / net brake", "yaw": "near-lossless motor", "roll": "pure damper"}
spw_body = ""
for v in ("pitch", "yaw", "roll"):
  s = SPW.get(v)
  if not s:
    continue
  spw_body += (f"<tr><td class='v v-{v}'>{v}</td>"
               f"<td>{s['neg_share']:.2f}</td><td>{s['net_mW']:+.1f}</td>"
               f"<td>{100 * s['spine_of_robot']:.0f}%</td><td style='text-align:left'>{_SPLABEL[v]}</td></tr>")
spw_tbl = ("<table><thead><tr><th>variant</th><th>neg-work share</th>"
           "<th>net spine power (mW)</th><th>share of robot's neg-work</th>"
           "<th style='text-align:left'>character</th></tr></thead>"
           f"<tbody>{spw_body}</tbody></table>")

# per-motor CoT table
cpm_body = ""
for v in ("rigid", "pitch", "yaw", "roll"):
  tot, lo = CPM[v]
  cpm_body += (f"<tr><td class='v v-{v}'>{v}</td><td>{NMOT[v]}</td>"
               f"<td>{A[v].cotp.mean():.2f}</td><td>{tot:.3f}</td><td>{lo:.3f}</td></tr>")
cpm_tbl = ("<table><thead><tr><th>variant</th><th>motors</th><th>CoT⁺ (total)</th>"
           "<th>CoT⁺ ÷ motors</th><th>leg-only CoT⁺ ÷ 8</th></tr></thead>"
           f"<tbody>{cpm_body}</tbody></table>")

# stability table
stab_tbl = ""
if STAB is not None:
  sb = ""
  for v in ("rigid", "pitch", "yaw", "roll"):
    r = STAB.loc[v]
    sb += (f"<tr><td class='v v-{v}'>{v}</td>"
           f"<td>{r['lat_rms_mm']:.1f}</td><td>{r['lat_peak_mm']:.1f}</td>"
           f"<td>{r['slow_wander_mm']:.1f}</td><td>{r['z_rms_mm']:.1f}</td>"
           f"<td>{100 * r['frac_ge3']:.0f}%</td></tr>")
  stab_tbl = ("<table><thead><tr><th>variant</th><th>CoM lat RMS (mm)</th>"
              "<th>CoM lat peak (mm)</th><th>slow wander (mm)</th>"
              "<th>CoM vert RMS (mm)</th><th>cycle with ≥3 feet</th></tr></thead>"
              f"<tbody>{sb}</tbody></table>")

FIG = lambda k, cap: f'<figure><img alt="{cap}" src="{figs[k]}"><figcaption>{cap}</figcaption></figure>'

HTML = f"""<title>Microtaur Spine Variants</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital@0;1&display=swap">
<style>
:root {{
  --ground:#eef2f5; --surface:#ffffff; --surface-2:#e7edf1; --raise:#f6f8fa;
  --ink:#131d25; --muted:#586a76; --faint:#8496a1; --line:#d9e0e5;
  --rigid:#33688f; --pitch:#bd6f1c; --yaw:#2f8a61; --roll:#8f4bbf;
  --ok:#2f8a61; --warn:#bd6f1c;
  --font-head:"IBM Plex Sans",system-ui,sans-serif;
  --font-body:"IBM Plex Sans",system-ui,sans-serif;
  --font-serif:"IBM Plex Serif",Georgia,serif;
  --font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0c1215; --surface:#141d21; --surface-2:#1c272c; --raise:#18221f;
    --ink:#e3ebef; --muted:#90a3ad; --faint:#6d818c; --line:#26343b;
    --rigid:#5da0c8; --pitch:#e0954c; --yaw:#45b585; --roll:#b78fd6;
    --ok:#45b585; --warn:#e0954c;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0c1215; --surface:#141d21; --surface-2:#1c272c; --raise:#18221f;
  --ink:#e3ebef; --muted:#90a3ad; --faint:#6d818c; --line:#26343b;
  --rigid:#5da0c8; --pitch:#e0954c; --yaw:#45b585; --roll:#b78fd6;
  --ok:#45b585; --warn:#e0954c;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--font-body); font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:clamp(1.5rem,4vw,4rem) clamp(1.1rem,3vw,2.5rem) 5rem; }}
.prose {{ max-width:68ch; }}
h1,h2,h3 {{ font-family:var(--font-head); font-weight:600; text-wrap:balance; line-height:1.25; margin:0; }}
h1 {{ font-size:clamp(1.9rem,4vw,2.7rem); letter-spacing:-0.02em; }}
h2 {{ font-size:1.35rem; letter-spacing:-0.01em; }}
h3 {{ font-size:1rem; }}
code {{ font-family:var(--font-mono); font-size:0.86em; background:var(--surface-2); padding:0.1em 0.36em; border-radius:3px; }}
.eyebrow {{ font-family:var(--font-mono); font-size:0.72rem; font-weight:600; letter-spacing:0.16em;
  text-transform:uppercase; color:var(--muted); margin:0 0 0.6rem; }}
header.masthead {{ border-bottom:2px solid var(--ink); padding-bottom:1.6rem; margin-bottom:2.4rem; }}
header.masthead .lede {{ font-family:var(--font-serif); font-size:1.15rem; line-height:1.55; color:var(--muted); margin:1.1rem 0 0; max-width:66ch; }}
header.masthead .lede em {{ color:var(--ink); font-style:italic; }}
.legend {{ display:flex; flex-wrap:wrap; gap:1rem 1.6rem; margin:1.4rem 0 0; font-family:var(--font-mono); font-size:0.8rem; }}
.legend span {{ display:inline-flex; align-items:center; gap:0.45rem; }}
.legend i {{ width:0.85rem; height:0.85rem; border-radius:3px; display:inline-block; }}
.i-rigid {{ background:var(--rigid); }} .i-pitch {{ background:var(--pitch); }}
.i-yaw {{ background:var(--yaw); }} .i-roll {{ background:var(--roll); }}
section {{ margin-top:3.2rem; }}
section > p {{ max-width:68ch; }}
figure {{ margin:1.6rem 0 0; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:0.9rem; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; }}
figcaption {{ font-size:0.83rem; color:var(--muted); margin-top:0.7rem; padding-top:0.6rem; border-top:1px solid var(--line); font-family:var(--font-mono); }}
.tablewrap {{ overflow-x:auto; margin:1.4rem 0 0; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-family:var(--font-mono); font-size:0.83rem; font-variant-numeric:tabular-nums; }}
th,td {{ padding:0.5rem 0.8rem; text-align:right; white-space:nowrap; }}
thead th {{ background:var(--surface-2); color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); }}
tbody tr:nth-child(even) {{ background:var(--raise); }}
td.v {{ text-align:left; font-weight:600; }}
.v-rigid {{ color:var(--rigid); }} .v-pitch {{ color:var(--pitch); }}
.v-yaw {{ color:var(--yaw); }} .v-roll {{ color:var(--roll); }}
td.frozen {{ color:var(--faint); }} td.frozen::after {{ content:" ✳"; }}
.verdict {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); margin-top:1.6rem; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:9px; padding:1.05rem 1.15rem; border-top:3px solid var(--line); }}
.card.win {{ border-top-color:var(--yaw); }}
.card h3 {{ margin-bottom:0.4rem; }} .card p {{ margin:0; font-size:0.9rem; color:var(--muted); }}
.note {{ background:var(--raise); border:1px solid var(--line); border-radius:8px; padding:1.05rem 1.25rem; margin-top:1.5rem; font-size:0.92rem; }}
.note strong {{ font-family:var(--font-head); }}
ul.tight {{ padding-left:1.2rem; }} ul.tight li {{ margin:0.35rem 0; max-width:66ch; }}
footer {{ margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--line); font-size:0.8rem; color:var(--faint); font-family:var(--font-mono); }}
</style>

<div class="wrap">
<header class="masthead prose">
  <p class="eyebrow">Policy evaluation &middot; morphology comparison</p>
  <h1>Microtaur Spine Variants</h1>
  <p class="lede">Four trained <em>microtaur_velocity</em> checkpoints &mdash; the rigid baseline and
  three active-spine morphologies (<em>pitch</em>, <em>yaw</em>, <em>roll</em>) &mdash; run in the
  mjlab play environment under identical commands. Forward sweep 0.08&ndash;0.20&nbsp;m/s &times; 3
  seeds &times; 18&nbsp;s, plus a &plusmn;0.2&nbsp;rad/s yaw sweep. 108 rollouts, 0 falls.</p>
  <div class="legend">
    <span><i class="i-rigid"></i>rigid &mdash; no spine</span>
    <span><i class="i-pitch"></i>pitch &mdash; active pitch spine</span>
    <span><i class="i-yaw"></i>yaw &mdash; active yaw spine (residual)</span>
    <span><i class="i-roll"></i>roll &mdash; active twist spine</span>
  </div>
</header>

<section>
  <p class="eyebrow">Bottom line</p>
  <h2>The yaw spine wins on every axis</h2>
  <div class="verdict">
    <div class="card win"><h3 class="v-yaw">yaw</h3><p>Lowest CoT (~0.8 vs ~1.4), tightest tracking
      (&lt;0.013&nbsp;m/s at every speed), lowest tilt (~1.9&deg;), and the only variant that
      realises a commanded yaw rate (~0.9&times;). Its spine is a <strong>near-lossless motor</strong>
      (~6% negative work). Heaviest at 0.72&nbsp;kg and still most efficient.</p></div>
    <div class="card"><h3 class="v-roll">roll</h3><p>Efficient when moving (CoT⁺ ~0.8, close to yaw)
      but <strong>freezes below 0.12&nbsp;m/s</strong> &mdash; locks into a tilted stand. Usable
      envelope 0.12&ndash;0.20. Turns moderately (~0.76&times;). Its spine is a <strong>near-pure
      damper</strong> &mdash; ~98% of its work is braking.</p></div>
    <div class="card"><h3 class="v-rigid">rigid</h3><p>Solid baseline. CoT⁺ ~1.4, tracks well at
      low speed then undershoots, tilt ~2.5&deg;, weak yaw tracking (~0.4&times;).</p></div>
    <div class="card"><h3 class="v-pitch">pitch</h3><p>The only variant that didn't learn a trot &mdash;
      it <strong>paces</strong> (lateral pairs). No efficiency gain over rigid, <strong>worst forward
      tracking</strong>, highest tilt (~3.9&deg;) and CoM roll, and cannot yaw-track. Its spine is a
      <strong>lossy spring</strong> (net brake) that churns power without net propulsion.</p></div>
  </div>
  <div class="tablewrap">{head_tbl}</div>
  <p style="font-size:0.82rem;color:var(--muted);margin-top:0.6rem;font-family:var(--font-mono)">
  Means over each variant's usable speed envelope. "yaw-rate realised" = achieved / commanded
  &omega;<sub>z</sub> at &plusmn;0.2&nbsp;rad/s (1.0 = perfect).</p>
</section>

<section>
  <p class="eyebrow">Forward locomotion</p>
  <h2>Cost of transport &amp; tracking</h2>
  <p>Both active spines that actually locomote &mdash; yaw and roll &mdash; transport the robot at
  roughly <strong>half</strong> the mechanical cost of the rigid and pitch designs, and the effect
  is flat across the trained speed band. yaw additionally tracks the forward command almost exactly;
  rigid and pitch increasingly undershoot as speed rises. The hollow marker is roll at
  0.08&nbsp;m/s, where it stops.</p>
  {FIG('c1_cot_tracking', 'Left: CoT⁺ vs commanded speed (hollow = roll frozen at 0.08). Right: forward-velocity tracking error; shaded band is ±FORWARD_TRACKING_SIGMA (0.04 m/s).')}
  <div class="tablewrap">{sp_tbl}</div>
  <p style="font-size:0.82rem;color:var(--muted);margin-top:0.6rem;font-family:var(--font-mono)">CoT⁺ by commanded speed. ✳ = &gt;50% of steps below v_eps (robot not locomoting).</p>

  <h3 style="margin-top:2rem">Normalised per motor</h3>
  <p>The spine variants carry 9 motors to rigid's 8, so part of any comparison is just the divisor.
  Dividing CoT⁺ by motor count keeps yaw and roll ~45% below rigid/pitch &mdash; and comparing
  <em>only the 8 leg motors</em> (excluding spine work entirely) still leaves yaw and roll ahead,
  so the yaw spine is genuinely making the leg gait cheaper, not just diluting the average.</p>
  <div class="tablewrap">{cpm_tbl}</div>
  {FIG('c6_cot_per_motor', 'Left: CoT⁺ ÷ number of actuated motors (8 rigid, 9 spine). Right: leg-only CoT⁺ ÷ 8 — spine work removed from the numerator for every variant.')}
</section>

<section>
  <p class="eyebrow">Trunk &amp; spine</p>
  <h2>Attitude and how much the spine moves</h2>
  <p>yaw keeps the trunk flattest and steadiest. pitch runs ~3&ndash;4&deg; off level (nose-up in
  pitch, plus roll sway) the whole time. roll is level once moving but tips to ~7&deg; during its
  low-speed freeze. Spine peak-to-peak: pitch swings its spine ~10&deg; to little effect; yaw's
  spine excursion grows with speed; roll's sits around 6&deg;.</p>
  {FIG('c2_attitude_spine', 'Left: mean trunk tilt vs speed. Right: spine-joint peak-to-peak (○) and mean |angle| (△) over each forward rollout.')}
  {FIG('c3_traces', 'Body-frame forward velocity and trunk tilt over an 18 s seed-0 rollout at 0.16 m/s.')}
</section>

<section>
  <p class="eyebrow">CoM orientation</p>
  <h2>Roll, pitch and yaw of the centre of mass</h2>
  <p>Tracked at the whole-robot CoM, world frame. <strong class="v-yaw">yaw</strong> is the
  steadiest on all three axes: smallest roll oscillation, pitch within ~0.5&deg; of level, and
  almost no heading drift. <strong class="v-rigid">rigid</strong> loses ~60&deg; of heading over
  18&nbsp;s and rolls ±5&deg;; <strong class="v-pitch">pitch</strong> holds a persistent ~+3&deg;
  pitch offset with the largest roll swing; <strong class="v-roll">roll</strong> sits ~1.5&deg;
  nose-down and drifts +2&deg;/s in heading.</p>
  {FIG('c7_com_rpy', 'CoM roll oscillation amplitude (1σ), CoM pitch mean (○) and amplitude (△), and CoM heading-drift rate vs commanded speed.')}
  {FIG('c8_com_rpy_traces', 'CoM roll / pitch / unwrapped-yaw over an 18 s seed-0 rollout at 0.16 m/s.')}
</section>

<section>
  <p class="eyebrow">Turning</p>
  <h2>Only the yaw spine actually turns</h2>
  <p>Commanded &plusmn;0.2&nbsp;rad/s at 0.12&nbsp;m/s forward. The yaw-spine variant realises
  ~90&ndash;98% of the commanded yaw rate, symmetric in both directions, and does it at the lowest
  cost of transport of any manoeuvre by any variant (CoT⁺ ~0.83). roll manages ~72&ndash;80%.
  rigid only reaches ~30&ndash;50% and pitch is erratic &mdash; several rollouts rotate opposite
  the command.</p>
  {FIG('c4_turning', 'Left: realised ÷ commanded yaw rate per variant (mean over 3 seeds, each command sign). Right: unwrapped heading, seed 0 (solid = +0.2 rad/s, dashed = −0.2).')}
</section>

<section>
  <p class="eyebrow">Spine power</p>
  <h2>Two of the three spines are brakes</h2>
  <p>Splitting each spine's per-step mechanical power <code>P = &tau;<sub>spine</sub> &middot;
  q&#775;<sub>spine</sub></code> into motoring work (<code>P &gt; 0</code>) and braking work
  (<code>P &lt; 0</code>) shows that only the yaw spine behaves like an actuator; the pitch and
  roll spines mostly absorb energy.</p>
  <div class="tablewrap">{spw_tbl}</div>
  <p style="font-size:0.82rem;color:var(--muted);margin-top:0.6rem;font-family:var(--font-mono)">
  Steady-state forward walking, usable speeds, seed-averaged. neg-work share =
  |W&#8315;| / (|W&#8314;| + |W&#8315;|): 1 = pure damper, 0 = pure motor.</p>
  <ul class="tight">
    <li><strong class="v-roll">roll</strong> &mdash; the twist spine is a <strong>near-pure
    damper</strong>: 97&ndash;99% of the energy it touches is negative, it opposes its own motion
    ~85&ndash;93% of steps, does essentially zero positive work, and absorbs ~40&nbsp;mW
    continuously. This is why roll's <code>CoT_abs</code> sits so far above its <code>CoT⁺</code>:
    the legs drive a body twist each stride and the spine brakes it.</li>
    <li><strong class="v-pitch">pitch</strong> &mdash; a <strong>lossy spring</strong>: it cycles
    real power both ways each stride (~16&nbsp;mW out, ~23&nbsp;mW absorbed) damping the trunk's
    fore-aft pitch oscillation, net negative. Consistent with pitch getting no locomotion benefit
    &mdash; the spine churns energy without net propulsion.</li>
    <li><strong class="v-yaw">yaw</strong> &mdash; a <strong>near-lossless motor</strong>: only
    5&ndash;7% negative work, net <em>positive</em> and growing with speed. It drives a small yaw
    motion phased with the gait that almost never fights it. This is the mechanism behind yaw's low
    cost of transport &mdash; the other two spines burn energy fighting themselves; yaw's doesn't,
    and its legs also do less work at matched speed.</li>
  </ul>
  {FIG('c5_spine_power', 'Left: spine actuator power vs speed — motoring rate (solid) above zero, braking rate (dashed) below. Right: negative-work share; shaded band ≥0.92 is "pure damper".')}
  <div class="note"><strong>What "negative work" costs.</strong> This is actuator mechanical power
  <code>&tau;&middot;q&#775;</code>. Negative <code>&tau;&middot;q&#775;</code> at an XL330
  position servo means it is resisting motion; that energy is dissipated in the driver, not
  usefully regenerated &mdash; a real efficiency cost, which is why <code>CoT_abs</code> counts it
  and <code>CoT⁺</code> does not. Treat the negative-work <em>shares</em> as solid; the absolute
  net wattages are indicative (the coupled five-bar plus passive joints make exact energy
  bookkeeping fiddly).</div>
</section>

<section>
  <p class="eyebrow">Gait</p>
  <h2>Three trot, pitch paces</h2>
  <p>From the foot-contact timing (verified against the hip geometry): <strong class="v-rigid">rigid</strong>,
  <strong class="v-yaw">yaw</strong> and <strong class="v-roll">roll</strong> all learned a
  <strong>trot</strong> &mdash; diagonal feet (FL+HR, FR+HL) swing together. <strong class="v-pitch">pitch</strong>
  learned a <strong>pace</strong> &mdash; the two <em>left</em> feet swing together, then the two
  <em>right</em>. This holds at every commanded speed (0.08&ndash;0.20&nbsp;m/s) and every seed.
  A pace has no diagonal bracing, so the trunk rolls toward each swinging side &mdash; which lines
  up with pitch also having the largest CoM roll oscillation, the worst forward tracking, and no
  usable yaw control. (If a video shows this checkpoint trotting, it is a different checkpoint or
  a different &mdash; e.g. training-time, domain-randomised &mdash; environment; gait mode is a
  known bistable outcome of locomotion RL.)</p>
  {FIG('g1_gait_diagram', 'Stance bars per foot over 4 s of an 0.14 m/s rollout. Coloured rows are one synchronised pair, grey the other. rigid/yaw/roll sync the diagonals (trot); pitch syncs the lateral pairs (pace). diag/lat = mean contact correlation for the diagonal vs lateral pairs.')}
</section>

<section>
  <p class="eyebrow">Static stability</p>
  <h2>All dynamic — but yaw's CoM barely moves</h2>
  <p>From a dedicated short rollout per variant at 0.14&nbsp;m/s, logging the four foot-site
  positions, per-foot contact, and the whole-robot CoM ground projection. Every variant keeps
  only two feet down ~85% of the cycle (a trot for three of them, a pace for pitch), a third foot
  down only 12&ndash;25% of the time &mdash; so the CoM sits outside the support most of the time
  for all of them. None is <em>statically</em> stable; they are dynamically balanced. Static-margin
  numbers therefore don't separate the variants.</p>
  <p>What does separate them is <strong>how much the CoM moves</strong>.
  <strong class="v-yaw">yaw</strong> walks with a near-stationary CoM &mdash; lateral RMS
  <strong>2.6&nbsp;mm</strong> vs 7.5&ndash;13&nbsp;mm for the others, and a slow body-drift
  component of just 2&nbsp;mm. <strong class="v-rigid">rigid</strong> wallows the most: a slow
  &plusmn;28&nbsp;mm side-to-side sway with occasional deep excursions it catches dynamically.
  pitch and roll oscillate at stride frequency in between.</p>
  <div class="tablewrap">{stab_tbl}</div>
  {FIG('s3_summary', 'CoM lateral RMS, the slow (0.6 s-smoothed) lateral wander, the 1st-percentile static margin (near-topple depth — comparable across variants), and the fraction of the cycle with ≥3 feet down.')}
  {FIG('s2_margin', 'Top: signed distance from CoM projection to the support (positive = inside a ≥3-foot polygon; negative otherwise). Bottom: heading-detrended CoM lateral offset — rigid slow-wallows, yaw stays within ±6 mm.')}
  {FIG('s1_polygons', 'Support polygon (hull of stance feet) and CoM ground projection (+) at eight phases of one gait cycle, per variant. Frame centred on the CoM; +x forward, +y left. Green + = CoM inside a ≥3-foot polygon, red + = 2-foot (line) support or CoM outside.')}
</section>

<section>
  <p class="eyebrow">Method &amp; caveats</p>
  <h2>How this was produced</h2>
  <ul class="tight">
    <li>Each variant runs its own <code>env_cfgs.py</code> (installed for the run, restored after)
    at the <code>ENV_CFG_REVISION</code> its checkpoint was trained against:
    <ul class="tight">
      <li>rigid &mdash; <code>{REV['rigid']}</code></li>
      <li>pitch &mdash; <code>{REV['pitch']}</code></li>
      <li>yaw &mdash; <code>{REV['yaw']}</code></li>
      <li>roll &mdash; <code>{REV['roll']}</code></li>
    </ul></li>
    <li>All three spine checkpoints trained at commit <code>b281dc3</code> with uncommitted
    edits; the exact env was reconstructed from each run's captured <code>git/microtaur.diff</code>.
    For yaw this matters &mdash; the repo's committed env was later switched to M288 motors, so the
    M077 table the policy trained with was restored.</li>
    <li>Play mode: observation corruption and randomised resets off &rarr; a fixed (seed, command)
    reproduces a rollout byte-for-byte. Seeds differ only via startup domain randomisation
    (foot friction, base CoM, encoder bias).</li>
    <li>CoT denominator floors <code>|v_body_x|</code> at 0.02&nbsp;m/s; roll's 0.08&nbsp;m/s rows
    are near-zero motion, so their CoT is not meaningful (flagged, not dropped).</li>
    <li>Every variant reproduces its training <code>forward_velocity_m_s</code> (~0.125) at
    cmd&nbsp;0.14, and logs 0 terminations across all 108 rollouts.</li>
  </ul>
</section>

<footer>
  microtaur_velocity &middot; 4 variants &times; (21 forward + 6 yaw) rollouts &middot; 50 Hz &middot;
  RTX 2060 / MuJoCo-Warp &middot; 2026-09-09
</footer>
</div>
"""

out = RO / "compare.html"
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out}  ({len(HTML)/1024:.0f} KB)")
print(H.round(3).to_string(index=False))
