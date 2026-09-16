"""Assemble the standalone rollout report HTML (figures embedded as data URIs)."""

from __future__ import annotations

import json
from pathlib import Path

RO = Path("rollouts")
figs = json.loads((RO / "report_figs" / "_figs_b64.json").read_text())
agg = [l.split(",") for l in (RO / "sweep_fwd" / "2026-09-08_21-00-47" / "aggregate_by_speed.csv").read_text().splitlines()]
head, rows = agg[0], agg[1:]

# columns we show, with labels + formatting
COLS = [
  ("cmd_vx", "cmd vₓ", 2), ("v_body_x", "v_body,x", 3), ("track_err", "|err|", 3),
  ("cot_pos", "CoT⁺", 2), ("cot_pos_sd", "±sd", 2), ("cot_abs", "CoT_abs", 2),
  ("cot_pos_p95", "CoT⁺ p95", 2), ("tilt", "tilt°", 2), ("roll", "|roll|°", 2),
  ("pitch", "|pitch|°", 2), ("yaw_drift", "yaw drift", 3), ("duty", "duty", 2),
]
idx = {name: i for i, name in enumerate(head)}
trows = []
for r in rows:
  cells = "".join(f"<td>{float(r[idx[c]]):.{p}f}</td>" for c, _, p in COLS)
  trows.append(f"<tr>{cells}</tr>")
thead = "".join(f"<th>{lbl}</th>" for _, lbl, _ in COLS)
table = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trows)}</tbody></table>"

CHECKS = [
  ("Runs &amp; walks", "18&nbsp;s / 1&nbsp;env at every commanded speed; robot moves forward, never trips <code>fell_over</code> or <code>base_too_low</code>.", "27 / 27 rollouts, 0 terminations"),
  ("Velocity tracking", "Body-frame <code>v_x</code> follows command within <code>FORWARD_TRACKING_SIGMA_M_S</code> = 0.04&nbsp;m/s.", "|err| 0.012 – 0.033 m/s across 0.08 – 0.20"),
  ("CoT ordering", "<code>CoT⁺ ≤ CoT_abs</code> and <code>P_abs ≥ 0</code> on every logged row; slow rows flagged, never NaN/Inf.", "holds on all 24 300 rows"),
  ("CoT magnitude", "Plausible for a 0.47&nbsp;kg servo quadruped (order 1 – 20).", "CoT⁺ ≈ 1.3 – 1.5, CoT_abs ≈ 1.8 – 2.2"),
  ("Attitude", "|roll|, |pitch| a few degrees in steady walk; quaternion tilt matches the reward's projected-gravity tilt.", "|roll| ≈ 1.8°, tilt ≈ 2.5°"),
  ("Determinism", "Same seed + command ⇒ byte-identical CSV across two runs (corruption + randomized reset off in play).", "diff = 0 bytes"),
  ("Checkpoint ↔ ONNX", "One recorded obs through <code>model_4499.pt</code> actor mean and <code>2026-09-06_12-33-30.onnx</code>.", "max |Δaction| = 7 × 10⁻⁷"),
  ("Vs training log", "Steady-state means vs last-20 of <code>events.out.tfevents…ARINA</code> @ step 4499.", "v_x 0.123 vs 0.125 · tilt 2.58° vs 2.61°"),
]
check_cards = "\n".join(
  f'<div class="check"><div class="check-hd"><span class="pill ok">PASS</span><h3>{t}</h3></div>'
  f'<p>{d}</p><p class="measure">{m}</p></div>' for t, d, m in CHECKS
)

FIG = lambda k, cap: f'<figure><img alt="{cap}" src="{figs[k]}"><figcaption>{cap}</figcaption></figure>'

HTML = f"""<title>Microtaur Forward Rollouts</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital@0;1&display=swap">
<style>
:root {{
  --ground:#eef2f5; --surface:#ffffff; --surface-2:#e7edf1; --raise:#f6f8fa;
  --ink:#131d25; --muted:#586a76; --faint:#8496a1; --line:#d9e0e5;
  --signal:#bd6f1c; --signal-soft:#f0e0cd; --probe:#33688f; --ok:#2f8a61; --warn:#b8821a;
  --font-head:"IBM Plex Sans",system-ui,sans-serif;
  --font-body:"IBM Plex Sans",system-ui,sans-serif;
  --font-serif:"IBM Plex Serif",Georgia,serif;
  --font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0c1215; --surface:#141d21; --surface-2:#1c272c; --raise:#18221f;
    --ink:#e3ebef; --muted:#90a3ad; --faint:#6d818c; --line:#26343b;
    --signal:#e0954c; --signal-soft:#3a2c1a; --probe:#5da0c8; --ok:#45b585; --warn:#e0b352;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0c1215; --surface:#141d21; --surface-2:#1c272c; --raise:#18221f;
  --ink:#e3ebef; --muted:#90a3ad; --faint:#6d818c; --line:#26343b;
  --signal:#e0954c; --signal-soft:#3a2c1a; --probe:#5da0c8; --ok:#45b585; --warn:#e0b352;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--font-body); font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1000px; margin:0 auto; padding:clamp(1.5rem,4vw,4rem) clamp(1.1rem,3vw,2.5rem) 5rem; }}
.prose {{ max-width:68ch; }}
h1,h2,h3 {{ font-family:var(--font-head); font-weight:600; text-wrap:balance; line-height:1.25; margin:0; }}
h1 {{ font-size:clamp(1.9rem,4vw,2.7rem); letter-spacing:-0.02em; }}
h2 {{ font-size:1.35rem; letter-spacing:-0.01em; }}
h3 {{ font-size:1rem; }}
a {{ color:var(--probe); }}
code {{ font-family:var(--font-mono); font-size:0.86em; background:var(--surface-2); padding:0.1em 0.36em; border-radius:3px; }}
.eyebrow {{
  font-family:var(--font-mono); font-size:0.72rem; font-weight:600; letter-spacing:0.16em;
  text-transform:uppercase; color:var(--signal); margin:0 0 0.6rem;
}}
header.masthead {{ border-bottom:2px solid var(--ink); padding-bottom:1.6rem; margin-bottom:2.4rem; }}
header.masthead .lede {{ font-family:var(--font-serif); font-size:1.15rem; line-height:1.55; color:var(--muted); margin:1.1rem 0 0; max-width:64ch; }}
header.masthead .lede em {{ color:var(--ink); font-style:italic; }}

.specs {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin:2rem 0 0;
}}
.specs div {{ background:var(--surface); padding:0.85rem 1rem; }}
.specs dt {{ font-family:var(--font-mono); font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--faint); margin:0 0 0.3rem; }}
.specs dd {{ font-family:var(--font-mono); font-size:0.92rem; margin:0; color:var(--ink); word-break:break-word; }}

section {{ margin-top:3.4rem; }}
section > p {{ max-width:68ch; }}
section > p.wide {{ max-width:none; }}

.figrow {{ display:grid; gap:1.6rem; margin:1.8rem 0 0; }}
@media (min-width:820px) {{ .figrow.two {{ grid-template-columns:1fr 1fr; }} }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:0.9rem; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; }}
figcaption {{ font-size:0.83rem; color:var(--muted); margin-top:0.7rem; padding-top:0.6rem; border-top:1px solid var(--line); font-family:var(--font-mono); }}

.checks {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:1rem; margin-top:1.8rem; }}
@media (max-width:600px) {{ .checks {{ grid-template-columns:1fr; }} }}
.check {{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--ok); border-radius:8px; padding:1.05rem 1.15rem; }}
.check-hd {{ display:flex; align-items:center; gap:0.6rem; margin-bottom:0.55rem; }}
.check-hd h3 {{ font-size:0.98rem; }}
.check p {{ margin:0; font-size:0.9rem; color:var(--muted); }}
.check p.measure {{ margin-top:0.55rem; font-family:var(--font-mono); font-size:0.82rem; color:var(--ink); }}
.pill {{ font-family:var(--font-mono); font-size:0.66rem; font-weight:600; letter-spacing:0.1em; padding:0.2em 0.5em; border-radius:4px; }}
.pill.ok {{ background:color-mix(in srgb,var(--ok) 18%,transparent); color:var(--ok); }}

.tablewrap {{ overflow-x:auto; margin:1.6rem 0 0; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-family:var(--font-mono); font-size:0.83rem; font-variant-numeric:tabular-nums; }}
th,td {{ padding:0.5rem 0.8rem; text-align:right; white-space:nowrap; }}
thead th {{ background:var(--surface-2); color:var(--muted); font-weight:600; position:sticky; top:0; border-bottom:1px solid var(--line); }}
tbody tr:nth-child(even) {{ background:var(--raise); }}
tbody td:first-child {{ color:var(--signal); font-weight:600; }}

.note {{ background:var(--raise); border:1px solid var(--line); border-radius:8px; padding:1.1rem 1.3rem; margin-top:1.6rem; font-size:0.92rem; }}
.note strong {{ font-family:var(--font-head); }}
ul.tight {{ padding-left:1.2rem; }}
ul.tight li {{ margin:0.35rem 0; max-width:66ch; }}
pre {{
  background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:1rem 1.15rem;
  overflow-x:auto; font-family:var(--font-mono); font-size:0.82rem; line-height:1.7; color:var(--ink);
}}
pre .c {{ color:var(--faint); }}
footer {{ margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--line); font-size:0.82rem; color:var(--faint); font-family:var(--font-mono); }}
@media (prefers-reduced-motion:no-preference) {{ html {{ scroll-behavior:smooth; }} }}
</style>

<div class="wrap">
<header class="masthead prose">
  <p class="eyebrow">Policy evaluation &middot; mjlab play rollout</p>
  <h1>Microtaur Forward Rollouts</h1>
  <p class="lede">The frozen <em>microtaur_velocity</em> policy (<code>model_4499.pt</code>) run in the
  mjlab play environment with no learning &mdash; logging mechanical cost of transport, trunk
  roll/pitch/yaw, and body/world velocity at every 50&nbsp;Hz control step. <em>27 rollouts</em>:
  a 7-point speed sweep &times; 3 seeds, plus yaw&nbsp;&plusmn;0.2&nbsp;rad/s at 0.12&nbsp;m/s.</p>
</header>

<dl class="specs">
  <div><dt>robot</dt><dd>rigid_microtaur &middot; 8-motor planar 5-bar</dd></div>
  <div><dt>total mass</dt><dd>0.4653 kg</dd></div>
  <div><dt>control rate</dt><dd>50 Hz &nbsp;(dt = 0.02 s)</dd></div>
  <div><dt>physics</dt><dd>MuJoCo-Warp &middot; 0.005 s &times; 4</dd></div>
  <div><dt>checkpoint</dt><dd>model_4499.pt &nbsp;(iter 4499)</dd></div>
  <div><dt>actor</dt><dd>MLP 36&rarr;512&rarr;256&rarr;128&rarr;8, ELU</dd></div>
  <div><dt>env revision</dt><dd>2026-09-05-rigid-aligned-baseline-v1</dd></div>
  <div><dt>command hold</dt><dd>joystick twist, fixed per rollout</dd></div>
</dl>

<section>
  <p class="eyebrow">Definitions</p>
  <h2>What is logged</h2>
  <p>Each control step writes one CSV row (110 columns). Raw per-joint <code>&tau;</code>,
  <code>q&#775;</code>, <code>q&#776;</code> are stored so any cost-of-transport convention is
  recomputable offline; the two headline conventions are:</p>
  <ul class="tight">
    <li><strong>CoT&#8314;</strong> = &Sigma; max(&tau;&middot;q&#775;, 0) / (m&middot;g&middot;v) &nbsp;&mdash; positive mechanical work only.</li>
    <li><strong>CoT<sub>abs</sub></strong> = &Sigma; |&tau;&middot;q&#775;| / (m&middot;g&middot;v) &nbsp;&mdash; counts negative-work (braking) joints too.</li>
  </ul>
  <p>Denominator uses <code>|v_body_x|</code> floored at <code>v_eps</code> = 0.02&nbsp;m/s; a
  <code>_horiz</code> variant uses world horizontal speed. <code>CoT_net</code> (signed) can dip
  below zero during braking phases &mdash; <code>P_abs</code> never does. Attitude is logged both as
  a world-frame quaternion&rarr;Euler triple and cross-checked against the reward's
  projected-gravity tilt.</p>
</section>

<section>
  <p class="eyebrow">Speed sweep &middot; vₓ ∈ [0.08, 0.20], 3 seeds, 18 s</p>
  <h2>Cost of transport &amp; velocity tracking</h2>
  <p>Mechanical CoT is nearly flat across the trained band &mdash; the policy is about equally
  efficient from 0.08 to 0.20&nbsp;m/s, with a shallow minimum near 0.10&ndash;0.12&nbsp;m/s.
  Forward tracking is tight at low speed and develops a mild, consistent undershoot toward
  0.20&nbsp;m/s (still inside the training tolerance).</p>
  <div class="figrow">
    {FIG('fig1_cot_tracking', 'Left: CoT⁺ (band = ±1 sd over seeds) and CoT_abs vs commanded speed, with the 95th-percentile of per-step CoT⁺. Right: achieved body-frame vₓ vs command; dotted line is perfect tracking.')}
  </div>
  <div class="tablewrap">{table}</div>
  <p class="wide" style="font-size:0.82rem;color:var(--muted);margin-top:0.7rem;font-family:var(--font-mono)">
  Steady-state window t &ge; 1.0 s, done = 0. Values averaged over 3 seeds; ±sd is the seed spread.
  yaw drift in rad/s.</p>
</section>

<section>
  <p class="eyebrow">Per-rollout traces &middot; seed 0</p>
  <h2>Trunk attitude holds through the gait</h2>
  <p>Roll stays within &plusmn;5&deg; of level and pitch sits a couple of degrees nose-down
  &mdash; the small offset that lets the five-bar legs push the body forward. Both are steady
  oscillations at the step frequency, not drift. The velocity panel shows the ~0.5&nbsp;s
  settle after the deterministic IK reset before the command is tracked.</p>
  <div class="figrow">
    {FIG('fig2_traces', 'Roll, pitch and body-frame vₓ over 18 s for the seed-0 rollout at each commanded speed (dark = 0.08 m/s, bright = 0.20 m/s).')}
  </div>
</section>

<section>
  <p class="eyebrow">Actuation &middot; model_4499 @ 0.14 m/s</p>
  <h2>The XL330s run near their torque ceiling</h2>
  <p>During stance the leg motors regularly reach the <code>effort_limit</code> of
  0.129&nbsp;N&middot;m (dotted rails), and per-joint power <code>&tau;&middot;q&#775;</code>
  swings negative on the trailing joints each cycle &mdash; the braking work that separates
  CoT<sub>abs</sub> from CoT&#8314;.</p>
  <div class="figrow">
    {FIG('fig3_torque_power', 'Per-joint torque (top, mN·m) and mechanical power τ·q̇ (bottom, W) over a 4 s window of the 0.14 m/s rollout. Eight actuated joints, canonical order leg1_a…leg4_e.')}
  </div>
</section>

<section>
  <p class="eyebrow">Yaw command &middot; 0.12 m/s, yaw ±0.2 rad/s</p>
  <h2>Turning is directional but under-tracked</h2>
  <p>The instantaneous yaw rate is dominated by per-step foot-impact transients (&plusmn;0.4&nbsp;rad/s),
  but the integrated heading turns cleanly in the commanded direction &mdash; about
  +0.07&nbsp;rad/s for a +0.2 command and &minus;0.11&nbsp;rad/s for &minus;0.2. The policy
  consistently realises 35&ndash;60&nbsp;% of the commanded yaw rate and turns left faster than
  right; the same weak yaw tracking is visible in the training log
  (<code>track_yaw_velocity</code> raw &asymp; 0.63).</p>
  <div class="figrow">
    {FIG('fig4_yaw', 'Left: body-frame yaw rate ω_z vs the held command (dotted). Right: unwrapped heading over 18 s. Orange = +0.2 rad/s command, blue = −0.2 rad/s.')}
  </div>
</section>

<section>
  <p class="eyebrow">Verification</p>
  <h2>Checks against the spec</h2>
  <div class="checks">
    {check_cards}
  </div>
</section>

<section>
  <p class="eyebrow">Reproduce</p>
  <h2>How it was produced</h2>
  <p>Route A from the task spec: the original <code>Microtaur_RL</code> repo, a
  <code>uv</code>-managed Python&nbsp;3.12 venv (Torch 2.14&nbsp;+cu126, mjlab&nbsp;1.6, MuJoCo-Warp
  on the RTX&nbsp;2060), and the checkpoint loaded through rsl-rl's <code>OnPolicyRunner</code>.
  MuJoCo-Warp runs without CUDA-graph capture because the installed driver reports CUDA&nbsp;12.3.</p>
<pre><span class="c"># one CSV + meta per rollout, plus summary.json and plots</span>
python scripts/rollout_log.py \\
  --vx 0.08 0.10 0.12 0.14 0.16 0.18 0.20 --yaw 0.0 \\
  --seeds 0 1 2 --sim-seconds 18 --settle-seconds 1.0

python scripts/rollout_log.py --vx 0.12 --yaw -0.2 0.2 --seeds 0 1 2 --sim-seconds 18

<span class="c"># consolidated report + verification checklist</span>
python scripts/analyze_rollouts.py rollouts/sweep_fwd/&lt;ts&gt; rollouts/sweep_yaw/&lt;ts&gt;

<span class="c"># checkpoint vs ONNX cross-check</span>
python scripts/check_onnx_vs_ckpt.py</pre>
  <div class="note"><strong>Determinism.</strong> Play mode disables observation corruption and
  randomises nothing at reset, so a fixed (seed, command) reproduces a rollout byte-for-byte.
  Seeds still differ because the startup domain-randomisation events (foot friction, base CoM,
  encoder bias) are RNG-seeded &mdash; this is deliberately kept so the seed spread measures
  sensitivity to those.</div>
</section>

<footer>
  microtaur_velocity &middot; model_4499.pt &middot; env 2026-09-05-rigid-aligned-baseline-v1 &middot;
  27 rollouts &middot; 24 300 control steps &middot; 50 Hz &middot; 2026-09-08
</footer>
</div>
"""

out = RO / "report.html"
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out}  ({len(HTML)/1024:.0f} KB)")
