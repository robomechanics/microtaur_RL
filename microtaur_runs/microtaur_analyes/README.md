Microtaur RL simulation package
================================

This archive was assembled for running analyses on a different training set.

simulation_scripts/
    Scripts that spawn MuJoCo/mjlab, drive/play trained policies, or run
    sweeps to produce raw rollout data (rollout_log.py, render_*.py,
    view_*.py, *_test.py, play/replay scripts, spine_ablation*.py sweep
    drivers, run_*.sh sweep shells, morphology variant registry, etc).
    These require the `src/microtaur_velocity` package, `pyproject.toml`,
    and `requirements.txt` from the original repo to actually run -- those
    were NOT copied into this package since only "simulation test scripts"
    were requested.

analysis_scripts/
    Scripts that consume existing rollout CSV/npz/json output and produce
    figures, phase portraits, comparison tables, and HTML reports
    (analyze_rollouts.py, compare_*.py, *_figs.py, resummarize.py,
    waypoint_analysis.py, spine_function.py, poincare_stability.py,
    paper_figures/, etc).


A handful of scripts blend both roles (e.g. poincare_stability.py and the
spine_ablation*.py files both run rollouts AND analyze them); they were
placed by primary purpose. Re-file anything that doesn't match your intent.
