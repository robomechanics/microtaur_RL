# v3 TensorDict fix

MJLab's `RslRlVecEnvWrapper` returns a `TensorDict` with batch shape
`[num_envs]`. The full TensorDict must be retained for RSL-RL teacher
inference, while aligned student distillation needs the `"actor"` feature tensor
inside it.

v2 incorrectly passed the full TensorDict directly to `AlignedStudentObsBuilder`,
which made the builder see shape `(num_envs,)`.

v3 fixes this by:

- preserving the full observation object in `get_initial_obs()` and `step_env()`;
- adding `get_actor_obs_tensor()` for explicit `"actor"` extraction;
- using that helper in BC, DAgger, student evaluation, and the viewer;
- continuing to pass the full TensorDict to the teacher.

For rigid, the BC startup line should now show:

```text
action_dim=8 actor_obs_dim=36 student_obs_dim=35
```

For active-spine variants:

```text
action_dim=9 actor_obs_dim=39 student_obs_dim=38
```
