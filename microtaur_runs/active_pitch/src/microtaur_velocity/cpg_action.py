"""CPG-RL leg action for Microtaur: one oscillator per leg, sinusoidal joint targets.

Each leg i carries its own phase phi_i. The oscillator drives the repo's five-bar
(swing, lift) coordinates with a single sinusoid each, and those map to the two
motor targets of the leg exactly as the hand-written CPG and the direct-action
policy do:

    swing_i = c_swing_i + A_swing_i * (1 + g_a * u_swing_i)  * sin(phi_i)
    lift_i  = c_lift_i  + A_lift_i  * (1 + g_a * u_lift_i)   * sin(phi_i + psi_i)
    a_i     = STAND_A_i + sign_i * (swing_i - lift_i)
    e_i     = STAND_E_i + sign_i * (swing_i + lift_i)
    d(phi_i)/dt = 2*pi * f_nom * (1 + g_f * u_freq_i)

The policy outputs three numbers per leg, u = (u_swing, u_lift, u_freq) in [-1, 1],
so the leg action is 12-dimensional instead of 8. The spine variants keep their
separate 1-D spine action term unchanged.

The nominal values (f_nom, A_*, c_*, per-leg phase offsets and the lift lag psi_i)
come from fitting this CPG to the flat-terrain policy's gait
(cpg_rl/fit_cpg_from_flat.py), so an untrained policy already walks roughly like
the flat policy it was fitted to. MICROTAUR_CPG_FIT points at that JSON file.

The five-bar safety filter, the joint-target delay and the target clamping are
inherited unchanged from MicrotaurOlympusWalkAction.
"""

from __future__ import annotations

import json
import math
import os

import torch

# Fraction by which the policy can scale an amplitude / the gait frequency.
CPG_AMPLITUDE_GAIN = float(os.environ.get("MICROTAUR_CPG_AMPLITUDE_GAIN", "0.5"))
CPG_FREQUENCY_GAIN = float(os.environ.get("MICROTAUR_CPG_FREQUENCY_GAIN", "0.5"))
CPG_MIN_AMPLITUDE_SCALE = 0.0
CPG_MAX_AMPLITUDE_SCALE = 1.0 + CPG_AMPLITUDE_GAIN


def load_fit(path: str | None = None) -> dict:
  """Read the CPG fit produced by cpg_rl/fit_cpg_from_flat.py."""
  path = path or os.environ.get("MICROTAUR_CPG_FIT")
  if not path:
    raise ValueError("Set MICROTAUR_CPG_FIT to the fitted-CPG JSON file.")
  with open(path) as handle:
    return json.load(handle)


def build_cpg_action_term(cfg, env):
  """Build the CPG leg-action term for the direct-action cfg `cfg`.

  Called from MicrotaurOlympusWalkActionCfg.build when MICROTAUR_CPG is set. The
  nominal CPG values are read from MICROTAUR_CPG_FIT here rather than carried on the
  cfg, because the CLI parser rebuilds cfg objects and would drop extra fields.
  """
  return _resolve_class()(cfg, env)


def _make_cpg_action_class():
  """Defined lazily so importing this module does not import env_cfgs."""
  from microtaur_velocity.env_cfgs import MicrotaurOlympusWalkAction

  class MicrotaurCpgAction(MicrotaurOlympusWalkAction):
    """Four leg oscillators; 12 actions (swing gain, lift gain, frequency per leg)."""

    def __init__(self, cfg, env):
      super().__init__(cfg, env)
      device = self.device
      fit = load_fit()
      legs = fit["legs"]
      tensor = lambda key: torch.tensor(
        [float(leg[key]) for leg in legs], device=device, dtype=torch.float32
      )
      self._cpg_freq_hz = float(fit["frequency_hz"])
      self._swing_amp = tensor("swing_amp")
      self._swing_offset = tensor("swing_offset")
      self._lift_amp = tensor("lift_amp")
      self._lift_offset = tensor("lift_offset")
      self._phase_offset = tensor("phase_offset")
      self._lift_lag = tensor("lift_lag")
      print(
        f"[cpg] {os.environ.get('MICROTAUR_CPG_FIT')}: f={self._cpg_freq_hz:.2f} Hz, "
        f"swing={self._swing_amp.tolist()}, phases={self._phase_offset.tolist()}"
      )
      self._dt = float(env.step_dt)
      self._phase = self._phase_offset[None, :].repeat(env.num_envs, 1).clone()
      # The base class sized both buffers by action_dim (12); the processed
      # actions are the eight motor targets.
      self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=device)
      self._processed_actions = torch.zeros(env.num_envs, 8, device=device)
      self._cpg_frequency = torch.full(
        (env.num_envs, 4), self._cpg_freq_hz, device=device
      )

    @property
    def action_dim(self) -> int:
      return 12

    @property
    def phase(self) -> torch.Tensor:
      """Leg phases, shape (num_envs, 4)."""
      return self._phase

    @property
    def cpg_frequency(self) -> torch.Tensor:
      """Commanded leg frequencies in Hz, shape (num_envs, 4)."""
      return self._cpg_frequency

    def process_actions(self, actions: torch.Tensor):
      if actions.shape[-1] != self.action_dim:
        raise ValueError(
          f"MicrotaurCpgAction expects {self.action_dim} actions "
          f"(4 legs x swing/lift/frequency); got {actions.shape[-1]}."
        )

      actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
      actions = torch.clamp(actions, -1.0, 1.0)
      u = actions.view(-1, 4, 3)
      u_swing, u_lift, u_freq = u[..., 0], u[..., 1], u[..., 2]

      reset = self._walk_env.episode_length_buf <= 1
      if torch.any(reset):
        self._phase[reset] = self._phase_offset[None, :].expand(int(reset.sum()), 4)

      frequency = self._cpg_freq_hz * (1.0 + CPG_FREQUENCY_GAIN * u_freq)
      frequency = frequency.clamp(min=0.1)
      self._cpg_frequency = frequency
      self._phase = torch.remainder(
        self._phase + 2.0 * math.pi * frequency * self._dt, 2.0 * math.pi
      )

      amp_scale = lambda u_: (1.0 + CPG_AMPLITUDE_GAIN * u_).clamp(
        CPG_MIN_AMPLITUDE_SCALE, CPG_MAX_AMPLITUDE_SCALE
      )
      swing = self._swing_offset + self._swing_amp * amp_scale(u_swing) * torch.sin(self._phase)
      lift = self._lift_offset + self._lift_amp * amp_scale(u_lift) * torch.sin(
        self._phase + self._lift_lag
      )

      requested = torch.empty(actions.shape[0], 8, device=actions.device)
      stand = self._stand.view(1, 4, 2)
      signs = self._leg_signs[None, :]
      requested[:, 0::2] = stand[:, :, 0] + signs * (swing - lift)
      requested[:, 1::2] = stand[:, :, 1] + signs * (swing + lift)
      requested = torch.clamp(requested, self._joint_lower, self._joint_upper)

      position_resolved = self._entity.data.joint_pos[:, self._target_ids]
      velocity_resolved = self._entity.data.joint_vel[:, self._target_ids]
      position = self.resolved_to_canonical(position_resolved)
      velocity = self.resolved_to_canonical(velocity_resolved)
      self._reset_delay_state(reset, position)

      safe, blend = self._filter_targets(position, velocity, requested)
      self._requested_targets = requested
      self._safe_targets = safe
      self._filter_correction = requested - safe
      self._filter_blend = blend

      self._delay_buffer[:, self._delay_write_index, :] = safe
      read_index = torch.remainder(
        self._delay_write_index - self._delay_steps, self._delay_buffer.shape[1]
      )
      env_index = torch.arange(self._delay_buffer.shape[0], device=actions.device)
      applied = self._delay_buffer[env_index, read_index]
      self._delay_write_index = (self._delay_write_index + 1) % self._delay_buffer.shape[1]
      self._applied_targets = applied

      self._raw_actions[:] = actions
      self._processed_actions[:] = self.canonical_to_resolved(applied)

  return MicrotaurCpgAction


MicrotaurCpgAction = None  # set on first use by _resolve_class()


def _resolve_class():
  global MicrotaurCpgAction
  if MicrotaurCpgAction is None:
    MicrotaurCpgAction = _make_cpg_action_class()
  return MicrotaurCpgAction


def cpg_phase_obs(env, action_name: str = "joint_pos") -> torch.Tensor:
  """Observation: sin and cos of each leg phase, shape (num_envs, 8)."""
  term = env.action_manager.get_term(action_name)
  phase = term.phase
  return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
