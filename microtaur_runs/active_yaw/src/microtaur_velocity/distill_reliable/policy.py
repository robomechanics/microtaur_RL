from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation {name!r}")


class StudentPolicy(nn.Module):
    """Small deployable MLP.

    Default reliable first pass:
      35 -> 128 -> 64 -> 8

    Final ESP32 candidate:
      35 -> 64 -> 64 -> 8

    Output is raw 8D Microtaur swing/lift action in [-1, 1].
    """

    def __init__(
        self,
        obs_dim: int = 35,
        action_dim: int = 8,
        hidden_dims: tuple[int, ...] = (128, 64),
        activation: str = "elu",
        output_tanh: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.activation = str(activation)
        self.output_tanh = bool(output_tanh)

        layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(self.activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        if self.output_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.detach().reshape(-1)
        std = std.detach().reshape(-1)
        if mean.numel() != self.obs_dim or std.numel() != self.obs_dim:
            raise ValueError(
                f"Normalization dim mismatch. Expected {self.obs_dim}, "
                f"got mean={mean.numel()}, std={std.numel()}."
            )
        self.obs_mean.copy_(mean.to(self.obs_mean.device, dtype=self.obs_mean.dtype))
        self.obs_std.copy_(torch.clamp(std.to(self.obs_std.device, dtype=self.obs_std.dtype), min=1e-4))

    def normalized_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalized_obs(obs))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
