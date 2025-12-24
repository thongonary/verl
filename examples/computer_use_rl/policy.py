from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor


class TinyCnnPolicy(nn.Module):
    """Minimal pixel->discrete-action policy.

    Input: (B, 3, H, W) uint8 or float32
    Output: logits over discrete action ids.
    """

    def __init__(self, num_actions: int, obs_hw: tuple[int, int] = (84, 84)):
        super().__init__()
        h, w = obs_hw
        self.obs_hw = (h, w)

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer flatten dim
        with torch.no_grad():
            dummy = torch.zeros((1, 3, h, w), dtype=torch.float32)
            out = self.net(dummy)
            flat_dim = out.shape[-1]

        self.head = nn.Linear(flat_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalize uint8 pixels to [0,1]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        x = self.net(obs)
        return self.head(x)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> PolicyOutput:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return PolicyOutput(action=action, log_prob=log_prob)

    def act_with_grad(self, obs: torch.Tensor) -> PolicyOutput:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return PolicyOutput(action=action, log_prob=log_prob)
