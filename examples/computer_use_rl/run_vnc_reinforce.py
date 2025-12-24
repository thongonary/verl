from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from examples.computer_use_rl.policy import TinyCnnPolicy
from examples.computer_use_rl.vnc_env import KvmVncEnv, VncAction


@dataclass(frozen=True)
class DiscreteAction:
    id: int
    spec: VncAction


def build_action_set(move_px: int) -> list[DiscreteAction]:
    # Minimal structured action set: 4-way move + click + enter.
    return [
        DiscreteAction(0, VncAction(kind="move", dx=-move_px, dy=0)),
        DiscreteAction(1, VncAction(kind="move", dx=move_px, dy=0)),
        DiscreteAction(2, VncAction(kind="move", dx=0, dy=-move_px)),
        DiscreteAction(3, VncAction(kind="move", dx=0, dy=move_px)),
        DiscreteAction(4, VncAction(kind="click_left")),
        DiscreteAction(5, VncAction(kind="key", key="enter")),
    ]


def to_torch_obs(obs_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    # HWC uint8 -> 1x3xHxW uint8
    # PIL-backed arrays can be read-only; copy to ensure torch tensor is safe.
    obs_chw = np.transpose(obs_hwc, (2, 0, 1)).copy()
    obs = torch.from_numpy(obs_chw).unsqueeze(0).to(device)
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal VNC/KVM computer-use REINFORCE demo")
    parser.add_argument("--vnc-host", default="127.0.0.1")
    parser.add_argument("--vnc-port", type=int, default=5901, help="VNC TCP port (display :1 -> 5901)")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--move-px", type=int, default=40)
    parser.add_argument("--resize", type=int, default=84)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dummy-reward", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    action_set = build_action_set(args.move_px)
    num_actions = len(action_set)

    env = KvmVncEnv(
        host=args.vnc_host,
        port=args.vnc_port,
        resize_hw=(args.resize, args.resize),
        step_sleep_s=0.05,
        connect_timeout_s=10.0,
        shutdown_reactor_on_close=True,
    )

    policy = TinyCnnPolicy(num_actions=num_actions, obs_hw=(args.resize, args.resize)).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            log_probs: list[torch.Tensor] = []

            for t in range(args.steps):
                obs_t = to_torch_obs(obs, device)
                out = policy.act_with_grad(obs_t)

                action_id = int(out.action.item())
                action = action_set[action_id].spec

                obs = env.step(action)
                log_probs.append(out.log_prob)

            # Sparse terminal reward: reward only at end.
            terminal_reward = torch.tensor(float(args.dummy_reward), device=device)

            # REINFORCE: maximize E[R * sum log pi(a_t|s_t)]
            # loss = -R * sum log pi
            loss = -(terminal_reward * torch.stack(log_probs).sum())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(
                f"episode={ep} steps={args.steps} reward={terminal_reward.item():.4f} loss={loss.item():.6f}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()
