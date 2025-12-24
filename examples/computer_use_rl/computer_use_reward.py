from __future__ import annotations

import json
from typing import Any, Optional


ParsedAction = dict[str, Any]


def _parse_actions(solution_str: str, max_actions: int) -> list[ParsedAction]:
    actions: list[ParsedAction] = []
    for line in solution_str.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        action_type = obj.get("type")
        if action_type == "move":
            dx = int(obj.get("dx", 0))
            dy = int(obj.get("dy", 0))
            actions.append({"type": "move", "dx": dx, "dy": dy})
        elif action_type == "click_left":
            actions.append({"type": "click_left"})
        elif action_type == "key":
            key = obj.get("key")
            if isinstance(key, str) and key:
                actions.append({"type": "key", "key": key})
        if len(actions) >= max_actions:
            break
    return actions


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    *,
    vnc_host: str = "127.0.0.1",
    vnc_port: int = 5901,
    max_actions: int = 8,
    step_sleep_s: float = 0.05,
    dummy_reward: float = 1.0,
) -> dict[str, Any] | float:
    """Custom reward for computer-use GRPO.

    Minimal behavior (as requested):
    - Connect to the already-running VNC VM
    - Execute up to `max_actions` parsed actions (if any)
    - Return a dummy terminal reward

    Notes:
    - This runs inside the reward manager process; keep it simple and robust.
    - If parsing yields no actions, we still return a reward (you can change that).
    """

    if data_source != "computer_use_dummy":
        # Let other datasets fall back to their default scoring logic.
        # Returning 0 avoids crashing; you can expand later.
        return 0.0

    actions = _parse_actions(solution_str, max_actions=max_actions)

    # Execute actions (best-effort). We intentionally do not attempt VM reset.
    executed = 0
    try:
        from examples.computer_use_rl.vnc_env import KvmVncEnv, VncAction

        env = KvmVncEnv(
            host=vnc_host,
            port=int(vnc_port),
            resize_hw=(84, 84),
            step_sleep_s=step_sleep_s,
            shutdown_reactor_on_close=False,
        )
        try:
            env.reset()
            for act in actions:
                act_type = act.get("type")
                if act_type == "move":
                    env.step(
                        VncAction(
                            kind="move",
                            dx=int(act.get("dx", 0)),
                            dy=int(act.get("dy", 0)),
                        )
                    )
                elif act_type == "click_left":
                    env.step(VncAction(kind="click_left"))
                elif act_type == "key":
                    key = act.get("key")
                    if isinstance(key, str) and key:
                        env.step(VncAction(kind="key", key=key))
                executed += 1
        finally:
            env.close()
    except Exception as exc:
        return {
            "score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "parsed_actions": len(actions),
            "executed_actions": executed,
        }

    return {
        "score": float(dummy_reward),
        "parsed_actions": len(actions),
        "executed_actions": executed,
    }
