# Minimal Computer-Use RL (KVM/VNC, pixels + structured actions)

This is a *minimal* demo that connects to an already-running KVM VM via VNC, collects a multi-step episode of **pixel observations** and **structured mouse/keyboard actions**, assigns a **dummy terminal reward**, and runs a **single REINFORCE policy update**.

## Assumptions

- Your VM is already running and exposes VNC on `:1`.
- VNC TCP port for display `:1` is typically `5901`.
- If you are running inside Docker/devcontainer, `127.0.0.1` refers to the container itself. You may need to use the host gateway IP (often `172.17.0.1`) or run the container with host networking.

## Install deps

```bash
pip install -r examples/computer_use_rl/requirements.txt
```

## Run one rollout + one update

If the container can reach the host VNC on `127.0.0.1:5901`:

```bash
python -m examples.computer_use_rl.run_vnc_reinforce --vnc-host 127.0.0.1 --vnc-port 5901 --steps 32 --episodes 1
```

If you’re inside Docker and VNC is on the host, try the Docker bridge gateway:

```bash
python -m examples.computer_use_rl.run_vnc_reinforce --vnc-host 172.17.0.1 --vnc-port 5901
```

## What this implements

- **Observation**: screenshot RGB pixels (resized to 84×84)
- **Actions**: a small discrete set mapped to structured commands:
  - move left/right/up/down
  - left click
  - press enter
- **Reward**: dummy terminal reward (default `1.0`)
- **Update**: single-step REINFORCE: `loss = -R * sum_t log π(a_t|s_t)`

This is intentionally minimal; once the VNC loop is stable, we can swap the policy/model to your real agent model and route rewards from a real task success signal.
