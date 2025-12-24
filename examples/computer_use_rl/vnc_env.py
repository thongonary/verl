from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VncAction:
    """A minimal structured action for a computer-use environment."""

    kind: str
    dx: int = 0
    dy: int = 0
    key: Optional[str] = None


class KvmVncEnv:
    """Minimal VNC-backed environment.

    This is intentionally tiny and synchronous:
    - Observation: RGB pixels (np.uint8 HxWx3)
    - Action: structured mouse/keyboard command
    - Reward: provided externally (we keep it env-agnostic)

    It assumes a VNC server is already running (your KVM VM). For display :1,
    the TCP port is typically 5901.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        resize_hw: tuple[int, int] = (84, 84),
        step_sleep_s: float = 0.05,
        connect_timeout_s: float = 10.0,
        shutdown_reactor_on_close: bool = False,
    ):
        self.host = host
        self.port = int(port)
        self.resize_hw = resize_hw
        self.step_sleep_s = float(step_sleep_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.shutdown_reactor_on_close = bool(shutdown_reactor_on_close)

        self._client = None
        self._cursor_xy = None  # lazily initialized to screen center

    def connect(self) -> None:
        try:
            from vncdotool import api  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "vncdotool is required for VNC control. "
                "Install with: pip install -r examples/computer_use_rl/requirements.txt"
            ) from exc

        deadline = time.time() + self.connect_timeout_s
        last_exc = None
        while time.time() < deadline:
            try:
                # vncdotool supports host::port format
                self._client = api.connect(f"{self.host}::{self.port}")
                return
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                time.sleep(0.5)

        raise RuntimeError(
            f"Failed to connect to VNC at {self.host}:{self.port}. "
            "If you are running inside Docker, note that '127.0.0.1' is the container itself; "
            "use the host gateway IP or run with --network=host."
        ) from last_exc

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

        if self.shutdown_reactor_on_close:
            # vncdotool spins up a Twisted reactor thread on first connect().
            # Some one-shot scripts may linger unless we explicitly stop it.
            try:
                from vncdotool import api  # type: ignore

                api.shutdown()
            except Exception:
                pass

    def reset(self) -> np.ndarray:
        if self._client is None:
            self.connect()
        obs = self._capture_obs()
        # Initialize cursor to center of the *original* screen resolution.
        # We don’t know exact screen dims, so we assume typical 1024x768-ish.
        # The cursor is only used for relative moves; absolute value is clamped.
        if self._cursor_xy is None:
            self._cursor_xy = (512, 384)
        return obs

    def step(self, action: VncAction) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("Environment not connected. Call reset() first.")

        if action.kind == "move":
            self._apply_mouse_move(action.dx, action.dy)
        elif action.kind == "click_left":
            self._apply_click_left()
        elif action.kind == "key":
            if not action.key:
                raise ValueError("key action requires action.key")
            self._client.keyPress(action.key)
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        if self.step_sleep_s > 0:
            time.sleep(self.step_sleep_s)
        return self._capture_obs()

    def _apply_mouse_move(self, dx: int, dy: int) -> None:
        assert self._cursor_xy is not None
        x, y = self._cursor_xy
        x = int(np.clip(x + dx, 0, 4095))
        y = int(np.clip(y + dy, 0, 4095))
        self._cursor_xy = (x, y)
        self._client.mouseMove(x, y)

    def _apply_click_left(self) -> None:
        # vncdotool 1.2.0 exposes mouseDown/mouseUp (no mouseRelease)
        self._client.mouseDown(1)
        self._client.mouseUp(1)

    def _capture_obs(self) -> np.ndarray:
        # vncdotool's captureScreen API writes to a file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self._client.captureScreen(tmp.name)
            img = Image.open(tmp.name).convert("RGB")
        if self.resize_hw is not None:
            h, w = self.resize_hw
            img = img.resize((w, h), resample=Image.BILINEAR)
        obs = np.asarray(img, dtype=np.uint8)
        return obs
