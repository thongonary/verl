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
            self._press_key(action.key)
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

        if self.step_sleep_s > 0:
            time.sleep(self.step_sleep_s)
        return self._capture_obs()

    def _press_key(self, key: str) -> None:
        """Press a key or type text.

        vncdotool accepts either:
        - a single character, e.g. "a"
        - a named keysym in vncdotool.client.KEYMAP, e.g. "enter", "esc", "left"

        Model outputs often contain "Enter" / "Backspace" / "Escape" etc.
        We normalize common synonyms and fall back to typing the string.
        """

        raw = str(key)
        normalized = _normalize_vnc_key(raw)
        if normalized is None:
            # Fallback: treat as literal text; type character-by-character.
            for ch in raw:
                if ch == "\n":
                    self._client.keyPress("enter")
                elif ch == "\t":
                    self._client.keyPress("tab")
                else:
                    self._client.keyPress(ch)
            return

        # vncdotool supports chords like "ctrl+c" as "ctrl+c" (with '+').
        self._client.keyPress(normalized)

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


def _normalize_vnc_key(key: str) -> Optional[str]:
    """Normalize a user/model key string into a vncdotool keysym.

    Returns None if it should be treated as literal text.
    """

    from vncdotool.client import KEYMAP

    s = key.strip()
    if not s:
        return None

    # Common separators for key chords.
    sep = None
    for candidate in ("+", "-"):
        if candidate in s:
            sep = candidate
            break

    parts = [s] if sep is None else [p for p in s.split(sep) if p]
    normalized_parts: list[str] = []

    for part in parts:
        p = part.strip().lower()
        if not p:
            continue

        # Synonyms / normalization.
        alias = {
            "escape": "esc",
            "esc": "esc",
            "return": "enter",
            "kpenter": "enter",
            "backspace": "bsp",
            "bksp": "bsp",
            "bs": "bsp",
            "spacebar": "space",
            "space": "space",
            "delete": "delete",
            "del": "delete",
            "control": "ctrl",
            "cmd": "meta",
            "command": "meta",
            "option": "alt",
            "pageup": "pgup",
            "pagedown": "pgdn",
        }.get(p, p)

        # Single characters are fine.
        if len(alias) == 1:
            normalized_parts.append(alias)
            continue

        # Named keysyms must exist in KEYMAP.
        if alias in KEYMAP:
            normalized_parts.append(alias)
            continue

        # Unknown multi-character key: treat as literal text.
        return None

    if not normalized_parts:
        return None

    joiner = "+" if sep is None else "+"
    return joiner.join(normalized_parts)
