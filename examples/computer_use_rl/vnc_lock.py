from __future__ import annotations

import contextlib
import hashlib
import os
import time
from typing import Iterator


# Re-entrant lock support per process.
#
# Some verl components (agent loop + reward fn) may both attempt to lock the same
# VNC target within the same Python process. On Linux, `flock` is associated with
# an open file description, so re-locking the same path via a separate `open()`
# can block. We avoid this by keeping a single open file handle per lock path
# and reference-counting nested acquisitions.
_REENTRANT_LOCKS: dict[str, tuple[object, int]] = {}


def _lock_path(host: str, port: int) -> str:
    key = f"{host}:{int(port)}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()  # stable, short
    return f"/tmp/verl_vnc_{digest}.lock"


@contextlib.contextmanager
def vnc_global_lock(
    *,
    host: str,
    port: int,
    timeout_s: float = 300.0,
    poll_s: float = 0.1,
) -> Iterator[None]:
    """A process-wide lock to prevent concurrent control of the same VNC VM.

    Uses an OS-level advisory file lock (flock) so it works across:
    - Ray workers / processes
    - multiple Python interpreters

    Limitations:
    - Only synchronizes processes on the same filesystem namespace (i.e., same machine/container).
    """

    # fcntl is Unix-only; this repo runs in Linux containers.
    import fcntl

    path = _lock_path(host, port)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing = _REENTRANT_LOCKS.get(path)
    if existing is not None:
        _, depth = existing
        _REENTRANT_LOCKS[path] = (existing[0], depth + 1)
        try:
            yield
        finally:
            f_obj, depth = _REENTRANT_LOCKS[path]
            if depth <= 1:
                _REENTRANT_LOCKS.pop(path, None)
            else:
                _REENTRANT_LOCKS[path] = (f_obj, depth - 1)
        return

    # Keep the file handle open for the duration of the lock.
    f = open(path, "a+")
    _REENTRANT_LOCKS[path] = (f, 1)
    start = time.time()

    try:
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Record current owner for debugging.
                f.seek(0)
                f.truncate()
                f.write(f"pid={os.getpid()} host={host} port={int(port)}\n")
                f.flush()
                break
            except BlockingIOError:
                if (time.time() - start) >= timeout_s:
                    # Best-effort debug hint: report recorded owner line.
                    owner_line = None
                    try:
                        f.seek(0)
                        owner_line = (f.readline() or "").strip() or None
                    except Exception:
                        owner_line = None
                    raise TimeoutError(
                        (
                            f"Timed out acquiring VNC lock for {host}:{int(port)} after {timeout_s:.1f}s"
                            + (f" (owner: {owner_line})" if owner_line else "")
                        )
                    )
                time.sleep(poll_s)

        yield

    finally:
        # Only release when the outermost context exits.
        try:
            current = _REENTRANT_LOCKS.get(path)
            if current is None:
                return
            _, depth = current
            if depth > 1:
                _REENTRANT_LOCKS[path] = (f, depth - 1)
                return
            _REENTRANT_LOCKS.pop(path, None)
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            finally:
                f.close()
