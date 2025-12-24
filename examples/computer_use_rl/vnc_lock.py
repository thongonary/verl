from __future__ import annotations

import contextlib
import hashlib
import os
import time
from typing import Iterator


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

    # Keep the file handle open for the duration of the lock.
    f = open(path, "a+")
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
                    raise TimeoutError(
                        f"Timed out acquiring VNC lock for {host}:{int(port)} after {timeout_s:.1f}s"
                    )
                time.sleep(poll_s)

        yield

    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()
