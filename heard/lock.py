"""Single-instance guard via an flock'd pidfile in XDG_RUNTIME_DIR.

The lock dies with the process (kernel releases flock), so stale files are
harmless -- whoever holds it is genuinely alive.
"""

import fcntl
import os
from pathlib import Path


class SingleInstance:
    def __init__(self, name: str = "heard"):
        runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
        self.path = Path(runtime) / f"{name}-{os.getuid()}.lock"
        self._fd: int | None = None

    def acquire(self) -> int | None:
        """Take the lock; return holder's pid when someone else owns it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_RDWR|O_CREAT, never O_TRUNC: truncating before the flock would
        # wipe the holder's pid right before we need to read it.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return self.holder_pid()
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return None

    def holder_pid(self) -> int | None:
        try:
            return int(self.path.read_text().strip() or 0) or None
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
