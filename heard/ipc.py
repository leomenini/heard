"""Unix-socket push-to-talk IPC.

The daemon (`heard listen`) serves $XDG_RUNTIME_DIR/heard.sock; compositor
keybindings call `heard hold down` / `heard hold up` as one-shot clients.
Protocol: one command word + newline in, one status line + newline out.
"""

import os
import socket
import threading
from pathlib import Path

from . import stt

SOCKET_NAME = "heard.sock"
CMD_TIMEOUT_S = 60.0          # "up" waits for the STT tail


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
    return Path(runtime) / SOCKET_NAME


class IpcServer:
    """Serves hold commands against a shared HoldController."""

    def __init__(self, controller: stt.HoldController,
                 on_result=None, path: Path | None = None):
        self._controller = controller
        self._on_result = on_result or (lambda r: None)
        self.path = path or socket_path()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()      # stale socket from a dead daemon
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        sock.listen(4)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (OSError, TimeoutError):
                continue
            with conn:
                conn.settimeout(CMD_TIMEOUT_S)
                try:
                    line = conn.makefile("r").readline().strip()
                    reply = self._handle(line)
                except Exception as e:
                    reply = f"err {e}"
                try:
                    conn.sendall((reply + "\n").encode())
                except OSError:
                    pass

    def _handle(self, line: str) -> str:
        cmd = line.strip().lower()
        with self._lock:
            if cmd == "ping":
                return "ok ready"
            if cmd == "down":
                started = self._controller.down()
                return "ok recording" if started else "err already-recording"
            if cmd == "up":
                result = self._controller.up()
                if result.text:
                    self._on_result(result)
                    return f"ok {result.text}"
                return "ok empty"
            return f"err unknown-command {cmd!r}"


def send(command: str, path: Path | None = None, timeout: float = CMD_TIMEOUT_S) -> str:
    """One-shot client: send a command line, return the reply."""
    target = path or socket_path()
    if not target.exists():
        raise RuntimeError(
            f"no daemon at {target}; start it with `heard listen`")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(target))
        sock.sendall((command.strip() + "\n").encode())
        reply = sock.makefile("r").readline().strip()
    finally:
        sock.close()
    if not reply:
        raise RuntimeError("empty reply from daemon")
    return reply
