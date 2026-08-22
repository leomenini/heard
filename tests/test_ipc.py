import json

import pytest

from heard import events, ipc, stt


def _controller_with_audio(monkeypatch):
    """A HoldController over a fake bus whose transcribe returns text."""
    bus = stt.AudioBus.__new__(stt.AudioBus)
    import threading
    from collections import deque

    bus._chunks = deque()
    bus._offsets = deque()
    bus._total = 0
    bus._lock = threading.Lock()

    def fake_transcribe(a):
        t = a[:100].sum()  # distinguish segments by first samples
        return f"words{int(abs(float(t)))}"

    monkeypatch.setattr(stt, "transcribe", fake_transcribe)
    return stt.HoldController(bus)


def _push(bus, arr):
    bus._on_audio(arr.reshape(-1, 1), len(arr), None, None)


@pytest.fixture
def server(monkeypatch, tmp_path):
    controller = _controller_with_audio(monkeypatch)
    received: list[str] = []
    srv = ipc.IpcServer(controller,
                        on_result=lambda cap: received.append(cap.text),
                        path=tmp_path / "heard.sock")
    srv.start()
    yield srv, controller, received
    srv.stop()


class TestIpcServer:
    def test_ping(self, server):
        srv, _, _ = server
        assert ipc.send("ping", path=srv.path) == "ok ready"

    def test_unknown_command(self, server):
        srv, _, _ = server
        assert "unknown-command" in ipc.send("frobnicate", path=srv.path)

    def test_up_without_down_is_empty_ok(self, server):
        srv, _, _ = server
        assert ipc.send("up", path=srv.path) == "ok empty"

    def test_full_hold_roundtrip(self, server, monkeypatch):
        import numpy as np

        srv, controller, received = server
        assert ipc.send("down", path=srv.path) == "ok recording"
        # second down is rejected while recording
        assert "already-recording" in ipc.send("down", path=srv.path)

        # speech arrives while "recording" (audio flows through the same bus)
        tone = np.sin(np.linspace(0, 400, 20000)).astype(np.float32)
        silence = np.zeros(6000, dtype=np.float32)
        _push(controller._bus, np.concatenate([tone, silence]))

        reply = ipc.send("up", path=srv.path, timeout=10)
        assert reply.startswith("ok ")
        assert received and received[0] == reply[3:]

    def test_socket_removed_on_stop(self, server):
        srv, _, _ = server
        assert srv.path.exists()
        srv.stop()
        assert not srv.path.exists()


class TestEvents:
    def test_log_event_appends_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        monkeypatch.setenv("HEARD_CONFIG", str(tmp_path / "c.toml"))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        try:
            events.log_event("command", transcript="hi", score=0.9)
            lines = (tmp_path / "heard" / "events.jsonl").read_text().splitlines()
            rec = json.loads(lines[-1])
            assert rec["kind"] == "command"
            assert rec["transcript"] == "hi"
        finally:
            cfg._cached_load.cache_clear()

    def test_disabled_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        c = tmp_path / "c.toml"
        c.write_text("events = false\n")
        monkeypatch.setenv("HEARD_CONFIG", str(c))
        from heard import config as cfg
        cfg._cached_load.cache_clear()
        try:
            events.log_event("command", transcript="x")
            assert not (tmp_path / "heard" / "events.jsonl").exists()
        finally:
            cfg._cached_load.cache_clear()
