import numpy as np

from heard import stt


def tone(seconds, amp=0.3):
    t = np.arange(int(stt.SAMPLE_RATE * seconds)) / stt.SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(stt.SAMPLE_RATE * seconds), dtype=np.float32)


class TestSpeechBounds:
    def test_finds_speech_between_silence(self):
        audio = np.concatenate([silence(0.4), tone(1.0), silence(0.5)])
        bounds = stt._speech_bounds(audio)
        assert bounds is not None
        start, end = bounds
        assert 0 < start < stt.SAMPLE_RATE * 0.6
        assert stt.SAMPLE_RATE * 1.2 < end <= len(audio)

    def test_pure_silence_returns_none(self):
        assert stt._speech_bounds(silence(1.0)) is None

    def test_tiny_input(self):
        assert stt._speech_bounds(np.zeros(100, dtype=np.float32)) is None


class TestTrailingSilence:
    def test_detects_long_pause(self):
        audio = np.concatenate([tone(1.0), silence(0.5)])
        cut = stt.trailing_silence_start(audio)
        assert cut is not None
        assert abs(cut - len(tone(1.0))) < stt.SAMPLE_RATE * 0.15

    def test_no_pause_when_speech_runs_to_end(self):
        assert stt.trailing_silence_start(tone(2.0)) is None


def _empty_bus():
    import threading
    from collections import deque

    bus = stt.AudioBus.__new__(stt.AudioBus)   # skip real stream
    bus._chunks = deque()
    bus._offsets = deque()
    bus._total = 0
    bus._lock = threading.Lock()
    return bus


def _push(bus, arr):
    n = len(arr)
    bus._on_audio(arr.reshape(-1, 1), n, None, None)


def make_session(segments):
    """Session first, then audio -- matching real arrival order."""
    bus = _empty_bus()
    session = stt.HoldSession(bus)
    for seg in segments:
        _push(bus, seg)
    return session, bus


class TestAudioBus:
    def test_read_slices_appended_chunks(self):
        bus = _empty_bus()
        a = np.ones(100, dtype=np.float32)
        b = np.full(50, 2.0, dtype=np.float32)
        _push(bus, a)
        _push(bus, b)

        assert bus.pos() == 150
        mid = bus.read(80, 120)
        assert list(mid) == [1.0] * 20 + [2.0] * 20

    def test_discard_before(self):
        bus = _empty_bus()
        _push(bus, np.ones(10, dtype=np.float32))
        _push(bus, np.ones(10, dtype=np.float32))
        bus.discard_before(10)
        assert bus.pos() == 20
        assert len(bus._chunks) == 1
        # reads overlapping the dropped region clamp to what remains
        got = bus.read(5, 15)
        assert list(got) == [1.0] * 5


class TestHoldSessionDrain:
    def test_release_transcribes_remainder(self, monkeypatch):
        audio = np.concatenate([tone(1.0), silence(0.3)])
        session, _bus = make_session([audio])
        session.release.set()

        def fake_transcribe(a):
            return "hello world" if float(np.abs(a).max()) > 0.01 else ""

        monkeypatch.setattr(stt, "transcribe", fake_transcribe)
        session.run()
        assert session.text() == "hello world"

    def test_pause_finalizes_mid_hold(self, monkeypatch):
        monkeypatch.setattr(stt, "HOLD_STREAM_THRESHOLD_S", 0)
        first = np.concatenate([tone(1.0), silence(0.35)])   # pause > TRAIL_SILENCE + pad
        session, bus = make_session([first])
        calls = []
        monkeypatch.setattr(
            stt, "transcribe", lambda a: (calls.append(len(a)), f"n={len(a)}")[1]
        )

        did = session._drain(bus.pos(), released=False)
        assert did and session.text().startswith("n=")

        # release with only post-pause silence left: no second transcription
        session.release.set()
        monkeypatch.setattr(stt, "transcribe", lambda a: "")
        session.run()
        assert session.text().startswith("n=")
        assert len(calls) == 1


class TestTranscribeSignature:
    def test_audio_is_only_positional_arg(self, monkeypatch):
        """Guards the SAMPLE_RATE-into-language regression: faster_whisper is
        mocked in conftest, which would otherwise swallow signature errors."""
        from unittest import mock

        fake_model = mock.Mock()
        fake_model.transcribe.return_value = (iter([]), None)
        monkeypatch.setattr(stt, "_model", lambda: fake_model)

        audio = np.concatenate([tone(1.0), silence(0.2)])
        out = stt.transcribe(audio)

        assert out == ""
        args, kwargs = fake_model.transcribe.call_args
        assert len(args) == 1, "audio must be the only positional argument"
        assert kwargs.get("language") == "en"
        assert kwargs.get("beam_size") == 1

    def test_silence_skips_model(self, monkeypatch):
        from unittest import mock

        fake_model = mock.Mock()
        monkeypatch.setattr(stt, "_model", lambda: fake_model)
        assert stt.transcribe(silence(1.0)) == ""
        fake_model.transcribe.assert_not_called()


class TestHoldSessionError:
    def test_worker_error_surfaced_not_lost(self, monkeypatch):
        session, _bus = make_session([np.concatenate([tone(1.0), silence(0.35)])])
        session.release.set()

        def boom(a):
            raise TypeError("WhisperModel.transcribe() got multiple values")

        monkeypatch.setattr(stt, "transcribe", boom)
        monkeypatch.setattr(stt, "POLL_S", 0.001)
        session.run()
        assert isinstance(session.error, TypeError)


class TestCaptureResult:
    def test_fields(self):
        r = stt.CaptureResult("hi", hold_ms=1200.0, tail_ms=300.0)
        assert r.text == "hi" and r.segments == 0

    def test_wrapper_returns_text_only(self):
        assert callable(stt.capture_while_held)
