import functools
import selectors
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from evdev import InputDevice, list_devices, ecodes
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
WHISPER_SIZE = "base"

FRAME_MS = 20                      # VAD frame size
TRAIL_SILENCE_S = 0.18             # pause length that finalizes a segment
MIN_SEGMENT_S = 0.60               # don't bother finalizing shorter segments
MIN_SPEECH_S = 0.15                # trim results shorter than this -> ""
SPEECH_RMS_RATIO = 0.07            # frame is speech if rms > ratio * peak
SPEECH_RMS_FLOOR = 3e-4            # ...and above this absolute floor
POLL_S = 0.03


@functools.lru_cache(maxsize=1)
def _model():
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def warmup() -> None:
    """Load whisper, open the capture stream, JIT one tiny transcription."""
    bus = _bus()
    model = _model()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    next(iter(model.transcribe(silence, language="en",
                               beam_size=1, temperature=0.0)[0]), None)


def _keycode(key_name: str) -> int:
    try:
        return getattr(ecodes, key_name)
    except AttributeError:
        raise ValueError(
            f"unknown key code {key_name!r}; "
            f"use e.g. KEY_LEFTSHIFT, KEY_LEFTCONTROL, KEY_SCROLLLOCK"
        )


_GROUP_HINT = (
    "Make sure you are in the 'input' group:\n"
    "  sudo usermod -aG input $USER\n"
    "Then log out and back in (or run via `sg input -c '...'`)."
)


def _find_input_devices(code: int, key_name: str) -> list[InputDevice]:
    """Every readable device that actually advertises the PTT keycode.

    Matching on the code -- not merely on EV_KEY capability -- is what keeps
    us off 'Power Button' and friends that never emit the key.
    """
    paths = list_devices()
    if not paths:
        raise RuntimeError(f"No input devices found in /dev/input/.\n{_GROUP_HINT}")

    matched: list[InputDevice] = []
    denied = False
    for path in sorted(paths):
        try:
            dev = InputDevice(path)
        except PermissionError:
            denied = True
            continue
        try:
            caps = dev.capabilities()
        except PermissionError:
            denied = True
            continue
        except Exception:
            continue
        if code in caps.get(ecodes.EV_KEY, []):
            matched.append(dev)

    if matched:
        return matched
    if denied:
        raise RuntimeError(
            f"Devices carrying {key_name} exist but none are readable.\n{_GROUP_HINT}"
        )
    raise RuntimeError(
        f"No input device advertises {key_name}.\n"
        f"Pick another push-to-talk key, e.g. KEY_LEFTCONTROL or KEY_SCROLLLOCK."
    )


def _ptt_events(devs: list[InputDevice], code: int):
    """Yield PTT key events from all matching devices until closed."""
    sel = selectors.DefaultSelector()
    for d in devs:
        sel.register(d, selectors.EVENT_READ)
    try:
        while True:
            for key, _mask in sel.select():
                for event in key.fileobj.read():
                    if event.type == ecodes.EV_KEY and event.code == code:
                        yield event
    finally:
        sel.close()


class AudioBus:
    """Persistent capture stream feeding an append-only sample buffer.

    The PortAudio stream stays open for the process lifetime: pressing the
    PTT key just marks a buffer offset instead of re-initializing the device.
    """

    def __init__(self):
        self._chunks: deque[np.ndarray] = deque()
        self._offsets: deque[int] = deque()   # cumulative start of each chunk
        self._total = 0
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status):
        with self._lock:
            self._offsets.append(self._total)
            self._total += frames
            self._chunks.append(indata[:, 0].copy())

    def pos(self) -> int:
        with self._lock:
            return self._total

    def read(self, start: int, end: int) -> np.ndarray:
        with self._lock:
            pieces = []
            for chunk, off in zip(self._chunks, self._offsets):
                cstart, cend = off, off + len(chunk)
                if cend <= start or cstart >= end:
                    continue
                lo = max(start, cstart) - cstart
                hi = min(end, cend) - cstart
                pieces.append(chunk[lo:hi])
        if not pieces:
            return np.empty(0, dtype=np.float32)
        out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return out[: end - start]

    def discard_before(self, pos: int) -> None:
        with self._lock:
            while self._chunks and self._offsets[0] + len(self._chunks[0]) <= pos:
                self._chunks.popleft()
                self._offsets.popleft()


@functools.lru_cache(maxsize=1)
def _bus() -> AudioBus:
    bus = AudioBus()
    bus._stream.start()
    return bus


def _frame_rms(audio: np.ndarray, frame: int) -> np.ndarray:
    n = len(audio) // frame
    if n == 0:
        return np.empty(0)
    usable = audio[: n * frame].reshape(n, frame).astype(np.float64)
    return np.sqrt((usable ** 2).mean(axis=1))


def _speech_bounds(audio: np.ndarray) -> tuple[int, int] | None:
    """Return (start, end) sample indices covering speech, or None.

    Bounds are padded by 120ms on each side; None when no frame clears the
    speech threshold (relative to peak, floored for dead mics).
    """
    frame = SAMPLE_RATE * FRAME_MS // 1000
    rms = _frame_rms(audio, frame)
    if not len(rms):
        return None
    thr = max(rms.max() * SPEECH_RMS_RATIO, SPEECH_RMS_FLOOR)
    voiced = np.flatnonzero(rms > thr)
    if not len(voiced):
        return None
    pad = SAMPLE_RATE * 120 // 1000
    start = max(0, int(voiced[0]) * frame - pad)
    end = min(len(audio), int(voiced[-1] + 1) * frame + pad)
    return start, end


def trailing_silence_start(audio: np.ndarray) -> int | None:
    """Index where trailing silence begins, if it spans >= TRAIL_SILENCE_S."""
    bounds = _speech_bounds(audio)
    if bounds is None:
        return None
    end = bounds[1]
    min_silence = int(SAMPLE_RATE * TRAIL_SILENCE_S)
    if len(audio) - end >= min_silence:
        return end
    return None


def transcribe(audio: np.ndarray) -> str:
    """STT over finished audio. Caller owns trimming."""
    bounds = _speech_bounds(audio)
    if bounds is None or (bounds[1] - bounds[0]) < SAMPLE_RATE * MIN_SPEECH_S * 0.5:
        return ""
    model = _model()
    segments, _ = model.transcribe(
        audio[bounds[0]:bounds[1]],
        language="en",
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segments).strip()


class HoldSession:
    """Streams STT across one PTT hold.

    While the key is down, any segment that ends in a natural pause is
    finalized and transcribed in a worker thread. Key release only has to
    transcribe whatever came after the last pause -- usually nothing.
    """

    def __init__(self, bus: AudioBus):
        self.bus = bus
        self.start_pos = bus.pos()
        self.release = threading.Event()
        self._parts: list[str] = []
        self._lock = threading.Lock()
        self._seg_start = self.start_pos
        self.error: Exception | None = None

    def text(self) -> str:
        with self._lock:
            return " ".join(p for p in self._parts if p).strip()

    def run(self) -> None:
        try:
            while True:
                end = self.bus.pos()
                progressed = self._drain(end, released=False)
                if self.release.is_set():
                    self._drain(max(end, self.bus.pos()), released=True)
                    return
                if not progressed:
                    time.sleep(POLL_S)
        except Exception as e:            # never die silently in a daemon thread
            self.error = e

    def _drain(self, end: int, released: bool) -> bool:
        """Transcribe finalized segments; returns True if any work was done."""
        did_work = False
        while True:
            seg = self.bus.read(self._seg_start, end)
            if len(seg) < SAMPLE_RATE * MIN_SEGMENT_S and not (
                released and len(seg) > SAMPLE_RATE // 20
            ):
                break
            cut = None if released else trailing_silence_start(seg)
            piece_end = len(seg) if cut is None else cut
            if cut is None:
                if not released:
                    break                       # no pause yet; keep waiting
                if piece_end <= 0:
                    break
            else:
                if piece_end < SAMPLE_RATE * MIN_SEGMENT_S:
                    self._seg_start += cut      # skip sub-minimum blip
                    continue

            piece = seg[:piece_end]
            text = transcribe(piece)
            with self._lock:
                self._parts.append(text)
            self._seg_start += piece_end
            did_work = True
            if not released:
                break                           # re-evaluate against live audio
        return did_work


def check_ptt(ptt_key: str = "KEY_LEFTSHIFT") -> list[InputDevice]:
    """Resolve PTT devices up front so failures surface before warmup."""
    code = _keycode(ptt_key)
    return _find_input_devices(code, ptt_key)


@dataclass(frozen=True)
class CaptureResult:
    text: str
    hold_ms: float          # key press -> release
    tail_ms: float          # release -> transcript ready (the felt STT cost)
    segments: int = 0       # pieces finalized during the hold (streaming wins)


def capture(ptt_key: str = "KEY_LEFTSHIFT") -> CaptureResult:
    """Block until a full PTT hold completes; return transcript + timings."""
    code = _keycode(ptt_key)
    devs = check_ptt(ptt_key)
    global _ptt_announced
    if not _ptt_announced:
        names = ", ".join(d.name for d in devs)
        print(f"heard: ptt on {ptt_key} via {names}", flush=True)
        _ptt_announced = True
    bus = _bus()
    events = _ptt_events(devs, code)

    session: HoldSession | None = None
    worker: threading.Thread | None = None
    press_t: float | None = None
    release_t: float | None = None

    def _finish() -> tuple[HoldSession, threading.Thread]:
        session.release.set()
        worker.join(timeout=30)
        return session, worker

    try:
        for event in events:
            if event.value == 1 and session is None:
                press_t = time.perf_counter()
                session = HoldSession(bus)
                worker = threading.Thread(target=session.run, daemon=True)
                worker.start()

            elif event.value == 0 and session is not None:
                release_t = time.perf_counter()
                session.release.set()
                worker.join(timeout=30)
                end = bus.pos()
                text = session.text()
                if session.error is not None:
                    print(f"heard: stt error: {session.error}", flush=True)
                    session = None
                    worker = None
                    continue
                bus.discard_before(end)
                hold_ms = (release_t - press_t) * 1000 if press_t else 0.0
                tail_ms = (time.perf_counter() - release_t) * 1000
                result = CaptureResult(text, hold_ms, tail_ms,
                                       segments=len(session._parts))
                session = None
                worker = None
                if text:
                    return result
    finally:
        if session is not None and worker is not None:
            session.release.set()
            worker.join(timeout=30)

    return CaptureResult("", 0.0, 0.0)


def capture_while_held(ptt_key: str = "KEY_LEFTSHIFT") -> str:
    """Compat wrapper: just the transcript."""
    return capture(ptt_key).text


_ptt_announced = False
