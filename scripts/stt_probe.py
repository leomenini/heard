"""Find the fastest STT settings for this machine.

Times transcribe() across model sizes and thread counts on short clips.
Run on the target hardware and pick the fastest combo whose accuracy you
accept, then set it in config:

  uv run python scripts/stt_probe.py [--audio clip.wav]
  heard config set stt_model_size tiny
  heard config set stt_cpu_threads 2
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
KW = dict(language="en", beam_size=1, temperature=0.0,
          condition_on_previous_text=False, without_timestamps=True)

SIZES = ["tiny", "base"]
THREADS = [1, 2, 4]


def load_audio(path: str) -> np.ndarray:
    import wave

    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (raw / 32768.0).astype(np.float32)


def synthetic() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(5)
    speech = (rng.standard_normal(SAMPLE_RATE * 2) * 0.2 *
              (np.hanning(SAMPLE_RATE * 2) ** 0.25)).astype(np.float32)
    tail = np.concatenate([
        (rng.standard_normal(SAMPLE_RATE) * 0.15).astype(np.float32),
        np.zeros(SAMPLE_RATE // 4, dtype=np.float32),
    ])
    return [("2s", speech), ("1s", tail)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="16kHz mono wav; else synthetic clips")
    args = ap.parse_args()

    clips = [("audio", load_audio(args.audio))] if args.audio else synthetic()

    print(f"{'model':<6} {'threads':>7} " +
          " ".join(f"{name:>12}" for name, _ in clips) + "   first-call")
    for size in SIZES:
        for threads in THREADS:
            model = WhisperModel(size, device="cpu", compute_type="int8",
                                 cpu_threads=threads)
            row = []
            t0 = time.perf_counter()
            next(iter(model.transcribe(clips[0][1], **KW)[0]), None)
            first_ms = (time.perf_counter() - t0) * 1000
            for _, audio in clips:
                model.transcribe(audio, **KW)      # warm this length bucket
                lat = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    next(iter(model.transcribe(audio, **KW)[0]), None)
                    lat.append((time.perf_counter() - t0) * 1000)
                row.append(statistics.median(lat))
            print(f"{size:<6} {threads:>7} " +
                  " ".join(f"{v:>10.0f}ms" for v in row) +
                  f"   {first_ms:>8.0f}ms")


if __name__ == "__main__":
    main()
