"""Local JSONL usage log -- the data source for threshold tuning and any
future finetune set. Opt-out via config `events = false`. Never leaves the
machine: one line per command in ~/.local/state/heard/events.jsonl.
"""

import json
import os
import time
from pathlib import Path


def events_path() -> Path:
    state = os.environ.get("XDG_STATE_HOME", "")
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "heard" / "events.jsonl"


def log_event(kind: str, **fields) -> None:
    try:
        from . import config

        enabled = bool(config.cached("events"))
    except Exception:
        enabled = True
    if not enabled:
        return

    record = {"ts": time.time(), "kind": kind, **fields}
    path = events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
