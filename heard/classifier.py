"""Fast intent path: embedding nearest-centroid classification + slot lookup.

Uses Needle's contrastive encoder (same 26M model, encode-only) to map the
transcript to one of a fixed set of classes -- one per tool and, where the
argument space is enumerable, one per action value. Arguments that cannot be
classified (app names, numbers) are resolved by lookup/extraction, never
generation.

Verdicts:
  - classified  -> tool_call dict ready for dispatch (<1ms of CPU after embed)
  - declined    -> confidently off-topic (below floor / unknown centroid wins)
  - fallback    -> uncertain; caller retries with generative Needle resolve
"""

from dataclasses import dataclass
import os
import re

import numpy as np
from rapidfuzz import fuzz, process

from .tools.helpers.apps import installed_apps

# Cosine thresholds. Tuned via scripts/benchmark_latency.py fast-path probe;
# override live with HEARD_FLOOR_DECLINE / HEARD_ACCEPT_SCORE / HEARD_MARGIN.
FLOOR_DECLINE = float(os.environ.get("HEARD_FLOOR_DECLINE", 0.35))
ACCEPT_SCORE = float(os.environ.get("HEARD_ACCEPT_SCORE", 0.55))
MARGIN = float(os.environ.get("HEARD_MARGIN", 0.04))

# launch_app corroboration: app nouns scatter in embedding space, so a strict
# dictionary hit against installed .desktop entries lets an utterance clear a
# lower bar than ACCEPT_SCORE -- but only when the best class is launch_app.
DICT_FLOOR = float(os.environ.get("HEARD_DICT_FLOOR", 0.40))
DICT_CUTOFF = int(os.environ.get("HEARD_DICT_CUTOFF", 90))


@dataclass(frozen=True)
class Verdict:
    tool_call: dict | None = None
    declined: bool = False
    score: float | None = None
    runner_up: tuple[str, float] | None = None


PROTOTYPES: dict[str, tuple[str, dict, list[str]]] = {
    # key -> (tool, static args, utterances)
    "system_query:battery": ("system_query", {"query": "battery"}, [
        "how's my battery",
        "check the battery",
        "battery status",
        "how much battery do i have left",
        "is my battery low",
        "battery level",
        "what's my battery at",
    ]),
    "system_query:time": ("system_query", {"query": "time"}, [
        "what time is it",
        "what's the time",
        "tell me the time",
        "current time",
        "time check",
        "what time is it right now",
    ]),
    "system_query:network": ("system_query", {"query": "network"}, [
        "check the network status",
        "how's my internet",
        "network status",
        "am i connected",
        "do i have internet",
        "connection status",
        "is the wifi working",
    ]),
    "system_query:disk": ("system_query", {"query": "disk"}, [
        "how much disk space is left",
        "check disk usage",
        "disk space",
        "how full is my drive",
        "storage status",
        "free disk space",
        "how much storage do i have",
    ]),
    "volume_control:up": ("volume_control", {"action": "up"}, [
        "turn the volume up",
        "volume up",
        "louder",
        "turn it up",
        "raise the volume",
        "it's too quiet",
        "pump up the volume",
        "a bit louder",
        "crank it up",
        "a little higher",
        "make it louder",
    ]),
    "volume_control:down": ("volume_control", {"action": "down"}, [
        "turn the volume down",
        "volume down",
        "quieter",
        "turn it down a bit",
        "lower the volume",
        "it's too loud",
        "decrease the volume",
        "not so loud",
        "bring it down",
        "drop the volume",
        "a little lower",
    ]),
    "volume_control:mute": ("volume_control", {"action": "mute"}, [
        "mute the audio",
        "mute",
        "silence",
        "mute the volume",
        "shut it up",
        "mute output",
        "kill the sound",
    ]),
    "volume_control:set": ("volume_control", {"action": "set"}, [
        "set the volume to fifty percent",
        "set volume to seventy percent",
        "volume to thirty percent",
        "make it forty percent",
        "set the audio level to sixty percent",
        "put volume at eighty percent",
        "volume twenty",
        "volume to ten",
        "set it to ninety",
    ]),
    "media_control:play": ("media_control", {"action": "play"}, [
        "play some music",
        "play",
        "resume playback",
        "resume the music",
        "start the music",
        "continue playing",
        "unpause",
        "hit play",
    ]),
    "media_control:pause": ("media_control", {"action": "pause"}, [
        "pause the playback",
        "pause",
        "pause the music",
        "stop the music",
        "hold on",
        "freeze playback",
        "pause it",
    ]),
    "media_control:next": ("media_control", {"action": "next"}, [
        "skip to the next track",
        "next track",
        "next song",
        "skip this one",
        "play the next one",
        "forward",
        "skip forward",
    ]),
    "media_control:previous": ("media_control", {"action": "previous"}, [
        "go back to the previous song",
        "previous track",
        "previous song",
        "go back a track",
        "replay the last song",
        "skip back",
        "rewind to the last track",
        "the one before",
    ]),
    "launch_app:app": ("launch_app", {}, [
        "open firefox",
        "launch the calculator",
        "start vs code",
        "run spotify",
        "open the file manager",
        "start the terminal",
        "fire up chrome",
        "launch gimp",
    ]),
    "window_action:close": ("window_action", {"action": "close"}, [
        "close this window",
        "close the window",
        "close it",
        "quit this application",
        "kill the window",
        "dismiss this window",
    ]),
    "window_action:focus": ("window_action", {"action": "focus"}, [],),  # target needed -> always fallback
    "window_action:fullscreen": ("window_action", {"action": "fullscreen"}, [
        "make it fullscreen",
        "go fullscreen",
        "fullscreen this window",
        "toggle fullscreen",
        "maximize to fullscreen",
    ]),
    "workspace_switch:ws": ("workspace_switch", {}, [
        "go to workspace three",
        "switch to workspace one",
        "jump to workspace five",
        "workspace two",
        "take me to workspace four",
        "change to workspace seven",
    ]),
    "unknown:x": ("__unknown__", {}, [
        "what is the weather tomorrow",
        "tell me a joke",
        "who won the game last night",
        "add milk to my shopping list",
        "what's the capital of france",
        "remind me to call mom",
        "how do you make pasta",
        "search for flights to tokyo",
    ]),
}

FALLBACK_CLASSES = frozenset({"window_action:focus"})  # need generated targets

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_FILLER_WORDS = {"the", "a", "an", "to", "please", "now"}
_LEADING_VERBS = {
    "open", "launch", "start", "run", "fire", "bring", "up", "switch",
    "go", "jump", "take", "me", "change", "set", "put",
}


def _strip_filler(text: str) -> str:
    tokens = [t for t in text.lower().split() if t not in _FILLER_WORDS]
    return " ".join(tokens)


def _strip_leading_verbs(text: str) -> str:
    tokens = text.split()
    while tokens and tokens[0] in _LEADING_VERBS:
        tokens.pop(0)
    return " ".join(tokens)


def parse_spoken_number(text: str) -> int | None:
    """Extract the last spoken number: digits, number words, 'half', 'quarter'."""
    tokens = text.lower().replace("-", " ").split()
    best = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        val = None
        if tok in _NUM_WORDS:
            val = _NUM_WORDS[tok]
            if i + 1 < len(tokens) and tokens[i + 1] == "hundred":
                val *= 100
                i += 1
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if nxt in _TENS_WORDS:
                    val += _TENS_WORDS[nxt]
                    i += 1
                    if i + 1 < len(tokens) and tokens[i + 1] in _NUM_WORDS and _NUM_WORDS[tokens[i + 1]] < 10:
                        val += _NUM_WORDS[tokens[i + 1]]
                        i += 1
                elif nxt in _NUM_WORDS and _NUM_WORDS[nxt] < 10:
                    val += _NUM_WORDS[nxt]
                    i += 1
        elif tok in _TENS_WORDS:
            val = _TENS_WORDS[tok]
            if i + 1 < len(tokens) and tokens[i + 1] in _NUM_WORDS and _NUM_WORDS[tokens[i + 1]] < 10:
                val += _NUM_WORDS[tokens[i + 1]]
                i += 1
        elif tok == "half":
            val = 50
        elif tok == "quarter":
            val = 25
        if val is not None:
            best = val
        i += 1
    return best


def extract_percent(text: str) -> int | None:
    """Pull a volume percentage out of the transcript ('50', '50 percent', 'fifty')."""
    m = re.search(r"\b(\d{1,3})\s*(?:%|percent|pc)\b", text)
    if m:
        n = int(m.group(1))
        return n if 0 <= n <= 100 else None
    n = parse_spoken_number(text)
    if n is not None and 0 <= n <= 100:
        return n
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        n = int(m.group(1))
        return n if 0 <= n <= 100 else None
    return None


def extract_workspace(text: str) -> str | None:
    """Pull a workspace number out of the transcript."""
    m = re.search(r"\b(?:workspace\s+)?(\d{1,2})\b", text)
    if m:
        return m.group(1)
    for tok in text.lower().split():
        if tok in _NUM_WORDS:
            return str(_NUM_WORDS[tok])
    return None


def extract_app_phrase(text: str) -> str | None:
    """Reduce 'open the thing please' to 'thing' for fuzzy app matching."""
    cleaned = _strip_filler(text)
    cleaned = _strip_leading_verbs(cleaned)
    return cleaned or None


def _dict_hit(phrase: str | None) -> bool:
    """Strict .desktop dictionary check used to corroborate launch_app."""
    if not phrase:
        return False
    choices = installed_apps()
    if not choices:
        return False
    match = process.extractOne(phrase.lower().strip(), choices.keys(),
                               scorer=fuzz.token_set_ratio, score_cutoff=DICT_CUTOFF)
    if match is None:
        return False
    key = match[0]
    # subset tokens reach 100 spuriously ('browser' inside 'avahi ... browser');
    # require a real word overlap with the spoken phrase
    return bool(set(phrase.lower().split()) & set(key.split()))


class Classifier:
    """Nearest-centroid over Needle contrastive embeddings."""

    def __init__(self, encode):
        """encode: Callable[[list[str]], np.ndarray] returning L2-normalized rows."""
        self._encode = encode
        keys = list(PROTOTYPES)
        texts = [u for k in keys for u in PROTOTYPES[k][2]]
        embs = encode(texts)
        self.centroids: dict[str, np.ndarray] = {}
        self._tool_of: dict[str, str] = {}
        i = 0
        for key in keys:
            utts = PROTOTYPES[key][2]
            if not utts:
                continue
            rows = embs[i:i + len(utts)]
            i += len(utts)
            centroid = rows.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.centroids[key] = centroid
            self._tool_of[key] = PROTOTYPES[key][0]

    def _rank(self, query: str, q_emb: np.ndarray | None = None) -> list[tuple[str, float]]:
        q = self._encode([query])[0] if q_emb is None else q_emb
        sims = {k: float(np.dot(q, c)) for k, c in self.centroids.items()}
        return sorted(sims.items(), key=lambda kv: kv[1], reverse=True)

    def match(self, query: str, q_emb: np.ndarray | None = None) -> Verdict:
        ranked = self._rank(query, q_emb)
        best_key, best_score = ranked[0]
        tool = self._tool_of[best_key]

        runner = next(((k, s) for k, s in ranked[1:] if self._tool_of[k] != tool),
                      None)

        unknown_score = next((s for k, s in ranked if k == "unknown:x"), -1.0)
        if best_score < FLOOR_DECLINE and unknown_score >= FLOOR_DECLINE:
            return Verdict(tool_call=None, declined=True,
                           score=best_score, runner_up=("unknown:x", unknown_score))
        if unknown_score >= best_score + MARGIN:
            return Verdict(tool_call=None, declined=True,
                           score=best_score, runner_up=("unknown:x", unknown_score))
        if tool == "__unknown__":
            return Verdict(tool_call=None, declined=True,
                           score=best_score, runner_up=runner)
        if best_key in FALLBACK_CLASSES:
            return Verdict(score=best_score, runner_up=runner)  # needs generated slots

        margin_ok = runner is None or best_score - runner[1] >= MARGIN
        strong = best_score >= ACCEPT_SCORE and margin_ok
        corroborated = (
            best_key == "launch_app:app"
            and best_score >= DICT_FLOOR
            and margin_ok
            and _dict_hit(extract_app_phrase(query))
        )
        if not (strong or corroborated):
            # too close to call -> generative fallback; scores kept for tuning
            return Verdict(score=best_score, runner_up=runner)

        return Verdict(tool_call=self._build_call(best_key, query),
                       score=best_score, runner_up=runner)

    def _build_call(self, key: str, query: str) -> dict | None:
        tool, static_args, _ = PROTOTYPES[key]
        args = dict(static_args)

        if tool == "volume_control" and args.get("action") == "set":
            pct = extract_percent(query)
            if pct is None:
                return None  # no number found -> generative fallback
            args["amount"] = str(pct)

        elif tool == "workspace_switch":
            ws = extract_workspace(query)
            if ws is None:
                return None
            args["workspace"] = ws

        elif tool == "launch_app":
            phrase = extract_app_phrase(query)
            if phrase is None:
                return None
            args["app"] = phrase

        return {"name": tool, "arguments": args}
