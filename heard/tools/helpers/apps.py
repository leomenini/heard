import os
import shlex
import shutil
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process


def app_exists(app: str) -> bool:
    return shutil.which(app) is not None


def _desktop_dirs() -> list[Path]:
    dirs = [Path.home() / ".local" / "share" / "applications"]
    for d in os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":"):
        if d:
            dirs.append(Path(d) / "applications")
    return dirs


def _exec_binary(exec_line: str) -> str | None:
    try:
        tokens = shlex.split(exec_line)
    except ValueError:
        return None
    for token in tokens:
        if token.startswith("%"):
            continue
        return Path(token).name
    return None


def _parse_desktop_entry(path: Path) -> list[tuple[str, str]]:
    """(display key, binary) pairs: Name, GenericName, and localized Names."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    section = None
    exec_token = None
    hidden = False
    names: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line
            continue
        if section != "[Desktop Entry]":
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "Exec":
            exec_token = value
        elif key == "Name" or key.startswith("Name["):
            if value:
                names.append(value)
        elif key == "GenericName" and value:
            names.append(value)
        elif key in ("Hidden", "NoDisplay") and value.lower() == "true":
            hidden = True

    if hidden or exec_token is None:
        return []
    binary = _exec_binary(exec_token)
    if binary is None:
        return []
    names.append(binary)
    return [(n.lower(), binary) for n in dict.fromkeys(names)]


@lru_cache(maxsize=1)
def installed_apps() -> dict[str, str]:
    """Map lowercase display names (incl. localized) and binaries to binaries."""
    apps: dict[str, str] = {}
    for d in _desktop_dirs():
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.desktop")):
            for name, binary in _parse_desktop_entry(path):
                apps.setdefault(name, binary)
    return apps


def resolve_app(query: str) -> str | None:
    """Fuzzy-match a spoken app name against installed .desktop entries."""
    query = query.strip().lower()
    if not query:
        return None
    choices = installed_apps()
    if not choices:
        return None
    match = process.extractOne(query, choices.keys(), scorer=fuzz.token_set_ratio, score_cutoff=70)
    if match is None:
        return None
    key, _score, _index = match
    if len(key) < 4 and query != key:
        return None
    return choices[key]
