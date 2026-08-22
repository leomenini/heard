import re


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable: TiB branch always returns")

_PHYSICAL_DISK_RE = re.compile(
    r"^(?:sd[a-z]+|hd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+|mmcblk\d+)$"
)


def is_physical_disk(device: str) -> bool:
    """Return True if the device appears to be a physical disk."""
    return _PHYSICAL_DISK_RE.fullmatch(device) is not None
