"""Host RAM that the kernel leaks when a large CUDA pin fails.

On Linux 6.11, a long-term pin (pin_user_pages with FOLL_LONGTERM) of more
than a couple of GB fails, and the pages pinned so far are never released
(kernel bug 219427, fixed by "mm/gup: stop leaking pinned pages in low memory
conditions"). llama.cpp asks for exactly that when it puts the CPU part of a
partially offloaded model into pinned host memory: the allocation fails, it
falls back to ordinary memory, and the pages it tried to pin stay allocated,
owned by no process, until the machine reboots. Restarting Ollama does not
give them back.

Seen with qwen3.6 (36B MoE) on an 8 GB GPU: 16 GB lost per load. The fix is
a newer kernel; yuzuru only warns and keeps a leaking model from being
offloaded again.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("yuzuru.pinleak")

GB = 1024**3
# Kernel series known to leak failed long-term pins (the fix went into 6.12).
LEAKING_SERIES = ("6.11",)
# Growth of unowned memory across one model load that counts as a leak.
LEAK_THRESHOLD = 1 * GB


def kernel_release() -> str:
    return os.uname().release


KERNEL_WARNING = ("Linux {release} leaks pinned host RAM until reboot when llama.cpp offloads part of a "
                  "large model to the CPU (kernel bug 219427, fixed in 6.12); update the kernel")


def kernel_leaks_pins(release: str | None = None) -> bool:
    """Whether this kernel is one known to leak failed long-term pins."""
    release = release or kernel_release()
    return any(release == s or release.startswith(s + ".") or release.startswith(s + "-") for s in LEAKING_SERIES)


def read_unowned(meminfo: str | Path = "/proc/meminfo") -> int | None:
    """Bytes of anonymous memory that no process maps.

    Anonymous pages stay on the LRU lists after their process exits only while
    something still holds them, such as a pin nobody released. Normally this
    is close to zero. None where /proc/meminfo is unavailable (macOS, Windows).
    """
    try:
        info = {}
        for line in Path(meminfo).read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) * 1024
        lru = info["Active(anon)"] + info["Inactive(anon)"]
        owned = info["AnonPages"] + info.get("Shmem", 0) + info.get("SwapCached", 0)
    except (OSError, KeyError, ValueError):
        return None
    return max(0, lru - owned)


class LeakRecord:
    """Model variants seen leaking pinned memory on the running kernel.

    Kept per kernel release, so booting a fixed kernel starts clean.
    """

    def __init__(self, path: Path, release: str | None = None) -> None:
        self.path = path
        self.release = release or kernel_release()
        self.variants: dict[str, float] = {}  # variant -> GB leaked by one load
        try:
            data = json.loads(path.read_text())
            if data.get("kernel") == self.release:
                self.variants = {str(k): float(v) for k, v in data.get("variants", {}).items()}
        except (OSError, ValueError, AttributeError, TypeError):
            pass

    def __contains__(self, variant: str) -> bool:
        return variant in self.variants

    def add(self, variant: str, leaked: int) -> None:
        self.variants[variant] = round(leaked / GB, 1)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"kernel": self.release, "updated": time.time(),
                                             "variants": self.variants}, indent=1))
        except OSError as e:
            log.warning("cannot record pinned-memory leak in %s: %r", self.path, e)


def record_path(state_dir: Path) -> Path:
    return state_dir / "pinned_leaks.json"

