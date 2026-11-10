"""Copies of weight files, made when a job starts, so that the next job cannot destroy them.

Told in AGENTS.md to save each experiment under its own name, agents still reuse one file: LoRA runs 2 and 3
lost their best weights that way, and Loop run 9 lost a model with val 73.58% to a job that reached 71.32%.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
KEPT_DIR = Path(".yuzuru") / "kept-weights"


@dataclass
class Kept:
    source: Path
    copy: Path
    mtime_ns: int
    size: int
    told: bool = False


def weight_files(cwd: Path) -> list[Path]:
    """Weight files under `cwd`, without hidden directories (`.venv`, `.yuzuru`, `.git`)."""
    found = []
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        found += [Path(root) / f for f in files if Path(f).suffix in SUFFIXES]
    return sorted(found)


class WeightKeeper:
    def __init__(self, max_gb: float) -> None:
        self.max_bytes = int(max_gb * 1024 ** 3)
        self.kept: list[Kept] = []

    def _unchanged(self, path: Path, st: os.stat_result) -> bool:
        return any(k.source == path and k.mtime_ns == st.st_mtime_ns and k.size == st.st_size for k in self.kept)

    def keep(self, cwd: Path) -> None:
        """Copy the weight files under `cwd` that have no copy yet. Never raises: a job matters more than a copy."""
        if self.max_bytes <= 0:
            return
        try:
            self._keep(cwd)
        except OSError as e:
            log.warning("keeping the weights in %s failed: %s", cwd, e)

    def _keep(self, cwd: Path) -> None:
        for path in weight_files(cwd):
            st = path.stat()
            if self._unchanged(path, st):
                continue
            if sum(k.size for k in self.kept) + st.st_size > self.max_bytes:
                # Removing an older copy could remove the one that matters: the newest goes without.
                log.warning("not keeping %s: over resume.keep_weights_gb", path)
                continue
            stamp = time.strftime("%H%M%S", time.localtime(st.st_mtime))
            name = "__".join(path.relative_to(cwd).with_suffix("").parts)
            copy = cwd / KEPT_DIR / f"{name}.{stamp}{path.suffix}"
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            after = path.stat()
            if (after.st_mtime_ns, after.st_size) != (st.st_mtime_ns, st.st_size):
                copy.unlink()  # a running job was writing it: half a file is no copy
                continue
            self.kept.append(Kept(path, copy, st.st_mtime_ns, st.st_size))

    def news(self) -> str:
        """Copies whose file has changed since, once each: the agent may not know what it overwrote."""
        lines = []
        for k in self.kept:
            if k.told:
                continue
            try:
                st = k.source.stat()
            except OSError:
                changed = True
            else:
                changed = (st.st_mtime_ns, st.st_size) != (k.mtime_ns, k.size)
            if changed:
                k.told = True
                saved = time.strftime("%H:%M", time.localtime(k.mtime_ns / 1e9))
                lines.append(f"[yuzuru] {k.source.name} was overwritten or removed. The version saved at {saved} is "
                             f"kept at {k.copy}.\n")
        if not lines:
            return ""
        return ("".join(lines) + "If an older version came from a better run, copy it back before using the file, "
                "and give each experiment its own weights file.\n\n")
