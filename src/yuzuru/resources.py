"""Resource probes.

A probe reports accelerator memory (any number of GPUs, possibly zero) and
system RAM. Everything above this module only sees `Snapshot`, so adding a
backend (ROCm, Apple unified memory, ...) means adding one probe class.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MB = 1024 * 1024


@dataclass
class GPU:
    index: int
    uuid: str
    name: str
    total: int  # bytes
    used: int  # bytes

    @property
    def free(self) -> int:
        return max(0, self.total - self.used)


@dataclass
class Snapshot:
    gpus: list[GPU] = field(default_factory=list)
    ram_total: int = 0
    ram_available: int = 0
    ollama_host: int | None = None  # bytes of host RAM held by Ollama, if measurable
    ts: float = field(default_factory=time.time)

    @property
    def gpu_total(self) -> int:
        return sum(g.total for g in self.gpus)

    @property
    def gpu_used(self) -> int:
        return sum(g.used for g in self.gpus)

    def gpu(self, key: str | int) -> GPU | None:
        for g in self.gpus:
            if str(key) in (str(g.index), g.uuid):
                return g
        return None


def read_ram() -> tuple[int, int]:
    """(total, available) bytes of system RAM."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info.get("MemFree", 0))
    except (OSError, KeyError, ValueError):
        pass
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except ImportError:
        page = os.sysconf("SC_PAGE_SIZE")
        return page * os.sysconf("SC_PHYS_PAGES"), page * os.sysconf("SC_AVPHYS_PAGES")


def _ollama_cgroup() -> Path | None:
    """cgroup v2 directory of the running `ollama serve` (systemd unit, docker, ...)."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() != "ollama":
                continue
            for line in (proc / "cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                    if (path / "memory.current").exists():
                        return path
        except OSError:
            continue
    return None


class OllamaHostMemory:
    """Host RAM actually held by Ollama and its runners.

    Ollama's /api/ps only reports `size - size_vram` as the RAM part, but with
    partial offload llama.cpp may keep far more in host memory (on CUDA it can
    hold the whole model). The cgroup's memory.current is the honest number;
    returns None where it cannot be read (macOS, Windows, no permission).
    """

    def __init__(self, setting: str = "auto") -> None:
        self.setting = setting
        self._path: Path | None = None
        self._resolved = False

    def path(self) -> Path | None:
        if self.setting in ("", "none", "off"):
            return None
        if self.setting != "auto":
            return Path(self.setting)
        if not self._resolved or (self._path and not self._path.exists()):
            self._path, self._resolved = _ollama_cgroup(), True
        return self._path

    def read(self) -> int | None:
        path = self.path()
        if path is None:
            return None
        try:
            current = int((path / "memory.current").read_text())
            stat = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
            return max(0, current - int(stat.get("file", 0)))  # page cache is reclaimable
        except (OSError, ValueError):
            self._resolved = False
            return None


class Probe:
    name = "none"

    def __init__(self, ollama_cgroup: str = "auto") -> None:
        self.host = OllamaHostMemory(ollama_cgroup)

    def snapshot(self) -> Snapshot:
        total, avail = read_ram()
        return Snapshot(gpus=self.gpus(), ram_total=total, ram_available=avail, ollama_host=self.host.read())

    def gpus(self) -> list[GPU]:
        return []


class NvidiaSmiProbe(Probe):
    """Uses nvidia-smi so there is no compiled dependency on NVML bindings."""

    name = "nvidia"

    def __init__(self, ollama_cgroup: str = "auto", binary: str = "nvidia-smi") -> None:
        super().__init__(ollama_cgroup)
        self.binary = binary

    def gpus(self) -> list[GPU]:
        out = subprocess.run(
            [
                self.binary,
                "--query-gpu=index,uuid,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        gpus = []
        for line in out.strip().splitlines():
            idx, uuid, name, total, used = (p.strip() for p in line.split(","))
            gpus.append(GPU(int(idx), uuid, name, int(total) * MB, int(used) * MB))
        return gpus


def make_probe(kind: str = "auto", ollama_cgroup: str = "auto") -> Probe:
    if kind == "nvidia" or (kind == "auto" and shutil.which("nvidia-smi")):
        return NvidiaSmiProbe(ollama_cgroup)
    return Probe(ollama_cgroup)
