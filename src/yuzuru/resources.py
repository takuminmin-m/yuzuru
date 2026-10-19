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


class Probe:
    name = "none"

    def snapshot(self) -> Snapshot:
        total, avail = read_ram()
        return Snapshot(gpus=self.gpus(), ram_total=total, ram_available=avail)

    def gpus(self) -> list[GPU]:
        return []


class NvidiaSmiProbe(Probe):
    """Uses nvidia-smi so there is no compiled dependency on NVML bindings."""

    name = "nvidia"

    def __init__(self, binary: str = "nvidia-smi") -> None:
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


def make_probe(kind: str = "auto") -> Probe:
    if kind == "nvidia" or (kind == "auto" and shutil.which("nvidia-smi")):
        return NvidiaSmiProbe()
    return Probe()
