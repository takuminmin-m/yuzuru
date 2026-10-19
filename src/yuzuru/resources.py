"""Resource probes.

A probe reports accelerator memory (any number of GPUs, possibly zero) and
system RAM. Everything above this module only sees `Snapshot`, so adding a
backend (ROCm, Apple unified memory, ...) means adding one probe class.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

MB = 1024 * 1024

log = logging.getLogger("yuzuru.resources")


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
    lost_gpus: list[str] = field(default_factory=list)  # present but not answering
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


def _cgroup_of(pid: int | str) -> Path | None:
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    except OSError:
        pass
    return None


def _process_tree_anon(root: int) -> int | None:
    """Anonymous + shared RSS of `root` and its descendants (Ollama runners are children)."""
    parents: dict[int, int] = {}
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            try:
                parents[int(proc.name)] = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
    if root not in parents:
        return None
    tree, frontier = {root}, [root]
    while frontier:
        kids = [p for p, pp in parents.items() if pp in frontier and p not in tree]
        tree.update(kids)
        frontier = kids
    total = 0
    for pid in tree:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("RssAnon:", "RssShmem:")):
                    total += int(line.split()[1]) * 1024
        except (OSError, ValueError):
            continue
    return total


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

    def __init__(self, setting: str = "auto", pid: Callable[[], int | None] | None = None) -> None:
        self.setting = setting
        # With a yuzuru-managed Ollama: its pid, so the system ollama.service is not measured.
        self.pid = pid
        self._path: Path | None = None
        self._resolved = False

    def path(self) -> Path | None:
        if self.setting in ("", "none", "off"):
            return None
        if self.setting != "auto":
            return Path(self.setting)
        if self.pid is not None:
            return self._managed_cgroup()
        if not self._resolved or (self._path and not self._path.exists()):
            self._path, self._resolved = _ollama_cgroup(), True
        return self._path

    def _managed_cgroup(self) -> Path | None:
        """The managed server's cgroup, only if it is its own (a systemd scope):
        one shared with yuzuru would count yuzuru and whatever else runs there."""
        pid = self.pid() if self.pid else None
        if pid is None:
            return None
        path = _cgroup_of(pid)
        if path is None or path == _cgroup_of("self") or not (path / "memory.current").exists():
            return None
        return path

    def read(self) -> int | None:
        path = self.path()
        if path is None:
            if self.pid is not None and self.setting == "auto" and (pid := self.pid()) is not None:
                return _process_tree_anon(pid)
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

    def __init__(self, ollama_cgroup: str = "auto", allow: Iterable[str] = ()) -> None:
        self.host = OllamaHostMemory(ollama_cgroup)
        # GPU allowlist (indices or UUIDs); empty = every GPU.
        self.allow = [str(a).lower() for a in allow]
        self._unmatched_warned = False

    def allowed(self, gpus: list[GPU]) -> list[GPU]:
        if not self.allow:
            return gpus
        keep = [g for g in gpus if str(g.index) in self.allow or g.uuid.lower() in self.allow]
        unmatched = set(self.allow) - {str(g.index) for g in keep} - {g.uuid.lower() for g in keep}
        if unmatched and not self._unmatched_warned:
            log.warning("resources.gpus: no such GPU %s (present: %s)", sorted(unmatched),
                        ", ".join(f"{g.index}={g.uuid}" for g in gpus))
            self._unmatched_warned = True
        return keep

    def snapshot(self) -> Snapshot:
        total, avail = read_ram()
        gpus = self.allowed(self.gpus())
        return Snapshot(gpus=gpus, ram_total=total, ram_available=avail, ollama_host=self.host.read(),
                        lost_gpus=list(getattr(self, "lost", [])))  # only NvidiaSmiProbe tracks this

    def gpus(self) -> list[GPU]:
        return []


class NvidiaSmiProbe(Probe):
    """Uses nvidia-smi so there is no compiled dependency on NVML bindings."""

    name = "nvidia"

    QUERY = "--query-gpu=index,uuid,name,memory.total,memory.used"

    def __init__(self, ollama_cgroup: str = "auto", binary: str = "nvidia-smi", allow: Iterable[str] = ()) -> None:
        super().__init__(ollama_cgroup, allow)
        self.binary = binary
        self.lost: list[str] = []  # GPUs that stopped answering (e.g. Xid 79, fallen off the bus)

    def _query(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.binary, *extra, self.QUERY, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )

    @staticmethod
    def _parse(out: str) -> list[GPU]:
        gpus = []
        for line in out.strip().splitlines():
            idx, uuid, name, total, used = (p.strip() for p in line.split(","))
            gpus.append(GPU(int(idx), uuid, name, int(total) * MB, int(used) * MB))
        return gpus

    def gpus(self) -> list[GPU]:
        r = self._query()
        if r.returncode == 0:
            self.lost = []
            return self._parse(r.stdout)
        # One dead GPU makes the bulk query fail; ask the others one by one so
        # the LLM keeps running on what is left instead of everything erroring.
        listing = subprocess.run([self.binary, "-L"], capture_output=True, text=True, timeout=10)
        # Query by index: selecting by UUID still touches the dead device and fails.
        indices = re.findall(r"^GPU (\d+): ", listing.stdout, re.M)
        if not indices:
            raise RuntimeError(f"nvidia-smi failed: {(r.stderr or r.stdout).strip()}")
        gpus = []
        for index in indices:
            one = self._query("-i", index)
            if one.returncode == 0:
                gpus.extend(self._parse(one.stdout))
        errors = "\n".join((r.stderr, r.stdout, listing.stdout, listing.stderr))
        found = re.findall(r"for (?:GPU|gpu) ?([0-9a-fA-F:.]+): (.+)", errors)
        self.lost = sorted({f"{bus} ({msg.strip()})" for bus, msg in found})
        return gpus


def make_probe(kind: str = "auto", ollama_cgroup: str = "auto", allow: Iterable[str] = ()) -> Probe:
    if kind == "nvidia" or (kind == "auto" and shutil.which("nvidia-smi")):
        return NvidiaSmiProbe(ollama_cgroup, allow=allow)
    return Probe(ollama_cgroup, allow)
