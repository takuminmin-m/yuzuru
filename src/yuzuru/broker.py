"""Memory broker.

Owns the question "how much memory may the LLM use right now?" and enforces
the answer by loading/unloading Ollama models. Training jobs take a *lease*:
the broker evicts GPU-resident models, waits until the job's memory use
settles, then lets the LLM use whatever is left (minus headroom).

Placement inside the accelerator pool is left to Ollama, which already splits
models across any number of GPUs and spills to RAM; the broker only decides
*how big* a model may be loaded. That is what keeps this hardware-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .catalog import Catalog, ModelSpec
from .config import MB, Config
from .ollama import LoadedModel, Ollama, OllamaError
from .resources import Probe, Snapshot

log = logging.getLogger("yuzuru.broker")


class LoadFailed(RuntimeError):
    """Ollama could not load a model (e.g. CUDA out of memory despite the estimate)."""


LOAD_FAILURE_COOLDOWN = 600.0  # seconds a failed variant is not offered again


class Placement(str, Enum):
    GPU = "gpu"  # fits entirely in accelerator memory
    OFFLOAD = "offload"  # partly in system RAM (slower, fine for MoE)
    CPU = "cpu"  # by design (e.g. the classifier)
    NONE = "none"


@dataclass
class Budget:
    fast: int  # bytes of accelerator memory the LLM may occupy in total
    slow: int  # bytes of system RAM the LLM may occupy in total
    unified: bool  # no discrete GPU: the fast pool *is* system RAM


@dataclass
class Lease:
    id: str
    pid: int
    label: str
    gpus: list[str]  # GPU uuids; empty = the whole pool
    vram: int | None  # explicit reservation in bytes; None = measure it
    session: str | None = None  # OpenCode session that started the job, if any
    kind: str = "job"  # job | reservation (user-forced, no job yet)
    cwd: str | None = None  # where the job runs (for resuming its session)
    log: str | None = None  # the job's log file, if any
    resume: bool = False  # resume `session` when the job ends
    note: str = ""  # what the agent meant to do next
    hard: bool = False  # reservation backed by a holder process that occupies the VRAM
    expires: float | None = None
    created: float = field(default_factory=time.time)
    # job: draining (LLM leaving) -> settling (job allocating) -> active; reservation: reserved
    state: str = "draining"
    baseline: int = 0  # memory in use on the leased devices once the LLM left
    level: int = 0
    last_change: float = field(default_factory=time.time)
    restore: list[str] = field(default_factory=list)

    @property
    def grown(self) -> int:
        return max(0, self.level - self.baseline)


class Broker:
    def __init__(self, cfg: Config, probe: Probe, ollama: Ollama, catalog: Catalog) -> None:
        self.cfg = cfg
        self.probe = probe
        self.ollama = ollama
        self.catalog = catalog
        self.snapshot = Snapshot()
        self.loaded: list[LoadedModel] = []
        self.leases: dict[str, Lease] = {}
        self.inflight: Counter[str] = Counter()
        self.last_used: dict[str, float] = {}
        self.lock = asyncio.Lock()
        self._settled = asyncio.Event()
        self._settled.set()
        self._restore_task: asyncio.Task | None = None
        self._holders: dict[int, asyncio.subprocess.Process] = {}
        # Called with (lease, exit code or None) whenever a lease ends.
        self.on_lease_end: list[Callable[[Lease, int | None], None]] = []
        self.failed: dict[str, float] = {}  # variant -> time until which it is not offered

    # ------------------------------------------------------------------ state

    async def refresh(self) -> None:
        self.snapshot = await asyncio.to_thread(self.probe.snapshot)
        self.loaded = await self.ollama.ps()
        for m in self.loaded:
            self.catalog.record_loaded(m.name, m.size)
        self._update_leases()

    def is_failed(self, variant: str) -> bool:
        until = self.failed.get(variant)
        if until is not None and time.time() >= until:
            del self.failed[variant]
            return False
        return until is not None

    def is_loaded(self, variant: str) -> LoadedModel | None:
        return next((m for m in self.loaded if m.name == variant), None)

    def _device_level(self, gpus: list[str]) -> int:
        snap = self.snapshot
        if not snap.gpus:
            return snap.ram_total - snap.ram_available
        devs = [g for g in snap.gpus if not gpus or g.uuid in gpus]
        return sum(g.used for g in devs)

    def _lease_reserve(self, lease: Lease) -> int:
        snap = self.snapshot
        if lease.kind == "reservation":
            # A hard reservation's holder shows up as growth, so it nets to ~0 here.
            return max(0, (lease.vram or 0) - lease.grown)
        if lease.state in ("draining", "settling"):
            # Hold every free byte on the leased devices until the job settles.
            devs = [g for g in snap.gpus if not lease.gpus or g.uuid in lease.gpus]
            free = sum(g.free for g in devs) if snap.gpus else snap.ram_available
            return max(free, (lease.vram or 0) - lease.grown)
        return max(0, (lease.vram or 0) - lease.grown) + self.cfg.lease.headroom_mb * MB

    def budget(self) -> Budget:
        snap, res = self.snapshot, self.cfg.resources
        ollama_vram = sum(m.size_vram for m in self.loaded)
        # Only what unloading would actually give back (not memory Ollama leaked).
        ollama_ram = sum(self.host_of(m) for m in self.loaded)
        reserved = sum(self._lease_reserve(l) for l in self.leases.values())
        ram_for_llm = snap.ram_available + ollama_ram - res.ram_reserve_mb * MB
        if not snap.gpus:
            # Unified memory / CPU-only: one pool.
            fast = snap.ram_available + ollama_vram + ollama_ram - res.ram_reserve_mb * MB - reserved
            return Budget(fast=max(0, fast), slow=0, unified=True)
        foreign = max(0, snap.gpu_used - ollama_vram)
        margin = res.vram_margin_mb * MB * len(snap.gpus)
        fast = snap.gpu_total - foreign - reserved - margin
        return Budget(fast=max(0, fast), slow=max(0, ram_for_llm), unified=False)

    def ollama_host(self) -> int:
        """Host RAM held by Ollama: measured via its cgroup, else what /api/ps implies."""
        if self.snapshot.ollama_host is not None:
            return self.snapshot.ollama_host
        return sum(self.host_of(m) for m in self.loaded)

    def ollama_unattributed(self) -> int:
        """Host RAM Ollama holds beyond its loaded models (e.g. leaked by killed runners)."""
        if self.snapshot.ollama_host is None:
            return 0
        return max(0, self.snapshot.ollama_host - sum(self.host_of(m) for m in self.loaded))

    def host_of(self, m: LoadedModel) -> int:
        spec = self.catalog.by_variant(m.name)
        placement = "cpu" if m.size_vram == 0 else ("gpu" if m.size_vram >= m.size * 0.98 else "offload")
        if spec and placement in spec.measured_host:
            return spec.measured_host[placement]
        return m.size_ram

    # -------------------------------------------------------------- placement

    def _evictable(self, keep: str) -> list[LoadedModel]:
        """Idle models in eviction order: GPU-resident first (LRU), CPU-only ones last.

        CPU-only models (the classifier) hold no VRAM, so evicting them only
        helps when host RAM is what is short; plan() reaches them last.
        """
        ms = [m for m in self.loaded if m.name != keep and self.inflight[m.name] == 0]
        return sorted(ms, key=lambda m: (m.size_vram == 0, self.last_used.get(m.name, 0.0)))

    def place(self, spec: ModelSpec, offload_ok: bool, evict: list[LoadedModel] | None = None) -> Placement:
        """Where `spec` would end up if loaded now (after evicting `evict`)."""
        if loaded := self.is_loaded(spec.variant):
            if spec.cpu_only:
                return Placement.CPU
            return Placement.GPU if loaded.size_vram >= loaded.size * 0.98 else Placement.OFFLOAD
        if self.is_failed(spec.variant):
            return Placement.NONE
        gone = {m.name for m in (evict or [])}
        others = [m for m in self.loaded if m.name not in gone]
        b = self.budget()
        need = spec.estimated_size
        free_fast = b.fast - sum(m.size_vram for m in others)
        free_slow = b.slow - sum(self.host_of(m) for m in others)
        if spec.cpu_only:
            pool = free_fast if b.unified else free_slow
            return Placement.CPU if need <= pool else Placement.NONE
        if need <= free_fast and (b.unified or spec.host_need("gpu") <= free_slow):
            return Placement.GPU
        tier = self.cfg.tier(spec.tier)
        spill = need - max(0, free_fast)
        max_frac = tier.max_cpu_fraction if tier and tier.max_cpu_fraction is not None else (0.7 if spec.moe else 0.25)
        allowed = offload_ok and (tier is None or tier.allow_cpu_offload)
        if allowed and not b.unified and spill <= need * max_frac and spec.host_need("offload") <= free_slow:
            return Placement.OFFLOAD
        return Placement.NONE

    def plan(self, spec: ModelSpec, offload_ok: bool) -> tuple[Placement, list[LoadedModel]]:
        """Best placement for `spec`, evicting as few idle models as possible."""
        best = self.place(spec, offload_ok)
        if best in (Placement.GPU, Placement.CPU) or self.is_loaded(spec.variant):
            return best, []
        best_evict: list[LoadedModel] = []
        evict: list[LoadedModel] = []
        for m in self._evictable(spec.variant):
            evict.append(m)
            p = self.place(spec, offload_ok, evict)
            if p in (Placement.GPU, Placement.CPU):
                return p, list(evict)
            if p is Placement.OFFLOAD and best is Placement.NONE:
                best, best_evict = p, list(evict)
        return best, best_evict

    async def acquire(self, spec: ModelSpec, offload_ok: bool = True, force: bool = False) -> Placement:
        """Make sure `spec` is loaded and mark it in use.

        Pair with release() unless this returns Placement.NONE (nothing acquired).
        """
        self.inflight[spec.variant] += 1
        try:
            if self.is_loaded(spec.variant):
                return self.place(spec, offload_ok)
            if self.is_failed(spec.variant):
                raise LoadFailed(f"{spec.variant} failed to load recently")
            await self.wait_settled()
            async with self.lock:
                await self.refresh()
                placement, evict = self.plan(spec, offload_ok)
                if placement is Placement.NONE and not force:
                    self.inflight[spec.variant] -= 1  # nothing acquired; no release()
                    return placement
                for m in evict:
                    log.info("evicting %s to make room for %s", m.name, spec.variant)
                    await self.ollama.unload(m.name)
                if not self.is_loaded(spec.variant):
                    log.info("loading %s (%s, ~%.1f GB)", spec.variant, placement.value, spec.estimated_size / 2**30)
                    await self.refresh()
                    host_before = self.snapshot.ollama_host
                    try:
                        await self.ollama.load(spec.variant, self.cfg.keep_alive)
                    except OllamaError as e:
                        # Ollama's own placement can disagree with the estimate (another
                        # process, fragmented VRAM, a GPU it should not use...). Stop
                        # offering this model for a while so routing falls back.
                        self.failed[spec.variant] = time.time() + LOAD_FAILURE_COOLDOWN
                        log.warning("loading %s failed; not offering it for %.0f min: %s",
                                    spec.variant, LOAD_FAILURE_COOLDOWN / 60, str(e)[:300])
                        raise LoadFailed(str(e)) from e
                    await self.refresh()
                    if self.is_loaded(spec.variant) and host_before is not None and self.snapshot.ollama_host is not None:
                        actual = self.place(spec, offload_ok)
                        taken = self.snapshot.ollama_host - host_before
                        self.catalog.record_host(spec.variant, actual.value, taken)
                        log.info("%s took %.1f GB VRAM + %.1f GB host RAM", spec.variant,
                                 (self.is_loaded(spec.variant).size_vram) / 2**30, max(0, taken) / 2**30)
                return self.place(spec, offload_ok) if self.is_loaded(spec.variant) else placement
        except BaseException:
            self.inflight[spec.variant] -= 1
            raise

    async def release(self, spec: ModelSpec, touch: bool = True) -> None:
        self.inflight[spec.variant] = max(0, self.inflight[spec.variant] - 1)
        self.last_used[spec.variant] = time.time()
        if not touch:
            return
        # OpenAI-endpoint requests reset keep_alive to Ollama's default; restore ours.
        # The same call loads a model that is not resident, so only touch one that
        # still is: a lease or a wait may have unloaded it while this request ran,
        # and reloading it here would take the memory a training job just got.
        try:
            if not self._settled.is_set() or self.inflight[spec.variant]:
                return
            if spec.variant in {m.name for m in await self.ollama.ps()}:
                await self.ollama.load(spec.variant, self.cfg.keep_alive)
        except Exception as e:  # noqa: BLE001 - best effort
            log.debug("keep_alive refresh failed: %s", e)

    async def unload(self, variants: list[str] | None = None) -> list[str]:
        async with self.lock:
            await self.refresh()
            targets = [m.name for m in self.loaded if variants is None or m.name in variants]
            for name in targets:
                await self.ollama.unload(name)
            await self.refresh()
            return targets

    # ----------------------------------------------------------------- leases

    async def wait_settled(self) -> None:
        if not self._settled.is_set():
            try:
                await asyncio.wait_for(self._settled.wait(), self.cfg.lease.settle_timeout + 5)
            except asyncio.TimeoutError:
                log.warning("lease settle wait timed out; continuing")

    def _resolve_gpus(self, gpus: list[str]) -> list[str]:
        resolved = []
        for g in gpus:
            gpu = self.snapshot.gpu(g)
            if gpu is None:
                raise ValueError(f"unknown GPU {g!r}")
            resolved.append(gpu.uuid)
        return resolved

    def visible_gpus(self, lease: Lease) -> list[str]:
        """CUDA_VISIBLE_DEVICES for a lease's job: its GPUs, else every allowed one.

        Empty means "do not restrict" (no allowlist configured)."""
        if lease.gpus or not self.cfg.resources.gpus:
            return lease.gpus
        return [g.uuid for g in self.snapshot.gpus] or list(self.cfg.resources.gpus)

    async def _clear_gpus(self, lease: Lease, drain: bool = True) -> None:
        """Unload GPU-resident models for `lease` and take its baseline."""
        deadline = time.time() + (self.cfg.lease.drain_timeout if drain else 0)
        while time.time() < deadline and any(self.inflight[m.name] for m in self.loaded if m.size_vram > 0):
            await asyncio.sleep(0.5)
        async with self.lock:
            await self.refresh()
            lease.restore = [m.name for m in self.loaded if m.size_vram > 0]
            for name in lease.restore:
                log.info("lease %s: unloading %s", lease.id, name)
                await self.ollama.unload(name)
            await asyncio.sleep(1.0)  # let the driver actually free memory
            await self.refresh()
            lease.baseline = lease.level = self._device_level(lease.gpus)
            lease.last_change = time.time()

    async def create_lease(
        self, pid: int, label: str, gpus: list[str], vram: int | None, session: str | None = None,
        drain: bool = True, **extra: Any,
    ) -> Lease:
        await self.refresh()
        resolved = self._resolve_gpus(gpus)
        lease = Lease(id=uuid.uuid4().hex[:8], pid=pid, label=label, gpus=resolved, vram=vram, session=session, **extra)
        self._settled.clear()
        self.leases[lease.id] = lease
        await self._clear_gpus(lease, drain)
        lease.state = "settling"
        log.info("lease %s (%s, pid %d) created", lease.id, label, pid)
        return lease

    async def create_reservation(
        self, gpus: list[str], vram: int | None, ttl: float | None, label: str, hard: bool = True,
    ) -> Lease:
        """User-forced reservation of VRAM/GPUs, with or without a job.

        hard: a holder process allocates the memory so nothing can take it
        (NVIDIA only). soft: the budget subtracts it (any hardware). With no
        `vram`, the whole free memory of the chosen GPUs is reserved.
        """
        await self.refresh()
        resolved = self._resolve_gpus(gpus)
        if not resolved and len(self.snapshot.gpus) == 1:
            resolved = [self.snapshot.gpus[0].uuid]
        if hard and not resolved:
            raise ValueError("a hard reservation needs --gpu (several GPUs present)")
        if vram is None and not resolved:
            raise ValueError("give --vram, --gpu, or both")
        lease = Lease(
            id=uuid.uuid4().hex[:8], pid=os.getpid(), label=label, gpus=resolved, vram=vram, kind="reservation",
            hard=hard, state="reserved", expires=time.time() + ttl if ttl else None,
        )
        self.leases[lease.id] = lease
        try:
            await self._clear_gpus(lease, drain=False)
            devs = [g for g in self.snapshot.gpus if g.uuid in resolved]
            margin = self.cfg.resources.vram_margin_mb * MB
            if vram is None:
                sizes = [max(0, g.free - margin) for g in devs]
                lease.vram = sum(sizes)
            else:
                sizes = [vram // len(devs)] * len(devs) if devs else []
            if hard:
                lease.pid = await self._spawn_holder(resolved, sizes, ttl)
                await asyncio.sleep(1.0)
                await self.refresh()
                lease.level = self._device_level(resolved)
        except BaseException:
            self.leases.pop(lease.id, None)
            raise
        log.info("reservation %s: %.1f GB on %s (%s)", lease.id, (lease.vram or 0) / 2**30,
                 ",".join(resolved) or "pool", "hard" if hard else "soft")
        return lease

    async def _spawn_holder(self, gpus: list[str], sizes: list[int], ttl: float | None) -> int:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "yuzuru.holder", "--bytes", ",".join(map(str, sizes)), "--ttl", str(ttl or 0),
            env=env, stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), 120)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("VRAM holder did not start in time")
        info = json.loads(line or b'{"ok": false, "error": "holder exited"}')
        if not info.get("ok"):
            await proc.wait()
            raise RuntimeError(info.get("error", "holder failed"))
        self._holders[proc.pid] = proc
        return proc.pid

    async def shutdown(self) -> None:
        """Free hard reservations: holders run in their own session and would
        otherwise keep the VRAM after the server is gone, with nothing tracking them."""
        for pid in list(self._holders):
            await self._stop_holder(pid)
        self.leases = {k: l for k, l in self.leases.items() if not l.hard}

    async def _stop_holder(self, pid: int) -> None:
        proc = self._holders.pop(pid, None)
        if proc is None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def handover(self, lease_id: str, pid: int, session: str | None = None) -> Lease:
        """Turn a reservation into the lease of the job that will use it."""
        lease = self.leases.get(lease_id)
        if lease is None or lease.kind != "reservation":
            raise KeyError(lease_id)
        self._settled.clear()
        # Switch owner first: a refresh while the holder is gone must not reap the lease.
        holder, was_hard = lease.pid, lease.hard
        lease.kind, lease.hard, lease.pid, lease.expires = "job", False, pid, None
        lease.state = "draining"
        if was_hard:
            await self._stop_holder(holder)
            await asyncio.sleep(1.0)
            await self.refresh()
        lease.session = session or lease.session
        lease.baseline = lease.level = self._device_level(lease.gpus)
        lease.created = lease.last_change = time.time()
        lease.state = "settling"
        log.info("reservation %s handed to pid %d", lease_id, pid)
        return lease

    async def release_lease(self, lease_id: str, rc: int | None = None) -> Lease | None:
        lease = self.leases.pop(lease_id, None)
        if lease is None:
            return None
        if lease.hard:
            await self._stop_holder(lease.pid)
        log.info("lease %s released (rc=%s)", lease_id, rc)
        self._ended(lease, rc)
        return lease

    def _ended(self, lease: Lease, rc: int | None = None) -> None:
        for cb in self.on_lease_end:
            cb(lease, rc)
        self._update_leases()
        if self.cfg.lease.restore_after and not self.leases and lease.restore:
            self._restore_task = asyncio.create_task(self._restore(lease.restore))

    async def _restore(self, variants: list[str]) -> None:
        await asyncio.sleep(2.0)
        for v in variants:
            spec = self.catalog.by_variant(v)
            if spec and not self.leases:
                try:
                    if await self.acquire(spec) is not Placement.NONE:
                        await self.release(spec)
                except Exception as e:  # noqa: BLE001
                    log.warning("restore of %s failed: %s", v, e)

    def _update_leases(self) -> None:
        now = time.time()
        for lease in list(self.leases.values()):
            gone = not _pid_alive(lease.pid) or (lease.hard and lease.pid in self._holders
                                                  and self._holders[lease.pid].returncode is not None)
            expired = lease.expires is not None and now >= lease.expires
            if gone or expired:
                log.info("lease %s: %s", lease.id, "expired" if expired else f"pid {lease.pid} gone")
                self.leases.pop(lease.id)
                for cb in self.on_lease_end:
                    cb(lease, None)
                if lease.hard:
                    holder = self._holders.pop(lease.pid, None)
                    if holder and holder.returncode is None:
                        holder.terminate()
                if self.cfg.lease.restore_after and not self.leases and lease.restore:
                    try:
                        self._restore_task = asyncio.get_running_loop().create_task(self._restore(lease.restore))
                    except RuntimeError:
                        pass
                continue
            if lease.state == "draining":
                continue
            level = self._device_level(lease.gpus)
            if abs(level - lease.level) > self.cfg.lease.settle_tolerance_mb * MB:
                lease.last_change = now
            lease.level = level
            if lease.state == "settling":
                quiet = now - lease.last_change >= self.cfg.lease.settle_seconds
                allocated = lease.grown > self.cfg.lease.settle_tolerance_mb * MB
                if (quiet and allocated) or now - lease.created >= self.cfg.lease.settle_timeout:
                    lease.state = "active"
                    log.info("lease %s settled at +%.1f GB", lease.id, lease.grown / 2**30)
        if all(l.state in ("active", "reserved") for l in self.leases.values()):
            self._settled.set()
        else:
            self._settled.clear()

    async def relieve_pressure(self) -> list[str]:
        """Unload idle models if foreign jobs grew into the LLM's share."""
        b = self.budget()
        used = sum(m.size_vram for m in self.loaded)
        dropped: list[str] = []
        if b.unified or used <= b.fast:
            return dropped
        async with self.lock:
            for m in self._evictable(keep=""):
                if m.size_vram == 0:
                    continue
                log.warning("memory pressure: unloading %s", m.name)
                await self.ollama.unload(m.name)
                dropped.append(m.name)
                used -= m.size_vram
                if used <= b.fast:
                    break
            await self.refresh()
        return dropped

    async def monitor(self, interval: float = 2.0) -> None:
        while True:
            try:
                await self.refresh()
                await self.relieve_pressure()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep monitoring
                log.warning("monitor: %r", e)
            await asyncio.sleep(interval)

    def status(self) -> dict[str, Any]:
        b = self.budget()
        return {
            "gpus": [asdict(g) | {"free": g.free} for g in self.snapshot.gpus],
            "ram": {"total": self.snapshot.ram_total, "available": self.snapshot.ram_available},
            "budget": asdict(b),
            "loaded": [asdict(m) | {"inflight": self.inflight[m.name]} for m in self.loaded],
            "leases": [asdict(l) | {"grown": l.grown} for l in self.leases.values()],
            "tiers": {
                name: {
                    "model": s.base,
                    "variant": s.variant,
                    "num_ctx": s.num_ctx,
                    "estimated_size": s.estimated_size,
                    "measured": bool(s.measured_size),
                    "placement": self.plan(s, True)[0].value,
                }
                for name, s in self.catalog.specs.items()
            },
            "classifier": self.catalog.router_spec.variant if self.catalog.router_spec else None,
            "failed": {v: until for v, until in self.failed.items()},
            "warnings": self.warnings(),
        }

    def warnings(self) -> list[str]:
        out = [
            f"GPU {g} is not responding; yuzuru is using the remaining GPUs. A reboot (or `sudo nvidia-smi -r`) "
            "usually brings it back; repeated Xid 79 points at power, PCIe seating or riser issues"
            for g in self.snapshot.lost_gpus
        ]
        if (leak := self.ollama_unattributed()) > 2 * 2**30:
            out.append(
                f"Ollama holds {leak / 2**30:.1f} GB of RAM that no loaded model accounts for "
                "(memory from killed runners is sometimes not returned); restarting Ollama frees it, "
                "e.g. `sudo systemctl restart ollama`"
            )
        return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
