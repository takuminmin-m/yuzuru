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
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .catalog import Catalog, ModelSpec
from .config import MB, Config
from .ollama import LoadedModel, Ollama
from .resources import Probe, Snapshot

log = logging.getLogger("yuzuru.broker")


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
    created: float = field(default_factory=time.time)
    state: str = "draining"  # draining (LLM leaving) -> settling (job allocating) -> active
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

    # ------------------------------------------------------------------ state

    async def refresh(self) -> None:
        self.snapshot = await asyncio.to_thread(self.probe.snapshot)
        self.loaded = await self.ollama.ps()
        for m in self.loaded:
            self.catalog.record_loaded(m.name, m.size)
        self._update_leases()

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
        if lease.state in ("draining", "settling"):
            # Hold every free byte on the leased devices until the job settles.
            devs = [g for g in snap.gpus if not lease.gpus or g.uuid in lease.gpus]
            free = sum(g.free for g in devs) if snap.gpus else snap.ram_available
            return max(free, (lease.vram or 0) - lease.grown)
        return max(0, (lease.vram or 0) - lease.grown) + self.cfg.lease.headroom_mb * MB

    def budget(self) -> Budget:
        snap, res = self.snapshot, self.cfg.resources
        ollama_vram = sum(m.size_vram for m in self.loaded)
        ollama_ram = sum(m.size_ram for m in self.loaded)
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

    # -------------------------------------------------------------- placement

    def _evictable(self, keep: str) -> list[LoadedModel]:
        ms = [m for m in self.loaded if m.name != keep and self.inflight[m.name] == 0]
        return sorted(ms, key=lambda m: self.last_used.get(m.name, 0.0))

    def place(self, spec: ModelSpec, offload_ok: bool, evict: list[LoadedModel] | None = None) -> Placement:
        """Where `spec` would end up if loaded now (after evicting `evict`)."""
        if loaded := self.is_loaded(spec.variant):
            if spec.cpu_only:
                return Placement.CPU
            return Placement.GPU if loaded.size_vram >= loaded.size * 0.98 else Placement.OFFLOAD
        gone = {m.name for m in (evict or [])}
        others = [m for m in self.loaded if m.name not in gone]
        b = self.budget()
        need = spec.estimated_size
        free_fast = b.fast - sum(m.size_vram for m in others)
        free_slow = b.slow - sum(m.size_ram for m in others)
        if spec.cpu_only:
            pool = free_fast if b.unified else free_slow
            return Placement.CPU if need <= pool else Placement.NONE
        if need <= free_fast:
            return Placement.GPU
        tier = self.cfg.tier(spec.tier)
        spill = need - max(0, free_fast)
        max_frac = tier.max_cpu_fraction if tier and tier.max_cpu_fraction is not None else (0.7 if spec.moe else 0.25)
        allowed = offload_ok and (tier is None or tier.allow_cpu_offload)
        if allowed and not b.unified and spill <= need * max_frac and spill <= free_slow:
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
                    await self.ollama.load(spec.variant, self.cfg.keep_alive)
                await self.refresh()
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
        try:
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

    async def create_lease(
        self, pid: int, label: str, gpus: list[str], vram: int | None, session: str | None = None
    ) -> Lease:
        await self.refresh()
        resolved = []
        for g in gpus:
            gpu = self.snapshot.gpu(g)
            if gpu is None:
                raise ValueError(f"unknown GPU {g!r}")
            resolved.append(gpu.uuid)
        lease = Lease(id=uuid.uuid4().hex[:8], pid=pid, label=label, gpus=resolved, vram=vram, session=session)
        self._settled.clear()
        self.leases[lease.id] = lease

        deadline = time.time() + self.cfg.lease.drain_timeout
        while time.time() < deadline and any(
            self.inflight[m.name] for m in self.loaded if m.size_vram > 0
        ):
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
            lease.state = "settling"
        log.info("lease %s (%s, pid %d) created", lease.id, label, pid)
        return lease

    async def release_lease(self, lease_id: str) -> Lease | None:
        lease = self.leases.pop(lease_id, None)
        if lease is None:
            return None
        log.info("lease %s released", lease_id)
        self._update_leases()
        if self.cfg.lease.restore_after and not self.leases and lease.restore:
            self._restore_task = asyncio.create_task(self._restore(lease.restore))
        return lease

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
            if not _pid_alive(lease.pid):
                log.info("lease %s: pid %d gone", lease.id, lease.pid)
                self.leases.pop(lease.id)
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
        if all(l.state == "active" for l in self.leases.values()):
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
        }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
