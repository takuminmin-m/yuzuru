"""Suspend the LLM while an agent waits on a process; resume the session when it ends.

A *watch* ties a process (a job's lease, or any PID) to an OpenCode session.
While anything is watched with `suspend`, every model is unloaded so the job
gets the whole machine. When the process ends, an event is recorded and, if
asked, the session's model is warmed and the session is continued with the
outcome (exit code, log tail, the agent's own note). OpenCode resends the
whole conversation on that turn, so the context comes back with it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import scope
from .broker import Broker, Lease, Placement, _pid_alive
from .config import Config

log = logging.getLogger("yuzuru.waits")


@dataclass
class Watch:
    id: str
    session: str | None
    pid: int | None = None
    lease: str | None = None
    label: str = "process"
    note: str = ""
    cwd: str | None = None
    log: str | None = None
    suspend: bool = True
    resume: bool = True
    created: float = field(default_factory=time.time)


@dataclass
class Event:
    id: int
    watch: str
    session: str | None
    label: str
    rc: int | None
    duration: float
    log: str | None
    log_tail: str
    note: str
    ts: float = field(default_factory=time.time)
    resumed: bool = False
    resume_error: str | None = None


def outside_env() -> dict[str, str]:
    """The environment for processes started for an agent (jobs, resumed sessions).

    A server started with `uv run yuzuru serve` has its own venv in VIRTUAL_ENV and
    first on PATH; a job inheriting them runs `python` / `uv run` in yuzuru's venv
    instead of the project's (ModuleNotFoundError for everything the project installed).
    """
    env = os.environ.copy()
    if sys.prefix == sys.base_prefix:
        return env
    if env.get("VIRTUAL_ENV") and Path(env["VIRTUAL_ENV"]).resolve() == Path(sys.prefix).resolve():
        del env["VIRTUAL_ENV"]
        env.pop("VIRTUAL_ENV_PROMPT", None)
    own_bin = (Path(sys.prefix) / "bin").resolve()
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p and Path(p).resolve() != own_bin)
    return env


def _proc_cwd(pid: int | None) -> str | None:
    if pid is None:
        return None
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def tail(path: str | None, cwd: str | None, lines: int) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute() and cwd:
        p = Path(cwd) / p
    try:
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64 * 1024))
            data = f.read().decode(errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])


class Waiter:
    def __init__(self, cfg: Config, broker: Broker, router: Any) -> None:
        self.cfg = cfg
        self.broker = broker
        self.router = router
        self.watches: dict[str, Watch] = {}
        self.events: deque[Event] = deque(maxlen=200)
        self._next = 1
        self._changed = asyncio.Event()
        self.suspended_models: list[str] = []
        self.suspend_pending = False  # suspend once the waiting sessions go quiet
        self._tasks: set[asyncio.Task] = set()
        # Outcomes held back until their session's turn ends: one resume reports them all.
        self._held: dict[str, list[tuple[Event, Watch]]] = {}
        broker.on_lease_end.append(self._lease_ended)

    # ---------------------------------------------------------------- watches

    async def add(self, **kw: Any) -> Watch:
        w = Watch(id=uuid.uuid4().hex[:8], **kw)
        if w.lease and w.lease not in self.broker.leases:
            raise KeyError(w.lease)
        if w.pid is not None and not _pid_alive(w.pid):
            raise ValueError(f"pid {w.pid} is not running")
        if not w.cwd:
            # A relative log ("train.log") is the process's, not the server's: resolve it where
            # the job runs, read now while the process is still there.
            lease = self.broker.leases.get(w.lease) if w.lease else None
            w.cwd = lease.cwd if lease and lease.cwd else _proc_cwd(lease.pid if lease else w.pid)
        if self._cpu_job(w):
            w.suspend = False  # e.g. `wait` on a job started with device "cpu"
        if same := self._watching(w):
            # e.g. `wait` on a job `train` already resumes for: one exit, one resume.
            log.info("watch %s already covers %s for session %s", same.id, w.label, w.session)
            same.note, same.log, same.cwd = same.note or w.note, same.log or w.log, same.cwd or w.cwd
            same.suspend, same.resume = same.suspend or w.suspend, same.resume or w.resume
            if w.suspend and self.cfg.resume.suspend_all:
                self.suspend_pending = True
            return same
        self.watches[w.id] = w
        log.info("watch %s: %s (session %s, suspend=%s, resume=%s)", w.id, w.label, w.session, w.suspend, w.resume)
        if w.suspend and self.cfg.resume.suspend_all:
            self.suspend_pending = True
        return w

    def watch_lease(self, lease: Lease) -> Watch | None:
        """Jobs started for a session with resume=True get a watch automatically."""
        if not (lease.resume and lease.session):
            return None
        for w in self.watches.values():
            if w.session == lease.session and w.pid == lease.pid and not w.lease:
                # The agent called `wait` on the job before its lease existed: take that watch over.
                w.lease, w.label = lease.id, lease.label
                w.note, w.log, w.cwd = w.note or lease.note, w.log or lease.log, w.cwd or lease.cwd
                w.resume = True
                w.suspend = w.suspend and not lease.cpu
                return w
        # A CPU-only job needs none of the LLM's memory: keep it loaded for the resumed turn.
        w = Watch(id=uuid.uuid4().hex[:8], session=lease.session, lease=lease.id, label=lease.label,
                  note=lease.note, cwd=lease.cwd, log=lease.log, suspend=not lease.cpu)
        self.watches[w.id] = w
        if w.suspend and self.cfg.resume.suspend_all:
            self.suspend_pending = True
        return w

    def _cpu_job(self, w: Watch) -> bool:
        """Whether `w` watches a CPU-only job's lease (by id, or by the job's pid)."""
        return any(l.cpu and (l.id == w.lease or w.pid is not None and l.pid == w.pid)
                   for l in self.broker.leases.values())

    def _watching(self, w: Watch) -> Watch | None:
        """An existing watch of the same session on the same process or lease."""
        def target(x: Watch) -> tuple[str, Any]:
            lease = self.broker.leases.get(x.lease) if x.lease else None
            return ("pid", lease.pid) if lease else (("lease", x.lease) if x.lease else ("pid", x.pid))
        return next((x for x in self.watches.values() if x.session == w.session and target(x) == target(w)), None)

    def _lease_ended(self, lease: Lease, rc: int | None) -> None:
        for w in [w for w in self.watches.values() if w.lease == lease.id]:
            self._finish(w, rc)

    async def poll(self) -> None:
        """Check plain-PID watches (no lease to tell us the exit); suspend when quiet."""
        for w in [w for w in self.watches.values() if w.pid is not None and not w.lease]:
            if not _pid_alive(w.pid):
                self._finish(w, None)
        # Only a watch that asked for it suspends (a CPU-only job's does not).
        if self.suspend_pending and any(w.suspend for w in self.watches.values()) and self._quiet():
            self.suspend_pending = False
            await self.suspend()

    def _quiet(self) -> bool:
        """No generation in flight and every waiting session idle for a while."""
        if any(self.broker.inflight.values()):
            return False
        cutoff = time.time() - self.cfg.resume.suspend_idle_seconds
        for w in self.watches.values():
            st = self.router.sessions.get(w.session) if w.session else None
            if st and st.updated > cutoff or self.router.in_turn(w.session, self.cfg.resume.turn_idle_seconds):
                return False
        return True

    def _finish(self, w: Watch, rc: int | None) -> None:
        self.watches.pop(w.id, None)
        ev = Event(
            id=self._next, watch=w.id, session=w.session, label=w.label, rc=rc,
            duration=time.time() - w.created, log=w.log,
            log_tail=tail(w.log, w.cwd, self.cfg.resume.log_tail_lines), note=w.note,
        )
        self._next += 1
        if not self.watches:
            self.suspend_pending = False
        self.events.append(ev)
        self._changed.set()
        self._changed = asyncio.Event()
        log.info("event %d: %s ended (rc=%s) after %.0fs", ev.id, w.label, rc, ev.duration)
        if w.resume and w.session and self.cfg.resume.enabled:
            self._spawn(self.resume(ev, w))
        elif not self.watches:
            self._spawn(self.unsuspend())

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -------------------------------------------------------- suspend/resume

    async def suspend(self) -> None:
        loaded = [m.name for m in self.broker.loaded]
        for name in loaded:
            if name not in self.suspended_models:
                self.suspended_models.append(name)
        if loaded:
            log.info("suspending LLM while waiting: unloading %s", ", ".join(loaded))
            await self.broker.unload()

    async def unsuspend(self) -> None:
        """Bring back what was resident before the wait (best effort)."""
        names, self.suspended_models = self.suspended_models, []
        for name in names:
            spec = self.broker.catalog.by_variant(name)
            if spec and not self.watches:
                try:
                    if await self.broker.acquire(spec) is not Placement.NONE:
                        await self.broker.release(spec)
                except Exception as e:  # noqa: BLE001
                    log.warning("reloading %s failed: %s", name, e)

    @staticmethod
    def _status(ev: Event) -> str:
        status = "finished" if ev.rc == 0 else (f"exited with code {ev.rc}" if ev.rc is not None else "exited")
        return f"{ev.label} {status} after {ev.duration / 60:.1f} min"

    def message(self, ev: Event, earlier: list[Event] | None = None) -> str:
        status = "finished" if ev.rc == 0 else (f"exited with code {ev.rc}" if ev.rc is not None else "exited")
        minutes = ev.duration / 60
        parts = [f"[yuzuru] The process you were waiting for ({ev.label}) {status} after {minutes:.1f} min."]
        if earlier:
            parts.append("Before that, while you were still working: " + "; ".join(self._status(e) for e in earlier) + ".")
        if ev.note:
            parts.append(f"Your note for this moment: {ev.note}")
        if ev.log_tail:
            parts.append(f"Last lines of {ev.log}:\n```\n{ev.log_tail}\n```")
        parts.append("Continue the task from here.")
        return "\n\n".join(parts)

    async def resume(self, ev: Event, w: Watch) -> None:
        earlier: list[Event] = []
        if ev.session in self._held:
            # Another outcome already waits for this turn to end; that one resume reports both.
            self._held[ev.session].append((ev, w))
            return
        if self.router.in_turn(ev.session, self.cfg.resume.turn_idle_seconds):
            self._held[ev.session] = [(ev, w)]
            try:
                await self._turn_over(ev.session)
            finally:
                held = self._held.pop(ev.session, [(ev, w)])
            # The agent kept working after the job ended and started another one: it has
            # seen this outcome, and the new job's own resume will continue the session.
            if any(x.session == ev.session for x in self.watches.values()):
                log.info("session %s moved on to another job; not resuming for %s",
                         ev.session, ", ".join(e.label for e, _ in held))
                return
            earlier = [e for e, _ in held[:-1]]
            ev, w = held[-1]
        if not self.watches:
            if self.cfg.resume.warm_before_resume:
                await self._warm(ev.session)
            self.suspended_models = []
        argv = [a.format(session=ev.session, message=self.message(ev, earlier), cwd=w.cwd or "")
                for a in self.cfg.resume.command]
        if await asyncio.to_thread(scope.use, self.cfg.isolation.scope):
            if outside := await asyncio.to_thread(scope.opencode_service_outside):
                log.warning("resuming session %s: %s", ev.session, scope.service_warning(outside))
            argv = scope.wrap(argv, scope.unit_name("agent"), self.cfg.isolation.memory_max, oom_continue=True)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=w.cwd or None, env=outside_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            ev.resumed = True
            log.info("resumed session %s (pid %d)", ev.session, proc.pid)
            # Reap it when the resumed turn ends, so long-running servers collect no zombies.
            self._spawn(proc.wait())
        except OSError as e:
            ev.resume_error = str(e)
            log.warning("could not resume session %s: %s", ev.session, e)

    async def _turn_over(self, session: str | None) -> None:
        """Wait until the session's client has finished its turn.

        `opencode run --session` on a session whose turn is still running starts a
        second agent loop next to the first: both edit the same files and start jobs.
        """
        idle = self.cfg.resume.turn_idle_seconds
        st = self.router.sessions.get(session) if session else None
        if st:
            log.info("session %s is still in its turn (%d replies being generated, last reply %s, %.0fs ago); "
                     "resuming once it ends", session, st.answering,
                     "called tools" if st.tool_calls else "ended", time.time() - st.updated)
        start = time.time()
        while self.router.in_turn(session, idle):
            await asyncio.sleep(1.0)
        log.info("session %s finished its turn after %.0fs", session, time.time() - start)

    async def _warm(self, session: str | None) -> None:
        st = self.router.sessions.get(session) if session else None
        tier = (st.pinned or st.tier) if st else None
        spec = self.broker.catalog.specs.get(tier or "") if tier else None
        names = [spec.variant] if spec else list(self.suspended_models)
        for name in names:
            s = self.broker.catalog.by_variant(name)
            if s:
                try:
                    if await self.broker.acquire(s) is not Placement.NONE:
                        await self.broker.release(s)
                except Exception as e:  # noqa: BLE001
                    log.warning("warming %s failed: %s", name, e)

    # ----------------------------------------------------------------- events

    async def wait_events(self, after: int, timeout: float) -> list[Event]:
        """Long-poll: events with id > after, waiting up to `timeout` for one."""
        found = [e for e in self.events if e.id > after]
        if found or timeout <= 0:
            return found
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except asyncio.TimeoutError:
            return []
        return [e for e in self.events if e.id > after]

    def status(self) -> dict[str, Any]:
        return {
            "watches": [asdict(w) for w in self.watches.values()],
            "suspended": bool(self.suspended_models),
            "recent_events": [asdict(e) | {"log_tail": e.log_tail[-500:]} for e in list(self.events)[-5:]],
        }
