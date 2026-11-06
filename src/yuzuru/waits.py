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
import contextlib
import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import scope
from .broker import Broker, Lease, Placement, _pid_alive
from .config import Config

log = logging.getLogger("yuzuru.waits")

STALLED = ("[yuzuru] Your last turn ended after a single reply with no tool call, and no job of yours is running: "
           "nothing continues this session unless you do. Continue the task from where you left off. If the task "
           "is finished, say so and summarize the result.")

LLM_BACK = ("[yuzuru] Your turn stopped because the local LLM could not be loaded. It can be loaded again now. "
            "Continue the task from where you left off.")

LOAD_RETRIES = 6  # resumes in a row retried after a failed LLM load, one per LOAD_FAILURE_COOLDOWN: about an hour

TIME_LIMIT_RC = 124  # what `timeout` exits with
EPOCHS_ARG = re.compile(r"--(?:num[-_])?epochs?[= ]+(\d+)")
OVERRUN_CHECK_SECONDS = 30.0  # how often a running job's log is read for its progress
OVERRUN_MIN_EPOCHS = 3  # epochs seen go by before their speed is trusted
# Options that only name a run or say where its outputs go: two commands that differ only in these train the same.
RUN_NAME_ARG = re.compile(
    r"\s--(?:name|run[-_]?name|exp(?:[-_]?name)?|tag|out(?:put)?(?:[-_]?dir)?|save(?:[-_]?(?:dir|path))?|"
    r"ckpt|checkpoint|log(?:[-_]?(?:dir|file))?)(?:[= ]+(?:\"[^\"]*\"|'[^']*'|\S+))")
SCRIPT_ARG = re.compile(r"[\w./-]+\.py\b")


@dataclass
class Started:
    """A GPU job as it was started: what it ran and the scripts it ran, to tell a repeat from a change."""

    session: str
    log: str
    command: str
    key: str
    scripts: str
    refused: bool = False


def run_key(command: str) -> str:
    """`command` without the options that only name the run, with its spaces evened out."""
    return " ".join(RUN_NAME_ARG.sub("", " " + command).split())


def scripts_digest(command: str, cwd: str | Path) -> str:
    """The contents of the Python scripts `command` names, so that a command run again after a fix is not a repeat."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(set(SCRIPT_ARG.findall(command))):
        p = Path(cwd) / name
        with contextlib.suppress(OSError):
            h.update(name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def epochs_done(text: str, total: int) -> int | None:
    """The last "k/total" in a log ("Ep 12/200", "[epoch] 44/45", "E3/100"): the epochs finished so far."""
    found = re.findall(rf"(?<![\d.])(\d+)\s*/\s*{total}(?![\d.])", text)
    done = [int(k) for k in found if 0 < int(k) <= total]
    return done[-1] if done else None


@dataclass
class Overrun:
    epochs: int  # the job's --epochs
    done: int  # epochs finished when it was stopped
    per_epoch: float
    need: float  # seconds the job would have taken
    limit: float
    fit: int  # epochs that end in about 8/10 of the limit

    def text(self) -> str:
        return (f"stopped early by yuzuru after {self.done} of {self.epochs} epochs: at {self.per_epoch:.0f} s per "
                f"epoch it needs about {self.need / 60:.0f} min, over the time limit of {self.limit / 60:.0f} min, "
                "and would have been stopped there with its learning rate still high. Start it again with "
                f"--epochs {self.fit} or fewer (about 8/10 of the limit on this GPU; a slower GPU needs fewer)")


TIME_LIMIT_TEXT = ("stopped by yuzuru: it reached the task's time limit for a job. It did not finish, so do not compare "
                   "it with finished runs; start the next job with fewer epochs")

UNTIL = ("[yuzuru] Your turn ended and no job of yours is running, but this task goes on until {until} "
         "({minutes} min from now). Do not finish early: check the end conditions in README.md and use the time "
         "left (start the next job, then end your turn). Keep the deliverables up to date as you go.")

FINISH = ("[yuzuru] Your turn ended and no job of yours is running. New GPU jobs ended at {until}, but the task "
          "ends at {finish_by} ({minutes} min from now). Check that every deliverable README.md and AGENTS.md ask "
          "for exists and is correct, and make what is missing now from the best results you have. Inference can "
          "run through yuzuru_train with device 'cpu'. If everything is there, say so and end your turn.")


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
    repeats: int = 0  # `wait` calls on it after the first


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
    delivered: bool = False  # answered a held `wait` call: the agent has it already
    told: bool = False  # a limit stop named in a resume or in another tool's answer
    refused: bool = False  # refused a job with as many epochs, which the agent may repeat
    overrun: Overrun | None = None  # stopped early: its epochs would not fit the time limit


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


def pin_gpus(env: dict[str, str], visible: str) -> None:
    """Show the process only `visible`, and keep its Python from switching to other GPUs
    (gpu_pin/sitecustomize.py, first on its PYTHONPATH)."""
    env["CUDA_VISIBLE_DEVICES"] = env["YUZURU_CUDA_VISIBLE_DEVICES"] = visible
    pin = str(Path(__file__).with_name("gpu_pin"))
    rest = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != pin]
    env["PYTHONPATH"] = os.pathsep.join([pin, *rest])


def agent_env(cfg: Config) -> dict[str, str]:
    """The environment for an agent (`yuzuru agent`, resumed turns): what it runs itself sees no GPU.

    An agent that runs `python train.py` from its shell instead of through the train tool trains
    without a lease, next to the LLM, until the shell tool's timeout kills it (in LoRA run 3 after
    it had overwritten the best weights). With no GPU visible such a script fails at once and
    points the agent at the train tool; jobs the train tool starts get their GPUs from yuzuru.
    OpenCode 2.x runs the agent's tools in its background service, so this has to reach
    `yuzuru agent -- opencode service start`.
    """
    env = outside_env()
    if cfg.isolation.hide_gpus_from_agent:
        pin_gpus(env, "")
        env["YUZURU_AGENT"] = "1"
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
        self.nudges: dict[str, int] = {}  # session -> stalled turns continued in a row
        self.finish_nudges: dict[str, int] = {}  # session -> turns continued between resume.until and finish_by
        # session -> its latest resume: a client yuzuru started earlier can stay attached while the session
        # goes on (HPO run 1 had eight), and only the latest one may decide to continue the session.
        self.latest_resume: dict[str, asyncio.subprocess.Process] = {}
        # session -> (client, replies before, time) of a resume whose turn has not finished a reply yet. Until
        # then the router does not see the turn (the client starts, the model loads), and a second job ending
        # there would resume the session a second time (HPO run 2: two clients 10 s apart).
        self.resuming: dict[str, tuple[asyncio.subprocess.Process, int, float]] = {}
        self.holding: Counter[str] = Counter()  # session -> `wait` calls being held
        self.load_retries: dict[str, int] = {}  # session -> resumes in a row lost to a failed LLM load
        self.job_epochs: dict[str, int] = {}  # log -> the job's --epochs
        self.job_limits: dict[str, float] = {}  # log -> the job's time limit in seconds
        self.progress: dict[str, tuple[float, int]] = {}  # watch -> (time, epochs) when first seen going
        self.progress_read: dict[str, float] = {}  # watch -> when its log was last read
        self.overruns: dict[str, Overrun] = {}  # lease -> why yuzuru stopped it
        self.started: list[Started] = []  # GPU jobs started, for `refuse_repeat`
        self.undelivered: dict[str, str] = {}  # session -> what such a resume was to say
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
        if w.session:
            self.nudges.pop(w.session, None)  # it is waiting for something: not stalled
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
        self.nudges.pop(lease.session, None)  # it started a job: not stalled
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
        self.check_overruns()
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

    async def hold(self, w: Watch, seconds: float) -> list[Event]:
        """Hold a `wait` call on `w` up to `seconds`, until a job of w's session ends; the events
        of those that did (w's own, or another job the agent would want to know about: a GPU freed).

        Their outcome goes back in the call, so they do not resume the session afterwards.
        """
        session = w.session or ""
        mine = {x.id for x in self.watches.values() if x.session == w.session} | {w.id}
        end = time.time() + seconds
        self.holding[session] += 1
        try:
            while mine <= set(self.watches) and (left := end - time.time()) > 0:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._changed.wait(), left)
        finally:
            self.holding[session] -= 1
            if self.holding[session] <= 0:
                del self.holding[session]
        return [e for e in self.events if e.watch in mine and e.delivered]

    def _finish(self, w: Watch, rc: int | None) -> None:
        self.watches.pop(w.id, None)
        ev = Event(
            id=self._next, watch=w.id, session=w.session, label=w.label, rc=rc,
            duration=time.time() - w.created, log=w.log,
            log_tail=tail(w.log, w.cwd, self.cfg.resume.log_tail_lines), note=w.note,
        )
        self._next += 1
        self.progress.pop(w.id, None)
        self.progress_read.pop(w.id, None)
        if w.lease and (over := self.overruns.pop(w.lease, None)):
            ev.rc, ev.overrun = TIME_LIMIT_RC, over  # a stop like the limit's, with the epochs that fit
        if not self.watches:
            self.suspend_pending = False
        if w.session and self.holding[w.session]:
            ev.delivered = True  # a held `wait` call of the session reports it
        self.events.append(ev)
        self._changed.set()
        self._changed = asyncio.Event()
        log.info("event %d: %s ended (rc=%s) after %.0fs", ev.id, w.label, rc, ev.duration)
        if w.resume and w.session and self.cfg.resume.enabled and not ev.delivered:
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
    def _ended(ev: Event) -> str:
        if over := getattr(ev, "overrun", None):
            return over.text()
        if ev.rc == TIME_LIMIT_RC:
            return TIME_LIMIT_TEXT
        return "finished" if ev.rc == 0 else (f"exited with code {ev.rc}" if ev.rc is not None else "exited")

    @staticmethod
    def _status(ev: Event) -> str:
        status = Waiter._ended(ev)
        return f"{ev.label} {status} after {ev.duration / 60:.1f} min"

    def note_job(self, log: str | None, command: str, limit: float | None = None) -> None:
        """Remember the epochs a job was started with (for `refuse_after_stop`) and its time limit."""
        if log and (m := EPOCHS_ARG.search(command)):
            self.job_epochs[log] = int(m.group(1))
            if limit is not None:
                self.job_limits[log] = limit
            else:
                self.job_limits.pop(log, None)

    def check_overruns(self, now: float | None = None) -> None:
        """Stop a GPU job early whose epochs will not fit its time limit, at the speed its log shows.

        Such a job runs its whole limit and ends with its learning rate still high: neither a result nor a
        comparison (Loop run 12 lost four of its five 45-min jobs so). The agent cannot stop a job itself.
        A job a little over (within `overrun_margin`) runs on: near its end the learning rate is low already.
        """
        margin = self.cfg.resume.overrun_margin
        if margin <= 0:
            return
        now = time.time() if now is None else now
        for w in list(self.watches.values()):
            epochs, limit = self.job_epochs.get(w.log or ""), self.job_limits.get(w.log or "")
            lease = self.broker.leases.get(w.lease) if w.lease else None
            if not (epochs and limit and lease) or lease.cpu or w.lease in self.overruns:
                continue
            elapsed = now - w.created
            if elapsed > limit / 2 or now - self.progress_read.get(w.id, 0.0) < OVERRUN_CHECK_SECONDS:
                continue
            self.progress_read[w.id] = now
            done = epochs_done(tail(w.log, w.cwd, 30), epochs)
            if done is None:
                continue
            first_at, first = self.progress.setdefault(w.id, (now, done))
            if done - first < OVERRUN_MIN_EPOCHS:
                continue
            per = (now - first_at) / (done - first)
            need = elapsed + (epochs - done) * per
            if need <= limit * (1 + margin):
                continue
            startup = max(0.0, elapsed - done * per)
            fit = max(1, int((0.8 * limit - startup) / per))
            over = Overrun(epochs=epochs, done=done, per_epoch=per, need=need, limit=limit, fit=fit)
            self.overruns[lease.id] = over
            log.info("stopping lease %s (pid %s) early: %s", lease.id, lease.pid, over.text())
            try:
                os.kill(lease.pid, signal.SIGTERM)  # `timeout`: it passes the signal on to the job's process group
            except OSError as e:
                log.warning("stopping pid %s: %s", lease.pid, e)
                self.overruns.pop(lease.id, None)

    def refuse_after_stop(self, session: str | None, command: str) -> str | None:
        """Why a job with as many epochs as a job stopped at the time limit should not start, once per stop.

        Loop run 8: told in every resume to use fewer epochs, the agent kept 150 and all five jobs were stopped.
        A refusal is read where a line in an answer is not; a job that fits a faster GPU can simply be sent again.
        """
        m = EPOCHS_ARG.search(command)
        if not session or not m:
            return None
        stops = [e for e in self.events if e.session == session and e.rc == TIME_LIMIT_RC and not e.refused
                 and self.job_epochs.get(e.log or "", int(m.group(1)) + 1) <= int(m.group(1))]
        for e in stops:
            e.refused = True
        if not stops:
            return None
        last = stops[-1]
        line = next((x for x in reversed(last.log_tail.splitlines()) if x.strip()), "")
        how = (f"was stopped early, after {last.overrun.done} epochs: at {last.overrun.per_epoch:.0f} s per epoch it "
               f"needed about {last.overrun.need / 60:.0f} min" if last.overrun else
               f"was stopped at the time limit after {last.duration / 60:.0f} min")
        return (f"not started: {last.log} used --epochs {self.job_epochs[last.log]} and {how}, "
                f"before its learning rate finished falling. Its last line: {line!r}. "
                f"This command has --epochs {m.group(1)}. From the seconds per epoch in that log, work out the epochs that "
                "fit, leave a margin, and use those from now on. If this job really fits (a faster GPU, a smaller "
                "model), send the same command again.")

    def note_started(self, session: str | None, log: str | None, command: str, cwd: str | Path) -> None:
        """Remember a GPU job of `session`, for `refuse_repeat`."""
        if session and log:
            self.started.append(Started(session=session, log=log, command=command, key=run_key(command),
                                        scripts=scripts_digest(command, cwd)))

    def refuse_repeat(self, session: str | None, command: str, cwd: str | Path) -> str | None:
        """Why a GPU job that trains the same as one this session already ran should not start, once per such job.

        Loop run 17: of the 3 hours, four 75-100 epoch jobs ran one setting under four names (r18, cutmix,
        r18_v2, r18_v3: 76.2-76.9%), while the rule to check RESULTS.md for the same setting was in AGENTS.md.
        The same command after a change to its script is not a repeat, nor one after a job that failed or was
        stopped at its time limit (`refuse_after_stop` answers that one).
        """
        if not session:
            return None
        key, digest = run_key(command), scripts_digest(command, cwd)
        ended = {e.log: e for e in self.events if e.session == session and e.log}
        same = [s for s in self.started if s.session == session and not s.refused and s.key == key
                and s.scripts == digest and (s.log not in ended or ended[s.log].rc == 0)]
        if not same:
            return None
        for s in same:
            s.refused = True
        last = same[-1]
        e = ended.get(last.log)
        line = next((x for x in reversed(tail(last.log, str(cwd), 20).splitlines()) if x.strip()), "")
        how = "is still running" if e is None else f"finished after {e.duration / 60:.0f} min"
        return (f"not started: this trains the same as {last.log}, which {how}: the same command apart from the "
                f"run's name ({last.command!r}), and the same scripts. Its last line: {line!r}. Running one setting "
                "again only shows how much results vary from run to run. Change one setting for the next experiment "
                "(look in RESULTS.md for what was tried), and check that the change is in the command or the script. "
                "If you do mean to measure that variation, send the same command again.")

    def stop_news(self, session: str | None) -> str:
        """Jobs of `session` stopped at the time limit that the agent has not been told about.

        A job that ends during a turn is held for the turn's end, and is dropped once the agent starts
        another job: in Loop run 7 two stops never reached it, and it started the next job with the same epochs.
        """
        unseen = [e for e in self.events
                  if session and e.session == session and e.rc == TIME_LIMIT_RC and not (e.delivered or e.told)]
        for e in unseen:
            e.told = True
        return "".join(f"[yuzuru] {e.log or e.label}: {self._ended(e)} (after {e.duration / 60:.1f} min)\n\n"
                       for e in unseen)

    def message(self, ev: Event, earlier: list[Event] | None = None) -> str:
        for e in [ev, *(earlier or [])]:
            e.told = True
        status = Waiter._ended(ev)
        minutes = ev.duration / 60
        parts = [f"[yuzuru] The process you were waiting for ({ev.label}) {status} after {minutes:.1f} min."]
        if earlier:
            parts.append("Before that, while you were still working: " + "; ".join(self._status(e) for e in earlier) + ".")
        if ev.note:
            parts.append(f"Your note for this moment: {ev.note}")
        if ev.log_tail:
            parts.append(f"Last lines of {ev.log}:\n```\n{ev.log_tail}\n```")
        if len(self.broker.snapshot.gpus) > 1:
            parts.append(self.broker.gpu_usage_line())
        parts.append("Continue the task from here.")
        return "\n\n".join(parts)

    async def resume(self, ev: Event, w: Watch) -> None:
        earlier: list[Event] = []
        if ev.session in self._held:
            # Another outcome already waits for this turn to end; that one resume reports both.
            self._held[ev.session].append((ev, w))
            return
        if self.in_turn(ev.session):
            self._held[ev.session] = [(ev, w)]
            try:
                await self._turn_over(ev.session)
            finally:
                held = self._held.pop(ev.session, [(ev, w)])
            # The agent kept working after the job ended and started another one: it has
            # seen this outcome, and the new job's own resume will continue the session.
            # Not so for a job that ended after the newest running one started (HPO run 3: of two
            # jobs started together, one failed in 4 s and the agent ended its turn unaware).
            running = [x.created for x in self.watches.values() if x.session == ev.session]
            if running:
                unseen = [(e, x) for e, x in held if e.ts > max(running)]
                if not unseen:
                    log.info("session %s moved on to another job; not resuming for %s",
                             ev.session, ", ".join(e.label for e, _ in held))
                    return
                log.info("session %s moved on to another job, but %d ended after it started; resuming",
                         ev.session, len(unseen))
                held = unseen
            earlier = [e for e, _ in held[:-1]]
            ev, w = held[-1]
        if not self.watches:
            if self.cfg.resume.warm_before_resume:
                await self._warm(ev.session)
            self.suspended_models = []
        ev.resume_error = await self._continue(ev.session, self.message(ev, earlier), w.cwd)
        ev.resumed = ev.resume_error is None

    async def _continue(self, session: str, message: str, cwd: str | None) -> str | None:
        """Run the resume command for `session` with `message`; the error if it could not start."""
        if earlier := self.undelivered.pop(session, None):
            message = f"{earlier}\n\n{message}"
        argv = [a.format(session=session, message=message, cwd=cwd or "") for a in self.cfg.resume.command]
        if await asyncio.to_thread(scope.use, self.cfg.isolation.scope):
            if outside := await asyncio.to_thread(scope.opencode_service_outside):
                log.warning("resuming session %s: %s", session, scope.service_warning(outside))
            argv = scope.wrap(argv, scope.unit_name("agent"), self.cfg.isolation.memory_max, oom_continue=True)
        st = self.router.sessions.get(session)
        before = st.replies if st else 0
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd or None, env=agent_env(self.cfg), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
        except OSError as e:
            log.warning("could not resume session %s: %s", session, e)
            return str(e)
        log.info("resumed session %s (pid %d)", session, proc.pid)
        self.latest_resume[session] = proc
        self.resuming[session] = (proc, before, time.time())
        # Reaped when the resumed turn ends (no zombies in a long-running server), then checked.
        self._spawn(self._after_turn(session, before, cwd, proc, message))
        return None

    async def _after_turn(self, session: str, before: int, cwd: str | None, proc: asyncio.subprocess.Process,
                          message: str) -> None:
        """Continue a resumed turn that stalled: one reply and no tool call, and no job left running.

        Nothing else would wake such a session: yuzuru resumes an agent only when a job of its ends.
        Only turns yuzuru resumed itself are checked, not a person's turns in their own client.
        """
        await proc.wait()
        # `opencode run` can exit while the turn goes on in the OpenCode service.
        while self.in_turn(session):
            await asyncio.sleep(1.0)
        if self.latest_resume.get(session) is not proc:
            return
        if await self._retry_load_failure(session, before, cwd, proc, message):
            return
        if session in self._held or any(w.session == session for w in self.watches.values()) \
                or any(l.session == session for l in self.broker.leases.values()):
            return
        deadline = self.cfg.resume.deadline()
        if deadline is not None and (left := deadline - time.time()) > 60:
            done, limit = self.nudges.get(session, 0), self.cfg.resume.until_nudges
            if done >= limit:
                log.warning("session %s ended its turn %.0f min before resume.until with no job running; "
                            "not continuing it again (%d in a row)", session, left / 60, done)
                return
            self.nudges[session] = done + 1
            log.warning("session %s ended its turn %.0f min before resume.until with no job running; "
                        "continuing it (%d/%d)", session, left / 60, done + 1, limit)
            if not self.watches and self.cfg.resume.warm_before_resume:
                await self._warm(session)
            text = UNTIL.format(until=self.cfg.resume.until, minutes=round(left / 60))
            if len(self.broker.snapshot.gpus) > 1:
                text += "\n\n" + self.broker.gpu_usage_line()
            await self._continue(session, text, cwd)
            return
        finish = self.cfg.resume.finish_time()
        if finish is not None and deadline is not None and time.time() >= deadline \
                and (left := finish - time.time()) > 60:
            done, limit = self.finish_nudges.get(session, 0), self.cfg.resume.finish_nudges
            if done >= limit:
                log.warning("session %s ended its turn %.0f min before resume.finish_by with no job running; "
                            "not continuing it again (%d so far)", session, left / 60, done)
                return
            self.finish_nudges[session] = done + 1
            log.warning("session %s ended its turn %.0f min before resume.finish_by with no job running; "
                        "continuing it (%d/%d)", session, left / 60, done + 1, limit)
            if not self.watches and self.cfg.resume.warm_before_resume:
                await self._warm(session)
            await self._continue(session, FINISH.format(until=self.cfg.resume.until, minutes=round(left / 60),
                                                        finish_by=self.cfg.resume.finish_by), cwd)
            return
        st = self.router.sessions.get(session)
        replies = (st.replies if st else 0) - before
        if replies > 1:
            self.nudges.pop(session, None)
            log.info("session %s ended its turn with no job running", session)
            return
        done, limit = self.nudges.get(session, 0), self.cfg.resume.stall_nudges
        if done >= limit:
            log.warning("session %s: resumed turn ended after %d replies with no job running; %s", session, replies,
                        f"not continuing it again ({done} in a row)" if limit > 0 else "resume.stall_nudges is 0")
            return
        self.nudges[session] = done + 1
        log.warning("session %s: resumed turn ended after %d replies with no job running; continuing it (%d/%d)",
                    session, replies, done + 1, limit)
        if not self.watches and self.cfg.resume.warm_before_resume:
            await self._warm(session)
        await self._continue(session, STALLED, cwd)

    async def _retry_load_failure(self, session: str, before: int, cwd: str | None,
                                  proc: asyncio.subprocess.Process, message: str) -> bool:
        """Resume again a turn the LLM could not be loaded for. A job still running would not wake it until it
        ends (Alloc run 1: the 3060 sat idle for 112 min), and with none running nothing would."""
        st = self.router.sessions.get(session)
        replies = (st.replies if st else 0) - before
        failed = [until for v, until in list(self.broker.failed.items()) if self.broker.is_failed(v)]
        if not failed or (replies > 0 and proc.returncode == 0):
            self.load_retries.pop(session, None)
            return False
        done = self.load_retries.get(session, 0)
        if done >= LOAD_RETRIES:
            log.warning("session %s: the LLM still cannot be loaded; not resuming it again (%d in a row)", session, done)
            return False
        self.load_retries[session] = done + 1
        self.undelivered[session] = message if replies == 0 else LLM_BACK
        wait = max(failed) - time.time() + 1.0
        log.warning("session %s: the LLM could not be loaded for its turn; resuming it in %.0f s (%d/%d)",
                    session, wait, done + 1, LOAD_RETRIES)
        await asyncio.sleep(max(wait, 0.0))
        if self.latest_resume.get(session) is proc and session in self.undelivered:
            await self._continue(session, self.undelivered.pop(session), cwd)
        return True

    async def _turn_over(self, session: str | None) -> None:
        """Wait until the session's client has finished its turn.

        `opencode run --session` on a session whose turn is still running starts a
        second agent loop next to the first: both edit the same files and start jobs.
        """
        st = self.router.sessions.get(session) if session else None
        if st:
            log.info("session %s is still in its turn (%d replies being generated, last reply %s, %.0fs ago); "
                     "resuming once it ends", session, st.answering,
                     "called tools" if st.tool_calls else "ended", time.time() - st.updated)
        start = time.time()
        while self.in_turn(session):
            await asyncio.sleep(1.0)
        log.info("session %s finished its turn after %.0fs", session, time.time() - start)

    def in_turn(self, session: str | None) -> bool:
        """The session's turn is running, counting one yuzuru resumed that has not finished a reply yet,
        and one holding a `wait` call (a long hold sends nothing to the LLM, so the router alone would
        call the turn over)."""
        idle = self.cfg.resume.turn_idle_seconds
        if self.router.in_turn(session, idle) or (session and self.holding[session]):
            return True
        started = self.resuming.get(session) if session else None
        if started is None:
            return False
        proc, before, at = started
        st = self.router.sessions.get(session)
        if (st.replies if st else 0) > before or proc.returncode is not None or time.time() - at > idle:
            del self.resuming[session]
            return False
        return True

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
