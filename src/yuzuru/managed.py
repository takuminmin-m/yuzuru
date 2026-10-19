"""A yuzuru-owned `ollama serve` (ollama.manage).

The system Ollama sees every GPU: a hard reservation keeps its weights off an
excluded GPU, but it still places compute buffers there and OOMs. A private
server started with CUDA_VISIBLE_DEVICES limited to `resources.gpus` cannot.
It shares the system model store through symlinked blobs, runs in its own
systemd user scope where possible (so its cgroup is its exact host RAM), is
restarted if it dies and stopped with yuzuru.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .config import Config

log = logging.getLogger("yuzuru.managed")

STORE_DIRS = ("manifests", "manifests-v2", "metadata")


def default_model_store() -> Path | None:
    """Where the system Ollama keeps its models."""
    for p in (os.environ.get("OLLAMA_MODELS"), "/usr/share/ollama/.ollama/models", Path.home() / ".ollama" / "models"):
        if p and (Path(p) / "blobs").is_dir():
            return Path(p)
    return None


def mirror_models(src: Path, dst: Path) -> int:
    """Make the models of store `src` usable from store `dst` without copying weights.

    Blobs become symlinks into `src`; manifests and metadata are copied (newer
    source wins), so `dst` stays writable for variants yuzuru creates there.
    Returns the number of entries added or updated.
    """
    if src.resolve() == dst.resolve():
        return 0
    changed = 0
    (dst / "blobs").mkdir(parents=True, exist_ok=True)
    for blob in (src / "blobs").iterdir():
        link = dst / "blobs" / blob.name
        if blob.name.endswith("-partial") or "-partial-" in blob.name or link.is_symlink() or link.exists():
            continue
        link.symlink_to(blob)
        changed += 1
    for sub in STORE_DIRS:
        for dirpath, _dirs, files in os.walk(src / sub):
            for name in files:
                try:
                    changed += _mirror_entry(Path(dirpath) / name, src, dst)
                except OSError as e:  # e.g. a manifest being rewritten by a pull right now
                    log.debug("mirror %s/%s: %r", dirpath, name, e)
    return changed


def _mirror_entry(s: Path, src: Path, dst: Path) -> int:
    d = dst / s.relative_to(src)
    d.parent.mkdir(parents=True, exist_ok=True)
    if s.is_symlink() and not os.path.isabs(target := os.readlink(s)):
        # manifests-v2 entries point at ../../blobs/...: keep them relative to `dst`.
        if d.is_symlink() and os.readlink(d) == target:
            return 0
        d.unlink(missing_ok=True)
        d.symlink_to(target)
    elif d.exists() and not d.is_symlink() and d.stat().st_mtime >= s.stat().st_mtime:
        return 0
    else:
        d.unlink(missing_ok=True)
        shutil.copy2(s, d)
    return 1


def _is_ollama_serve(pid: int) -> bool:
    """An `ollama serve` of ours (never the system service's)."""
    try:
        if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
            return False
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return any(a.endswith(b"ollama") for a in argv) and b"serve" in argv


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


class ManagedOllama:
    def __init__(self, cfg: Config, gpus: Callable[[], list[str]]) -> None:
        self.cfg = cfg
        self.oc = cfg.ollama
        self.gpus = gpus  # UUIDs of the allowed GPUs (blocking: runs nvidia-smi)
        self.proc: asyncio.subprocess.Process | None = None
        self.started_at = 0.0
        self.restarts = 0
        self.scoped = False
        self.visible: str | None = None
        self._scope_failed = False
        self._stopping = False
        self._task: asyncio.Task | None = None

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc and self.proc.returncode is None else None

    @property
    def pidfile(self) -> Path:
        return self.cfg.state_dir / "ollama.pid"

    def source(self) -> Path | None:
        if self.oc.mirror_from in ("", "none", "off"):
            return None
        if self.oc.mirror_from != "auto":
            return Path(self.oc.mirror_from).expanduser()
        return default_model_store()

    def sync(self) -> None:
        self.oc.models.mkdir(parents=True, exist_ok=True)
        if (src := self.source()) is not None:
            try:
                if n := mirror_models(src, self.oc.models):
                    log.info("managed ollama: mirrored %d entries from %s", n, src)
            except OSError as e:
                log.warning("managed ollama: cannot mirror %s: %r", src, e)

    async def env(self) -> dict[str, str]:
        env = {**os.environ, **self.oc.env, "OLLAMA_HOST": f"{self.oc.host}:{self.oc.port}",
               "OLLAMA_MODELS": str(self.oc.models)}
        allow = self.cfg.resources.gpus
        if allow:
            try:
                uuids = await asyncio.to_thread(self.gpus)
            except Exception as e:  # noqa: BLE001
                log.warning("managed ollama: cannot resolve GPU UUIDs (%r); using %s as given", e, allow)
                uuids = []
            env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids or allow)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # if indices had to be used as given
        self.visible = env.get("CUDA_VISIBLE_DEVICES")
        return env

    def _use_scope(self) -> bool:
        if self.oc.scope == "on":
            return True
        return self.oc.scope == "auto" and not self._scope_failed and shutil.which("systemd-run") is not None

    async def _launch(self) -> None:
        argv = [self.oc.binary, "serve"]
        self.scoped = self._use_scope()
        if self.scoped:
            # Runs the command in place (same pid) inside a transient scope = its own cgroup.
            unit = f"yuzuru-ollama-{os.getpid()}-{self.restarts}"
            argv = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", "--", *argv]
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.cfg.state_dir / "ollama.log").open("ab") as out:
            # Own session: Ctrl-C on yuzuru's terminal must not kill it before yuzuru stops it.
            self.proc = await asyncio.create_subprocess_exec(
                *argv, env=await self.env(), stdin=asyncio.subprocess.DEVNULL, stdout=out, stderr=out,
                start_new_session=True,
            )
        self.started_at = time.time()
        self.pidfile.write_text(str(self.proc.pid))
        log.info("managed ollama: pid %d on %s, CUDA_VISIBLE_DEVICES=%s%s", self.proc.pid, self.oc.url,
                 self.visible or "(all)", " (own scope)" if self.scoped else "")

    async def _ready(self) -> bool:
        deadline = time.time() + self.oc.startup_timeout
        async with httpx.AsyncClient(timeout=2) as c:
            while time.time() < deadline:
                if self.proc is None or self.proc.returncode is not None:
                    return False
                try:
                    if (await c.get(self.oc.url + "/api/version")).status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        return False

    def _kill_stale(self) -> None:
        """An instance left behind by a yuzuru that was killed would hold the port."""
        try:
            pid = int(self.pidfile.read_text())
        except (OSError, ValueError):
            return
        if not _is_ollama_serve(pid):
            return
        log.warning("managed ollama: stopping stale pid %d", pid)
        _kill_group(pid, signal.SIGTERM)
        for _ in range(100):
            if not Path(f"/proc/{pid}").exists() or not _is_ollama_serve(pid):
                return
            time.sleep(0.1)
        _kill_group(pid, signal.SIGKILL)

    async def start(self) -> None:
        self._stopping = False
        await asyncio.to_thread(self._kill_stale)
        await asyncio.to_thread(self.sync)
        await self._launch()
        if not await self._ready() and self.scoped and self.proc and self.proc.returncode is not None:
            # No user systemd instance (e.g. a container): run it without a scope.
            log.warning("managed ollama: systemd-run --scope failed (rc %s); starting without it",
                        self.proc.returncode)
            self._scope_failed = True
            await self._launch()
            await self._ready()
        if self.proc and self.proc.returncode is not None:
            log.warning("managed ollama exited with %s at startup; see %s", self.proc.returncode,
                        self.cfg.state_dir / "ollama.log")
        self._task = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        backoff = 1.0
        while not self._stopping:
            assert self.proc is not None
            try:
                rc = await asyncio.wait_for(self.proc.wait(), 60)
            except asyncio.TimeoutError:
                await asyncio.to_thread(self.sync)  # pick up models pulled for the system Ollama
                continue
            if self._stopping:
                return
            if time.time() - self.started_at > 60:
                backoff = 1.0
            log.warning("managed ollama exited with %s; restarting in %.0fs", rc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            if self._stopping:
                return
            self.restarts += 1
            try:
                await asyncio.to_thread(self.sync)
                await self._launch()
            except OSError as e:
                log.warning("managed ollama: restart failed: %r", e)

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        proc = self.proc
        if proc is not None and proc.returncode is None:
            _kill_group(proc.pid, signal.SIGTERM)  # the group: its runners too
            try:
                await asyncio.wait_for(proc.wait(), 10)
            except asyncio.TimeoutError:
                _kill_group(proc.pid, signal.SIGKILL)
                await proc.wait()
        self.pidfile.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        return {"managed": True, "url": self.oc.url, "pid": self.pid, "running": self.pid is not None,
                "restarts": self.restarts, "cuda_visible_devices": self.visible, "own_cgroup": self.scoped,
                "models": str(self.oc.models)}
