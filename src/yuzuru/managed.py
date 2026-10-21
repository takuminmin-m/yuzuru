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
import json
import logging
import os
import shutil
import signal
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from . import scope
from .config import Config
from .ollama import Ollama, OllamaError

log = logging.getLogger("yuzuru.managed")

STORE_DIRS = ("manifests", "manifests-v2", "metadata")
SYSTEM_OLLAMA_PORT = 11434  # never taken by `port: auto`, even while ollama.service is stopped


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


def port_free(host: str, port: int) -> bool:
    """Whether `ollama serve` could listen on host:port right now."""
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as Go's net.Listen does
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def state_file(cfg: Config) -> Path:
    return cfg.state_dir / "ollama.json"


def running_url(cfg: Config) -> str | None:
    """URL of the managed server a running yuzuru started (its port may be auto-picked)."""
    try:
        info = json.loads(state_file(cfg).read_text())
    except (OSError, ValueError):
        return None
    return info.get("url") if _is_ollama_serve(int(info.get("pid", 0))) else None


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state in ("Z", "X")


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


class ManagedOllama:
    def __init__(self, cfg: Config, gpus: Callable[[], list[str]],
                 on_url: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.oc = cfg.ollama
        self.gpus = gpus  # UUIDs of the allowed GPUs (blocking: runs nvidia-smi)
        self.on_url = on_url  # told where the server listens whenever that changes
        self.proc: asyncio.subprocess.Process | None = None
        self.port: int | None = None
        self.started_at = 0.0
        self.restarts = 0
        self.scoped = False
        self.visible: str | None = None
        self._scope_ok: bool | None = None
        self._stopping = False
        self.hidden: set[str] = set()  # GPU UUIDs kept from it (dedicated to a job)
        self._swapping = False  # being restarted on purpose: the supervisor leaves it alone
        self._task: asyncio.Task | None = None
        self.system_url = f"http://127.0.0.1:{SYSTEM_OLLAMA_PORT}"  # owner of the mirrored blobs

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc and self.proc.returncode is None else None

    @property
    def url(self) -> str:
        return f"http://{self.oc.host}:{self.port}" if self.port else self.oc.url

    @property
    def pidfile(self) -> Path:
        return state_file(self.cfg)

    def candidate_ports(self) -> list[int]:
        if isinstance(self.oc.port, int):
            return [self.oc.port]
        lo, hi = self.oc.port_range
        taken = {self.cfg.port, SYSTEM_OLLAMA_PORT}
        ports = [p for p in range(lo, hi + 1) if p not in taken]
        # Keep the previous port across restarts so the URL stays put when possible.
        return ([self.port] if self.port in ports else []) + ports

    def pick_port(self) -> int | None:
        return next((p for p in self.candidate_ports() if port_free(self.oc.host, p)), None)

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
        env = {**os.environ, **self.oc.env, "OLLAMA_HOST": f"{self.oc.host}:{self.port}",
               "OLLAMA_MODELS": str(self.oc.models)}
        allow = self.cfg.resources.gpus
        if allow or self.hidden:
            try:
                uuids = await asyncio.to_thread(self.gpus)
            except Exception as e:  # noqa: BLE001
                log.warning("managed ollama: cannot resolve GPU UUIDs (%r); using %s as given", e, allow)
                uuids = []
            visible = [g for g in (uuids or allow) if g not in self.hidden]
            # "-1" hides every GPU (an empty value would be read as "unset" by some runtimes).
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible) or "-1"
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # if indices had to be used as given
        self.visible = env.get("CUDA_VISIBLE_DEVICES")
        return env

    async def _use_scope(self) -> bool:
        if self.oc.scope != "auto":
            return self.oc.scope == "on"
        if self._scope_ok is None:
            # No user systemd instance (a container, a bare ssh session): run it without a scope.
            self._scope_ok = await asyncio.to_thread(scope.available)
            if not self._scope_ok:
                log.info("managed ollama: systemd-run --user --scope unavailable; measuring its RAM by process")
        return self._scope_ok

    async def _launch(self) -> bool:
        """Start the server on a free port; False when there is none."""
        port = await asyncio.to_thread(self.pick_port)
        if port is None:
            ports = self.candidate_ports()
            what = f"port {ports[0]} is" if len(ports) == 1 else f"ports {self.oc.port_range[0]}-{self.oc.port_range[1]} are"
            log.warning("managed ollama: %s in use; set ollama.port (or port_range) to free ones", what)
            self.proc = None
            return False
        if port != self.port:
            self.port = port
            self.cfg.ollama_url = self.url
            if self.on_url:
                self.on_url(self.url)
        argv = [self.oc.binary, "serve"]
        self.scoped = await self._use_scope()
        if self.scoped:
            # Runs the command in place (same pid) inside a transient scope = its own cgroup.
            unit = f"yuzuru-ollama-{os.getpid()}-{self.restarts}"
            argv = scope.wrap(argv, unit)
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.cfg.state_dir / "ollama.log").open("ab") as out:
            # Own session: Ctrl-C on yuzuru's terminal must not kill it before yuzuru stops it.
            self.proc = await asyncio.create_subprocess_exec(
                *argv, env=await self.env(), stdin=asyncio.subprocess.DEVNULL, stdout=out, stderr=out,
                start_new_session=True,
            )
        self.started_at = time.time()
        self.pidfile.write_text(json.dumps({"pid": self.proc.pid, "url": self.url}))
        log.info("managed ollama: pid %d on %s, CUDA_VISIBLE_DEVICES=%s%s", self.proc.pid, self.url,
                 self.visible or "(all)", " (own scope)" if self.scoped else "")
        return True

    async def _ready(self) -> bool:
        deadline = time.time() + self.oc.startup_timeout
        async with httpx.AsyncClient(timeout=2) as c:
            while time.time() < deadline:
                if self.proc is None or self.proc.returncode is not None:
                    return False
                try:
                    if (await c.get(self.url + "/api/version")).status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        return False

    def _kill_stale(self) -> None:
        """An instance left behind by a yuzuru that was killed would hold the port."""
        try:
            pid = int(json.loads(self.pidfile.read_text())["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        if not _is_ollama_serve(pid):
            return
        log.warning("managed ollama: stopping stale pid %d", pid)
        _kill_group(pid, signal.SIGTERM)
        for _ in range(100):
            if _gone(pid):  # only then is its port closed (cmdline empties earlier)
                return
            time.sleep(0.1)
        _kill_group(pid, signal.SIGKILL)

    async def start(self) -> None:
        self._stopping = False
        await asyncio.to_thread(self._kill_stale)
        await asyncio.to_thread(self.sync)
        for _ in range(3):
            if not await self._launch() or await self._ready():
                break
            if self.proc is None or self.proc.returncode is None:
                break  # still starting (slow disk): the supervisor keeps watching it
            # Most likely another server took the port between the check and the bind
            # (two yuzurus starting at once); `auto` then moves on to the next one.
            log.warning("managed ollama exited with %s at startup on port %s; see %s", self.proc.returncode,
                        self.port, self.cfg.state_dir / "ollama.log")
            if isinstance(self.oc.port, int):
                break
        self._task = asyncio.create_task(self._supervise())

    async def hide(self, gpus: set[str]) -> None:
        """Restart with `gpus` hidden (or shown again), on the same port when it is still free."""
        if gpus == self.hidden and self.pid is not None:
            return
        self.hidden = set(gpus)
        self._swapping = True
        try:
            await self._terminate()
            self.restarts += 1
            if await self._launch():
                await self._ready()
        finally:
            self._swapping = False

    async def _terminate(self) -> None:
        proc = self.proc
        if proc is not None and proc.returncode is None:
            _kill_group(proc.pid, signal.SIGTERM)  # the group: its runners too
            try:
                await asyncio.wait_for(proc.wait(), 10)
            except asyncio.TimeoutError:
                _kill_group(proc.pid, signal.SIGKILL)
                await proc.wait()

    async def _supervise(self) -> None:
        backoff = 1.0
        while not self._stopping:
            if self.proc is not None and self.proc.returncode is None:
                try:
                    rc = await asyncio.wait_for(self.proc.wait(), 60)
                except asyncio.TimeoutError:
                    await asyncio.to_thread(self.sync)  # pick up models pulled for the system Ollama
                    continue
                if self._stopping:
                    return
                if self._swapping or self.proc.returncode is None:
                    # hide() stopped it and starts the next one itself.
                    while self._swapping:
                        await asyncio.sleep(0.2)
                    continue
                if time.time() - self.started_at > 60:
                    backoff = 1.0
                log.warning("managed ollama exited with %s; restarting in %.0fs", rc, backoff)
            # else: not started (no free port, or it died at startup): retry the same way.
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

    async def create_variant(self, ollama: Ollama, name: str, base: str, parameters: dict[str, Any]) -> None:
        """Create a variant on the managed server, or else where its blobs live.

        Ollama touches (chtimes) every layer a create reuses. Mirrored blobs are
        symlinks to files of the system Ollama's user, so the managed server gets
        EPERM on any create. The system Ollama owns them: it creates the variant
        in the source store and the mirror brings it over.
        """
        try:
            await ollama.create_variant(name, base, parameters)
            return
        except OllamaError as e:
            if self.source() is None or ("chtimes" not in str(e) and "not permitted" not in str(e)):
                raise
            log.info("managed ollama: cannot touch the shared blobs; creating %s with the system Ollama", name)
        system = Ollama(self.system_url)
        try:
            await system.create_variant(name, base, parameters)
        except httpx.HTTPError as e:
            raise OllamaError(f"creating {name}: the managed Ollama cannot write the shared blobs and the "
                              f"system Ollama (which owns them) is not reachable: {e!r}") from e
        finally:
            await system.aclose()
        await asyncio.to_thread(self.sync)
        try:
            await ollama.show(name)
        except OllamaError as e:  # the system Ollama keeps its models somewhere else than mirror_from
            raise OllamaError(f"creating {name}: the system Ollama created it, but not in {self.source()} "
                              "(ollama.mirror_from), so the managed Ollama cannot see it") from e

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._terminate()
        self.pidfile.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        return {"managed": True, "url": self.url, "pid": self.pid, "running": self.pid is not None,
                "restarts": self.restarts, "cuda_visible_devices": self.visible, "own_cgroup": self.scoped,
                "models": str(self.oc.models)}
