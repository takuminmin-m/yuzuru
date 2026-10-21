"""Run a command in its own systemd user scope, i.e. its own cgroup.

A process's memory counts against the unit it was started from. When the kernel
OOM-kills one, systemd stops that whole unit (OOMPolicy=stop is the default), so
an agent's runaway script takes down the terminal, IDE or service that started
the agent. In a scope of its own with OOMPolicy=continue and a MemoryMax, the
script is killed there alone and the agent sees it exit with 137.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import uuid
from pathlib import Path


@functools.cache
def available() -> bool:
    """Whether `systemd-run --user --scope` works (not in a container or a bare ssh session)."""
    if shutil.which("systemd-run") is None:
        return False
    try:
        return subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "true"],
                              capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def use(mode: str) -> bool:
    """auto | on | off -> whether to wrap commands in a scope."""
    return mode == "on" or (mode == "auto" and available())


def unit_name(kind: str) -> str:
    """yuzuru-<kind>-<pid>-<random>: a scope can outlive the pid that made it (one holding
    the OpenCode service, a detached job), so the pid alone may be reused."""
    return f"yuzuru-{kind}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def wrap(argv: list[str], unit: str, memory_max: str | None = None, oom_continue: bool = False) -> list[str]:
    """`argv` run in place (same pid) inside a transient scope named `unit`."""
    props = []
    if memory_max:
        props += ["-p", f"MemoryMax={memory_max}"]
    if oom_continue:
        props += ["-p", "OOMPolicy=continue"]
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", *props, "--", *argv]


def opencode_service_outside(proc: Path = Path("/proc")) -> list[tuple[int, str]]:
    """OpenCode 2.x background services not running in an agent scope, as (pid, cgroup).

    `opencode run` hands the turn to that service, so the agent's tools run in the
    service's cgroup, not in the scope `opencode run` itself was started in. The
    service lands wherever the first command that needed it ran (a Claude Code shell, a
    terminal), unless it was started with `yuzuru agent -- opencode service start`.
    """
    found = []
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
            if not (argv[0].endswith(b"opencode") and b"serve" in argv and b"--service" in argv):
                continue
            cgroup = (d / "cgroup").read_text().strip().split(":", 2)[-1]
        except (OSError, IndexError):
            continue
        if "/yuzuru-agent-" not in cgroup:
            found.append((int(d.name), cgroup))
    return found


def service_warning(outside: list[tuple[int, str]]) -> str:
    pid, cgroup = outside[0]
    return (f"the OpenCode service (pid {pid}) runs in {cgroup.rsplit('/', 1)[-1]}, not in an agent scope: "
            "the agent's tools run there, without the memory cap. Restart it in one: "
            "opencode service stop && yuzuru agent -- opencode service start")
