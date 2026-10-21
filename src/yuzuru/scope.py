"""Run a command in its own systemd user scope, i.e. its own cgroup.

A process's memory counts against the unit it was started from. When the kernel
OOM-kills one, systemd stops that whole unit (OOMPolicy=stop is the default), so
an agent's runaway script takes down the terminal, IDE or service that started
the agent. In a scope of its own with OOMPolicy=continue and a MemoryMax, the
script is killed there alone and the agent sees it exit with 137.
"""

from __future__ import annotations

import functools
import shutil
import subprocess


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


def wrap(argv: list[str], unit: str, memory_max: str | None = None, oom_continue: bool = False) -> list[str]:
    """`argv` run in place (same pid) inside a transient scope named `unit`."""
    props = []
    if memory_max:
        props += ["-p", f"MemoryMax={memory_max}"]
    if oom_continue:
        props += ["-p", "OOMPolicy=continue"]
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", *props, "--", *argv]
