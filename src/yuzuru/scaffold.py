"""`yuzuru init`: set up a task directory for an agent in one go.

Writes what each run used to need by hand: a yuzuru.yaml for this directory (GPU UUIDs and the
model detected), opencode.json (yuzuru's provider and MCP server, an allowlist of permissions,
no cloud fallback), AGENTS.md (how to use yuzuru's tools), .gitignore, then a uv venv and a git
repository. Files that already exist are left alone unless --force (which keeps a .bak).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import httpx

from .config import Config, load_config
from .opencode import add_mcp, mcp_config, opencode_major, provider_config
from .waits import outside_env

# Unattended, so an allowlist: the agent may run its Python through uv and look around, nothing else.
PERMISSION: dict[str, Any] = {
    "*": "allow",
    "question": "deny",  # nobody is there to answer
    "doom_loop": "deny",
    "execute": "deny",  # OpenCode 2.x's JavaScript tool would get past the bash allowlist
    "external_directory": {
        "*": "deny",  # e.g. a grader next to the task
        "~/.local/share/opencode/shell/*": "allow",  # where OpenCode 2.x keeps long command output for the agent
    },
    "bash": {
        "*": "deny",
        **{cmd: "allow" for cmd in (
            "uv run *", "uv pip install *", "uv pip list*", "uv pip show *",
            "cd *", "pwd", "ls", "ls *", "cat *", "head *", "tail *", "grep *", "wc *", "find *", "diff *",
            "sed -n *", "echo *", "mkdir *", "cp *", "mv *", "rm *", "touch *",
            "nvidia-smi*", "free*", "df *",
            "git status*", "git diff*", "git log*", "git show*", "git add *", "git commit *",
        )},
        **{cmd: "deny" for cmd in ("rm -r*", "rm -f*", "rm -rf*", "*--break-system-packages*", "sudo *")},
    },
}

GITIGNORE = [
    "__pycache__/", ".venv/", ".yuzuru/", ".opencode/", "opencode.json.bak", "yuzuru.yaml.bak", "AGENTS.md.bak",
    "data/", "*.pt", "*.pth", "*.ckpt", "*.safetensors",
]


def template(name: str) -> str:
    return resources.files("yuzuru").joinpath(f"templates/{name}").read_text()


def gpus() -> list[tuple[str, str, str]]:
    """(index, uuid, name) of every NVIDIA GPU, in nvidia-smi order; none without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [tuple(f.strip() for f in line.split(",", 2)) for line in out.splitlines() if line.strip()]  # type: ignore[misc]


def resolve_gpus(wanted: list[str] | None) -> list[str]:
    """UUIDs for `wanted` (indices or UUIDs); every GPU when nothing is asked for."""
    if wanted and all(w.startswith("GPU-") for w in wanted):
        return wanted
    found = gpus()
    if not wanted:
        return [uuid for _, uuid, _ in found]
    by_index = {index: uuid for index, uuid, _ in found}
    missing = [w for w in wanted if not w.startswith("GPU-") and w not in by_index]
    if missing:
        listed = ", ".join(f"{i} {n} ({u})" for i, u, n in found) or "none (no nvidia-smi)"
        raise SystemExit(f"yuzuru init: no GPU {', '.join(missing)}; GPUs: {listed}")
    return [w if w.startswith("GPU-") else by_index[w] for w in wanted]


def default_model(ollama_url: str, state_dir: Path) -> str:
    """The model yuzuru last served as the agent's (large) tier, else the biggest installed chat model:
    agents need a large one (4B-14B ones keep misusing tools), but the biggest may not fit."""
    from .catalog import _is_embedding

    try:
        tags = httpx.get(ollama_url.rstrip("/") + "/api/tags", timeout=10).json().get("models", [])
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"yuzuru init: cannot list Ollama's models at {ollama_url} ({e.__class__.__name__}); "
                         "start Ollama or pass --model")
    chat = [m for m in tags if not m["name"].startswith("yuzuru-") and not _is_embedding(m)]
    if not chat:
        raise SystemExit("yuzuru init: Ollama has no models; `ollama pull <model>` first or pass --model")
    try:
        profiles = json.loads((state_dir / "profiles.json").read_text())
    except (OSError, ValueError):
        profiles = {}
    # yuzuru creates a variant (yuzuru-large:<model>-64k, ...) for each model and context it serves;
    # the newest one's weights name the base model.
    variants = sorted((m for m in tags if m["name"].startswith("yuzuru-large:")),
                      key=lambda m: m.get("modified_at", ""), reverse=True)
    for v in variants:
        digest = profiles.get(v["name"], {}).get("digest")
        for m in chat:
            if digest and profiles.get(m["name"], {}).get("digest") == digest:
                return m["name"]
    return max(chat, key=lambda m: m.get("size", 0))["name"]


def yuzuru_yaml(task: Path, args: argparse.Namespace) -> str:
    model = args.model or default_model(os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"), Config().state_dir)
    log = shlex.quote(str(task / ".yuzuru" / "resume.log"))
    resume = ["bash", "-c", f'opencode run --session "$0" "$1" >> {log} 2>&1', "{session}", "{message}"]
    return (template("yuzuru.yaml")
            .replace("{port}", str(args.port))
            .replace("{gpus}", json.dumps(resolve_gpus(args.gpu)))
            .replace("{model}", json.dumps(model))
            .replace("{num_ctx}", str(args.ctx))
            .replace("{memory_max}", json.dumps(args.memory_max))
            .replace("{resume}", json.dumps(resume, ensure_ascii=False)))


def opencode_json(cfg: Config) -> str:
    conf: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"yuzuru": provider_config(cfg)},
    }
    add_mcp(conf, mcp_config(cfg), opencode_major())
    top = cfg.tiers[-1].name if len(cfg.tiers) == 1 else "auto"
    conf |= {
        "model": f"yuzuru/{top}",
        "small_model": f"yuzuru/{cfg.tiers[0].name}",
        # If yuzuru is down, OpenCode would otherwise fall back to its free cloud models with the code.
        "disabled_providers": ["opencode"],
        "permission": PERMISSION,
    }
    return json.dumps(conf, indent=2, ensure_ascii=False) + "\n"


def agents_md(task: Path, packages: list[str]) -> str:
    have = f"- `.venv` には {' / '.join(packages)} が入っている\n" if packages else ""
    return template("AGENTS.md").replace("{packages}", have).replace("{cwd}", str(task))


def write(path: Path, text: str, force: bool, report: list[str]) -> None:
    if path.exists() and not force:
        report.append(f"kept {path.name} (exists; --force to replace)")
        return
    if path.exists():
        shutil.copy(path, path.with_name(path.name + ".bak"))
    path.write_text(text)
    report.append(f"wrote {path.name}")


def run(cmd: list[str], cwd: Path, report: list[str]) -> bool:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    try:
        # outside_env: under `uv run yuzuru init`, VIRTUAL_ENV is yuzuru's venv and `uv pip install` would go there.
        ok = subprocess.run(cmd, cwd=cwd, env=outside_env()).returncode == 0
    except OSError as e:
        ok = False
        print(f"  {e}", file=sys.stderr)
    if not ok:
        report.append(f"FAILED: {shlex.join(cmd)}")
    return ok


def setup_venv(task: Path, args: argparse.Namespace, report: list[str]) -> list[str]:
    """Create the task's .venv; the --packages now in it."""
    if not shutil.which("uv"):
        report.append("skipped the venv: uv is not on PATH")
        return []
    if (task / "pyproject.toml").exists():
        if run(["uv", "sync"], task, report):
            report.append("synced .venv from pyproject.toml")
        return []
    if not (task / ".venv").exists():
        run(["uv", "venv", *(["--python", args.python] if args.python else [])], task, report)
    pip = list(args.packages or [])
    if (task / "requirements.txt").exists():
        pip += ["-r", "requirements.txt"]
    if pip and run(["uv", "pip", "install", *pip], task, report):
        report.append(f"installed {' '.join(pip)} into .venv")
        return list(args.packages or [])
    return []


def setup_git(task: Path, report: list[str]) -> None:
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=task, capture_output=True, text=True)
    if inside.returncode == 0:
        report.append("git: already a repository, nothing committed")
        return
    # Commit what is there, so `git diff` shows what the agent changed.
    if (run(["git", "init", "-q"], task, report) and run(["git", "add", "-A"], task, report)
            and run(["git", "commit", "-qm", "Task set up with yuzuru init"], task, report)):
        report.append("git: new repository with the task committed")


def cmd_init(args: argparse.Namespace) -> None:
    task = Path(args.dir).expanduser().resolve()
    task.mkdir(parents=True, exist_ok=True)
    (task / ".yuzuru").mkdir(exist_ok=True)
    report: list[str] = []

    config = task / "yuzuru.yaml"
    if args.force or not config.exists():  # detecting the model needs Ollama: only when writing
        write(config, yuzuru_yaml(task, args), args.force, report)
    else:
        report.append("kept yuzuru.yaml (exists; --force to replace)")
    cfg = load_config(config)
    write(task / "opencode.json", opencode_json(cfg), args.force, report)
    installed = [] if args.no_venv else setup_venv(task, args, report)
    write(task / "AGENTS.md", agents_md(task, installed), args.force, report)

    ignore = task / ".gitignore"
    text = ignore.read_text() if ignore.exists() else ""
    if add := [line for line in GITIGNORE if line not in text.splitlines()]:
        ignore.write_text(text + ("\n" if text and not text.endswith("\n") else "") + "\n".join(add) + "\n")
        report.append(f"added {len(add)} entries to .gitignore")

    if not args.no_git:
        setup_git(task, report)

    for line in report:
        print(line)
    model = cfg.tiers[-1].models[0] if cfg.tiers[-1].models else "auto"
    print(f"\n{task}: port {cfg.port}, model {model}, GPUs {', '.join(cfg.resources.gpus) or 'all'}, "
          f"memory_max {cfg.isolation.memory_max}")
    here = "" if task == Path.cwd() else f"cd {shlex.quote(str(task))}\n"
    print(f"""
next (in {task}, which makes yuzuru use its yuzuru.yaml):
{here}yuzuru serve                                   # in a terminal of its own
yuzuru status                                  # if the tier shows `none`: yuzuru warm {cfg.tiers[-1].name} --force
opencode service stop; yuzuru agent -- opencode service start
yuzuru agent -- opencode run "README.md と AGENTS.md を読んで、課題を最後までやり遂げてください。"
""".rstrip())
