"""A record of every GPU job an agent starts, kept by yuzuru in the task's `.yuzuru/runs/`.

Agents lose what they ran: a training script edited in place loses the model definition of the best
run, a fixed weights file name loses the best weights, RESULTS.md is written from memory. Each job
started by the train tool gets a directory here, written whatever the agent does:

    .yuzuru/runs/007-wd02/
        run.json   the command, where it ran, its log, the setting it trains (`args`), when it
                   started and ended and how (rc), and the git commit if the task is a repository
        args.json  the arguments the script actually parsed, defaults included (written by the job's
                   own Python: gpu_pin/sitecustomize.py, when the script uses argparse)
        code/      the scripts and config files the command names, as they were when it started

The same record tells a repeat from a change: a job whose setting (`args`, below) and scripts are
those of a job already running or finished is refused once (Loop run 17 ran one setting under four
names, Loop run 20 the same setting with its defaults spelled out and another seed).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

RUNS_DIR = Path(".yuzuru") / "runs"
RUN_DIR_ENV = "YUZURU_RUN_DIR"
# Arguments that only name a run, say where its outputs go, or set the seed: two jobs that differ only in
# these train the same setting (a seed alone tells nothing about what to change next).
NAME_ARG = re.compile(
    r"(?:name|run_?name|exp(?:_?name)?|tag|out(?:put)?(?:_?dir)?|save(?:_?(?:dir|path))?|ckpt|checkpoint|"
    r"log(?:_?(?:dir|file))?|seed|weights?(?:_?(?:path|file))?)")
CODE_FILE = re.compile(r"[\w./-]+\.(?:py|ya?ml|json|toml|cfg)\b")


def _dest(option: str) -> str:
    return option.lstrip("-").replace("-", "_")


def _value(text: Any) -> Any:
    """A command-line or default value in one form: 0.0005, "5e-4" and 5e-4 compare equal."""
    if isinstance(text, bool) or text is None:
        return text
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text)
    with contextlib.suppress(ValueError):
        return float(s)
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return ast.unparse(node)  # a name or call: compared by its text


def parser_defaults(script: Path) -> dict[str, tuple[list[str], Any, str | None]] | None:
    """The options a script's argparse parser takes: dest -> (option strings, default, action).

    Read from the source, not run: a job is judged before it starts. None when the script has no
    `add_argument` call this can read.
    """
    try:
        tree = ast.parse(script.read_text(errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None
    out: dict[str, tuple[list[str], Any, str | None]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        options = [n for n in names if n.startswith("-")]
        if not options:
            continue  # a positional argument: its value is in the command anyway
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        action = _literal(kw["action"]) if "action" in kw else None
        if "dest" in kw:
            dest = str(_literal(kw["dest"]))
        else:
            dest = _dest(next((o for o in options if o.startswith("--")), options[0]))
        if "default" in kw:
            default = _literal(kw["default"])
        else:
            default = {"store_true": False, "store_false": True}.get(action)
        out[dest] = (options, default, action)
    return out or None


def _words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def setting(command: str, cwd: str | Path) -> dict[str, Any]:
    """What `command` trains, as {"head": the words before its options, "args": {dest: value}}.

    The values are the script's defaults (read from its argparse calls) overridden by the command's
    options, so `--epochs 60 --lr 0.1` and `--lr=0.1 --epochs 60 --wd 5e-4` are the same setting when
    the script's default weight decay is 5e-4. Run names, output paths and the seed are left out.
    A script whose parser cannot be read is compared by its options as given.
    """
    words = _words(command)
    script = next((Path(cwd) / w for w in words if w.endswith(".py")), None)
    known = parser_defaults(script) if script else None
    by_option = {o: d for d, (opts, _, _) in (known or {}).items() for o in opts}
    actions = {d: a for d, (_, _, a) in (known or {}).items()}
    args: dict[str, Any] = {d: _value(v) for d, (_, v, _) in (known or {}).items()}
    def is_option(w: str) -> bool:
        return w.startswith("-") and w not in ("-", "--") and not w[1:2].isdigit() and w[1:2] != "."

    i = next((i for i, w in enumerate(words) if is_option(w)), len(words))
    head = words[:i]
    while i < len(words):
        w = words[i]
        if not is_option(w):
            head.append(w)
            i += 1
            continue
        name, eq, value = w.partition("=")
        dest = by_option.get(name, _dest(name))
        flag = actions.get(dest) in ("store_true", "store_false")
        if not eq and not flag:
            values = []
            while i + 1 < len(words) and not is_option(words[i + 1]):
                values.append(words[i + 1])
                i += 1
            value = " ".join(values)
        if flag:
            args[dest] = actions[dest] == "store_true"
        else:
            args[dest] = _value(value) if value != "" else True
        i += 1
    kept = {d: v for d, v in args.items() if not NAME_ARG.fullmatch(d)}
    return {"head": [w for w in head if w not in ("uv", "run", "python", "python3")], "args": kept}


def code_files(command: str, cwd: str | Path) -> list[Path]:
    """The scripts and config files `command` names that exist under `cwd`."""
    out = []
    for name in dict.fromkeys(CODE_FILE.findall(command)):
        p = Path(cwd) / name
        if p.is_file():
            out.append(p)
    return out


def digest(files: list[Path], cwd: str | Path) -> dict[str, str]:
    out = {}
    for p in files:
        with contextlib.suppress(OSError):
            out[str(p.relative_to(cwd)) if p.is_relative_to(cwd) else str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _git(cwd: str | Path) -> dict[str, Any] | None:
    """The commit checked out in the repository `cwd` is in, read from .git (no `git` process: this runs in the
    server's event loop for every job)."""
    with contextlib.suppress(OSError, ValueError):
        for d in (Path(cwd).resolve(), *Path(cwd).resolve().parents):
            git = d / ".git"
            if git.is_file():  # a worktree: "gitdir: <path>"
                git = (d / git.read_text().split(":", 1)[1].strip()).resolve()
            if not git.is_dir():
                continue
            head = (git / "HEAD").read_text().strip()
            if not head.startswith("ref:"):
                return {"commit": head}
            ref = head.split(":", 1)[1].strip()
            common = git
            if (git / "commondir").is_file():
                common = (git / (git / "commondir").read_text().strip()).resolve()
            for base in (git, common):
                if (base / ref).is_file():
                    return {"commit": (base / ref).read_text().strip(), "branch": ref.removeprefix("refs/heads/")}
            packed = common / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return {"commit": line.split()[0], "branch": ref.removeprefix("refs/heads/")}
            return {"branch": ref.removeprefix("refs/heads/")}  # no commit yet
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "-", text).strip("-")[:40] or "run"


def run_name(command: str, log: str | None) -> str:
    words = _words(command)
    for i, w in enumerate(words):
        name, eq, value = w.partition("=")
        if _dest(name) in ("name", "run_name", "exp", "exp_name", "tag"):
            return _slug(value if eq else (words[i + 1] if i + 1 < len(words) else ""))
    return _slug(Path(log).stem) if log else "run"


def runs_dir(cwd: str | Path) -> Path:
    return Path(cwd) / RUNS_DIR


def load(cwd: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    """The task's records, oldest first."""
    out = []
    root = runs_dir(cwd)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        with contextlib.suppress(OSError, ValueError):
            out.append((d, json.loads((d / "run.json").read_text())))
    return out


def _save(d: Path, run: dict[str, Any]) -> None:
    tmp = d / "run.json.tmp"
    tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, d / "run.json")


def begin(command: str, cwd: str | Path, session: str | None, log: str | None, gpu: bool) -> Path:
    """Make the record of a job about to start: its directory, run.json and a copy of its code."""
    root = runs_dir(cwd)
    root.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.name.split("-", 1)[0]) for p in root.iterdir() if p.name.split("-", 1)[0].isdigit()), default=0)
    d = root / f"{n:03d}-{run_name(command, log)}"
    d.mkdir()
    files = code_files(command, cwd)
    for p in files:
        rel = p.relative_to(cwd) if p.is_relative_to(cwd) else Path(p.name)
        (d / "code" / rel).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            shutil.copy2(p, d / "code" / rel)
    _save(d, {"id": d.name, "command": command, "cwd": str(cwd), "session": session, "gpu": gpu, "log": log,
              "setting": setting(command, cwd), "code": digest(files, cwd), "git": _git(cwd),
              "started": time.time(), "ended": None, "rc": None})
    return d


def started(d: Path, pid: int, log: str) -> None:
    """Add what the job got when it started."""
    with contextlib.suppress(OSError, ValueError):
        run = json.loads((d / "run.json").read_text())
        run.update(pid=pid, log=log)
        _save(d, run)


def ended(cwd: str | Path | None, log: str | None, rc: int | None, last_line: str = "") -> None:
    """Close the record of the job that wrote `log`."""
    if not cwd or not log:
        return
    for d, run in reversed(load(cwd)):
        if run.get("log") == log and run.get("ended") is None:
            run.update(ended=time.time(), rc=rc, last_line=last_line)
            with contextlib.suppress(OSError):
                _save(d, run)
            return


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refuse_repeat(command: str, cwd: str | Path) -> str | None:
    """Why a GPU job that trains the same as one already running or finished here should not start; once per
    such earlier job, so that sending the same command again starts it.

    The same command after a change to its scripts or config files is not a repeat, nor one after a job that
    failed (rc other than 0): running it again after a crash or a kill is what one does.
    """
    new = setting(command, cwd)
    code = digest(code_files(command, cwd), cwd)
    same = []
    for d, run in load(cwd):
        if not run.get("gpu") or run.get("repeat_allowed") or run.get("setting") != new or run.get("code") != code:
            continue
        running = run.get("ended") is None and _pid_alive(run.get("pid"))
        if running or run.get("rc") == 0:
            same.append((d, run, running))
    if not same:
        return None
    for d, run, _ in same:
        run["repeat_allowed"] = True
        with contextlib.suppress(OSError):
            _save(d, run)
    d, run, running = same[-1]
    how = "is still running" if running else f"finished after {(run['ended'] - run['started']) / 60:.0f} min"
    line = run.get("last_line") or ""
    return (f"not started: this trains the same as {run['id']} ({run['command']!r}), which {how}: the same "
            "setting once the script's defaults are filled in, apart from the run's name, output paths and the "
            f"seed, and the same scripts. {'Its last line: ' + repr(line) + '. ' if line else ''}"
            "Running one setting again, or with another seed, only shows how much results vary from run to run. "
            f"Change one setting for the next experiment (the records of what ran are in {RUNS_DIR}/*/run.json), "
            "and check that the change is in the command or the script. If you do mean to run it again, send the "
            "same command again.")


def table(cwd: str | Path) -> str:
    """The task's records, one line each, with the arguments that differ between them."""
    runs = load(cwd)
    if not runs:
        return f"no records in {runs_dir(cwd)}"
    keys = sorted({k for _, r in runs for k in (r.get("setting") or {}).get("args", {})})
    varying = [k for k in keys
               if len({json.dumps((r.get("setting") or {}).get("args", {}).get(k), default=str) for _, r in runs}) > 1]
    lines = []
    for _, r in runs:
        args = (r.get("setting") or {}).get("args", {})
        if r.get("ended"):
            minutes = f"{(r['ended'] - r['started']) / 60:.0f} min"
        else:  # no end recorded: a job refused for want of a free GPU after it was started, or a server restart
            minutes = "running" if _pid_alive(r.get("pid")) else "gone"
        changed = " ".join(f"{k}={args[k]}" for k in varying if k in args)
        lines.append(f"{r['id']:<24} rc={r.get('rc')!s:<4} {minutes:>8}  {changed}")
    return "\n".join(lines)
