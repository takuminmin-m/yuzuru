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

from . import metrics

RUNS_DIR = Path(".yuzuru") / "runs"
JOURNAL = Path(".yuzuru") / "journal.jsonl"
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


def begin(command: str, cwd: str | Path, session: str | None, log: str | None, gpu: bool,
          verify: dict[str, Any] | None = None) -> Path:
    """Make the record of a job about to start: its directory, run.json and a copy of its code.

    `verify` (from `verify_target`) makes it a job that evaluates an earlier job's saved weights on the
    validation data: when it ends, what it reports is compared with that job's log.
    """
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
    # Jobs given the same log file append to it: what this one writes starts here.
    offset = 0
    with contextlib.suppress(OSError):
        offset = (Path(cwd) / log).stat().st_size if log else 0
    _save(d, {"id": d.name, "command": command, "cwd": str(cwd), "session": session, "gpu": gpu, "log": log,
              "log_offset": offset, "setting": setting(command, cwd), "code": digest(files, cwd), "git": _git(cwd),
              "started": time.time(), "ended": None, "rc": None, **({"verify": verify} if verify else {})})
    return d


def started(d: Path, pid: int, log: str, gpus: list[dict[str, Any]] | None = None) -> None:
    """Add what the job got when it started: its pid and log, and its GPUs as [{"index", "name"}]."""
    with contextlib.suppress(OSError, ValueError):
        run = json.loads((d / "run.json").read_text())
        run.update(pid=pid, log=log)
        if gpus is not None:
            run["gpus"] = gpus
        _save(d, run)


def ended(cwd: str | Path | None, log: str | None, rc: int | None, last_line: str = "") -> dict[str, Any] | None:
    """Close the record of the job that wrote `log`; the record, or None if there is none."""
    if not cwd or not log:
        return None
    for d, run in reversed(load(cwd)):
        if run.get("log") == log and run.get("ended") is None:
            run.update(ended=time.time(), rc=rc, last_line=last_line)
            # The validation metric its log reports, for `yuzuru report`.
            run["metric"] = metrics.read(Path(cwd) / run["log"], run.get("log_offset", 0))
            if run.get("verify"):
                _check(cwd, run)
            with contextlib.suppress(OSError):
                _save(d, run)
            return run
    return None


# ------------------------------------------------------------------ verify

def log_ends(records: list[dict[str, Any]]) -> dict[str, int | None]:
    """Where each job's part of its log ends, by job id: where the next job given the same file began.

    A record older than `log_offset` cannot be told apart from the others in its file: -1 (nothing to read),
    but for the last job that wrote there.
    """
    out: dict[str, int | None] = {}
    by_log: dict[str, list[dict[str, Any]]] = {}
    for run in records:
        if run.get("log"):
            by_log.setdefault(str(Path(run.get("cwd") or "") / run["log"]), []).append(run)
    for sharing in by_log.values():
        for i, run in enumerate(sharing):
            later = [r["log_offset"] for r in sharing[i + 1:] if "log_offset" in r]
            out[run["id"]] = min(later) if "log_offset" in run and later else (-1 if sharing[i + 1:] and "log_offset" not in run else None)
    return out


def own_text(cwd: str | Path, run: dict[str, Any]) -> str:
    """What the job `run` wrote to its log, without what other jobs given the same file wrote there."""
    if not run.get("log"):
        return ""
    end = log_ends([r for _, r in load(cwd)]).get(run["id"])
    return metrics.read_text(Path(cwd) / run["log"], run.get("log_offset", 0), end)


def find(cwd: str | Path, ref: str) -> tuple[Path, dict[str, Any]] | None:
    """The record `ref` names: its id ("024-cutmix50"), number ("24", "024") or name ("cutmix50"; the newest)."""
    ref = ref.strip()
    records = load(cwd)
    for d, run in records:
        if run["id"] == ref:
            return d, run
    head, _, name = ref.partition("-")
    if head.isdigit():
        for d, run in records:
            if int(run["id"].split("-", 1)[0]) == int(head) and (not name or run["id"].split("-", 1)[1] == name):
                return d, run
    return next(((d, run) for d, run in reversed(records) if run["id"].split("-", 1)[-1] == ref), None)


def verify_target(cwd: str | Path, ref: str, tolerance: float) -> dict[str, Any]:
    """What a job that evaluates `ref`'s saved weights is checked against. Raises ValueError for a job that cannot be."""
    found = find(cwd, ref)
    if not found:
        raise ValueError(f"no job {ref!r} in {RUNS_DIR}: give its id as `yuzuru_train` named it (e.g. '024-cutmix50')")
    _, run = found
    if run.get("verify"):
        raise ValueError(f"{run['id']} itself evaluated {run['verify']['of']}: name the training job")
    if run.get("ended") is None:
        raise ValueError(f"{run['id']} has not ended: verify its weights once it has")
    if "metric" in run:
        m = run["metric"]
    else:  # a record from before yuzuru read metrics at a job's end
        found = metrics.series(own_text(cwd, run))
        name = metrics.choose(found)
        m = metrics.summarize(name, found[name]) if name else None
    if not m:
        raise ValueError(f"{run['id']}'s log has no validation metric yuzuru can read (a `name=value` pair whose "
                         "name says validation, such as val_acc=0.81): there is nothing to compare with")
    return {"of": run["id"], "name": m["name"], "lower": m["lower"], "expected": m["best"], "best_at": m["best_at"],
            "n": m["n"], "tolerance": tolerance}


def _core(name: str) -> str:
    """A metric name without its split: `va_acc`, `val/acc` and `[val] acc` are all `acc`."""
    return re.sub(r"(?:^|[^a-z])(?:val|valid|validation|va|eval|dev|best|final|test)(?=[^a-z]|$)", "", name).strip("_/.- ")


def _pick(found: dict[str, list[float]], want: str) -> str | None:
    """The pair of a verify job's log that reports the metric `want` of the job it checks."""
    if want in found:
        return want
    core = _core(want)
    alike = [n for n in found if not metrics.COUNT.search(n) and core
             and ((c := _core(n)) and (c.startswith(core) or core.startswith(c)))]
    if alike:  # `val_accuracy` for `va_acc`
        return min(alike, key=lambda n: (not metrics.VAL.search(n), -len(found[n])))
    return metrics.choose(found)


def _close(a: float, b: float, rel: float) -> bool:
    a, b = metrics.same_scale(a, b)
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def _check(cwd: str | Path, run: dict[str, Any]) -> None:
    """Compare what a verify job reported with the log of the job whose weights it evaluated."""
    v = run["verify"]
    if run.get("rc") != 0:
        v["verdict"] = "failed"
        return
    found = metrics.series(metrics.read_text(Path(cwd) / run["log"], run.get("log_offset", 0)))
    name = _pick(found, v["name"])
    if not name:
        v["verdict"] = "unread"
    else:
        got = found[name][-1]
        v.update(got=got, got_name=name, verdict="match" if _close(got, v["expected"], v["tolerance"]) else "mismatch")
    target = find(cwd, v["of"])
    if v["verdict"] == "mismatch" and target and v["best_at"] < v["n"]:
        # The last value, not the best: the file was written at the end of training. (Any value below the best
        # is also somewhere on a rising curve, so only the last one says something.)
        values = metrics.series(own_text(cwd, target[1])).get(v["name"], [])
        if values and _close(v["got"], values[-1], v["tolerance"] / 4):
            v["like_last"] = values[-1]
    if target:
        d, t = target
        t.setdefault("verified", []).append({"by": run["id"], "verdict": v["verdict"], "got": v.get("got")})
        with contextlib.suppress(OSError):
            _save(d, t)
    note(cwd, "verify", of=v["of"], by=run["id"], verdict=v["verdict"], got=v.get("got"), expected=v["expected"])


def _num(x: float) -> str:
    return f"{x:.4g}" if abs(x) < 1 else f"{x:.2f}"


def verify_text(run: dict[str, Any] | None) -> str:
    """For the agent, when a verify job ends: whether the saved weights give what the training log says."""
    v = (run or {}).get("verify")
    if not v or v.get("verdict") in (None, "failed"):
        return ""
    of, want = v["of"], f"{v['name']} {_num(v['expected'])}"
    if v["verdict"] == "unread":
        return (f"[yuzuru] Verify of {of}: yuzuru found no metric in this job's log to compare with {of}'s best "
                f"{want}. Print it on the validation data under the same name, as `{v['name']}=<value>`, and run "
                "the check again.")
    got = f"{v['got_name']} {_num(v['got'])}"
    if v["verdict"] == "match":
        return (f"[yuzuru] Verify of {of}: the saved weights give {got} on the validation data, as its log's best "
                f"({want}). Build the submission from these weights with the same script.")
    epoch = (f" It is the log's last value ({_num(v['like_last'])}): the file holds the weights from the end of "
             "training, not those of the best point." if "like_last" in v else "")
    return (f"[yuzuru] Verify of {of} FAILED: the saved weights give {got} on the validation data, but its log's "
            f"best is {want} (value {v['best_at']} of {v['n']}).{epoch} A submission made from them will not score "
            "what the log says. Usual causes: the weights saved at another point than the best, BatchNorm buffers "
            "or other state not saved, preprocessing (normalization, resizing, tokenization) not the same as in "
            "validation during training, the model not in eval mode, or another model definition. Find the cause, "
            "fix it, and verify again before you build the submission.")


def note(cwd: str | Path | None, kind: str, **data: Any) -> None:
    """Add a line to the task's journal of what yuzuru did to its agent (a nudge, a refusal), for `yuzuru report`."""
    if not cwd:
        return
    with contextlib.suppress(OSError):
        path = Path(cwd) / JOURNAL
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "kind": kind, **data}, ensure_ascii=False, default=str) + "\n")


def journal(cwd: str | Path) -> list[dict[str, Any]]:
    out = []
    with contextlib.suppress(OSError):
        for line in (Path(cwd) / JOURNAL).read_text().splitlines():
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    return out


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
        if (not run.get("gpu") or run.get("verify") or run.get("repeat_allowed") or run.get("setting") != new
                or run.get("code") != code):
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
