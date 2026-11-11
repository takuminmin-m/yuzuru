"""`yuzuru report`: what an agent's run of a task came to, read from yuzuru's own records.

After a run one wants to know which job was best and where its weights are, what each job changed, how
long the GPUs sat idle and what went wrong. Until now that meant reading the logs and RESULTS.md, which the
agent writes from memory. This reads `.yuzuru/runs/` (one record per job, written by yuzuru), the jobs' logs
and `.yuzuru/journal.jsonl` instead.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import metrics, runs
from .weights import KEPT_DIR, SUFFIXES, weight_files

GPU_LINE = re.compile(r"cuda:\d+ = GPU (\d+) \(([^)]*)\)")
IDLE_WORTH_SAYING = 15 * 60  # a GPU free this long between two jobs is listed
WRITTEN_AFTER = 120  # a weights file written this long after its job ended still counts as that job's


def _gpus(run: dict[str, Any], head: str) -> list[dict[str, Any]]:
    """The job's GPUs: recorded when it started, or (older records) read from the line `yuzuru run` logs."""
    if run.get("gpus") is not None:
        return run["gpus"]
    return [{"index": int(i), "name": name} for i, name in GPU_LINE.findall(head)]


def _state(run: dict[str, Any]) -> str:
    if run.get("ended"):
        return "ok" if run.get("rc") == 0 else "failed"
    return "running" if runs._pid_alive(run.get("pid")) else "gone"


def _fmt(value: float) -> str:
    return f"{value:.4g}" if abs(value) < 10 else f"{value:.2f}"


def _weights(cwd: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    """The weight files the job wrote (last written while it ran), and whether they are still as it left them.

    A job running next to it writes its own in the same hours: when some carry the job's name, only those.
    """
    start, end = run["started"], (run.get("ended") or time.time()) + WRITTEN_AFTER
    out, seen = [], set()
    for p in weight_files(cwd):
        if start <= p.stat().st_mtime <= end:
            out.append({"path": str(p.relative_to(cwd)), "state": "as the job left it"})
            seen.add("__".join(p.relative_to(cwd).with_suffix("").parts))
    kept = cwd / KEPT_DIR
    if kept.is_dir():
        for p in sorted(kept.iterdir()):
            # weights.py names a copy <path with __>.<HHMMSS><suffix> and keeps the file's time.
            name = p.name.rsplit(".", 2)[0]
            if p.suffix in SUFFIXES and start <= p.stat().st_mtime <= end and name not in seen:
                out.append({"path": str(p.relative_to(cwd)), "state": "the original was overwritten or removed "
                            "later; this is yuzuru's copy"})
    name = run["id"].split("-", 1)[-1]
    return [w for w in out if name in w["path"]] or out


def _idle(jobs: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
    """Per GPU: how long it ran jobs and how long it was free, between the task's first GPU job and its last."""
    spans = [(j["started"], j["ended"] or now, j) for j in jobs if j["gpus"] and j["state"] != "gone"]
    if not spans:
        return []
    first, last = min(s for s, _, _ in spans), max(e for _, e, _ in spans)
    names = {g["index"]: g["name"] for _, _, j in spans for g in j["gpus"]}
    out = []
    for index in sorted(names):
        mine = sorted((s, e) for s, e, j in spans if any(g["index"] == index for g in j["gpus"]))
        busy, gaps, at = 0.0, [], first
        for s, e in mine:
            if s > at:
                gaps.append((at, s))
            if e > at:
                busy += e - max(s, at)
                at = e
        if last > at:
            gaps.append((at, last))
        out.append({"index": index, "name": names[index], "jobs": len(mine), "busy": busy, "idle": last - first - busy,
                    "gaps": [{"from": a, "to": b} for a, b in sorted(gaps, key=lambda g: g[0] - g[1])
                             if b - a >= IDLE_WORTH_SAYING]})
    return out


def build(cwd: str | Path, metric: str | None = None, lower: bool | None = None) -> dict[str, Any]:
    """The report as data. `metric` names the log's `name=value` pair to compare jobs by (default: guessed)."""
    cwd = Path(cwd).resolve()
    now = time.time()
    jobs = []
    records = [run for _, run in runs.load(cwd)]
    ends = runs.log_ends(records)
    for run in records:
        log = Path(run.get("cwd") or cwd) / run["log"] if run.get("log") else None
        offset, end, state = run.get("log_offset", 0), ends.get(run["id"]), _state(run)
        m = run.get("metric")
        if metric or "metric" not in run:  # asked for another pair, or a record older than the metric
            m = metrics.read(log, offset, metric, lower, end)
        if state == "gone":  # it wrote nothing of its own
            m = None
        head = metrics.read_text(log, offset, end)[:2000] if run.get("gpus") is None and run.get("gpu") else ""
        jobs.append({"id": run["id"], "command": run["command"], "state": state, "rc": run.get("rc"),
                     "started": run["started"], "ended": run.get("ended"), "gpus": _gpus(run, head) if run.get("gpu") else [],
                     "gpu": bool(run.get("gpu")), "metric": m, "last_line": run.get("last_line") or "",
                     "setting": run.get("setting") or {}, "code": run.get("code") or {},
                     "verify": run.get("verify"), "verified": run.get("verified") or []})
    report: dict[str, Any] = {"task": str(cwd), "jobs": jobs, "best": None, "metric": None, "weights": [],
                              "gpus": _idle(jobs, now), "journal": dict(Counter(e.get("kind") for e in runs.journal(cwd)))}
    # Jobs are compared by the metric most of them report; a job reporting another one is not ranked.
    trained = [j for j in jobs if j["gpu"] and j["state"] == "ok" and j["metric"] and not j["verify"]]
    if trained:
        name = Counter(j["metric"]["name"] for j in trained).most_common(1)[0][0]
        ranked = [j for j in trained if j["metric"]["name"] == name]
        low = ranked[0]["metric"]["lower"]
        best = (min if low else max)(ranked, key=lambda j: j["metric"]["best"])
        report.update(best=best["id"], metric={"name": name, "lower": low},
                      weights=_weights(cwd, next(r for r in records if r["id"] == best["id"])))
    report["notes"] = _notes(report)
    return report


def _notes(r: dict[str, Any]) -> list[str]:
    """What a person should look at."""
    jobs, out = r["jobs"], []
    if not r["best"] and jobs:
        out.append("no finished GPU job reported a validation metric yuzuru could read from its log "
                   "(name the log's `name=value` pair with --metric)")
    if r["best"] and not r["weights"]:
        out.append(f"no weights file written by the best job ({r['best']}) is left, nor a copy of one")
    elif r["best"] and all(w["path"].startswith(str(KEPT_DIR)) for w in r["weights"]):
        out.append(f"the weights of the best job ({r['best']}) were overwritten or removed; yuzuru's copy is listed above")
    best = next((j for j in jobs if j["id"] == r["best"]), None)
    if best:
        checks = best["verified"]
        if not checks:
            out.append(f"the best job's weights ({best['id']}) were not evaluated again on the validation data "
                       "(`yuzuru_train` with verify): a submission made from them is unchecked")
        elif checks[-1]["verdict"] == "mismatch":
            out.append(f"the best job's saved weights gave {checks[-1]['got']:g} on the validation data "
                       f"({checks[-1]['by']}), not the {best['metric']['best']:g} of its log")
    bad = [j for j in jobs if j["verify"] and j["verify"].get("verdict") == "mismatch" and j["verify"]["of"] != r["best"]]
    if bad:
        out.append("saved weights that did not match their log: " + ", ".join(j["verify"]["of"] for j in bad))
    failed = [j["id"] for j in jobs if j["state"] == "failed"]
    if failed:
        out.append(f"{len(failed)} of {len(jobs)} jobs failed: " + ", ".join(failed[:6]) + (" ..." if len(failed) > 6 else ""))
    gone = [j["id"] for j in jobs if j["state"] == "gone"]
    if gone:
        out.append(f"{len(gone)} jobs have no end recorded (not started for want of a free GPU, or yuzuru restarted): "
                   + ", ".join(gone[:6]) + (" ..." if len(gone) > 6 else ""))
    same: dict[str, list[str]] = {}
    for j in jobs:
        if j["gpu"] and j["state"] == "ok" and not j["verify"]:
            same.setdefault(json.dumps([j["setting"], j["code"]], sort_keys=True, default=str), []).append(j["id"])
    for ids in same.values():
        if len(ids) > 1:
            out.append("the same setting and scripts ran more than once: " + ", ".join(ids))
    unknown = [j["id"] for j in jobs if j["gpu"] and j["state"] != "gone" and not j["gpus"]]
    if unknown and r["gpus"]:
        out.append(f"the GPU of {len(unknown)} jobs is not recorded (older records): the busy and free times "
                   "above leave them out")
    for g in r["gpus"]:
        if g["gaps"]:
            longest = g["gaps"][0]
            out.append(f"GPU {g['index']} was free for {(longest['to'] - longest['from']) / 60:.0f} min from "
                       f"{_clock(longest['from'])} ({len(g['gaps'])} gaps of 15 min or more)")
    said = {"stalled_nudge": "continued the agent after a turn that did nothing",
            "idle_gpu_nudge": "continued the agent to use a free GPU", "repeat_refused": "refused a repeated setting",
            "verify_nudge": "continued the agent to check its best weights"}
    for kind, n in r["journal"].items():
        if kind in said:
            out.append(f"yuzuru {said[kind]}: {n}×")
    return out


def _clock(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _hours(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def _changed(job: dict[str, Any], best: dict[str, Any] | None) -> str:
    """How the job's setting differs from the best job's."""
    if best is None or job is best:
        return ""
    if job["setting"].get("head") != best["setting"].get("head"):
        return "other command: " + " ".join(job["setting"].get("head") or [])
    mine, ref = job["setting"].get("args", {}), best["setting"].get("args", {})
    parts = [f"{k}={mine[k]:g}" if isinstance(mine[k], float) else f"{k}={mine[k]}"
             for k in mine if mine[k] != ref.get(k)]
    parts += [f"no {k}" for k in ref if k not in mine]
    if job["code"] != best["code"]:
        parts.append("(other code)")
    return " ".join(parts)


def text(r: dict[str, Any], width: int = 150) -> str:
    jobs = r["jobs"]
    if not jobs:
        return f"no records in {Path(r['task']) / runs.RUNS_DIR}"
    by_id = {j["id"]: j for j in jobs}
    best = by_id.get(r["best"])
    first, last = min(j["started"] for j in jobs), max(j["ended"] or j["started"] for j in jobs)
    out = [f"Task  {r['task']}",
           f"Jobs  {len(jobs)} ({sum(j['state'] == 'ok' for j in jobs)} finished, "
           f"{sum(j['state'] == 'failed' for j in jobs)} failed), {_clock(first)} to {_clock(last)} ({_hours(last - first)})"]
    if best:
        m = best["metric"]
        out.append(f"Best  {best['id']}: {m['name']} {_fmt(m['best'])} at {m['best_at']} of {m['n']} values "
                   f"(last {_fmt(m['last'])}; {'lower' if m['lower'] else 'higher'} is better)")
        out.append(f"      {best['command']}")
        for w in r["weights"]:
            out.append(f"      weights: {w['path']} ({w['state']})")
        for c in best["verified"]:
            got = f" {_fmt(c['got'])}" if c.get("got") is not None else ""
            out.append(f"      verified by {c['by']}: {c['verdict']}{got}")
    if r["gpus"]:
        out.append("")
        for g in r["gpus"]:
            total = g["busy"] + g["idle"]
            out.append(f"GPU {g['index']} {g['name']:<26} {g['jobs']:>3} jobs, busy {_hours(g['busy'])} "
                       f"({100 * g['busy'] / total if total else 0:.0f}%), free {_hours(g['idle'])}")
    out += ["", f"{'JOB':<26} {'STATE':<8} {'TIME':>8} {'GPU':<4} {'BEST':>8} {'LAST':>8}  DIFFERS FROM THE BEST JOB IN"]
    for j in jobs:
        took = _hours((j["ended"] or time.time()) - j["started"]) if j["state"] != "gone" else "-"
        m = j["metric"] if j["metric"] and r["metric"] and j["metric"]["name"] == r["metric"]["name"] else None
        state = f"rc={j['rc']}" if j["state"] == "failed" else j["state"]
        where = ",".join(str(g["index"]) for g in j["gpus"]) or ("cpu" if not j["gpu"] else "?")
        if j["verify"]:
            v = j["verify"]
            what = f"verify of {v['of']}: {v.get('verdict', '-')}" + (f" ({_fmt(v['got'])})" if "got" in v else "")
        else:
            what = "(best)" if j is best else _changed(j, best)
        if j["state"] == "failed" and j["last_line"]:
            what = j["last_line"]
        line = (f"{j['id']:<26.26} {state:<8} {took:>8} {where:<4} {_fmt(m['best']) if m else '-':>8} "
                f"{_fmt(m['last']) if m else '-':>8}  {what}")
        out.append(line[:width])
    if r["notes"]:
        out += ["", "Worth a look:"] + [f"  - {n}" for n in r["notes"]]
    return "\n".join(out)
