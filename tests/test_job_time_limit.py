"""A GPU job is stopped at resume.job_max_minutes and at finish_by: rules in README.md did not hold (Loop run 4)."""

import datetime
import json
import os
import subprocess
import sys
import time

from yuzuru.mcp import MCPServer
from yuzuru.server import State
from yuzuru.waits import TIME_LIMIT_RC, TIME_LIMIT_TEXT, Event, Waiter


def minutes_from_now(n):
    return (datetime.datetime.now() + datetime.timedelta(minutes=n)).isoformat(timespec="seconds")


async def started(cfg, tmp_path, monkeypatch, **job):
    cfg.isolation.scope = "off"
    ran = []

    def run(argv, cwd, **kw):
        ran.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"pid": 1, "log": "/l"}), "")

    monkeypatch.setattr(subprocess, "run", run)
    text = await MCPServer(State(cfg)).tools["train"][1](
        {"command": "python train.py", "cwd": str(tmp_path), "resume_when_done": False, **job}, {})
    return ran[0], text


async def test_no_limit_by_default(cfg, tmp_path, monkeypatch):
    argv, text = await started(cfg, tmp_path, monkeypatch, vram="4G")
    assert "timeout" not in argv and "stopped after" not in text


async def test_a_gpu_job_gets_the_limit_and_is_told(cfg, tmp_path, monkeypatch):
    cfg.resume.job_max_minutes = 45
    argv, text = await started(cfg, tmp_path, monkeypatch, vram="4G")
    assert argv[argv.index("timeout"):][-4:] == ["2700s", "bash", "-c", "python train.py"]
    assert "stopped after 45 min" in text


async def test_finish_by_comes_first_when_it_is_sooner(cfg, tmp_path, monkeypatch):
    cfg.resume.job_max_minutes = 45
    cfg.resume.until, cfg.resume.finish_by = minutes_from_now(5), minutes_from_now(20)
    _, text = await started(cfg, tmp_path, monkeypatch, vram="4G")
    assert "stopped after 20 min" in text


async def test_finish_by_alone_stops_a_job(cfg, tmp_path, monkeypatch):
    cfg.resume.until, cfg.resume.finish_by = minutes_from_now(5), minutes_from_now(200)
    _, text = await started(cfg, tmp_path, monkeypatch, vram="4G")
    assert "stopped after 200 min" in text


async def test_a_cpu_job_has_no_limit(cfg, tmp_path, monkeypatch):
    cfg.resume.job_max_minutes = 45
    argv, _ = await started(cfg, tmp_path, monkeypatch, device="cpu")
    assert "timeout" not in argv


def test_the_group_is_stopped_with_the_exit_code_we_report(tmp_path):
    # A child of the command, like a data loader worker, goes too.
    pidfile = tmp_path / "child"
    code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
            f"open(r'{pidfile}','w').write(str(p.pid)); time.sleep(60)")
    t = time.time()
    r = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30", "2s", sys.executable, "-c", code])
    assert r.returncode == TIME_LIMIT_RC and time.time() - t < 10
    child = int(pidfile.read_text())
    deadline = time.time() + 3
    while os.path.exists(f"/proc/{child}") and open(f"/proc/{child}/stat").read().split()[2] != "Z":
        assert time.time() < deadline, "the child survived"
        time.sleep(0.1)


def test_the_outcome_says_why_it_stopped():
    ev = Event(id=1, watch="w", session="s1", label="train", rc=TIME_LIMIT_RC, duration=2700.0, log=None, log_tail="",
               note="")
    assert TIME_LIMIT_TEXT in Waiter._status(ev)
