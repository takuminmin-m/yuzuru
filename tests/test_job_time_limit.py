"""A GPU job is stopped at resume.job_max_minutes and at finish_by: rules in README.md did not hold (Loop run 4)."""

import datetime
import json
import os
import subprocess
import sys
import time

import pytest

from yuzuru.mcp import MCPServer
from yuzuru.server import State
from yuzuru.waits import TIME_LIMIT_RC, TIME_LIMIT_TEXT, Event, Waiter, Watch

from .test_repeated_wait import tools


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


async def test_a_stop_during_a_turn_reaches_the_next_answer(cfg, world, http, tmp_path):
    # Loop run 7: the agent was polling, and the stop text reached it only at the end of the task.
    cfg.resume.enabled = False
    cfg.resume.repeat_wait_seconds = 0.0
    st, wait = await tools(cfg, world, http)
    w = Watch(id="w1", session="s1", label="train", log="exp2.log")
    st.waiter.watches[w.id] = w
    st.waiter._finish(w, TIME_LIMIT_RC)
    other = Watch(id="w2", session="s2", label="train", log="other.log")
    st.waiter.watches[other.id] = other
    st.waiter._finish(other, TIME_LIMIT_RC)
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        headers = {"x-opencode-session": "s1"}
        first = await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert "exp2.log" in first and TIME_LIMIT_TEXT in first and "other.log" not in first  # the other session's stays
        assert TIME_LIMIT_TEXT not in await wait({"pid": job.pid, "device": "cpu"}, headers)  # said once
    finally:
        job.kill()
        job.wait()


async def test_a_resume_counts_as_telling(cfg, world, http, tmp_path):
    cfg.resume.enabled = False
    st, _ = await tools(cfg, world, http)
    w = Watch(id="w1", session="s1", label="train", log="exp2.log")
    st.waiter.watches[w.id] = w
    st.waiter._finish(w, TIME_LIMIT_RC)
    assert TIME_LIMIT_TEXT in st.waiter.message(st.waiter.events[-1])
    assert st.waiter.stop_news("s1") == ""


async def test_a_job_with_as_many_epochs_is_refused_once_after_a_stop(cfg, world, http, tmp_path, monkeypatch):
    # Loop run 8: told in each resume to use fewer epochs, the agent kept 150; all five jobs were stopped.
    cfg.resume.enabled = False
    cfg.isolation.scope = "off"
    st, _ = await tools(cfg, world, http)
    train = MCPServer(st).tools["train"][1]
    headers = {"x-opencode-session": "s1"}
    logs = iter(["/exp1.log", "/exp2.log", "/exp3.log", "/exp4.log", "/exp5.log"])
    monkeypatch.setattr(subprocess, "run", lambda argv, cwd, **kw: subprocess.CompletedProcess(
        argv, 0, json.dumps({"pid": os.getpid(), "log": next(logs)}), ""))

    async def start(epochs, **job):
        return await train({"command": f"python train.py --epochs {epochs}", "cwd": str(tmp_path), "gpus": "0",
                            "vram": "1G", "resume_when_done": False, **job}, headers)

    def stop(log):
        w = Watch(id=log, session="s1", label="train", log=log)
        st.waiter.watches[w.id] = w
        st.waiter._finish(w, TIME_LIMIT_RC)

    await start(150)
    stop("/exp1.log")
    with pytest.raises(ValueError, match="exp1.log used --epochs 150.*has --epochs 150"):
        await start(150)
    await start(150)  # sent again: it may fit a faster GPU
    stop("/exp2.log")
    await start(100)  # fewer epochs than the stopped job: no second look
    stop("/exp3.log")
    with pytest.raises(ValueError, match="exp3.log used --epochs 100"):
        await start(120)  # more than the stopped 100; exp2's 150 is not reached
