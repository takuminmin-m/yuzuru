"""`wait` on a process the session already waits for is held instead of answered at once
(HPO run 2's agent checked a running job every few seconds, the LLM busy and a GPU idle)."""

import asyncio
import contextlib
import subprocess
import sys
import time

from .test_core import make
from yuzuru.mcp import MCPServer
from yuzuru.server import State


async def tools(cfg, world, http):
    st = State(cfg, http=http)
    _, st.broker, st.router = await make(cfg, world, http)
    from yuzuru.waits import Waiter
    st.waiter = Waiter(cfg, st.broker, st.router)
    return st, MCPServer(st).tools["wait"][1]


async def test_a_second_wait_is_held_and_tells_the_agent_to_end_its_turn(cfg, world, http, tmp_path):
    cfg.resume.repeat_wait_seconds = 0.5
    st, wait = await tools(cfg, world, http)
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        headers = {"x-opencode-session": "s1"}
        t = time.time()
        first = await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert time.time() - t < 0.3 and "End your turn" in first
        t = time.time()
        again = await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert time.time() - t >= 0.5
        assert "still running" in again and "End your turn" in again
        assert len(st.waiter.watches) == 1
    finally:
        job.kill()
        job.wait()


async def test_a_held_wait_returns_the_outcome_when_the_process_ends(cfg, world, http, tmp_path):
    cfg.resume.repeat_wait_seconds = 10
    cfg.resume.enabled = False
    st, wait = await tools(cfg, world, http)
    log = tmp_path / "train.log"
    log.write_text("epoch 9 val 0.71\n")
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])

    async def poll():  # the server's loop: reap the child, then check plain-PID watches
        while True:
            job.poll()
            await st.waiter.poll()
            await asyncio.sleep(0.1)

    poller = asyncio.create_task(poll())
    try:
        headers = {"x-opencode-session": "s1"}
        await wait({"pid": job.pid, "device": "cpu", "log": str(log)}, headers)
        t = time.time()
        again = await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert time.time() - t < 8
        assert "ended after" in again and "val 0.71" in again
    finally:
        job.kill()
        job.wait()
        poller.cancel()


async def test_a_third_wait_holds_until_the_process_ends(cfg, world, http, tmp_path):
    # HPO run 3's agent ignored "end your turn" and called wait 81 times: hold it until the end.
    cfg.resume.repeat_wait_seconds = 0.2
    cfg.resume.repeat_wait_max_seconds = 30
    cfg.resume.enabled = False
    st, wait = await tools(cfg, world, http)
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])

    async def poll():
        while True:
            job.poll()
            await st.waiter.poll()
            await asyncio.sleep(0.1)

    poller = asyncio.create_task(poll())
    try:
        headers = {"x-opencode-session": "s1"}
        await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert "still running" in await wait({"pid": job.pid, "device": "cpu"}, headers)
        t = time.time()
        again = await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert 0.5 < time.time() - t < 20
        assert "ended after" in again
    finally:
        job.kill()
        job.wait()
        poller.cancel()


async def test_a_held_wait_answers_when_another_job_of_the_session_ends(cfg, world, http, tmp_path):
    # The other GPU's job ended: the agent wants to know now (to refill it), not when its own job does.
    cfg.resume.repeat_wait_seconds = 20
    st, wait = await tools(cfg, world, http)
    resumed = []

    async def fake_continue(session, message, cwd):
        resumed.append(message)

    st.waiter._continue = fake_continue
    long = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    short = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])

    async def poll():
        while True:
            short.poll()
            await st.waiter.poll()
            await asyncio.sleep(0.1)

    poller = asyncio.create_task(poll())
    try:
        headers = {"x-opencode-session": "s1"}
        await wait({"pid": short.pid, "device": "cpu"}, headers)
        await wait({"pid": long.pid, "device": "cpu"}, headers)
        t = time.time()
        again = await wait({"pid": long.pid, "device": "cpu"}, headers)
        assert time.time() - t < 10
        assert f"pid {short.pid} ended" in again and f"pid {long.pid} is still running" in again
        await asyncio.sleep(0.5)
        assert resumed == []  # the call reported it: no resume for it afterwards
        assert len(st.waiter.watches) == 1
    finally:
        for p in (long, short):
            p.kill()
            p.wait()
        poller.cancel()


async def test_a_session_holding_a_wait_counts_as_in_its_turn(cfg, world, http, tmp_path):
    # A long hold sends no LLM request: past turn_idle_seconds the router alone would call the turn
    # over, and an outcome arriving then would resume a second agent loop next to the first.
    cfg.resume.repeat_wait_seconds = 5
    cfg.resume.turn_idle_seconds = 0
    st, wait = await tools(cfg, world, http)
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        headers = {"x-opencode-session": "s1"}
        await wait({"pid": job.pid, "device": "cpu"}, headers)
        assert not st.waiter.in_turn("s1")
        held = asyncio.create_task(wait({"pid": job.pid, "device": "cpu"}, headers))
        await asyncio.sleep(0.2)
        assert st.waiter.in_turn("s1")
        held.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await held
        assert not st.waiter.in_turn("s1")
    finally:
        job.kill()
        job.wait()
