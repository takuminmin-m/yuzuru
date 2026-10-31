"""`wait` on a process the session already waits for is held instead of answered at once
(HPO run 2's agent checked a running job every few seconds, the LLM busy and a GPU idle)."""

import asyncio
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
