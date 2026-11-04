"""A turn that ends with a job running and a GPU free is continued, a bounded number of times.

In Overfit run 2 the agent ended its turn "waiting for Exp2": the 3050 sat idle for 42 min.
"""

import asyncio
import os
import subprocess
import sys

import pytest

from .conftest import GB
from .test_core import make, wait_for
from .test_resume_until import in_minutes
from .test_stall_nudge import agent, reply, turns
from yuzuru.mcp import MCPServer
from yuzuru.server import State
from yuzuru.waits import IDLE_GPU, Waiter


@pytest.fixture
def two_gpus(world):
    world.gpu_sizes = [8 * GB, 12 * GB]
    world.foreign = [0, 0]
    return world


async def start_job(broker, waiter, tmp_path):
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                      session="s1", cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    return lease


async def setup(cfg, world, http, tmp_path, until=120):
    out, go = agent(cfg, tmp_path, nudges=0)
    if until is not None:
        cfg.resume.until = in_minutes(until)
    catalog, broker, router = await make(cfg, world, http)
    return out, go, broker, router, Waiter(cfg, broker, router)


async def test_a_turn_ending_with_a_free_gpu_is_continued(cfg, two_gpus, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    cfg.resume.idle_gpu_nudges = 2
    reply(router, "tool_calls")
    lease = await start_job(broker, waiter, tmp_path)
    reply(router)  # "waiting for the job": the first turn ends with the other GPU free
    await wait_for(lambda: len(turns(out)) == 1, timeout=8)
    message = turns(out)[0][1]
    assert message.startswith(IDLE_GPU[:30]) and cfg.resume.until in message and "GPU 0 (Fake 0): free" in message
    reply(router)
    go.touch()
    await wait_for(lambda: len(turns(out)) == 2, timeout=8)  # that turn ended the same way: again
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 2 and waiter.idle_gpu_nudges["s1"] == 2  # then it is left alone
    # A new job shows it is using the GPUs: the count starts again.
    other = await start_job(broker, waiter, tmp_path)
    assert "s1" not in waiter.idle_gpu_nudges
    await broker.release_lease(other.id, rc=0)
    await broker.release_lease(lease.id, rc=0)


async def test_no_nudge_with_every_gpu_busy(cfg, two_gpus, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    leases = [await start_job(broker, waiter, tmp_path) for _ in range(2)]
    await asyncio.sleep(3.0)
    assert not turns(out) and "s1" not in waiter.idle_gpu_nudges
    for l in leases:
        await broker.release_lease(l.id, rc=0)


@pytest.mark.parametrize("until", [None, -5, 5])  # no time box, past it, and too late for another job
async def test_no_nudge_outside_the_time_for_new_jobs(cfg, two_gpus, http, tmp_path, until):
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path, until=until)
    lease = await start_job(broker, waiter, tmp_path)
    await asyncio.sleep(3.0)
    assert not turns(out)
    await broker.release_lease(lease.id, rc=0)


async def test_no_nudge_on_one_gpu(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, drain=False, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    await asyncio.sleep(3.0)
    assert not turns(out)
    await broker.release_lease(lease.id, rc=0)


async def test_zero_turns_it_off(cfg, two_gpus, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    cfg.resume.idle_gpu_nudges = 0
    lease = await start_job(broker, waiter, tmp_path)
    await asyncio.sleep(3.0)
    assert not turns(out)
    await broker.release_lease(lease.id, rc=0)


async def wait_answers(cfg, world, http, tmp_path, until):
    """What `wait` says first and when called again, with one job on one of two GPUs."""
    cfg.resume.repeat_wait_seconds = 0.1
    cfg.resume.until = in_minutes(until) if until is not None else None
    catalog, broker, router = await make(cfg, world, http)
    st = State(cfg, http=http)
    st.catalog, st.broker, st.router = catalog, broker, router
    st.waiter = Waiter(cfg, broker, router)
    wait = MCPServer(st).tools["wait"][1]
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lease = await broker.create_lease(pid=job.pid, label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                          session="s1", cwd=str(tmp_path), resume=True, note="")
        headers = {"x-opencode-session": "s1"}
        first = await wait({"pid": job.pid, "device": "cpu"}, headers)
        again = await wait({"pid": job.pid, "device": "cpu"}, headers)
        await broker.release_lease(lease.id, rc=0)
        return first, again
    finally:
        job.kill()
        job.wait()


async def test_wait_says_a_gpu_is_free(cfg, two_gpus, http, tmp_path):
    # An agent that keeps calling wait stays in its turn: the nudge above never comes (Loop run 3).
    for answer in await wait_answers(cfg, two_gpus, http, tmp_path, until=120):
        assert "A GPU is free" in answer and "GPU 0 (Fake 0): free" in answer and cfg.resume.until in answer


@pytest.mark.parametrize("until", [None, -5, 5])
async def test_wait_says_nothing_without_time_to_use_it(cfg, two_gpus, http, tmp_path, until):
    for answer in await wait_answers(cfg, two_gpus, http, tmp_path, until=until):
        assert "A GPU is free" not in answer


async def test_wait_says_nothing_when_turned_off(cfg, two_gpus, http, tmp_path):
    cfg.resume.idle_gpu_nudges = 0
    for answer in await wait_answers(cfg, two_gpus, http, tmp_path, until=120):
        assert "A GPU is free" not in answer

