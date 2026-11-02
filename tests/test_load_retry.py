"""A resumed turn the LLM could not be loaded for is resumed again, even with another job still running.

In Alloc run 1 the resume after exp0 died on "every local model failed to load recently". exp1 was still
running, so nothing woke the agent until it ended: the 3060 sat idle for 112 min.
"""

import asyncio
import os
import time

from .test_core import make, wait_for
from .test_stall_nudge import AGENT, agent, job_ends, reply, turns
from yuzuru.waits import LLM_BACK, LOAD_RETRIES, Waiter


async def setup(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path, nudges=0)
    catalog, broker, router = await make(cfg, world, http)
    return out, go, broker, router, Waiter(cfg, broker, router)


async def running_job(broker, waiter, tmp_path):
    lease = await broker.create_lease(pid=os.getpid(), label="other", gpus=[], vram=None, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    return lease


async def test_failed_load_resumes_again_with_the_same_message(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    go.touch()
    broker.failed["m"] = time.time() + 3.0
    await job_ends(broker, waiter, tmp_path, rc=0)
    await wait_for(lambda: len(turns(out)) == 2, timeout=8)
    assert turns(out)[1][1] == turns(out)[0][1] and "train" in turns(out)[0][1]


async def test_retries_while_another_job_runs(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    go.touch()
    lease = await running_job(broker, waiter, tmp_path)
    broker.failed["m"] = time.time() + 0.3
    await waiter._continue("s1", "first", str(tmp_path))
    await wait_for(lambda: len(turns(out)) == 2, timeout=8)
    assert turns(out)[1][1] == "first"
    await broker.release_lease(lease.id, rc=0)


async def test_a_later_resume_carries_the_undelivered_message(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    go.touch()
    broker.failed["m"] = time.time() + 30
    await waiter._continue("s1", "first", str(tmp_path))
    await wait_for(lambda: "s1" in waiter.undelivered, timeout=8)
    go.unlink()
    broker.failed.clear()
    await waiter._continue("s1", "second", str(tmp_path))
    await wait_for(lambda: len(turns(out)) == 2, timeout=8)
    assert turns(out)[1][1] == "first\n\nsecond"


async def test_a_turn_that_replied_and_exited_cleanly_is_not_retried(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    broker.failed["m"] = time.time() + 0.2
    await waiter._continue("s1", "first", str(tmp_path))
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router)
    reply(router)
    go.touch()
    await wait_for(lambda: waiter.latest_resume["s1"].returncode is not None)
    await wait_for(lambda: not waiter.in_turn("s1"), timeout=8)
    time_up = time.time() + 1.0
    while time.time() < time_up:
        assert len(turns(out)) == 1
        await asyncio.sleep(0.05)


async def test_retries_are_bounded(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    go.touch()
    waiter.load_retries["s1"] = LOAD_RETRIES
    broker.failed["m"] = time.time() + 30
    await waiter._continue("s1", "first", str(tmp_path))
    await wait_for(lambda: len(turns(out)) == 1)
    await asyncio.sleep(1.0)
    assert len(turns(out)) == 1 and "s1" not in waiter.undelivered


async def test_a_turn_that_replied_and_then_died_is_told_what_happened(cfg, world, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, world, http, tmp_path)
    cfg.resume.command[2] = AGENT + "\nsys.exit(1)\n"
    broker.failed["m"] = time.time() + 3.0
    await waiter._continue("s1", "first", str(tmp_path))
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router)
    go.touch()
    await wait_for(lambda: len(turns(out)) == 2, timeout=8)
    assert turns(out)[1][1] == LLM_BACK
