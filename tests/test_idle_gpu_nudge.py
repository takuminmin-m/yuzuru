"""A turn that ends with a job of the agent running and a GPU free is continued, a bounded number of times.

In Overfit run 2 the agent ended its turn "waiting for Exp2": the 3050 sat idle for 42 min.
"""

import asyncio
import os
import subprocess
import sys
import time

import pytest

from .conftest import GB
from .test_core import make, wait_for
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


async def setup(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path, nudges=0)
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
    assert message.startswith(IDLE_GPU[:30]) and "GPU 0 (Fake 0): free" in message
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


def request(text="result", role="tool"):
    return {"messages": [{"role": "user", "content": "go"}, {"role": role, "content": text}]}


async def test_a_long_free_gpu_reaches_an_agent_in_the_middle_of_a_turn(cfg, two_gpus, http, tmp_path, monkeypatch):
    # Loop run 10: 109 min of debugging in one long turn, with both GPUs almost unused.
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    cfg.resume.idle_gpu_remind_minutes = 15
    clock = [1000.0 + time.time()]
    monkeypatch.setattr("yuzuru.waits.time.time", lambda: clock[0])
    running = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                        session="s1", cwd=str(tmp_path), resume=False, note="")

    def tick(minutes):
        clock[0] += minutes * 60
        body = request()
        waiter.remind_of_free_gpu(body)
        return body["messages"][-1]["content"]

    assert tick(0) == "result"  # the clock starts here
    assert tick(14) == "result"
    assert "free for 15 min" in tick(1)
    assert tick(14) == "result"  # not in every request
    assert "free for 30 min" in tick(1)
    # With every GPU busy the wait starts again.
    other = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                      session="s1", cwd=str(tmp_path), resume=False, note="")
    assert tick(60) == "result"
    await broker.release_lease(other.id, rc=0)
    assert tick(0) == "result"
    assert tick(14) == "result"
    assert "free for 15 min" in tick(1)
    # With no job of an agent running, the task may be over: no reminder.
    await broker.release_lease(running.id, rc=0)
    assert tick(60) == "result"


async def test_the_reminder_keeps_the_form_of_the_message(cfg, two_gpus, http, tmp_path, monkeypatch):
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    cfg.resume.idle_gpu_remind_minutes = 0.0001
    await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                              session="s1", cwd=str(tmp_path), resume=False, note="")
    waiter.remind_of_free_gpu(request())  # starts the clock
    await asyncio.sleep(0.05)
    parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    waiter.remind_of_free_gpu(parts)
    assert parts["messages"][0]["content"][0]["text"] == "hi" and "A GPU has been free" in parts["messages"][0]["content"][1]["text"]
    await asyncio.sleep(0.05)
    own = request(role="assistant")  # not put in the agent's own mouth
    waiter.remind_of_free_gpu(own)
    assert own["messages"][-1]["content"] == "result"


async def test_no_reminder_without_a_job_running(cfg, two_gpus, http, tmp_path):
    # Every job has ended: the agent may be done, and a free GPU is not a reason to keep it going.
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    cfg.resume.idle_gpu_remind_minutes = 0.0001
    waiter.remind_of_free_gpu(request())
    await asyncio.sleep(0.05)
    body = request()
    waiter.remind_of_free_gpu(body)
    assert body["messages"][-1]["content"] == "result"


async def test_zero_turns_the_reminder_off(cfg, two_gpus, http, tmp_path):
    out, go, broker, router, waiter = await setup(cfg, two_gpus, http, tmp_path)
    cfg.resume.idle_gpu_remind_minutes = 0
    waiter.remind_of_free_gpu(request())
    await asyncio.sleep(0.05)
    body = request()
    waiter.remind_of_free_gpu(body)
    assert body["messages"][-1]["content"] == "result"


async def wait_answers(cfg, world, http, tmp_path, session="s1"):
    """What `wait` says first and when called again, with one job (of `session`) on one of two GPUs."""
    cfg.resume.repeat_wait_seconds = 0.1
    catalog, broker, router = await make(cfg, world, http)
    st = State(cfg, http=http)
    st.catalog, st.broker, st.router = catalog, broker, router
    st.waiter = Waiter(cfg, broker, router)
    wait = MCPServer(st).tools["wait"][1]
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lease = await broker.create_lease(pid=job.pid, label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                          session=session, cwd=str(tmp_path), resume=True, note="")
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
    for answer in await wait_answers(cfg, two_gpus, http, tmp_path):
        assert "A GPU is free" in answer and "GPU 0 (Fake 0): free" in answer


async def test_a_repeated_wait_is_not_held_while_a_gpu_is_free(cfg, two_gpus, http, tmp_path):
    # Loop run 5: each answer that named the free GPU came after a hold of 4 min and more; the 3050 sat idle for 18 min.
    cfg.resume.repeat_wait_seconds = 1.0
    cfg.resume.repeat_wait_max_seconds = 1.0
    cfg.resume.idle_gpu_nudges = 2
    catalog, broker, router = await make(cfg, two_gpus, http)
    st = State(cfg, http=http)
    st.catalog, st.broker, st.router = catalog, broker, router
    st.waiter = Waiter(cfg, broker, router)
    wait = MCPServer(st).tools["wait"][1]
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    headers = {"x-opencode-session": "s1"}

    async def timed_wait():
        start = asyncio.get_running_loop().time()
        answer = await wait({"pid": job.pid, "device": "cpu"}, headers)
        return asyncio.get_running_loop().time() - start, answer

    try:
        lease = await broker.create_lease(pid=job.pid, label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                          session="s1", cwd=str(tmp_path), resume=True, note="")
        answers = [await timed_wait() for _ in range(4)]  # the first call, then 2 at once, then held again
        assert [took < 0.5 for took, _ in answers] == [True, True, True, False]
        assert all("A GPU is free" in answer for _, answer in answers)
        # A new job shows it listened: the next free GPU is named at once again.
        other = await broker.create_lease(pid=job.pid, label="train", gpus=[], vram=None, drain=False, gpu_count=1,
                                          session="s1", cwd=str(tmp_path), resume=True, note="")
        st.waiter.watch_lease(other)
        assert "s1" not in st.waiter.idle_gpu_answers
        await broker.release_lease(other.id, rc=0)
        assert (await timed_wait())[0] < 0.5
        await broker.release_lease(lease.id, rc=0)
    finally:
        job.kill()
        job.wait()


async def test_wait_says_nothing_about_another_agents_job(cfg, two_gpus, http, tmp_path):
    for answer in await wait_answers(cfg, two_gpus, http, tmp_path, session="s2"):
        assert "A GPU is free" not in answer


async def test_wait_says_nothing_when_turned_off(cfg, two_gpus, http, tmp_path):
    cfg.resume.idle_gpu_nudges = 0
    for answer in await wait_answers(cfg, two_gpus, http, tmp_path):
        assert "A GPU is free" not in answer

