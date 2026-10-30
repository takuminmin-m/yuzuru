"""A resumed turn that stalls (one reply, no tool call, no job left) is continued, a bounded number of times.

In LoRA run 3 yuzuru resumed the agent after a failed job; the turn ended after a single empty reply,
and with no job running nothing woke the agent again for 13 hours.
"""

import asyncio
import json
import os
import sys

from .test_core import make, wait_for
from yuzuru.waits import STALLED, Waiter

# The fake agent: records its arguments (and whether it sees a GPU), then keeps its turn open
# until the test lets it end by creating `go`.
AGENT = r"""
import json, os, sys, time
out, go = sys.argv[1], sys.argv[2]
open(out, "a").write(json.dumps([sys.argv[3], sys.argv[4], os.environ.get("CUDA_VISIBLE_DEVICES")]) + "\n")
end = time.time() + 10
while not os.path.exists(go) and time.time() < end:
    time.sleep(0.02)
"""


def agent(cfg, tmp_path, nudges=1):
    out, go = tmp_path / "turns.jsonl", tmp_path / "go"
    cfg.resume.command = [sys.executable, "-c", AGENT, str(out), str(go), "{session}", "{message}"]
    cfg.resume.stall_nudges = nudges
    cfg.resume.turn_idle_seconds = 5
    return out, go


def turns(out):
    return [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []


async def job_ends(broker, waiter, tmp_path, rc=1):
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    await broker.release_lease(lease.id, rc=rc)


def reply(router, finish="stop"):
    router.answering("s1")
    router.answered("s1", finish)


async def test_stalled_turn_is_continued_once(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router)  # one reply, no tool call: the turn did nothing
    go.touch()
    await wait_for(lambda: len(turns(out)) == 2)
    session, message, visible = turns(out)[1]
    assert session == "s1" and message == STALLED
    assert visible == ""  # a resumed agent sees no GPU either
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 2  # the continued turn stalled too (no reply): not again
    assert waiter.nudges["s1"] == 1
    # Starting a job again shows it is working: a later stall may be continued again.
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    assert "s1" not in waiter.nudges
    await broker.release_lease(lease.id, rc=0)


async def test_turn_that_did_work_is_left_alone(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router, "tool_calls")  # read the log...
    reply(router)  # ...and reported: the agent decided to stop here
    go.touch()
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1


async def test_turn_with_a_job_running_is_left_alone(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path)
    await wait_for(lambda: len(turns(out)) == 1)
    # A job of the session runs (started some other way); its end is what wakes the agent.
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1")
    reply(router)
    go.touch()
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1
    await broker.release_lease(lease.id, rc=0)


async def test_nudges_can_be_turned_off(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path, nudges=0)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router)
    go.touch()
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1


async def test_turn_still_running_in_the_service_is_waited_for(cfg, world, http, tmp_path):
    # `opencode run` exited, but the last reply called tools: the turn goes on in the OpenCode service.
    out, go = agent(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router, "tool_calls")
    go.touch()
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1  # not judged yet
    reply(router, "tool_calls")
    reply(router)
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1  # three replies: it worked


async def test_a_job_ending_while_a_resumed_turn_starts_does_not_resume_it_again(cfg, world, http, tmp_path):
    # The resumed client has not sent a request yet (it starts, the model loads): the router sees no
    # turn, but a second resume now would put two clients on the session (HPO run 2).
    out, go = agent(cfg, tmp_path, nudges=0)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path)
    await wait_for(lambda: len(turns(out)) == 1)
    await job_ends(broker, waiter, tmp_path, rc=0)
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1  # held for the turn that is starting
    reply(router)
    go.touch()
    await wait_for(lambda: len(turns(out)) == 2)
    assert "finished" in turns(out)[1][1]
