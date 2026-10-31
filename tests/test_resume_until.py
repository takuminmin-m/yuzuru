"""resume.until: a time-boxed task is not left to end early (HPO run 1 stopped with 3.5 hours to go)."""

import asyncio
import datetime

import pytest

from .test_core import make, wait_for
from .test_stall_nudge import agent, job_ends, reply, turns
from yuzuru.config import ResumeConfig
from yuzuru.waits import Waiter


def in_minutes(m):
    return (datetime.datetime.now() + datetime.timedelta(minutes=m)).isoformat(timespec="minutes")


async def test_a_turn_that_ends_before_until_is_continued(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path)
    cfg.resume.until, cfg.resume.until_nudges = in_minutes(120), 2
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path, rc=0)
    await wait_for(lambda: len(turns(out)) == 1)
    for _ in range(3):
        reply(router, "tool_calls")  # wrote RESULTS.md...
    reply(router)  # ...and declared the task done
    go.touch()
    await wait_for(lambda: len(turns(out)) == 3)  # continued, and that turn ended the same way: again
    message = turns(out)[1][1]
    assert message.startswith("[yuzuru]") and cfg.resume.until in message and "min from now" in message
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 3  # until_nudges in a row: then it is left alone
    assert waiter.nudges["s1"] == 2


async def test_after_until_a_finished_turn_is_left_alone(cfg, world, http, tmp_path):
    out, go = agent(cfg, tmp_path)
    cfg.resume.until = in_minutes(-5)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await job_ends(broker, waiter, tmp_path, rc=0)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router, "tool_calls")
    reply(router)
    go.touch()
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1


async def test_only_the_latest_resume_continues_a_session(cfg, world, http, tmp_path, caplog):
    out, go = agent(cfg, tmp_path)
    cfg.resume.until, cfg.resume.until_nudges = in_minutes(120), 1
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await waiter._continue("s1", "first", str(tmp_path))
    await waiter._continue("s1", "second", str(tmp_path))  # the first client is still attached
    await wait_for(lambda: len(turns(out)) == 2)
    reply(router)
    go.touch()
    await wait_for(lambda: len(turns(out)) == 3)
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 3 and waiter.nudges["s1"] == 1
    # The first client ending too decided nothing; only the continued turn's end hit the limit.
    assert caplog.text.count("continuing it (1/1)") == 1
    assert caplog.text.count("not continuing it again") == 1


def test_until_is_parsed_as_local_time():
    assert ResumeConfig().deadline() is None
    t = datetime.datetime(2026, 10, 31, 6, 15)
    assert ResumeConfig(until="2026-10-31T06:15").deadline() == t.timestamp()
    assert ResumeConfig(until=t).deadline() == t.timestamp()  # what YAML makes of 2026-10-31T06:15:00
    with pytest.raises(ValueError, match="resume.until"):
        ResumeConfig(until="06:15 tomorrow")
