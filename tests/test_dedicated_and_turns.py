"""A GPU a job was given stays free of the LLM; a resume never joins a turn still running."""

import asyncio
import json
import os
import sys

from .conftest import FakeProbe
from .test_core import make, wait_for
from .test_gpu_allowlist import managed_cfg, starts
from yuzuru.broker import Broker
from yuzuru.catalog import Catalog
from yuzuru.config import GB
from yuzuru.managed import ManagedOllama
from yuzuru.ollama import Ollama
from yuzuru.router import SessionState
from yuzuru.server import State, create_app
from yuzuru.waits import Waiter

# ------------------------------------------------------------------ dedicated GPUs


async def test_dedicated_gpu_is_hidden_from_ollama_until_the_job_ends(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    hidden: list[set[str]] = []

    async def restrict(gpus):
        hidden.append(set(gpus))

    broker.restrict = restrict
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=["1"], vram=None)
    assert hidden == [{"GPU-1"}]
    world.foreign = [0, 3 * GB]  # the job settles small at first...
    await broker.refresh()
    await broker.refresh()
    assert lease.state == "active"
    # ...yet its 9 GB left are not offered to the LLM: only GPU0 is.
    assert broker.budget().fast < 8 * GB
    await broker.release_lease(lease.id)
    await wait_for(lambda: len(hidden) == 2)
    assert hidden[-1] == set()
    assert broker.budget().fast > 16 * GB


async def test_pool_lease_does_not_restart_ollama(cfg, world, http):
    _, broker, _ = await make(cfg, world, http)
    calls = []

    async def restrict(gpus):
        calls.append(gpus)

    broker.restrict = restrict
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None)
    await broker.release_lease(lease.id)
    await asyncio.sleep(0.1)
    assert calls == []


async def test_requests_wait_while_ollama_restarts(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    gate = asyncio.Event()

    async def restrict(gpus):
        await gate.wait()

    broker.restrict = restrict
    lease_task = asyncio.create_task(broker.create_lease(pid=os.getpid(), label="t", gpus=["1"], vram=None))
    await wait_for(lambda: broker._restricting)
    acquire = asyncio.create_task(broker.acquire(catalog.specs["small"]))
    await asyncio.sleep(0.2)
    assert not acquire.done()  # held back until Ollama is up again
    gate.set()
    lease = await lease_task
    world.foreign = [0, 4 * GB]
    await broker.refresh()
    await broker.refresh()
    await asyncio.wait_for(acquire, 5)
    await broker.release(catalog.specs["small"], touch=False)
    await broker.release_lease(lease.id)


async def test_managed_ollama_restarts_without_hidden_gpus(cfg, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    managed_cfg(cfg, binary, free_port, tmp_path / "models")
    cfg.resources.gpus = ["GPU-aaa", "GPU-bbb"]
    m = ManagedOllama(cfg, lambda: ["GPU-aaa", "GPU-bbb"])
    await m.start()
    try:
        assert starts(record)[-1]["CUDA_VISIBLE_DEVICES"] == "GPU-aaa,GPU-bbb"
        await m.hide({"GPU-bbb"})
        assert starts(record)[-1] == starts(record)[-1] | {"pid": m.pid, "CUDA_VISIBLE_DEVICES": "GPU-aaa"}
        await m.hide({"GPU-aaa", "GPU-bbb"})
        assert starts(record)[-1]["CUDA_VISIBLE_DEVICES"] == "-1"
        await m.hide(set())
        assert starts(record)[-1]["CUDA_VISIBLE_DEVICES"] == "GPU-aaa,GPU-bbb"
        await asyncio.sleep(1.5)  # the supervisor must not start a second copy of its own
        assert len(starts(record)) == 4 and starts(record)[-1]["pid"] == m.pid
        assert m.url == f"http://127.0.0.1:{free_port}"
    finally:
        await m.stop()


# ------------------------------------------------------------------ turns and resume


def resume_capture(cfg, tmp_path):
    out = tmp_path / "resumed.jsonl"
    cfg.resume.command = [sys.executable, "-c",
                          f"import sys, json; open({str(out)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')",
                          "{session}", "{message}"]
    return out


async def test_router_tracks_turns():
    from yuzuru.router import Router

    r = Router.__new__(Router)
    r.sessions = {}
    assert not r.in_turn("s", 600)
    r.answering("s")
    assert r.in_turn("s", 600)
    r.answered("s", "tool_calls")
    assert r.in_turn("s", 600)  # the tool runs; the next request is due
    assert not r.in_turn("s", 0)  # ...unless the client went away
    r.answering("s")
    r.answered("s", "stop")
    assert not r.in_turn("s", 600)


async def test_resume_waits_for_the_running_turn(cfg, world, http, tmp_path):
    out = resume_capture(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    router.answering("s1")
    router.answered("s1", "tool_calls")  # the agent is still in the turn that started the job
    await broker.release_lease(lease.id, rc=1)  # ...which crashes at once
    await asyncio.sleep(1.5)
    assert not out.exists()  # no second client next to the running one
    router.answering("s1")
    router.answered("s1", "stop")  # the turn ends
    await wait_for(out.exists)
    assert json.loads(out.read_text())[0] == "s1"


async def test_no_resume_when_the_turn_moved_on_to_another_job(cfg, world, http, tmp_path):
    out = resume_capture(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    first = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(first)
    router.answering("s1")
    router.answered("s1", "tool_calls")
    await broker.release_lease(first.id, rc=1)
    # The agent read the crash in its turn and started the job again.
    second = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                       cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(second)
    router.answering("s1")
    router.answered("s1", "stop")
    await asyncio.sleep(1.5)
    assert not out.exists()
    await broker.release_lease(second.id, rc=0)
    await wait_for(out.exists)
    assert len(out.read_text().splitlines()) == 1 and "finished" in out.read_text()


async def test_suspend_waits_for_the_turn(cfg, world, http, tmp_path):
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await broker.acquire(catalog.specs["small"])
    await broker.release(catalog.specs["small"], touch=False)
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                      cwd=str(tmp_path), resume=True, note="")
    waiter.watch_lease(lease)
    cfg.resume.suspend_idle_seconds = 0
    router.sessions["s1"] = SessionState(tool_calls=True)  # a long tool call of the same turn
    await broker.acquire(catalog.specs["small"])
    await broker.release(catalog.specs["small"], touch=False)
    await waiter.poll()
    assert world.loaded  # the agent is still working: keep its model
    router.answered("s1", "stop")
    await waiter.poll()
    assert not world.loaded
    await broker.release_lease(lease.id, rc=0)


async def test_stream_records_tool_call_finish(cfg, world, http):
    import httpx

    chunks = (b'data: {"choices":[{"delta":{"tool_calls":[]}}]}\n\n'
              b'data: {"choices":[{"delta":{},"finish_reason":"tool_ca')
    rest = b'lls"}]}\n\ndata: [DONE]\n\n'

    async def split():  # the finish_reason arrives cut in two
        yield chunks
        yield rest

    def handler(request):
        if request.url.path == "/v1/chat/completions" and json.loads(request.content).get("stream"):
            world.load(json.loads(request.content)["model"])
            return httpx.Response(200, content=split(), headers={"content-type": "text/event-stream"})
        return world.handler(request)

    st = State(cfg, FakeProbe(world), httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler)))
    await st.start()
    try:
        app = create_app(cfg, st)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            body = {"model": "small", "stream": True, "messages": [{"role": "user", "content": "ls"}]}
            r = await c.post("/v1/chat/completions", json=body, headers={"x-yuzuru-session": "s9"})
            assert r.status_code == 200
        assert st.router.sessions["s9"].tool_calls and st.router.sessions["s9"].answering == 0
        assert st.router.in_turn("s9", 600)
    finally:
        await st.stop()


async def test_outcomes_held_for_one_turn_resume_once(cfg, world, http, tmp_path):
    out = resume_capture(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    router.answering("s1")
    router.answered("s1", "tool_calls")  # one long turn: the agent starts job after job and reads their logs
    for rc in (1, 1, 0):
        lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="s1",
                                          cwd=str(tmp_path), resume=True, note="")
        waiter.watch_lease(lease)
        await broker.release_lease(lease.id, rc=rc)
    await asyncio.sleep(1.5)
    assert not out.exists()
    router.answering("s1")
    router.answered("s1", "stop")
    await wait_for(out.exists)
    await asyncio.sleep(1.5)
    lines = out.read_text().splitlines()
    assert len(lines) == 1  # one client, not three
    message = json.loads(lines[0])[1]
    assert message.startswith("[yuzuru] The process you were waiting for (train) finished")
    assert message.count("exited with code 1") == 2
