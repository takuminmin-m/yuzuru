"""When jobs give their GPUs back, a generation the LLM began on the CPU is redone on them
instead of keeping them idle for drain_timeout (HPO run 1: 13 minutes on the CPU)."""

import asyncio
import json
import os

import httpx

from .conftest import FakeProbe
from .test_core import make, wait_for
from yuzuru.config import RouterConfig
from yuzuru.server import State, create_app


async def dedicated(broker):
    hidden: list[set[str]] = []

    async def restrict(gpus):
        hidden.append(set(gpus))

    broker.restrict = restrict
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=["1"], vram=None)
    return lease, hidden


async def test_giving_back_does_not_wait_for_a_generation_before_its_first_token(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    lease, hidden = await dedicated(broker)
    broker.inflight["m"] += 1
    broker.unstarted["m"] += 1  # e.g. a long prompt still being processed on the CPU
    await broker.release_lease(lease.id)
    await wait_for(lambda: len(hidden) == 2)
    assert hidden[-1] == set()


async def test_giving_back_waits_for_a_generation_already_streaming(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    lease, hidden = await dedicated(broker)
    broker.inflight["m"] += 1  # tokens are reaching a client: cutting it would lose them
    await broker.release_lease(lease.id)
    await asyncio.sleep(0.3)
    assert len(hidden) == 1
    broker.inflight["m"] -= 1
    await wait_for(lambda: len(hidden) == 2)


async def test_taking_gpus_still_waits_for_a_generation_before_its_first_token(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    hidden: list[set[str]] = []

    async def restrict(gpus):
        hidden.append(set(gpus))

    broker.restrict = restrict
    broker.inflight["m"] += 1
    broker.unstarted["m"] += 1  # on the GPUs now; redoing it on what is left would be slower
    task = asyncio.create_task(broker.create_lease(pid=os.getpid(), label="train", gpus=["1"], vram=None))
    await asyncio.sleep(0.3)
    assert hidden == []
    broker.inflight["m"] -= 1
    broker.unstarted["m"] -= 1
    lease = await asyncio.wait_for(task, 5)
    await broker.release_lease(lease.id)


def req():
    return {"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


async def serve(cfg, world, cut):
    """yuzuru whose first upstream completion is cut, as a restart of Ollama would."""
    cfg.router = RouterConfig(classifier="heuristic")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                cut(st)
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
        return world.handler(request)

    http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    st = State(cfg, probe=FakeProbe(world), http=http)
    await st.start()
    return st, create_app(cfg, st), calls


async def post(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
        return await c.post("/v1/chat/completions", json=req())


async def test_a_stream_cut_by_a_restart_before_its_first_token_is_sent_again(cfg, world):
    def restart(st):
        assert sum(st.broker.unstarted.values()) == 1
        st.broker.restarts += 1

    st, app, calls = await serve(cfg, world, restart)
    try:
        r = await post(app)
        assert len(calls) == 2
        assert b"[DONE]" in r.content and b'"error"' not in r.content
        assert not any(st.broker.inflight.values()) and not any(st.broker.unstarted.values())
    finally:
        await st.stop()


async def test_a_stream_cut_without_a_restart_is_an_error(cfg, world):
    st, app, calls = await serve(cfg, world, lambda st: None)  # Ollama crashed on its own
    try:
        r = await post(app)
        assert len(calls) == 1
        assert b'"error"' in r.content
        assert not any(st.broker.inflight.values()) and not any(st.broker.unstarted.values())
    finally:
        await st.stop()
