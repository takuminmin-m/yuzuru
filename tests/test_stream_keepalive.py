import asyncio
import json

import httpx

from .conftest import FakeProbe
from yuzuru import server
from yuzuru.config import RouterConfig
from yuzuru.server import State, create_app


def req(text="hi"):
    return {"model": "auto", "stream": True, "messages": [{"role": "user", "content": text}]}


async def make(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    st = State(cfg, probe=FakeProbe(world), http=http)
    await st.start()
    return st, create_app(cfg, st)


async def test_keepalive_while_the_model_loads(cfg, world, http, monkeypatch):
    monkeypatch.setattr(server, "KEEPALIVE_SECONDS", 0.05)
    st, app = await make(cfg, world, http)
    orig = st.broker.acquire

    async def slow_acquire(spec, *a, **kw):
        await asyncio.sleep(0.3)  # a slow model load
        return await orig(spec, *a, **kw)

    st.broker.acquire = slow_acquire
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            r = await c.post("/v1/chat/completions", json=req())
        assert r.status_code == 200
        assert r.content.count(b": keep-alive") >= 2
        assert b"[DONE]" in r.content
        assert not any(st.broker.inflight.values())
    finally:
        await st.stop()


async def test_errors_arrive_as_an_sse_error_event(cfg, world, http):
    st, app = await make(cfg, world, http)
    st.catalog.specs.clear()  # nothing to serve
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            r = await c.post("/v1/chat/completions", json=req())
        assert r.status_code == 200
        event = next(l for l in r.text.splitlines() if l.startswith("data: "))
        assert "error" in json.loads(event[6:])
    finally:
        await st.stop()
