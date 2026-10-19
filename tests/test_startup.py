import asyncio
import json
import sys

import httpx

from .conftest import FakeProbe
from yuzuru.server import State, create_app
from yuzuru.waits import Event, Waiter, Watch


async def test_server_starts_while_ollama_is_down_and_recovers(cfg, world):
    up = False

    def handler(request):
        if not up:
            raise httpx.ConnectError("connection refused")
        return world.handler(request)

    http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    st = State(cfg, probe=FakeProbe(world), http=http)
    await st.start()  # must not raise
    try:
        assert st.catalog.specs == {}
        app = create_app(cfg, st)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            r = await c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 503
        up = True
        await st.catalog.refresh()
        assert "small" in st.catalog.specs
    finally:
        await st.stop()


async def test_resumed_process_is_reaped(cfg, world, http, tmp_path):
    from yuzuru.broker import Broker
    from yuzuru.catalog import Catalog
    from yuzuru.ollama import Ollama
    from yuzuru.router import Router

    cfg.resume.warm_before_resume = False
    cfg.resume.command = [sys.executable, "-c", "pass", "{session}", "{message}"]
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    waiter = Waiter(cfg, broker, Router(cfg, catalog, broker, ollama))
    ev = Event(1, "w", "ses", "job", 0, 1.0, None, "", "")
    await waiter.resume(ev, Watch(id="w", session="ses"))
    assert ev.resumed
    for _ in range(100):
        if not waiter._tasks:
            break
        await asyncio.sleep(0.05)
    assert not waiter._tasks  # proc.wait() finished: no zombie left behind
