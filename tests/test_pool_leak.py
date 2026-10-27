"""Cancellations must not leak Ollama connections or skip the clean-up of a completion.

A cancellation that interrupts httpx while it finishes a response leaves that connection busy for
good. In a long agent run all 100 of the pool's connections ended up like that and every request
to Ollama hung: the LLM was never moved back to the GPU after a training job, and the agent's turn
timed out.
"""

import asyncio
import contextlib
import json
import random

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .test_stream_keepalive import make, req
from yuzuru.ollama import Ollama


async def test_cancelled_calls_do_not_use_up_the_pool():
    rng = random.Random(0)

    async def ps(request):
        await asyncio.sleep(rng.uniform(0, 0.004))
        return JSONResponse({"models": []})

    server = uvicorn.Server(uvicorn.Config(Starlette(routes=[Route("/api/ps", ps)]), port=0, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    ollama = Ollama(url, httpx.AsyncClient(base_url=url, limits=httpx.Limits(max_connections=5)))
    try:
        for _ in range(200):
            call = asyncio.create_task(ollama.ps())
            await asyncio.sleep(rng.uniform(0, 0.006))
            call.cancel()
            await asyncio.sleep(0)
            call.cancel()  # what a cancelled stream_chat used to do to its producer
            with contextlib.suppress(asyncio.CancelledError):
                await call
        await asyncio.sleep(0.2)
        assert await asyncio.wait_for(ollama.ps(), 5) == []  # a leaked pool would wait forever
    finally:
        await ollama.aclose()
        server.should_exit = True
        await serving


async def test_client_leaving_after_the_answer_does_not_skip_the_clean_up(cfg, world, http):
    st, app = await make(cfg, world, http)
    released = asyncio.Event()
    release = st.broker.release

    async def slow_release(spec, *a, **kw):
        await asyncio.sleep(0.2)  # e.g. the keep_alive refresh
        await release(spec, *a, **kw)
        released.set()

    st.broker.release = slow_release
    body = json.dumps(req()).encode()
    answered = asyncio.Event()
    requested = False

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": body, "more_body": False}
        await answered.wait()
        return {"type": "http.disconnect"}  # OpenCode hangs up as soon as it has [DONE]

    async def send(message):
        if b"[DONE]" in message.get("body", b""):
            answered.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
        "query_string": b"", "root_path": "", "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 11435),
        "headers": [(b"host", b"127.0.0.1:11435"), (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    }
    try:
        await app(scope, receive, send)
        await asyncio.wait_for(released.wait(), 2)
        assert not any(st.broker.inflight.values())
    finally:
        await st.stop()
