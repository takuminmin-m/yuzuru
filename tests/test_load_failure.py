import httpx

from .conftest import FakeProbe
from yuzuru.config import RouterConfig
from yuzuru.server import State, create_app


def user(text):
    return {"model": "auto", "messages": [{"role": "user", "content": text}]}


async def test_failed_load_falls_back_and_is_not_retried(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    orig = world.handler
    attempts = []

    def handler(request):
        if request.url.path == "/api/generate" and b"yuzuru-medium" in request.content and b"keep_alive\": 0" not in request.content:
            attempts.append(1)
            return httpx.Response(500, json={"error": "cudaMalloc failed: out of memory"})
        return orig(request)

    http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    st = State(cfg, probe=FakeProbe(world), http=http)
    app = create_app(cfg, st)
    await st.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            r = await c.post("/v1/chat/completions", json=user("この関数にテストを書いて"))  # wants medium
            assert r.status_code == 200
            assert r.headers["x-yuzuru-tier"] != "medium"
            r = await c.post("/v1/chat/completions", json=user("この関数にテストを書いて"))
            assert r.status_code == 200
        assert len(attempts) == 1  # the failed model is not retried during its cooldown
        assert st.broker.is_failed(st.catalog.specs["medium"].variant)
        assert st.broker.inflight[st.catalog.specs["medium"].variant] == 0
    finally:
        await st.stop()
