import dataclasses
import http.server
import json
import threading

import httpx
import pytest

from yuzuru.config import OllamaConfig
from yuzuru.managed import ManagedOllama
from yuzuru.ollama import Ollama, OllamaError

CHTIMES = '{"error":"chtimes /m/blobs/sha256-ab: operation not permitted"}'


def managed_ollama(cfg, tmp_path, mirror=True):
    src = tmp_path / "system"
    (src / "blobs").mkdir(parents=True)
    cfg = dataclasses.replace(cfg)
    cfg.ollama = OllamaConfig(manage=True, models=tmp_path / "models", mirror_from=str(src) if mirror else "none")
    return ManagedOllama(cfg, lambda: []), src


def answering(status, body, created=None):
    """An Ollama /api/create that answers `status` and records what it was asked."""
    def handler(request: httpx.Request) -> httpx.Response:
        if created is not None:
            created.append(json.loads(request.content))
        return httpx.Response(status, text=body)
    return Ollama("http://managed", httpx.AsyncClient(base_url="http://managed", transport=httpx.MockTransport(handler)))


class SystemOllama(http.server.BaseHTTPRequestHandler):
    """Creates the variant in the source store, as the real system Ollama would."""
    store = None
    asked = []

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SystemOllama.asked.append(req)
        name, tag = req["model"].split(":")
        d = self.store / "manifests" / "registry.ollama.ai" / "library" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / tag).write_text("{}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"success"}')

    def log_message(self, *a):
        pass


@pytest.fixture
def system_ollama():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SystemOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SystemOllama.asked = []
    yield srv
    srv.shutdown()


async def test_created_on_the_managed_server_when_it_can(cfg, tmp_path):
    m, _ = managed_ollama(cfg, tmp_path)
    created = []
    await m.create_variant(answering(200, '{"status":"success"}', created), "yuzuru-small:x-8k", "x", {"num_ctx": 8192})
    assert created == [{"model": "yuzuru-small:x-8k", "from": "x", "parameters": {"num_ctx": 8192}, "stream": False}]


async def test_shared_blobs_it_cannot_touch_go_through_the_system_ollama(cfg, tmp_path, system_ollama):
    m, src = managed_ollama(cfg, tmp_path)
    SystemOllama.store = src
    m.system_url = f"http://127.0.0.1:{system_ollama.server_port}"
    await m.create_variant(answering(500, CHTIMES), "yuzuru-medium:x-32k", "x", {"num_ctx": 32768, "num_gpu": 0})
    assert SystemOllama.asked == [{"model": "yuzuru-medium:x-32k", "from": "x",
                                   "parameters": {"num_ctx": 32768, "num_gpu": 0}, "stream": False}]
    # Mirrored right away, so the managed server can load it.
    assert (tmp_path / "models" / "manifests" / "registry.ollama.ai" / "library" / "yuzuru-medium" / "x-32k").exists()


async def test_no_system_ollama_says_why(cfg, tmp_path):
    m, _ = managed_ollama(cfg, tmp_path)
    m.system_url = "http://127.0.0.1:9"  # nothing listens
    with pytest.raises(OllamaError, match="system Ollama .* not reachable"):
        await m.create_variant(answering(500, CHTIMES), "yuzuru-small:x-8k", "x", {"num_ctx": 8192})


async def test_other_errors_and_unshared_stores_are_not_rerouted(cfg, tmp_path):
    m, _ = managed_ollama(cfg, tmp_path)
    with pytest.raises(OllamaError, match="no such model"):
        await m.create_variant(answering(404, '{"error":"no such model"}'), "yuzuru-small:x-8k", "x", {})
    m, _ = managed_ollama(cfg, tmp_path / "own", mirror=False)
    with pytest.raises(OllamaError, match="chtimes"):
        await m.create_variant(answering(500, CHTIMES), "yuzuru-small:x-8k", "x", {})


def test_catalog_creates_variants_through_the_managed_server(cfg, tmp_path):
    from yuzuru.server import State
    cfg.ollama = OllamaConfig(manage=True, models=tmp_path / "models", mirror_from="none")
    st = State(cfg)
    assert st.catalog.create_variant.func == st.managed.create_variant
