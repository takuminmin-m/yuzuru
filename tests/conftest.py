"""Fakes: an in-memory Ollama and a GPU probe that share one memory model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest

from yuzuru.config import GB, Config, LeaseConfig, ResourceConfig, ResumeConfig
from yuzuru.resources import GPU, Probe, Snapshot


@dataclass
class FakeWorld:
    gpu_sizes: list[int] = field(default_factory=lambda: [8 * GB, 12 * GB])
    ram_total: int = 32 * GB
    foreign: list[int] = field(default_factory=lambda: [0, 0])  # e.g. training jobs
    installed: dict[str, dict] = field(default_factory=dict)
    loaded: dict[str, dict] = field(default_factory=dict)  # name -> {size, size_vram}
    variants: dict[str, dict] = field(default_factory=dict)  # name -> {from, parameters}
    difficulty: str = "medium"
    chat_requests: list[dict] = field(default_factory=list)
    unloads: list[str] = field(default_factory=list)
    busy: set[str] = field(default_factory=set)  # generating: Ollama defers their unload
    pending_unload: set[str] = field(default_factory=set)
    measure_host: bool = True

    def add_model(self, name: str, params_b: float, size_gb: float, moe: bool = False) -> None:
        self.installed[name] = {
            "name": name,
            "size": int(size_gb * GB),
            "digest": f"d-{name}",
            "details": {"parameter_size": f"{params_b}B", "family": "qwen3moe" if moe else "llama"},
        }

    def base_of(self, name: str) -> str:
        return self.variants[name]["from"] if name in self.variants else name

    def finish(self, name: str) -> None:
        """A generation ends: a deferred unload happens now."""
        self.busy.discard(name)
        if name in self.pending_unload:
            self.pending_unload.discard(name)
            self.loaded.pop(name, None)

    def ollama_vram(self) -> int:
        return sum(m["size_vram"] for m in self.loaded.values())

    def load(self, name: str) -> None:
        if name in self.loaded:
            return
        params = self.variants.get(name, {}).get("parameters", {})
        size = self.installed[self.base_of(name)]["size"] + int(0.5 * GB)
        if params.get("num_gpu") == 0:
            vram = 0
        else:
            free = sum(self.gpu_sizes) - sum(self.foreign) - self.ollama_vram()
            vram = max(0, min(size, free))
        self.loaded[name] = {"size": size, "size_vram": vram}

    def snapshot(self) -> Snapshot:
        # Spread Ollama's VRAM over GPUs in order, after foreign usage.
        rest = self.ollama_vram()
        gpus = []
        for i, (total, foreign) in enumerate(zip(self.gpu_sizes, self.foreign)):
            take = min(rest, total - foreign)
            rest -= take
            gpus.append(GPU(i, f"GPU-{i}", f"Fake {i}", total, foreign + take))
        host = self.ollama_host()
        return Snapshot(gpus=gpus, ram_total=self.ram_total, ram_available=self.ram_total - 4 * GB - host,
                        ollama_host=host if self.measure_host else None)

    def ollama_host(self) -> int:
        # Like llama.cpp on CUDA: partial offload keeps the whole model in host RAM.
        return sum(
            m["size"] if m["size_vram"] < m["size"] else int(0.1 * m["size"]) for m in self.loaded.values()
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path == "/api/tags":
            return httpx.Response(200, json={"models": list(self.installed.values()) + [
                {"name": v, "size": 0, "details": {}} for v in self.variants
            ]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n, **m} for n, m in self.loaded.items()]})
        if path == "/api/show":
            arch = self.installed[body["model"]]["details"]["family"]
            return httpx.Response(200, json={"model_info": {
                "general.architecture": arch, f"{arch}.block_count": 32,
                f"{arch}.attention.head_count": 32, f"{arch}.attention.head_count_kv": 8,
                f"{arch}.embedding_length": 4096,
            }})
        if path == "/api/create":
            self.variants[body["model"]] = {"from": body["from"], "parameters": body["parameters"]}
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/generate":
            if body.get("keep_alive") == 0:
                self.unloads.append(body["model"])
                if body["model"] in self.busy:
                    self.pending_unload.add(body["model"])
                else:
                    self.loaded.pop(body["model"], None)
            else:
                self.load(body["model"])
            return httpx.Response(200, json={"done": True})
        if path == "/api/chat":
            self.load(body["model"])
            return httpx.Response(200, json={"message": {"content": json.dumps({"difficulty": self.difficulty})}})
        if path == "/v1/chat/completions":
            self.load(body["model"])
            self.chat_requests.append(body)
            if body.get("stream"):
                chunks = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
                return httpx.Response(200, content=chunks, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"model": body["model"], "choices": [{"message": {"content": "hi"}}]})
        return httpx.Response(404, json={"error": path})


class FakeProbe(Probe):
    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def snapshot(self) -> Snapshot:
        return self.world.snapshot()


@pytest.fixture
def world() -> FakeWorld:
    w = FakeWorld()
    w.add_model("qwen3:4b", 4.0, 2.5)
    w.add_model("qwen3:14b", 14.0, 9.0)
    w.add_model("qwen3.6:latest", 36.0, 23.0, moe=True)
    return w


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        state_dir=tmp_path,
        resources=ResourceConfig(vram_margin_mb=256, ram_reserve_mb=4096),
        lease=LeaseConfig(settle_seconds=0.0, settle_timeout=5, drain_timeout=0.1, unload_timeout=0.5,
                          restore_after=False),
        resume=ResumeConfig(scope="off"),  # tests that want a scope fake systemd-run
    )


@pytest.fixture
def http(world: FakeWorld) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(world.handler))


# ------------------------------------------------------------ GPU allowlist


class AllowlistProbe(FakeProbe):
    """FakeProbe that applies `resources.gpus` like the real probes do."""

    def __init__(self, world: FakeWorld, allow: list[str]) -> None:
        super().__init__(world)
        self.allow = [a.lower() for a in allow]
        self._unmatched_warned = False

    def snapshot(self) -> Snapshot:
        snap = self.world.snapshot()
        snap.gpus = self.allowed(snap.gpus)
        return snap


FAKE_NVIDIA_SMI = """#!{python}
import sys
if "-L" in sys.argv:
    print("GPU 0: Fake 3050 (UUID: GPU-aaa)\\nGPU 1: Fake 3060 (UUID: GPU-bbb)")
else:
    print("0, GPU-aaa, Fake 3050, 8192, 100\\n1, GPU-bbb, Fake 3060, 12288, 11000")
"""

# Stands in for `ollama serve`: records its environment and answers /api/version.
FAKE_OLLAMA = """#!{python}
import http.server, json, os, sys
assert sys.argv[1:] == ["serve"], sys.argv
host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OLLAMA_MODELS", "OLLAMA_HOST", "OLLAMA_FLASH_ATTENTION")
with open({record!r}, "a") as f:
    f.write(json.dumps({{"pid": os.getpid(), **{{k: os.environ.get(k) for k in keys}}}}) + "\\n")

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{{"version": "0.0.0"}}')

    def log_message(self, *a):
        pass

http.server.HTTPServer((host, int(port)), H).serve_forever()
"""


def write_script(path, template: str, **kw) -> str:
    import stat
    import sys

    path.write_text(template.format(python=sys.executable, **kw))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def fake_nvidia_smi(tmp_path) -> str:
    return write_script(tmp_path / "nvidia-smi", FAKE_NVIDIA_SMI)


@pytest.fixture
def fake_ollama(tmp_path):
    """(binary, record file): each start appends its pid and environment to the record."""
    d = tmp_path / "bin"
    d.mkdir()
    record = tmp_path / "ollama-starts.jsonl"
    return write_script(d / "ollama", FAKE_OLLAMA, record=str(record)), record


@pytest.fixture
def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
