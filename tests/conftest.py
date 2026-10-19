"""Fakes: an in-memory Ollama and a GPU probe that share one memory model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest

from yuzuru.config import GB, Config, LeaseConfig, ResourceConfig
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

    def add_model(self, name: str, params_b: float, size_gb: float, moe: bool = False) -> None:
        self.installed[name] = {
            "name": name,
            "size": int(size_gb * GB),
            "digest": f"d-{name}",
            "details": {"parameter_size": f"{params_b}B", "family": "qwen3moe" if moe else "llama"},
        }

    def base_of(self, name: str) -> str:
        return self.variants[name]["from"] if name in self.variants else name

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
        ram_used = 4 * GB + sum(m["size"] - m["size_vram"] for m in self.loaded.values())
        return Snapshot(gpus=gpus, ram_total=self.ram_total, ram_available=self.ram_total - ram_used)

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
                self.loaded.pop(body["model"], None)
                self.unloads.append(body["model"])
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
        lease=LeaseConfig(settle_seconds=0.0, settle_timeout=5, drain_timeout=0.1, restore_after=False),
    )


@pytest.fixture
def http(world: FakeWorld) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(world.handler))
