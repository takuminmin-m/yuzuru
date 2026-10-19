from __future__ import annotations

import os

import httpx
import pytest

from yuzuru.broker import Broker, Placement
from yuzuru.catalog import Catalog, kv_bytes_per_token, variant_name
from yuzuru.config import GB, RouterConfig, TierConfig
from yuzuru.ollama import Ollama
from yuzuru.router import Router, heuristic_level
from yuzuru.server import State, create_app

from .conftest import FakeProbe


async def make(cfg, world, http):
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    await catalog.refresh()
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    await broker.refresh()
    return catalog, broker, Router(cfg, catalog, broker, ollama)


def user(text):
    return {"model": "auto", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]}


def test_kv_estimate_llama_gqa():
    info = {
        "general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096,
    }
    assert kv_bytes_per_token(info) == 32 * 8 * 256 * 2


def test_kv_estimate_hybrid_attention_interval():
    info = {
        "general.architecture": "x", "x.block_count": 40, "x.attention.head_count": 16,
        "x.attention.head_count_kv": 2, "x.attention.key_length": 256, "x.full_attention_interval": 4,
    }
    assert kv_bytes_per_token(info) == 10 * 2 * 512 * 2


async def test_auto_assigns_tiers_by_size(cfg, world, http):
    catalog, _, _ = await make(cfg, world, http)
    assert {t: s.base for t, s in catalog.specs.items()} == {
        "small": "qwen3:4b", "medium": "qwen3:14b", "large": "qwen3.6:latest",
    }
    assert catalog.specs["large"].moe
    assert catalog.router_spec is not None and catalog.router_spec.cpu_only
    assert world.variants[variant_name("large", "qwen3.6:latest", 32768)]["parameters"] == {"num_ctx": 32768}


async def test_single_big_model_disables_classifier(cfg, world, http):
    del world.installed["qwen3:4b"], world.installed["qwen3:14b"]
    catalog, _, _ = await make(cfg, world, http)
    assert catalog.tiers == ["large"]
    assert catalog.router_spec is None
    assert catalog.nearest_tier("small") == "large"


async def test_explicit_candidates_win(cfg, world, http):
    cfg.tiers = [TierConfig("small", models=["qwen3:14b"]), TierConfig("large", models=["missing", "qwen3:4b"])]
    catalog, _, _ = await make(cfg, world, http)
    assert catalog.specs["small"].base == "qwen3:14b"
    assert catalog.specs["large"].base == "qwen3:4b"


async def test_placement_moe_offloads_dense_does_not(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    assert broker.plan(catalog.specs["medium"], True)[0] is Placement.GPU
    assert broker.plan(catalog.specs["large"], True)[0] is Placement.OFFLOAD
    catalog.specs["large"].moe = False
    assert broker.plan(catalog.specs["large"], True)[0] is Placement.NONE


async def test_acquire_evicts_idle_model(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    world.foreign = [0, 6 * GB]
    await broker.refresh()
    assert await broker.acquire(catalog.specs["medium"]) is Placement.GPU
    await broker.release(catalog.specs["medium"], touch=False)
    # small no longer fits next to medium within 14 GB -> medium gets evicted.
    world.foreign = [0, 10 * GB]
    await broker.refresh()
    assert await broker.acquire(catalog.specs["small"]) is Placement.GPU
    assert catalog.specs["medium"].variant in world.unloads


async def test_lease_unloads_settles_and_shrinks_budget(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    await broker.acquire(catalog.specs["medium"])
    await broker.release(catalog.specs["medium"], touch=False)
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=["1"], vram=None)
    assert lease.gpus == ["GPU-1"]
    assert catalog.specs["medium"].variant in world.unloads
    assert lease.state == "settling"
    # While settling, nothing new may land on the leased GPU's free memory.
    assert broker.budget().fast <= 8 * GB
    world.foreign = [0, 10 * GB]  # training allocated 10 GB on GPU1
    await broker.refresh()
    await broker.refresh()
    assert lease.state == "active"
    b = broker.budget()
    # 20 GB total - 10 GB training - 1 GB headroom - margins
    assert 8 * GB < b.fast < 9 * GB
    assert broker.plan(catalog.specs["medium"], True)[0] is not Placement.GPU
    assert broker.plan(catalog.specs["small"], True)[0] is Placement.GPU
    await broker.release_lease(lease.id)
    assert not broker.leases


async def test_dead_pid_lease_is_reaped(cfg, world, http):
    _, broker, _ = await make(cfg, world, http)
    lease = await broker.create_lease(pid=2**22 + 12345, label="x", gpus=[], vram=None)
    assert lease.id not in broker.leases


def test_heuristics():
    assert heuristic_level("このモジュールの設計を見直したい")[0] == 2
    assert heuristic_level("ありがとう")[0] == 0
    assert heuristic_level("この関数にテストを書いて")[0] == 1


async def test_router_classifier_and_stickiness(cfg, world, http):
    catalog, broker, router = await make(cfg, world, http)
    world.difficulty = "easy"
    d = await router.route(user("ls して"), {"x-yuzuru-session": "s1"})
    assert d.tier == "small" and "classifier:easy" in d.reason
    # A tool-result continuation keeps the turn's tier without reclassifying.
    world.difficulty = "hard"
    body = user("ls して")
    body["messages"] += [{"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "content": "ok"}]
    d = await router.route(body, {"x-yuzuru-session": "s1"})
    assert d.wanted == "small" and "continuation" in d.reason


async def test_router_escalates_after_tool_errors(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic", error_escalation=2)
    _, _, router = await make(cfg, world, http)
    body = user("テストを直して")
    await router.route(body, {"x-yuzuru-session": "s"})
    for _ in range(2):
        body["messages"] += [{"role": "assistant", "content": ""}, {"role": "tool", "content": "Traceback (most recent call last):\n..."}]
    d = await router.route(body, {"x-yuzuru-session": "s"})
    assert d.wanted == "large" and "escalated" in d.reason


async def test_router_context_floor_and_overrides(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    _, _, router = await make(cfg, world, http)
    router.pin("long", "small")  # small has 16k context
    d = await router.route(user("x" * 60000), {"x-yuzuru-session": "long"})
    assert d.wanted == "medium" and "context" in d.reason
    d = await router.route(user("ありがとう @large"), {})
    assert d.wanted == "large"
    router.pin("p", "small")
    d = await router.route(user("設計して"), {"x-yuzuru-session": "p"})
    assert d.wanted == "small"
    d = await router.route({**user("x"), "model": "yuzuru/medium"}, {})
    assert d.tier == "medium"


async def test_router_downgrades_when_training_holds_memory(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    catalog, broker, router = await make(cfg, world, http)
    world.foreign = [0, 11 * GB]
    await broker.refresh()
    catalog.specs["large"].moe = False
    d = await router.route(user("アーキテクチャを設計して"), {})
    assert d.wanted == "large" and d.tier in ("small", "medium") and "downgraded" in d.reason


async def test_server_routes_and_streams(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    st = State(cfg, probe=FakeProbe(world), http=http)
    app = create_app(cfg, st)
    await st.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://yuzuru") as c:
            r = await c.get("/v1/models")
            assert [m["id"] for m in r.json()["data"]] == ["auto", "small", "medium", "large"]
            r = await c.post("/v1/chat/completions", json=user("ありがとう"), headers={"x-yuzuru-session": "a"})
            assert r.status_code == 200 and r.headers["x-yuzuru-tier"] == "small"
            assert world.chat_requests[-1]["model"] == catalog_variant(st, "small")
            r = await c.post("/v1/chat/completions", json={**user("ok"), "stream": True})
            assert r.status_code == 200 and b"[DONE]" in r.content
            assert st.broker.inflight[catalog_variant(st, "small")] == 0
            r = await c.get("/yuzuru/status")
            assert set(r.json()["tiers"]) == {"small", "medium", "large"}
    finally:
        await st.stop()


async def test_token_required_when_configured(cfg, world, http):
    cfg.token = "secret"
    st = State(cfg, probe=FakeProbe(world), http=http)
    app = create_app(cfg, st)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://y") as c:
        assert (await c.get("/v1/models")).status_code == 401
        r = await c.get("/v1/models", headers={"authorization": "Bearer secret"})
        assert r.status_code == 200


def catalog_variant(st, tier):
    return st.catalog.specs[tier].variant


async def test_mcp_tools(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    st = State(cfg, probe=FakeProbe(world), http=http)
    app = create_app(cfg, st)
    await st.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://y") as c:
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            assert r.json()["result"]["serverInfo"]["name"] == "yuzuru"
            assert (await c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})).status_code == 202
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            names = {t["name"] for t in r.json()["result"]["tools"]}
            assert {"yuzuru_status", "yuzuru_model", "yuzuru_train"} <= names
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                           "params": {"name": "yuzuru_status", "arguments": {}}})
            assert "LLM budget" in r.json()["result"]["content"][0]["text"]
            # Pin the session OpenCode identifies via its own header.
            await c.post("/v1/chat/completions", json=user("hi"), headers={"x-opencode-session": "ses_1"})
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                           "params": {"name": "yuzuru_model", "arguments": {"tier": "large"}}})
            assert not r.json()["result"]["isError"]
            assert st.router.sessions["ses_1"].pinned == "large"
    finally:
        await st.stop()
