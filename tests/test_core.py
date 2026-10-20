from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from yuzuru.broker import Broker, Placement
from yuzuru.catalog import Catalog, kv_bytes_per_token, variant_name
from yuzuru.config import GB, RouterConfig, TierConfig
from yuzuru.ollama import Ollama
from yuzuru.router import Router, SessionState, heuristic_level
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
    world.ram_total = 64 * GB
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
    cfg.tiers[0].num_ctx = 16384
    router.pin("long", "small")  # small now has 16k context
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
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:11435") as c:
            r = await c.get("/v1/models")
            ids = [m["id"] for m in r.json()["data"]]
            assert ids[:4] == ["auto", "small", "medium", "large"] and "qwen3:14b" in ids
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
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
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
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            assert r.json()["result"]["serverInfo"]["name"] == "yuzuru"
            assert (await c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})).status_code == 202
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            names = {t["name"] for t in r.json()["result"]["tools"]}
            assert {"status", "model", "models", "train", "wait"} <= names
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                           "params": {"name": "status", "arguments": {}}})
            assert "LLM budget" in r.json()["result"]["content"][0]["text"]
            # Pin the session OpenCode identifies via its own header.
            await c.post("/v1/chat/completions", json=user("hi"), headers={"x-opencode-session": "ses_1"})
            r = await c.post("/mcp", json={"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                           "params": {"name": "model", "arguments": {"tier": "large"}}})
            assert not r.json()["result"]["isError"]
            assert st.router.sessions["ses_1"].pinned == "large"
    finally:
        await st.stop()


async def test_offload_needs_host_ram_for_whole_model(cfg, world, http):
    # 32 GB RAM: offloading the 28 GB MoE would need ~28 GB of host RAM -> refuse.
    catalog, broker, _ = await make(cfg, world, http)
    assert broker.plan(catalog.specs["large"], True)[0] is Placement.NONE
    world.ram_total = 64 * GB
    await broker.refresh()
    spec = catalog.specs["large"]
    assert await broker.acquire(spec) is Placement.OFFLOAD
    await broker.release(spec, touch=False)
    # The measured host footprint is remembered per placement.
    assert spec.measured_host["offload"] == world.loaded[spec.variant]["size"]
    assert broker.ollama_host() == world.ollama_host()


async def test_leaked_ollama_memory_is_not_counted_as_freeable(cfg, world, http):
    world.ram_total = 64 * GB
    catalog, broker, _ = await make(cfg, world, http)
    before = broker.budget().slow
    orig = world.ollama_host
    world.ollama_host = lambda: orig() + 16 * GB  # held by Ollama with nothing loaded
    await broker.refresh()
    assert broker.budget().slow <= before - 15 * GB
    assert broker.warnings() and "restart" in broker.warnings()[0]

async def test_model_selection_assign_disable_persist(cfg, world, http):
    catalog, _, _ = await make(cfg, world, http)
    catalog.assign("medium", "qwen3.6")  # ':latest' optional
    catalog.set_enabled("qwen3:4b", False)
    await catalog.refresh()
    assert catalog.specs["medium"].base == "qwen3.6:latest"
    assert "small" not in catalog.specs  # only candidate disabled
    assert "qwen3:4b" not in catalog.enabled
    # Survives a restart.
    catalog2, _, _ = await make(cfg, world, http)
    assert catalog2.specs["medium"].base == "qwen3.6:latest"
    catalog2.assign("medium", None)
    catalog2.set_enabled("qwen3:4b", True)
    await catalog2.refresh()
    assert catalog2.specs["medium"].base == "qwen3:14b" and catalog2.specs["small"].base == "qwen3:4b"
    with pytest.raises(ValueError):
        catalog2.assign("medium", "nope:1b")


async def test_route_direct_model_and_model_pin(cfg, world, http):
    cfg.router = RouterConfig(classifier="heuristic")
    catalog, _, router = await make(cfg, world, http)
    d = await router.route({**user("hi"), "model": "yuzuru/qwen3:14b"}, {})
    assert d.tier == "qwen3:14b" and d.spec.tier == "direct" and d.spec.num_ctx == cfg.tiers[-1].num_ctx
    router.pin("s", "qwen3:4b")
    d = await router.route(user("設計して"), {"x-yuzuru-session": "s"})
    assert d.spec.base == "qwen3:4b" and "pin" in d.reason
    with pytest.raises(LookupError):
        await router.route({**user("hi"), "model": "missing:7b"}, {})
    with pytest.raises(ValueError):
        router.pin("s", "missing:7b")


async def test_soft_reservation_shrinks_budget_and_expires(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    before = broker.budget().fast
    lease = await broker.create_reservation(gpus=["1"], vram=8 * GB, ttl=None, label="mine", hard=False)
    assert lease.kind == "reservation" and lease.state == "reserved"
    assert before - broker.budget().fast >= 8 * GB - 1
    assert broker._settled.is_set()  # a reservation never blocks routing
    lease.expires = 0  # expire now
    await broker.refresh()
    assert lease.id not in broker.leases


async def test_whole_gpu_reservation_and_handover(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    await broker.acquire(catalog.specs["medium"])
    await broker.release(catalog.specs["medium"], touch=False)
    lease = await broker.create_reservation(gpus=["1"], vram=None, ttl=3600, label="gpu1", hard=False)
    assert catalog.specs["medium"].variant in world.unloads  # forced out
    assert 11 * GB < lease.vram <= 12 * GB
    assert broker.plan(catalog.specs["medium"], True)[0] is not Placement.GPU
    job = await broker.handover(lease.id, pid=os.getpid())
    assert job.kind == "job" and job.state == "settling" and job.vram == lease.vram
    with pytest.raises(KeyError):
        await broker.handover(lease.id, pid=os.getpid())


async def test_reservation_validation(cfg, world, http):
    _, broker, _ = await make(cfg, world, http)
    with pytest.raises(ValueError):
        await broker.create_reservation(gpus=[], vram=4 * GB, ttl=None, label="x", hard=True)  # 2 GPUs: which?
    with pytest.raises(ValueError):
        await broker.create_reservation(gpus=["7"], vram=None, ttl=None, label="x", hard=False)


async def test_handover_from_hard_reservation_keeps_lease(cfg, world, http, monkeypatch):
    _, broker, _ = await make(cfg, world, http)
    holder_pid = 2**22 + 777  # not alive, like a holder that just exited

    async def fake_spawn(gpus, sizes, ttl):
        return os.getpid()

    async def fake_stop(pid):
        world.foreign[1] = 0

    monkeypatch.setattr(broker, "_spawn_holder", fake_spawn)
    monkeypatch.setattr(broker, "_stop_holder", fake_stop)
    lease = await broker.create_reservation(gpus=["1"], vram=6 * GB, ttl=None, label="h", hard=True)
    lease.pid = holder_pid  # the holder is gone by the time the job takes over
    job = await broker.handover(lease.id, pid=os.getpid())
    assert job.id in broker.leases and job.state == "settling"


def enable_cloud(cfg, monkeypatch, **kw):
    from yuzuru.config import CloudConfig

    monkeypatch.setenv("TEST_CLOUD_KEY", "sk-test")
    cfg.cloud = CloudConfig(enabled=True, base_url="http://cloud/v1", api_key_env="TEST_CLOUD_KEY",
                            model="frontier-1", **kw)
    cfg.router = RouterConfig(classifier="heuristic", error_escalation=2)


def failing_turn(text="テストを直して", n=2):
    body = user(text)
    for _ in range(n):
        body["messages"] += [{"role": "assistant", "content": ""},
                             {"role": "tool", "content": "Traceback (most recent call last):\n..."}]
    return body


async def test_cloud_after_escalation_past_largest_tier(cfg, world, http, monkeypatch):
    enable_cloud(cfg, monkeypatch)
    _, _, router = await make(cfg, world, http)
    router.pin("s", "large")
    await router.route(user("テストを直して"), {"x-yuzuru-session": "s"})
    router.sessions["s"].pinned = None
    d = await router.route(failing_turn(), {"x-yuzuru-session": "s"})
    assert d.cloud and d.model == "frontier-1" and "tool errors" in d.reason
    # The rest of the turn stays on the cloud.
    body = failing_turn()
    body["messages"] += [{"role": "assistant", "content": ""}, {"role": "tool", "content": "ok"}]
    assert (await router.route(body, {"x-yuzuru-session": "s"})).cloud
    # A new turn is local again.
    assert not (await router.route(user("ありがとう"), {"x-yuzuru-session": "s"})).cloud


async def test_cloud_auto_off_and_missing_key(cfg, world, http, monkeypatch):
    enable_cloud(cfg, monkeypatch, auto="off")
    _, _, router = await make(cfg, world, http)
    router.pin("s", "large")
    await router.route(user("x"), {"x-yuzuru-session": "s"})
    router.sessions["s"].pinned = None
    d = await router.route(failing_turn("x"), {"x-yuzuru-session": "s"})
    assert not d.cloud
    assert (await router.route(user("これは @cloud で"), {})).cloud  # explicit still works
    monkeypatch.delenv("TEST_CLOUD_KEY")
    with pytest.raises(RuntimeError):
        await router.route({**user("x"), "model": "yuzuru/cloud"}, {})
    assert not (await router.route(user("これは @cloud で"), {})).cloud


async def test_cloud_auto_hard_rate_limit_context_and_no_memory(cfg, world, http, monkeypatch):
    enable_cloud(cfg, monkeypatch, auto="hard", max_requests_per_hour=1, when_no_memory=True)
    catalog, broker, router = await make(cfg, world, http)
    assert (await router.route(user("アーキテクチャを設計して"), {})).cloud
    d = await router.route(user("アーキテクチャを設計して"), {})  # over the hourly limit
    assert not d.cloud and "limit" in d.reason
    router.cloud_calls.clear()
    cfg.cloud.max_requests_per_hour = 100
    cfg.cloud.auto = "escalation"
    d = await router.route(user("x" * 200000), {})  # beyond every local context
    assert d.cloud and "context" in d.reason
    world.foreign = [8 * GB, 12 * GB]  # training holds every GPU
    await broker.refresh()
    catalog.specs["large"].moe = False
    d = await router.route(user("この関数にテストを書いて"), {})
    assert d.cloud and "cannot load" in d.reason


async def test_server_forwards_cloud_with_key(cfg, world, http, monkeypatch):
    enable_cloud(cfg, monkeypatch)
    seen = []

    def cloud_handler(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "deep answer"}}]})

    st = State(cfg, probe=FakeProbe(world), http=http)
    st.cloud_http = httpx.AsyncClient(transport=httpx.MockTransport(cloud_handler))
    app = create_app(cfg, st)
    await st.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            assert "cloud" in [m["id"] for m in (await c.get("/v1/models")).json()["data"]]
            r = await c.post("/v1/chat/completions", json={**user("hi"), "model": "yuzuru/cloud"})
            assert r.status_code == 200 and r.headers["x-yuzuru-tier"] == "cloud"
            assert str(seen[0].url) == "http://cloud/v1/chat/completions"
            assert seen[0].headers["authorization"] == "Bearer sk-test"
            assert json.loads(seen[0].content)["model"] == "frontier-1"
            assert not world.chat_requests  # nothing went to Ollama
    finally:
        await st.stop()


async def wait_for(cond, timeout=5.0):
    import asyncio
    import time as _t

    end = _t.time() + timeout
    while not cond():
        if _t.time() > end:
            raise AssertionError("condition not met")
        await asyncio.sleep(0.05)


def resume_capture(cfg, tmp_path):
    import sys

    out = tmp_path / "resumed.json"
    cfg.resume.command = [sys.executable, "-c",
                          f"import sys, json; open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]))",
                          "{session}", "{message}"]
    return out


async def test_job_lease_suspends_llm_and_resumes_session(cfg, world, http, tmp_path):
    from yuzuru.waits import Waiter

    out = resume_capture(cfg, tmp_path)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await broker.acquire(catalog.router_spec)  # CPU classifier resident
    await broker.release(catalog.router_spec, touch=False)
    log = tmp_path / "train.log"
    log.write_text("epoch 1 loss 2.0\nepoch 2 loss 1.1\nsaved checkpoint\n")
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None, session="ses_9",
                                      cwd=str(tmp_path), log=str(log), resume=True, note="評価を回す")
    assert waiter.watch_lease(lease) is not None
    router.sessions["ses_9"] = SessionState()  # the session that just started the job
    await waiter.poll()
    assert world.loaded  # the session just spoke: give it time to wrap up its turn
    cfg.resume.suspend_idle_seconds = 0
    await waiter.poll()
    assert not world.loaded  # everything unloaded, CPU model too
    await broker.release_lease(lease.id, rc=0)
    await wait_for(out.exists)
    session, message = json.loads(out.read_text())
    assert session == "ses_9"
    assert "finished" in message and "saved checkpoint" in message and "評価を回す" in message
    ev = list(waiter.events)[-1]
    assert ev.rc == 0 and ev.resumed and not waiter.watches


async def test_pid_watch_and_long_poll(cfg, world, http, tmp_path):
    import asyncio
    import subprocess
    import sys

    from yuzuru.waits import Waiter

    cfg.resume.enabled = False
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    await broker.acquire(catalog.specs["small"])
    await broker.release(catalog.specs["small"], touch=False)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    cfg.resume.suspend_idle_seconds = 0
    w = await waiter.add(session="s", pid=proc.pid, label="prep", note="")
    await waiter.poll()
    assert not world.loaded  # suspended
    poll_task = asyncio.create_task(waiter.wait_events(0, 5))
    proc.wait()
    await waiter.poll()
    evs = await poll_task
    assert [e.watch for e in evs] == [w.id] and evs[0].rc is None
    # Nothing to resume -> what was resident comes back.
    await wait_for(lambda: catalog.specs["small"].variant in world.loaded)
    with pytest.raises(ValueError):
        await waiter.add(session="s", pid=proc.pid)  # already gone


def test_default_tiers_fit_an_agent_prompt():
    from yuzuru.config import default_tiers

    assert all(t.num_ctx >= 32768 for t in default_tiers())


async def test_classifier_survives_truncated_json(cfg, world, http, monkeypatch):
    catalog, broker, router = await make(cfg, world, http)

    async def truncated(payload, timeout=None):
        return {"message": {"content": '{"difficulty": "hard", "reason": "The user asks to redes'}}

    monkeypatch.setattr(router.ollama, "chat", truncated)
    level, reason = await router.classify("このモジュールを再設計して", 2)
    assert level == 2 and reason.startswith("classifier:hard")


async def test_cpu_classifier_is_evicted_last(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    for spec in (catalog.router_spec, catalog.specs["small"]):
        await broker.acquire(spec)
        await broker.release(spec, touch=False)
    broker.last_used[catalog.router_spec.variant] = 0  # least recently used
    order = [m.name for m in broker._evictable(keep="")]
    assert order[-1] == catalog.router_spec.variant
    world.foreign = [0, 12 * GB - 1]  # only GPU0 left: medium must evict small, not the classifier
    await broker.refresh()
    _, evict = broker.plan(catalog.specs["medium"], True)
    assert catalog.router_spec.variant not in [m.name for m in evict]


def test_opencode_install_v2_codemode_off_and_local_only(cfg, tmp_path, monkeypatch):
    from yuzuru import opencode

    monkeypatch.setattr(opencode, "opencode_major", lambda: 2)
    monkeypatch.setattr(opencode, "installed_models", lambda cfg: ["qwen3:14b"])
    (tmp_path / "opencode.json").write_text(json.dumps({"model": "other/x", "disabled_providers": ["foo"]}))
    opencode.install(cfg, tmp_path, set_default=False, local_only=True)
    conf = json.loads((tmp_path / "opencode.json").read_text())
    assert conf["mcp"]["servers"]["yuzuru"]["codemode"] is False
    assert conf["model"] == "other/x"  # kept without --set-default
    assert conf["disabled_providers"] == ["foo", "opencode"]
    assert "qwen3:14b" in conf["provider"]["yuzuru"]["models"]
    assert not (tmp_path / ".opencode" / "plugins" / "yuzuru.ts").exists()  # 2.x cannot load it


def test_nvidia_probe_survives_a_dead_gpu(monkeypatch):
    import subprocess

    from yuzuru.resources import NvidiaSmiProbe

    dead = "Unable to determine the device handle for GPU0000:04:00.0: Unknown Error"

    def fake_run(argv, **kw):
        if "-L" in argv:
            return subprocess.CompletedProcess(argv, 255, f"GPU 1: RTX 3060 (UUID: GPU-403d)\n", dead)
        if "-i" in argv:
            return subprocess.CompletedProcess(argv, 0, "1, GPU-403d, RTX 3060, 12288, 205\n", "")
        return subprocess.CompletedProcess(argv, 255, "", dead)

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = NvidiaSmiProbe(ollama_cgroup="none")
    snap = probe.snapshot()
    assert [g.uuid for g in snap.gpus] == ["GPU-403d"]
    assert snap.lost_gpus and "0000:04:00.0" in snap.lost_gpus[0]


async def test_web_pages_cannot_drive_the_api(cfg, world, http):
    st = State(cfg, probe=FakeProbe(world), http=http)
    app = create_app(cfg, st)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
        assert (await c.post("/mcp", json=call)).status_code == 200
        # A page on another site: the browser adds Origin.
        r = await c.post("/mcp", json=call, headers={"origin": "https://evil.example"})
        assert r.status_code == 403
        # A "simple" cross-site request needs no preflight; refuse non-JSON bodies.
        r = await c.post("/mcp", content=json.dumps(call), headers={"content-type": "text/plain"})
        assert r.status_code == 415
    # DNS rebinding: the attacker's name resolves to 127.0.0.1, Host stays theirs.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://evil.example:11435") as c:
        assert (await c.post("/mcp", json=call)).status_code == 403
    cfg.allowed_origins = ["http://localhost:3000"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:11435") as c:
        r = await c.post("/mcp", json=call, headers={"origin": "http://localhost:3000"})
        assert r.status_code == 200


async def test_lease_waits_until_a_busy_model_really_leaves(cfg, world, http):
    # The drain timed out while the agent's last reply was still generating:
    # Ollama only unloads once that generation ends.
    cfg.lease.unload_timeout = 10
    catalog, broker, _ = await make(cfg, world, http)
    await broker.acquire(catalog.specs["medium"])
    await broker.release(catalog.specs["medium"], touch=False)
    variant = catalog.specs["medium"].variant
    world.busy.add(variant)
    asyncio.get_running_loop().call_later(1.5, world.finish, variant)
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None)
    assert variant not in world.loaded
    assert lease.resident == []
    assert lease.baseline == 0  # taken after the model left


async def test_lease_reports_a_model_that_never_left(cfg, world, http):
    catalog, broker, _ = await make(cfg, world, http)
    await broker.acquire(catalog.specs["medium"])
    await broker.release(catalog.specs["medium"], touch=False)
    variant = catalog.specs["medium"].variant
    world.busy.add(variant)
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram=None)
    assert lease.resident == [variant]
