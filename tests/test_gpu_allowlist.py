import argparse
import asyncio
import dataclasses
import json
import os
import signal
import sys
import time

import httpx
import pytest

from .conftest import AllowlistProbe, FakeProbe
from yuzuru import cli
from yuzuru.broker import Broker
from yuzuru.catalog import Catalog
from yuzuru.config import GB, Config, OllamaConfig, ResourceConfig, load_config
from yuzuru.managed import ManagedOllama, mirror_models
from yuzuru.ollama import Ollama
from yuzuru.resources import NvidiaSmiProbe, OllamaHostMemory
from yuzuru.server import State, create_app


def make_broker(cfg, world, http, allow):
    cfg = dataclasses.replace(cfg, resources=dataclasses.replace(cfg.resources, gpus=allow))
    ollama = Ollama("http://ollama", http)
    probe = AllowlistProbe(world, allow) if allow else FakeProbe(world)
    return Broker(cfg, probe, ollama, Catalog(cfg, ollama))


# ------------------------------------------------------------------ config / probe


def test_config_reads_allowlist_and_managed_ollama(tmp_path, monkeypatch):
    path = tmp_path / "y.yaml"
    path.write_text("resources:\n  gpus: [0, GPU-abc]\nollama:\n  manage: true\n  port: 11500\n"
                    "  env: {OLLAMA_FLASH_ATTENTION: 1}\n")
    monkeypatch.setenv("OLLAMA_HOST_URL", "http://elsewhere:11434")
    cfg = load_config(path)
    assert cfg.resources.gpus == ["0", "GPU-abc"]
    assert cfg.ollama_url == "http://127.0.0.1:11500"  # the managed server, not OLLAMA_HOST_URL
    assert cfg.ollama.env == {"OLLAMA_FLASH_ATTENTION": "1"}
    assert Config().ollama_url == "http://127.0.0.1:11434"  # unmanaged by default


def test_nvidia_probe_reports_only_allowed_gpus(fake_nvidia_smi):
    assert [g.uuid for g in NvidiaSmiProbe("none", fake_nvidia_smi).snapshot().gpus] == ["GPU-aaa", "GPU-bbb"]
    by_index = NvidiaSmiProbe("none", fake_nvidia_smi, allow=["0"]).snapshot()
    assert [(g.index, g.uuid) for g in by_index.gpus] == [(0, "GPU-aaa")]
    by_uuid = NvidiaSmiProbe("none", fake_nvidia_smi, allow=["gpu-BBB"]).snapshot()
    assert [g.index for g in by_uuid.gpus] == [1]
    assert NvidiaSmiProbe("none", fake_nvidia_smi, allow=["7"]).snapshot().gpus == []


# ------------------------------------------------------------------ broker / leases


async def test_budget_and_leases_ignore_excluded_gpu(cfg, world, http):
    world.foreign = [0, 11 * GB]  # the excluded 12 GB GPU is nearly full
    full = make_broker(cfg, world, http, [])
    await full.refresh()
    only0 = make_broker(cfg, world, http, ["0"])
    await only0.refresh()
    assert [g.index for g in only0.snapshot.gpus] == [0]
    assert only0.budget().fast < full.budget().fast
    assert only0.budget().fast <= 8 * GB

    with pytest.raises(ValueError, match="unknown GPU"):
        await only0.create_lease(pid=os.getpid(), label="job", gpus=["1"], vram=None)
    lease = await only0.create_lease(pid=os.getpid(), label="job", gpus=[], vram=None)
    assert only0.visible_gpus(lease) == ["GPU-0"]
    lease_full = await full.create_lease(pid=os.getpid(), label="job", gpus=[], vram=None)
    assert full.visible_gpus(lease_full) == []  # no allowlist: jobs are not restricted

    # `yuzuru reserve` without --gpu takes the only allowed GPU.
    res = await only0.create_reservation(gpus=[], vram=None, ttl=None, label="r", hard=False)
    assert res.gpus == ["GPU-0"]


async def test_lease_endpoint_tells_job_which_gpus_to_see(cfg, world, http):
    cfg.resources.gpus = ["GPU-1"]
    st = State(cfg, probe=AllowlistProbe(world, ["GPU-1"]), http=http)
    app = create_app(cfg, st)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
        r = await c.post("/yuzuru/leases", json={"pid": os.getpid(), "label": "train"})
        assert r.status_code == 200
        assert r.json()["gpus"] == [] and r.json()["visible"] == ["GPU-1"]
        status = (await c.get("/yuzuru/status")).json()
        assert [g["uuid"] for g in status["gpus"]] == ["GPU-1"]
        assert status["gpu_allowlist"] == ["GPU-1"]


# ------------------------------------------------------------------ yuzuru run


def run_args(tmp_path, out, config=None, cpu=False):
    code = f"import os; open({str(out)!r}, 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))"
    return argparse.Namespace(
        cmd=["--", sys.executable, "-c", code], detach=False, log=None, session=None, resume=False, note="",
        use=None, label=None, gpu=None, vram=None, config=config, cpu=cpu,
    )


def call_run(args, client):
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with pytest.raises(SystemExit) as e:
            cli.cmd_run(args, client)
        assert e.value.code == 0
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


class FakeClient:
    def __init__(self, lease=None):
        self.lease = lease
        self.calls = []

    def call(self, method, path, payload=None, timeout=30):
        self.calls.append((method, path))
        if self.lease is None:
            raise httpx.ConnectError("refused")
        return self.lease if method == "POST" else {}


def test_run_limits_job_to_allowed_gpus_without_gpu_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")  # must not leak through
    out = tmp_path / "seen"
    call_run(run_args(tmp_path, out), FakeClient({"id": "abc", "gpus": [], "visible": ["GPU-0"]}))
    assert out.read_text() == "GPU-0"


async def test_cpu_lease_endpoint_hides_every_gpu(cfg, world, http):
    cfg.resources.gpus = ["GPU-1"]
    st = State(cfg, probe=AllowlistProbe(world, ["GPU-1"]), http=http)
    app = create_app(cfg, st)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
        r = await c.post("/yuzuru/leases", json={"pid": os.getpid(), "label": "lgbm", "cpu": True})
        assert r.status_code == 200 and r.json()["visible"] == ["-1"] and r.json()["state"] == "active"
        assert (await c.get("/yuzuru/status")).json()["leases"][0]["cpu"]
        r = await c.post("/yuzuru/leases", json={"pid": os.getpid(), "cpu": True, "gpus": ["GPU-1"]})
        assert r.status_code == 400


def test_run_cpu_hides_gpus_with_or_without_server(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    out = tmp_path / "seen"
    client = FakeClient({"id": "abc", "gpus": [], "visible": ["-1"]})
    call_run(run_args(tmp_path, out, cpu=True), client)
    assert out.read_text() == "-1"
    conf = tmp_path / "y.yaml"
    conf.write_text("resources:\n  gpus: [0]\n")
    call_run(run_args(tmp_path, out, str(conf), cpu=True), FakeClient(None))
    assert out.read_text() == "-1"  # not the allowlist: the job is CPU only


def test_run_without_server_still_applies_allowlist(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    conf = tmp_path / "y.yaml"
    conf.write_text("resources:\n  gpus: [0]\n")
    out = tmp_path / "seen"
    call_run(run_args(tmp_path, out, str(conf)), FakeClient(None))
    assert out.read_text() == "0"


# ------------------------------------------------------------------ managed Ollama


def test_mirror_symlinks_blobs_and_copies_manifests(tmp_path):
    src, dst = tmp_path / "system", tmp_path / "mine"
    (src / "blobs").mkdir(parents=True)
    (src / "blobs" / "sha256-aa").write_text("weights")
    (src / "blobs" / "sha256-bb-partial").write_text("downloading")
    man = src / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "4b"
    man.parent.mkdir(parents=True)
    man.write_text("{}")
    v2 = src / "manifests-v2" / "ollama.com" / "library" / "qwen3" / "4b"
    v2.parent.mkdir(parents=True)
    v2.symlink_to("../../../../blobs/sha256-aa")
    (src / "metadata").mkdir()
    (src / "metadata" / "sha256-aa.json").write_text("{}")

    assert mirror_models(src, dst) == 4
    blob = dst / "blobs" / "sha256-aa"
    assert blob.is_symlink() and blob.resolve() == (src / "blobs" / "sha256-aa").resolve()
    assert not (dst / "blobs" / "sha256-bb-partial").exists()
    mine = dst / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "4b"
    assert mine.read_text() == "{}" and not mine.is_symlink()  # a copy: dst stays writable
    assert (dst / "manifests-v2" / "ollama.com" / "library" / "qwen3" / "4b").read_text() == "weights"
    assert mirror_models(src, dst) == 0

    os.utime(man, (time.time() + 10, time.time() + 10))
    man.write_text('{"new": 1}')
    os.utime(man, (time.time() + 10, time.time() + 10))
    assert mirror_models(src, dst) == 1
    assert mine.read_text() == '{"new": 1}'


def managed_cfg(cfg, binary, port, models):
    cfg.resources = ResourceConfig(gpus=["0"])
    cfg.ollama = OllamaConfig(manage=True, binary=binary, port=port, models=models, mirror_from="none",
                              scope="off", env={"OLLAMA_FLASH_ATTENTION": "1"}, startup_timeout=10)
    return cfg


def starts(record):
    return [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []


async def test_managed_ollama_sees_only_allowed_gpus_and_is_restarted(cfg, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    managed_cfg(cfg, binary, free_port, tmp_path / "models")
    m = ManagedOllama(cfg, lambda: ["GPU-a0d7"])
    await m.start()
    try:
        first = starts(record)[-1]
        assert first == {"pid": m.pid, "CUDA_VISIBLE_DEVICES": "GPU-a0d7", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                         "OLLAMA_MODELS": str(tmp_path / "models"), "OLLAMA_HOST": f"127.0.0.1:{free_port}",
                         "OLLAMA_FLASH_ATTENTION": "1"}
        os.kill(first["pid"], signal.SIGKILL)
        for _ in range(100):
            if m.pid and m.pid != first["pid"] and len(starts(record)) == 2:
                break
            await asyncio.sleep(0.1)
        assert m.restarts == 1 and starts(record)[-1]["pid"] == m.pid
    finally:
        pid = m.pid
        await m.stop()
    assert pid is not None and not os.path.exists(f"/proc/{pid}")
    assert not m.pidfile.exists()


async def test_managed_ollama_replaces_stale_instance(cfg, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    managed_cfg(cfg, binary, free_port, tmp_path / "models")
    old = ManagedOllama(cfg, lambda: ["GPU-x"])
    await old.start()
    stale = old.pid
    old._task.cancel()  # yuzuru "killed": the server is left behind, pidfile and all
    m = ManagedOllama(cfg, lambda: ["GPU-x"])
    await m.start()
    try:
        assert m.pid != stale and m.pid is not None
        await old.proc.wait()  # the stale one was stopped (it held the port)
    finally:
        await m.stop()


async def test_state_runs_managed_ollama_on_allowed_gpus(cfg, world, http, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    managed_cfg(cfg, binary, free_port, tmp_path / "models")
    cfg.resources.gpus = ["GPU-0"]
    st = State(cfg, probe=AllowlistProbe(world, ["GPU-0"]), http=http)
    await st.start()
    try:
        assert st.managed is not None and st.managed.pid is not None
        assert starts(record)[-1]["CUDA_VISIBLE_DEVICES"] == "GPU-0"
        app = create_app(cfg, st)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            o = (await c.get("/yuzuru/status")).json()["ollama"]
        assert o["managed"] and o["running"] and o["cuda_visible_devices"] == "GPU-0"
    finally:
        pid = st.managed.pid
        await st.stop()
    assert not os.path.exists(f"/proc/{pid}")


def test_host_memory_targets_managed_process_not_system_ollama():
    # Same cgroup as us (no scope): measure the process tree instead of the whole cgroup.
    host = OllamaHostMemory("auto", pid=os.getpid)
    assert host.path() is None
    assert host.read() > 0
    assert OllamaHostMemory("auto", pid=lambda: None).read() is None  # managed server not running
    assert OllamaHostMemory("none", pid=os.getpid).read() is None
