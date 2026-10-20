"""Linux 6.11 leaks pinned host RAM when a large CUDA pin fails (see yuzuru.pinleak)."""

import json

import pytest

from .conftest import FakeProbe
from .test_core import make
from .test_gpu_allowlist import managed_cfg, starts
from yuzuru import pinleak
from yuzuru.broker import Broker, LoadFailed, Placement
from yuzuru.config import GB
from yuzuru.managed import ManagedOllama
from yuzuru.pinleak import kernel_leaks_pins, read_unowned, record_path

MEMINFO = """MemTotal:       32786432 kB
MemAvailable:   13350912 kB
SwapCached:        10240 kB
Active(anon):   12000000 kB
Inactive(anon):  6000000 kB
AnonPages:       1000000 kB
Shmem:             90000 kB
"""


def test_unowned_is_anon_lru_that_no_process_maps(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text(MEMINFO)
    assert read_unowned(p) == (18000000 - 1000000 - 90000 - 10240) * 1024
    p.write_text(MEMINFO.replace("AnonPages:       1000000", "AnonPages:      19000000"))
    assert read_unowned(p) == 0  # never negative
    assert read_unowned(tmp_path / "missing") is None


def test_known_leaking_kernels():
    assert kernel_leaks_pins("6.11.0-29-generic")
    assert kernel_leaks_pins("6.11.11")
    assert not kernel_leaks_pins("6.12.0-1-generic")
    assert not kernel_leaks_pins("6.1.0")
    assert not kernel_leaks_pins("7.0.0-38-generic")


async def test_leaking_load_is_unloaded_and_not_offloaded_again(cfg, world, http):
    world.ram_total = 64 * GB
    catalog, broker, _ = await make(cfg, world, http)
    large = catalog.specs["large"]
    assert broker.plan(large, True)[0] is Placement.OFFLOAD
    world.leak_on_load[large.variant] = 16 * GB

    with pytest.raises(LoadFailed):
        await broker.acquire(large)
    assert not broker.is_loaded(large.variant) and world.unloads[-1] == large.variant
    assert broker.inflight[large.variant] == 0
    assert broker.plan(large, True)[0] is Placement.NONE
    assert any("held by no process" in w and "only a reboot" in w for w in broker.warnings())

    # Remembered across restarts of yuzuru, for this kernel only.
    data = json.loads(record_path(cfg.state_dir).read_text())
    assert data["variants"][large.variant] == 16.0
    assert Broker(cfg, FakeProbe(world), broker.ollama, catalog).leaks_when_offloaded(large.variant)
    data["kernel"] = "7.0.0-38-generic"  # a leak recorded under another kernel does not count
    record_path(cfg.state_dir).write_text(json.dumps(data))
    assert not Broker(cfg, FakeProbe(world), broker.ollama, catalog).leaks_when_offloaded(large.variant)


async def test_unowned_ram_is_not_blamed_on_ollama(cfg, world, http):
    _, broker, _ = await make(cfg, world, http)
    assert broker.warnings() == []
    world.unowned = 16 * GB
    orig = world.ollama_host
    world.ollama_host = lambda: orig() + 16 * GB  # the leaked pages are charged to Ollama's cgroup
    await broker.refresh()
    warnings = broker.warnings()
    assert len(warnings) == 1 and "held by no process" in warnings[0] and "only a reboot" in warnings[0]


async def test_leaking_kernel_is_warned_about(cfg, world, http, monkeypatch):
    monkeypatch.setattr(pinleak, "kernel_release", lambda: "6.11.0-29-generic")
    _, broker, _ = await make(cfg, world, http)
    assert any("Linux 6.11.0-29-generic" in w and "update the kernel" in w for w in broker.warnings())


async def test_managed_ollama_hides_vulkan_when_gpus_are_limited(cfg, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    managed_cfg(cfg, binary, free_port, tmp_path / "models")
    m = ManagedOllama(cfg, lambda: ["GPU-a0d7"])
    await m.start()
    try:
        env = starts(record)[-1]
        assert env["OLLAMA_VULKAN"] == "0"  # Vulkan would show the excluded GPU again
        assert env["GGML_CUDA_NO_PINNED"] is None
    finally:
        await m.stop()

    cfg.ollama.env["OLLAMA_VULKAN"] = "1"  # the user's choice wins
    m = ManagedOllama(cfg, lambda: ["GPU-a0d7"])
    await m.start()
    try:
        assert starts(record)[-1]["OLLAMA_VULKAN"] == "1"
    finally:
        await m.stop()
