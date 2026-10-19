import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import socket

import pytest

from .conftest import AllowlistProbe
from yuzuru.config import Config, OllamaConfig, ResourceConfig, load_config
from yuzuru.managed import ManagedOllama, port_free, running_url
from yuzuru.server import State


@contextlib.contextmanager
def occupied(port):
    """Something else listening on `port` (another Ollama, a hand-started server...)."""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen()
    try:
        yield
    finally:
        s.close()


def managed(cfg, binary, tmp_path, port="auto", port_range=None, state=None):
    cfg = dataclasses.replace(cfg, state_dir=state or cfg.state_dir)
    cfg.ollama = OllamaConfig(manage=True, binary=binary, port=port, port_range=port_range or [11436, 11499],
                              models=tmp_path / "models", mirror_from="none", scope="off", startup_timeout=10)
    urls = []
    return ManagedOllama(cfg, lambda: [], urls.append), urls


async def wait_for(cond, seconds=8.0):
    for _ in range(int(seconds / 0.1)):
        if cond():
            return True
        await asyncio.sleep(0.1)
    return False


def test_port_config(tmp_path):
    assert OllamaConfig().port == "auto"
    assert OllamaConfig(port="11500").port == 11500
    with pytest.raises(ValueError):
        OllamaConfig(port="any")
    with pytest.raises(ValueError):
        OllamaConfig(port_range=[11500, 11400])
    path = tmp_path / "y.yaml"
    path.write_text("ollama:\n  manage: true\n  port_range: [12000, 12010]\n")
    assert load_config(path).ollama_url == "http://127.0.0.1:12000"  # a guess until started


def test_auto_never_takes_yuzuru_or_system_ollama_port(cfg):
    m, _ = managed(Config(port=11436), "ollama", cfg.state_dir, port_range=[11433, 11438])
    assert m.candidate_ports() == [11433, 11435, 11437, 11438]  # not 11434 (Ollama) or 11436 (yuzuru)
    m.port = 11437
    assert m.candidate_ports()[0] == 11437  # sticky across restarts


async def test_auto_skips_busy_port_and_reports_url(cfg, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    m, urls = managed(cfg, binary, tmp_path, port_range=[free_port, free_port + 20])
    with occupied(free_port):
        await m.start()
        try:
            assert m.pid is not None and m.port != free_port
            assert free_port < m.port <= free_port + 20
            url = f"http://127.0.0.1:{m.port}"
            assert urls == [url] and m.cfg.ollama_url == url and m.status()["url"] == url
            assert json.loads(record.read_text().splitlines()[-1])["OLLAMA_HOST"] == f"127.0.0.1:{m.port}"
            assert running_url(m.cfg) == url  # what `yuzuru opencode` reads
        finally:
            await m.stop()
    assert running_url(m.cfg) is None


async def test_two_managed_servers_get_different_ports(cfg, fake_ollama, free_port, tmp_path):
    binary, _ = fake_ollama
    a, _ = managed(cfg, binary, tmp_path, port_range=[free_port, free_port + 20], state=tmp_path / "a")
    b, _ = managed(cfg, binary, tmp_path, port_range=[free_port, free_port + 20], state=tmp_path / "b")
    await a.start()
    await b.start()
    try:
        assert a.pid and b.pid and a.port != b.port
    finally:
        await a.stop()
        await b.stop()


async def test_fixed_busy_port_waits_instead_of_crash_looping(cfg, fake_ollama, free_port, tmp_path):
    binary, record = fake_ollama
    m, urls = managed(cfg, binary, tmp_path, port=free_port)
    try:
        with occupied(free_port):
            await m.start()
            assert m.pid is None and not record.exists()  # nothing spawned onto a taken port
        assert await wait_for(lambda: m.pid is not None)  # starts once the port is free
        assert m.port == free_port and urls == [f"http://127.0.0.1:{free_port}"]
    finally:
        await m.stop()


async def test_restart_moves_to_another_port_if_its_own_was_taken(cfg, fake_ollama, free_port, tmp_path):
    binary, _ = fake_ollama
    m, urls = managed(cfg, binary, tmp_path, port_range=[free_port, free_port + 20])
    await m.start()
    try:
        first = m.port
        os.kill(m.pid, signal.SIGKILL)
        await m.proc.wait()
        with occupied(first):
            assert await wait_for(lambda: m.pid is not None and m.port != first)
        assert urls == [f"http://127.0.0.1:{first}", f"http://127.0.0.1:{m.port}"]
    finally:
        await m.stop()


async def test_state_follows_the_picked_port(cfg, world, http, fake_ollama, free_port, tmp_path):
    binary, _ = fake_ollama
    cfg.resources = ResourceConfig(gpus=["0"])
    cfg.ollama = OllamaConfig(manage=True, binary=binary, port_range=[free_port, free_port + 20],
                              models=tmp_path / "models", mirror_from="none", scope="off", startup_timeout=10)
    st = State(cfg, probe=AllowlistProbe(world, ["0"]), http=http)
    with occupied(free_port):
        await st.start()
    try:
        url = st.managed.url
        assert url != f"http://127.0.0.1:{free_port}"
        assert str(st.http.base_url).rstrip("/") == url and st.ollama.base_url == url and cfg.ollama_url == url
    finally:
        await st.stop()


async def test_scope_detected_up_front(cfg, monkeypatch, tmp_path):
    import yuzuru.managed as managed_mod

    m, _ = managed(cfg, "ollama", tmp_path)
    m.oc.scope = "auto"
    monkeypatch.setattr(managed_mod.shutil, "which", lambda name: None)
    assert await m._use_scope() is False  # decided before launching, not from an early exit
    assert port_free("127.0.0.1", 0)
