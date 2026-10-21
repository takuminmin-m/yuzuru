import asyncio
import json
import os
import sys

import httpx

from .conftest import FakeProbe
from yuzuru.server import State, create_app
from yuzuru.waits import Event, Waiter, Watch


async def test_server_starts_while_ollama_is_down_and_recovers(cfg, world):
    up = False

    def handler(request):
        if not up:
            raise httpx.ConnectError("connection refused")
        return world.handler(request)

    http = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    st = State(cfg, probe=FakeProbe(world), http=http)
    await st.start()  # must not raise
    try:
        assert st.catalog.specs == {}
        app = create_app(cfg, st)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:11435") as c:
            r = await c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 503
        up = True
        await st.catalog.refresh()
        assert "small" in st.catalog.specs
    finally:
        await st.stop()


async def test_resumed_process_is_reaped(cfg, world, http, tmp_path):
    from yuzuru.broker import Broker
    from yuzuru.catalog import Catalog
    from yuzuru.ollama import Ollama
    from yuzuru.router import Router

    cfg.resume.warm_before_resume = False
    cfg.resume.command = [sys.executable, "-c", "pass", "{session}", "{message}"]
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    waiter = Waiter(cfg, broker, Router(cfg, catalog, broker, ollama))
    ev = Event(1, "w", "ses", "job", 0, 1.0, None, "", "")
    await waiter.resume(ev, Watch(id="w", session="ses"))
    assert ev.resumed
    for _ in range(100):
        if not waiter._tasks:
            break
        await asyncio.sleep(0.05)
    assert not waiter._tasks  # proc.wait() finished: no zombie left behind


def fake_systemd_run(tmp_path, monkeypatch):
    """A `systemd-run` on PATH that records its arguments and runs the command after `--`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "systemd-run.json"
    script = bin_dir / "systemd-run"
    script.write_text(f"#!{sys.executable}\n"
                      "import json, os, sys\n"
                      f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                      "cmd = sys.argv[sys.argv.index('--') + 1:]\n"
                      "os.execvp(cmd[0], cmd)\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return argv_file


async def test_resumed_agent_runs_in_its_own_capped_scope(cfg, world, http, tmp_path, monkeypatch):
    from yuzuru.broker import Broker
    from yuzuru.catalog import Catalog
    from yuzuru.ollama import Ollama
    from yuzuru.router import Router

    argv_file = fake_systemd_run(tmp_path, monkeypatch)
    out = tmp_path / "resumed"
    cfg.resume.warm_before_resume = False
    cfg.resume.scope, cfg.resume.memory_max = "on", "24G"
    cfg.resume.command = [sys.executable, "-c", f"open({str(out)!r}, 'w').write('ok')", "{session}"]
    ollama = Ollama("http://ollama", http)
    catalog = Catalog(cfg, ollama)
    broker = Broker(cfg, FakeProbe(world), ollama, catalog)
    waiter = Waiter(cfg, broker, Router(cfg, catalog, broker, ollama))
    await waiter.resume(Event(1, "w", "ses", "job", 0, 1.0, None, "", ""), Watch(id="w", session="ses"))
    for _ in range(100):
        if out.exists():
            break
        await asyncio.sleep(0.05)
    assert out.read_text() == "ok"  # the agent itself still ran
    args = json.loads(argv_file.read_text())
    assert args[:2] == ["--user", "--scope"] and any(a.startswith("--unit=yuzuru-agent-") for a in args)
    # Its own cap, and a script it runs being OOM-killed does not stop the agent's scope.
    assert "MemoryMax=24G" in args and "OOMPolicy=continue" in args


def test_agent_command_execs_into_a_scope(tmp_path, monkeypatch):
    from yuzuru import cli, scope

    calls = []
    monkeypatch.setattr(cli.os, "execvpe", lambda f, argv, env: calls.append((argv, env)))
    monkeypatch.setattr(scope, "available", lambda: True)
    monkeypatch.setattr(scope, "opencode_service_outside", lambda: [])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))  # no user config: the defaults
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)  # as under `uv run yuzuru agent`
    cli.main(["agent", "--", "opencode", "run", "go"])
    argv, env = calls[0]
    assert argv[0] == "systemd-run" and "MemoryMax=80%" in argv and "OOMPolicy=continue" in argv
    assert argv[argv.index("--") + 1:] == ["opencode", "run", "go"]
    # The agent (and the OpenCode service it may start) must not run in yuzuru's venv.
    if sys.prefix != sys.base_prefix:
        assert "VIRTUAL_ENV" not in env


def test_opencode_service_outside_an_agent_scope_is_found(tmp_path):
    from yuzuru import scope

    def proc(pid, argv, cgroup):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
        (d / "cgroup").write_text(f"0::{cgroup}\n")

    app = "/user.slice/user-1000.slice/user@1000.service/app.slice"
    proc(10, [b"/usr/bin/opencode", b"serve", b"--service"], f"{app}/app-com.anthropic.Claude-1.scope")
    proc(11, [b"/usr/bin/opencode", b"serve", b"--service"], f"{app}/yuzuru-agent-5.scope")
    proc(12, [b"/usr/bin/opencode", b"run", b"--session", b"ses"], f"{app}/app-com.anthropic.Claude-1.scope")
    (tmp_path / "self").mkdir()
    outside = scope.opencode_service_outside(tmp_path)
    assert outside == [(10, f"{app}/app-com.anthropic.Claude-1.scope")]
    assert "app-com.anthropic.Claude-1.scope" in scope.service_warning(outside)
