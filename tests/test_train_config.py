import json
import subprocess

from yuzuru.config import load_config
from yuzuru.mcp import MCPServer
from yuzuru.server import State


async def test_train_jobs_get_the_servers_config_wherever_they_run(cfg, tmp_path, monkeypatch):
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "y.yaml").write_text("port: 11999\n")
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "srv")
    monkeypatch.setenv("YUZURU_CONFIG", "y.yaml")  # relative, as `YUZURU_CONFIG=y.yaml yuzuru serve`
    loaded = load_config()
    assert loaded.source == tmp_path / "srv" / "y.yaml"

    st = State(loaded)
    ran = []

    def run(argv, cwd, **kw):
        ran.append((argv, cwd))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"pid": 1, "log": "/l"}), "")

    monkeypatch.setattr(subprocess, "run", run)
    await MCPServer(st).tools["train"][1](
        {"command": "python train.py", "cwd": str(tmp_path / "project"), "resume_when_done": False}, {})
    argv, cwd = ran[0]
    assert cwd == tmp_path / "project"
    assert argv[argv.index("--config") + 1] == str(tmp_path / "srv" / "y.yaml")


def test_no_config_file_passes_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YUZURU_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert load_config().source is None


async def test_jobs_do_not_inherit_the_servers_venv(cfg, monkeypatch, tmp_path):
    import os
    import sys

    from yuzuru.waits import outside_env

    if sys.prefix == sys.base_prefix:
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    own = os.path.join(sys.prefix, "bin")
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setenv("PATH", os.pathsep.join([own, "/usr/bin"]))
    env = outside_env()
    assert "VIRTUAL_ENV" not in env and env["PATH"] == "/usr/bin"

    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "project-venv"))  # someone else's venv stays
    assert outside_env()["VIRTUAL_ENV"] == str(tmp_path / "project-venv")

    st = State(cfg)
    seen = []

    def run(argv, cwd, env=None, **kw):
        seen.append(env)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"pid": 1, "log": "/l"}), "")

    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.setattr(subprocess, "run", run)
    await MCPServer(st).tools["train"][1]({"command": "uv run python train.py", "cwd": str(tmp_path),
                                           "resume_when_done": False}, {})
    assert "VIRTUAL_ENV" not in seen[0] and own not in seen[0]["PATH"].split(os.pathsep)
