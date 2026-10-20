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
