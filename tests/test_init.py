import json
import shlex

import httpx
import pytest

from yuzuru import scaffold
from yuzuru.cli import main
from yuzuru.config import load_config

UUID = "GPU-403df14d-2132-46af-4c18-052a4bdb3642"


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("YUZURU_CONFIG", raising=False)
    monkeypatch.setattr(scaffold, "opencode_major", lambda: 2)
    return tmp_path / "my task"  # a space, to check the resume command's quoting


def init(task, *extra):
    main(["init", str(task), "--no-venv", "--no-git", "--gpu", UUID, "--model", "big:latest", *extra])


def test_init_writes_a_config_that_yuzuru_picks_up_in_the_task(task, monkeypatch):
    init(task, "--port", "11475", "--memory-max", "8G")

    monkeypatch.chdir(task)
    cfg = load_config()
    assert cfg.source == task / "yuzuru.yaml"
    assert (cfg.port, cfg.resources.gpus, cfg.isolation.memory_max) == (11475, [UUID], "8G")
    assert cfg.ollama.manage and [t.models for t in cfg.tiers] == [["big:latest"]]
    script = cfg.resume.command[2]
    assert shlex.split(script)[-2:] == [str(task / ".yuzuru" / "resume.log"), "2>&1"]
    assert (task / ".yuzuru").is_dir()  # the redirect fails without it

    oc = json.loads((task / "opencode.json").read_text())
    assert oc["provider"]["yuzuru"]["options"]["baseURL"] == "http://127.0.0.1:11475/v1"
    assert oc["mcp"]["servers"]["yuzuru"] == {"type": "remote", "url": "http://127.0.0.1:11475/mcp", "codemode": False}
    assert (oc["model"], oc["small_model"], oc["disabled_providers"]) == ("yuzuru/large", "yuzuru/large", ["opencode"])
    assert oc["permission"]["bash"]["*"] == "deny" and oc["permission"]["execute"] == "deny"

    agents = (task / "AGENTS.md").read_text()
    assert f'`cwd: "{task}"`' in agents and "{" not in agents
    assert ".venv` には" not in agents  # nothing was installed


def test_init_keeps_what_is_there_unless_forced(task):
    task.mkdir()
    (task / "AGENTS.md").write_text("task rules\n")
    (task / ".gitignore").write_text("data/\nout")
    init(task)
    assert (task / "AGENTS.md").read_text() == "task rules\n"
    lines = (task / ".gitignore").read_text().splitlines()
    assert lines[:2] == ["data/", "out"] and lines.count("data/") == 1 and ".venv/" in lines

    init(task)
    assert (task / ".gitignore").read_text().splitlines() == lines

    init(task, "--force", "--port", "11480")
    assert (task / "AGENTS.md.bak").read_text() == "task rules\n"
    assert load_config(task / "yuzuru.yaml").port == 11480
    assert load_config(task / "yuzuru.yaml.bak").port == 11435


def test_gpu_indices_become_uuids(monkeypatch):
    monkeypatch.setattr(scaffold, "gpus", lambda: [("0", "GPU-aaa", "RTX 3050"), ("1", UUID, "RTX 3060")])
    assert scaffold.resolve_gpus(["1"]) == [UUID]
    assert scaffold.resolve_gpus(None) == ["GPU-aaa", UUID]
    with pytest.raises(SystemExit, match="no GPU 2"):
        scaffold.resolve_gpus(["2"])


def tags(monkeypatch, models):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json={"models": models},
                                                                        request=httpx.Request("GET", url)))


def test_default_model_is_the_one_last_served_as_large(tmp_path, monkeypatch):
    tags(monkeypatch, [
        {"name": "huge:latest", "size": 23e9},
        {"name": "fits:latest", "size": 13e9},
        {"name": "yuzuru-large:huge-32k", "size": 23e9, "modified_at": "2026-10-20T10:00:00+09:00"},
        {"name": "yuzuru-large:fits-64k", "size": 13e9, "modified_at": "2026-10-29T08:00:00+09:00"},
    ])
    (tmp_path / "profiles.json").write_text(json.dumps({
        "huge:latest": {"digest": "h"}, "fits:latest": {"digest": "f"},
        "yuzuru-large:huge-32k": {"digest": "h"}, "yuzuru-large:fits-64k": {"digest": "f"},
    }))
    assert scaffold.default_model("http://ollama", tmp_path) == "fits:latest"
    assert scaffold.default_model("http://ollama", tmp_path / "none") == "huge:latest"  # no history: the largest
