"""A `yuzuru run` job's Python cannot switch CUDA_VISIBLE_DEVICES to GPUs outside its lease.

An agent's training script that set CUDA_VISIBLE_DEVICES itself used a GPU yuzuru was told to
keep free; yuzuru's own value only reached the process environment.
"""

import argparse
import json
import sys

from .test_gpu_allowlist import FakeClient, call_run

# Tries each way of changing CUDA_VISIBLE_DEVICES and records what the job ends up with.
PROBE = r"""
import json, os, sys
K = "CUDA_VISIBLE_DEVICES"
seen = {"start": os.environ.get(K)}

def attempt(name, change):
    try:
        change()
        seen[name] = os.environ.get(K)
    except RuntimeError:
        seen[name] = "refused"

attempt("other", lambda: os.environ.__setitem__(K, "GPU-9"))
attempt("index", lambda: os.environ.update({K: "0"}))
attempt("putenv", lambda: os.putenv(K, "GPU-9"))
attempt("narrow", lambda: os.environ.__setitem__(K, "GPU-1"))
attempt("delete", lambda: os.environ.pop(K))
saved = dict(os.environ)
os.environ.clear()  # what accelerate's clear_environment() does around HF Trainer setup
seen["clear"] = sorted(os.environ)
os.environ.update(saved)
seen["restored"] = dict(os.environ) == saved
attempt("hide", lambda: os.environ.__setitem__(K, "-1"))
open(sys.argv[1], "w").write(json.dumps(seen))
"""


def job(tmp_path, *prefix, config=None, cpu=False):
    out = tmp_path / "seen.json"
    cmd = ["--", *prefix, sys.executable, "-c", PROBE, str(out)]
    args = argparse.Namespace(cmd=cmd, detach=False, log=None, session=None, resume=False, note="",
                              use=None, label=None, gpu=None, vram=None, config=config, cpu=cpu)
    return args, out


def test_job_cannot_leave_its_gpus(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    args, out = job(tmp_path)
    call_run(args, FakeClient({"id": "abc", "gpus": [], "visible": ["GPU-0", "GPU-1"]}))
    assert json.loads(out.read_text()) == {
        "start": "GPU-0,GPU-1", "other": "refused", "index": "refused", "putenv": "refused",
        "narrow": "GPU-1", "delete": "GPU-1", "clear": ["CUDA_VISIBLE_DEVICES"], "restored": True, "hide": "-1",
    }


def test_value_set_before_python_starts_is_replaced(tmp_path, capfd):
    args, out = job(tmp_path, "env", "CUDA_VISIBLE_DEVICES=GPU-9")
    call_run(args, FakeClient({"id": "abc", "gpus": ["GPU-0"]}))
    assert json.loads(out.read_text())["start"] == "GPU-0"
    assert "CUDA_VISIBLE_DEVICES=GPU-9 is outside this job's GPUs; using GPU-0" in capfd.readouterr().err


def test_cpu_job_cannot_take_a_gpu(tmp_path):
    args, out = job(tmp_path, cpu=True)
    call_run(args, FakeClient({"id": "abc", "gpus": [], "visible": ["-1"]}))
    seen = json.loads(out.read_text())
    assert seen["start"] == "-1" and seen["other"] == seen["narrow"] == "refused"


def test_pinned_without_server_too(tmp_path):
    conf = tmp_path / "y.yaml"
    conf.write_text("resources:\n  gpus: [1]\n")
    args, out = job(tmp_path, config=str(conf))
    call_run(args, FakeClient(None))
    seen = json.loads(out.read_text())
    assert seen["start"] == "1" and seen["index"] == "refused"


def test_shadowed_sitecustomize_still_runs(tmp_path, monkeypatch):
    own = tmp_path / "own"
    own.mkdir()
    (own / "sitecustomize.py").write_text(f"open({str(tmp_path / 'ran')!r}, 'w').close()\n")
    monkeypatch.setenv("PYTHONPATH", str(own))
    args, out = job(tmp_path)
    call_run(args, FakeClient({"id": "abc", "gpus": ["GPU-0"]}))
    assert (tmp_path / "ran").exists()
    assert json.loads(out.read_text())["other"] == "refused"


def test_unrestricted_job_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args, out = job(tmp_path)
    call_run(args, FakeClient({"id": "abc", "gpus": [], "visible": []}))  # no resources.gpus
    seen = json.loads(out.read_text())
    assert seen["start"] is None and seen["other"] == "GPU-9"


# What an agent started through yuzuru runs itself (`python train.py` from its shell) sees no GPU:
# in LoRA run 3 such a run trained without a lease and overwrote the best weights.

def agent_python(tmp_path, cfg, *code_args, extra_env=None):
    import subprocess

    from yuzuru.waits import agent_env

    env = agent_env(cfg) | (extra_env or {})
    return subprocess.run([sys.executable, *code_args], env=env, capture_output=True, text=True, timeout=60)


def test_agent_commands_see_no_gpu(tmp_path, cfg, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    out = tmp_path / "seen.json"
    r = agent_python(tmp_path, cfg, "-c", PROBE, str(out))
    assert r.returncode == 0, r.stderr
    seen = json.loads(out.read_text())
    assert seen["start"] == "" and seen["hide"] == "-1"
    assert seen["other"] == seen["index"] == seen["putenv"] == seen["narrow"] == "refused"
    r = agent_python(tmp_path, cfg, "-c", "import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'")
    assert "cannot be set to '0' here" in r.stderr and "yuzuru_train" in r.stderr


def test_agent_value_set_before_python_starts_is_ignored(tmp_path, cfg):
    r = agent_python(tmp_path, cfg, "-c", "import os; print(repr(os.environ['CUDA_VISIBLE_DEVICES']))",
                     extra_env={"CUDA_VISIBLE_DEVICES": "0"})
    assert r.stdout.strip() == "''"
    assert "CUDA_VISIBLE_DEVICES=0 ignored" in r.stderr


def test_agent_cuda_error_points_at_the_train_tool(tmp_path, cfg):
    r = agent_python(tmp_path, cfg, "-c", "raise RuntimeError('No CUDA GPUs are available')")
    assert r.returncode == 1
    assert r.stderr.index("No CUDA GPUs are available") < r.stderr.index("yuzuru_train")
    r = agent_python(tmp_path, cfg, "-c", "raise ValueError('bad shape')")
    assert "yuzuru_train" not in r.stderr


def test_agent_gpus_can_stay_visible(tmp_path, cfg, monkeypatch):
    from yuzuru.waits import agent_env

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0")
    cfg.isolation.hide_gpus_from_agent = False
    env = agent_env(cfg)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-0" and "YUZURU_AGENT" not in env and "PYTHONPATH" not in env


def test_yuzuru_agent_starts_without_gpus(cfg, monkeypatch):
    from yuzuru import cli

    started = {}
    monkeypatch.setattr(cli.os, "execvpe", lambda f, argv, env: started.update(argv=argv, env=env))
    cli.cmd_agent(argparse.Namespace(cmd=["--", "opencode", "service", "start"]), cfg)
    assert started["argv"] == ["opencode", "service", "start"]
    assert started["env"]["CUDA_VISIBLE_DEVICES"] == "" and started["env"]["YUZURU_AGENT"] == "1"


def test_job_started_from_the_agents_shell_gets_its_gpus(tmp_path, monkeypatch):
    # `yuzuru run` from inside the agent: the lease's GPUs, not the agent's hiding.
    monkeypatch.setenv("YUZURU_AGENT", "1")
    monkeypatch.setenv("YUZURU_CUDA_VISIBLE_DEVICES", "")
    args, out = job(tmp_path)
    call_run(args, FakeClient({"id": "abc", "gpus": ["GPU-0"]}))
    seen = json.loads(out.read_text())
    assert seen["start"] == "GPU-0" and seen["other"] == "refused"
