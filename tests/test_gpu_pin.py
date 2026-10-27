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
