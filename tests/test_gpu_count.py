import json
import os
import subprocess
from types import SimpleNamespace

import pytest

import yuzuru.mcp as mcp_mod
from yuzuru.config import GB, RouterConfig
from yuzuru.server import State

from .conftest import FakeProbe
from .test_gpu_allowlist import make_broker


@pytest.fixture
def two_gpus(world):
    world.gpu_sizes = [8 * GB, 12 * GB]  # nvidia-smi order, like the 3050 and the 3060
    world.foreign = [0, 0]
    return world


async def lease(broker, **kw):
    return await broker.create_lease(pid=os.getpid(), label="job", gpus=[], vram=None, drain=False, **kw)


async def test_gpu_count_takes_free_gpus_most_memory_first(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    first = await lease(broker, gpu_count=1)
    assert first.gpus == ["GPU-1"] and first.whole
    second = await lease(broker, gpu_count=1)
    assert second.gpus == ["GPU-0"]
    with pytest.raises(ValueError, match=r"1 free GPU\(s\) needed but 0 available.*GPU 1 .*pid"):
        await lease(broker, gpu_count=1)


async def test_gpu_count_two_lists_gpus_in_nvidia_smi_order(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    assert (await lease(broker, gpu_count=2)).gpus == ["GPU-0", "GPU-1"]


async def test_gpu_count_with_vram_takes_the_gpu_whole(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    placed = await broker.create_lease(pid=os.getpid(), label="job", gpus=[], vram=4 * GB, drain=False, gpu_count=1)
    assert placed.gpus == ["GPU-1"] and placed.whole and placed.vram is None
    assert broker.dedicated() == {"GPU-1"}
    # Its full size is enough: nothing else is on a GPU taken whole (Alloc run 1 asked 8G of the 3050).
    other = await broker.create_lease(pid=os.getpid(), label="job", gpus=[], vram=8 * GB, drain=False, gpu_count=1)
    assert other.gpus == ["GPU-0"] and other.whole


async def test_refusal_says_why_a_free_gpu_is_not_enough(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    await broker.create_lease(pid=os.getpid(), label="job", gpus=[], vram=None, drain=False, gpu_count=1)
    with pytest.raises(ValueError, match=r"(?s)with 10\.0 GB needed but 0 available: GPU 0 \(.*\) is free but has only "
                                         r"8\.0 GB.*ask for less now; otherwise end your turn"):
        await broker.create_lease(pid=os.getpid(), label="job", gpus=[], vram=10 * GB, drain=False, gpu_count=1)


async def test_refusal_does_not_say_wait_when_waiting_cannot_help(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    for kw in ({"vram": 13 * GB, "gpu_count": 1}, {"vram": None, "gpu_count": 3}, {"vram": 10 * GB, "gpu_count": 2}):
        with pytest.raises(ValueError, match="this machine has") as e:
            await broker.create_lease(pid=os.getpid(), label="job", gpus=[], drain=False, **kw)
        assert "End your turn" not in str(e.value) and "end your turn" not in str(e.value)


async def test_gpu_count_waits_for_a_job_that_may_be_anywhere(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    await lease(broker)
    with pytest.raises(ValueError, match="may be on any GPU"):
        await lease(broker, gpu_count=1)


async def test_gpu_count_excludes_named_gpus(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    with pytest.raises(ValueError, match="not both"):
        await broker.create_lease(pid=os.getpid(), label="job", gpus=["0"], vram=None, gpu_count=1)


async def test_usage_line_names_each_gpus_job(cfg, two_gpus, http):
    broker = make_broker(cfg, two_gpus, http, [])
    await broker.refresh()
    assert broker.gpu_usage_line() == "GPUs now: GPU 0 (Fake 0): free; GPU 1 (Fake 1): free"
    await lease(broker, gpu_count=1, log="/x/exp7.log")
    assert broker.gpu_usage_line() == f"GPUs now: GPU 0 (Fake 0): free; GPU 1 (Fake 1): exp7.log (pid {os.getpid()})"


@pytest.fixture
async def train(cfg, two_gpus, http, tmp_path, monkeypatch):
    cfg.router = RouterConfig(classifier="heuristic")
    cfg.isolation.scope = "off"
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"pid": os.getpid(), "log": "x.log"}), stderr="")

    monkeypatch.setattr(mcp_mod.subprocess, "run", fake_run)
    st = State(cfg, probe=FakeProbe(two_gpus), http=http)
    await st.broker.refresh()
    base = {"command": "python train.py", "cwd": str(tmp_path), "resume_when_done": False}
    return SimpleNamespace(call=lambda **kw: mcp_mod.MCPServer(st).tools["train"][1](base | kw, {}), argv=calls, st=st)


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


async def test_train_picks_one_gpu_when_there_are_several(train):
    await train.call()
    assert flag(train.argv[-1], "--gpu-count") == "1" and "--vram" not in train.argv[-1]
    await train.call(vram="6G")  # DDP run 2: `vram` alone stacked three jobs on both GPUs
    assert flag(train.argv[-1], "--gpu-count") == "1" and flag(train.argv[-1], "--vram") == "6G"
    await train.call(gpu_count=2)
    assert flag(train.argv[-1], "--gpu-count") == "2"
    await train.call(gpus="1", vram="6G")  # naming the GPU is how to share one on purpose
    assert "--gpu-count" not in train.argv[-1] and flag(train.argv[-1], "--gpu") == "1"
    await train.call(device="cpu")
    assert "--gpu-count" not in train.argv[-1]


async def test_train_refuses_gpu_count_with_gpus_or_vram(train):
    for extra in ({"gpus": "1"}, {"vram": "6G"}, {"device": "cpu"}):
        with pytest.raises(ValueError, match="gpu_count"):
            await train.call(gpu_count=1, **extra)
    assert not train.argv


async def test_train_says_where_the_job_runs(train):
    await lease(train.st.broker, gpu_count=1)
    text = await train.call()
    assert "cuda:0 = GPU 1 (Fake 1)" in text and "GPU 0 (Fake 0): free" in text


async def test_train_reports_a_job_that_was_refused(train, tmp_path, monkeypatch):
    log = tmp_path / "x.log"
    log.write_text('yuzuru: 400 {"detail":"1 free GPU(s) needed but 0 available"}\n')
    server = mcp_mod.MCPServer(train.st)
    gone = subprocess.Popen(["true"])
    gone.wait()
    with pytest.raises(RuntimeError, match="(?s)did not start.*needed but 0 available"):
        await server._placement(gone.pid, str(log))


def test_resume_message_lists_free_gpus(cfg, two_gpus, http):
    st = State(cfg, probe=FakeProbe(two_gpus), http=http)
    st.broker.snapshot = FakeProbe(two_gpus).snapshot()
    ev = SimpleNamespace(rc=0, duration=60, label="job", note="", log=None, log_tail="")
    assert "GPU 0 (Fake 0): free; GPU 1 (Fake 1): free" in st.waiter.message(ev)
