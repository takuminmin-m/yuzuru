"""yuzuru's record of each job (.yuzuru/runs/), and refusing once a job that trains the same as one already run.

Loop run 17 ran one setting under four names; Loop run 20 ran it again with its defaults spelled out
(exp05) and with another seed and the options reordered (exp07).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from yuzuru import runs

SCRIPT = """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--weight-decay", "--wd", dest="wd", type=float, default=5e-4)
p.add_argument("--mixup", action="store_true")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="runs")
args = p.parse_args()
print(args)
"""


@pytest.fixture
def task(tmp_path):
    (tmp_path / "train.py").write_text(SCRIPT)
    return tmp_path


@pytest.mark.parametrize("a,b,same", [
    ("uv run python train.py --name a --epochs 75", "uv run python train.py --name=b --epochs 75", True),
    ("python train.py --name a", "python train.py --name e5 --epochs 60 --lr 0.1 --wd 5e-4", True),  # exp05
    ("python train.py --name a --epochs 60 --seed 1 --lr 0.1", "python train.py --lr=0.10 --name b --seed 2", True),
    ("python train.py --name a --weight-decay 0.0005", "python train.py --name b --wd 5e-4", True),
    ("python train.py --name a --out x/a", "python train.py --name b --out x/b", True),
    ("python train.py --name a", "python train.py --name a --epochs 100", False),
    ("python train.py --name a", "python train.py --name b --mixup", False),
    ("python train.py --name a --wd 5e-4", "python train.py --name b --wd 1e-3", False),
])
def test_setting(task, a, b, same):
    assert (runs.setting(a, task) == runs.setting(b, task)) is same


def test_a_script_without_a_parser_is_compared_by_its_options(tmp_path):
    (tmp_path / "train.py").write_text("import sys\n")
    s = lambda c: runs.setting(c, tmp_path)  # noqa: E731
    assert s("python train.py --name a --lr 0.1") == s("python train.py --lr 0.1 --name b")
    assert s("python train.py --lr 0.1") != s("python train.py --lr 0.1 --epochs 60")


def finish(d, rc=0):
    run = json.loads((d / "run.json").read_text())
    runs.started(d, 999_999_999, run["log"] or str(d / "x.log"))
    runs.ended(run["cwd"], json.loads((d / "run.json").read_text())["log"], rc, "best val_acc=0.7656")


def test_a_record_keeps_the_command_the_setting_and_the_code(task):
    d = runs.begin("uv run python train.py --name wd02 --wd 0.02", task, "s1", None, gpu=True)
    assert d.name == "001-wd02" and (d / "code" / "train.py").read_text() == SCRIPT
    (task / "train.py").write_text(SCRIPT.replace("0.1", "0.05"))  # edited in place afterwards: the copy stays
    assert (d / "code" / "train.py").read_text() == SCRIPT
    runs.started(d, 123, str(task / "wd02.log"))
    runs.ended(task, str(task / "wd02.log"), 0, "best val_acc=0.7494")
    run = json.loads((d / "run.json").read_text())
    assert run["setting"]["args"] == {"epochs": 60.0, "lr": 0.1, "wd": 0.02, "mixup": False}
    assert run["rc"] == 0 and run["pid"] == 123 and run["ended"] >= run["started"] and run["last_line"].startswith("best")
    assert runs.begin("python train.py --name x", task, "s1", None, gpu=True).name == "002-x"
    assert "002-x" in runs.table(task) and "wd=0.02" in runs.table(task)


def test_the_same_setting_is_refused_once(task):
    finish(runs.begin("python train.py --name exp02 --epochs 60", task, "s1", None, gpu=True))
    refusal = runs.refuse_repeat("python train.py --name exp05 --epochs 60 --lr 0.1 --wd 5e-4 --seed 123", task)
    assert refusal and "001-exp02" in refusal and "val_acc=0.7656" in refusal
    # Sent again: the agent means it.
    assert runs.refuse_repeat("python train.py --name exp05 --epochs 60 --lr 0.1 --wd 5e-4 --seed 123", task) is None
    assert runs.refuse_repeat("python train.py --name exp06 --epochs 60 --mixup", task) is None


def test_a_running_job_counts(task):
    d = runs.begin("python train.py --name a", task, "s1", None, gpu=True)
    runs.started(d, os.getpid(), str(task / "a.log"))
    assert "is still running" in runs.refuse_repeat("python train.py --name b", task)


def test_not_a_repeat_after_a_failure_a_fix_or_on_cpu(task):
    finish(runs.begin("python train.py --name a", task, "s1", None, gpu=True), rc=1)
    assert runs.refuse_repeat("python train.py --name b", task) is None  # it failed: run it again
    finish(runs.begin("python train.py --name c", task, "s1", None, gpu=True))
    (task / "train.py").write_text(SCRIPT + "# mixup labels fixed\n")
    assert runs.refuse_repeat("python train.py --name d", task) is None  # the script changed
    finish(runs.begin("python predict.py --split val", task, "s1", None, gpu=False))
    assert runs.refuse_repeat("python predict.py --split val", task) is None  # CPU jobs are not judged


def test_the_job_writes_the_arguments_it_parsed(task):
    d = runs.begin("python train.py --name a --lr 0.05", task, "s1", None, gpu=True)
    pin = Path(runs.__file__).with_name("gpu_pin")
    env = {**os.environ, "PYTHONPATH": str(pin), runs.RUN_DIR_ENV: str(d)}
    subprocess.run([sys.executable, "train.py", "--name", "a", "--lr", "0.05"], cwd=task, env=env, check=True,
                   capture_output=True)
    got = json.loads((d / "args.json").read_text())
    assert got["args"] == {"name": "a", "epochs": 60, "lr": 0.05, "wd": 5e-4, "mixup": False, "seed": 42, "out": "runs"}
    assert got["argv"] == ["--name", "a", "--lr", "0.05"]
    # Only the first parse of the job is kept.
    subprocess.run([sys.executable, "train.py", "--name", "b"], cwd=task, env=env, check=True, capture_output=True)
    assert json.loads((d / "args.json").read_text())["args"]["name"] == "a"


async def test_the_record_is_closed_when_the_jobs_lease_ends(cfg, world, http, task):
    from yuzuru.waits import Waiter

    from .test_core import make

    catalog, broker, router = await make(cfg, world, http)
    Waiter(cfg, broker, router)
    log = task / "a.log"
    log.write_text("epoch 60 val_acc=0.7656\n")
    d = runs.begin("python train.py --name a", task, "s1", str(log), gpu=True)
    runs.started(d, os.getpid(), str(log))
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram="1G", drain=False, session="s1",
                                      cwd=str(task), log=str(log), resume=False, note="")
    await broker.release_lease(lease.id, rc=0)
    run = json.loads((d / "run.json").read_text())
    assert run["rc"] == 0 and run["last_line"] == "epoch 60 val_acc=0.7656"
