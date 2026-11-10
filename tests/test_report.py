"""`yuzuru report`: the best job, its weights, GPU idle time and what went wrong, from yuzuru's own records.

Until now each run was audited by hand from its logs (All-PR run: 45 jobs over 23 hours).
"""

import json
import os
import time

import pytest

from yuzuru import metrics, report, runs

from .test_runs import SCRIPT


@pytest.mark.parametrize("line,name,value", [
    ("  e= 16/150 tr_loss=2.6353 tr_acc=26.36% va_loss=2.4784 va_acc=46.38% lr=0.096867 time=100.4s", "va_acc", 46.38),
    ("[epoch 15/60] train_loss=1.1499 train_acc=0.6653 val_loss=1.7318 val_acc=0.5504 lr=0.085370", "val_acc", 0.5504),
    ("ep=23 tl=2.1382 tacc=0.5432 val=0.6464 lr=0.002025 ep_time=26.4s", "val", 0.6464),
    ("[val] accuracy=74.94% (3747/5000)", "val_accuracy", 74.94),
    ("epoch 3: train loss: 0.41, val loss: 0.52", None, None),  # "val loss" is two words: not a pair
    ("epoch 3 val_loss: 0.52 lr: 1e-3", "val_loss", 0.52),
    ("[info] n_train=2000 n_val=5000 bs=128", None, None),  # a count, not a metric
])
def test_the_validation_metric_of_a_log_line(tmp_path, line, name, value):
    (tmp_path / "a.log").write_text(line + "\n")
    m = metrics.read(tmp_path / "a.log")
    assert (m and (m["name"], m["last"])) == (name and (name, value))


def test_a_score_is_read_before_a_loss_and_the_best_is_the_highest(tmp_path):
    (tmp_path / "a.log").write_text("".join(
        f"ep={i} val_loss={1 / i:.3f} val_acc={a}\n[save] best_val_acc={b}\n"
        for i, (a, b) in enumerate([(0.5, 0.5), (0.7, 0.7), (0.6, 0.7)], 1)))
    m = metrics.read(tmp_path / "a.log")
    assert m == {"name": "val_acc", "lower": False, "best": 0.7, "last": 0.6, "n": 3, "best_at": 2}
    assert metrics.read(tmp_path / "a.log", name="val_loss")["best"] == pytest.approx(0.333)


@pytest.fixture
def task(tmp_path):
    (tmp_path / "train.py").write_text(SCRIPT)
    return tmp_path


def job(task, command, log_text, rc=0, gpus=(0,), minutes=30.0, ago=0.0, log="", gpu=True):
    """A finished job that took `minutes` and ended `ago` minutes ago."""
    d = runs.begin(command, task, "s1", log or None, gpu=gpu)
    path = task / (log or f"{d.name}.log")
    with path.open("a") as f:
        f.write(log_text)
    runs.started(d, 999_999_999, str(path), [{"index": i, "name": f"RTX {i}"} for i in gpus])
    runs.ended(task, str(path), rc, log_text.strip().splitlines()[-1] if log_text.strip() else "")
    run = json.loads((d / "run.json").read_text())
    run["ended"] = time.time() - ago * 60
    run["started"] = run["ended"] - minutes * 60
    (d / "run.json").write_text(json.dumps(run))
    return d


def test_the_best_job_its_weights_and_what_the_others_changed(task):
    job(task, "python train.py --name base", "ep=1 val_acc=0.70\nep=2 val_acc=0.74\n", ago=100)
    job(task, "python train.py --name wd --wd 1e-3", "ep=1 val_acc=0.71\nep=2 val_acc=0.7656\nep=3 val_acc=0.76\n", ago=60)
    job(task, "python train.py --name bad --lr 10", "Traceback\nValueError: loss is NaN\n", rc=1, ago=20)
    (task / "best_wd.pth").write_bytes(b"w")
    os.utime(task / "best_wd.pth", (time.time() - 70 * 60,) * 2)  # written while 002-wd ran
    (task / "best_base.pth").write_bytes(b"w")
    os.utime(task / "best_base.pth", (time.time() - 75 * 60,) * 2)  # so was this, by a job next to it
    r = report.build(task)
    assert r["best"] == "002-wd" and r["metric"] == {"name": "val_acc", "lower": False}
    assert r["weights"] == [{"path": "best_wd.pth", "state": "as the job left it"}]
    out = report.text(r)
    assert "002-wd: val_acc 0.7656 at 2 of 3 values (last 0.76" in out
    assert "wd=0.0005" in out and "(best)" in out and "ValueError: loss is NaN" in out
    assert "1 of 3 jobs failed: 003-bad" in out


def test_overwritten_weights_point_to_yuzurus_copy(task):
    job(task, "python train.py --name a", "val_acc=0.7\n", ago=60)
    kept = task / ".yuzuru" / "kept-weights"
    kept.mkdir(parents=True)
    (kept / "best.101500.pth").write_bytes(b"w")
    os.utime(kept / "best.101500.pth", (time.time() - 65 * 60,) * 2)
    (task / "best.pth").write_bytes(b"w")  # written now, by something later
    r = report.build(task)
    assert [w["path"] for w in r["weights"]] == [".yuzuru/kept-weights/best.101500.pth"]
    assert "were overwritten or removed" in report.text(r)


def test_a_lower_metric_is_better_for_a_loss(task):
    job(task, "python train.py --name a", "val_mae=0.12\n", ago=60)
    job(task, "python train.py --name b --lr 0.2", "val_mae=0.10\nval_mae=0.11\n", ago=20)
    r = report.build(task)
    assert r["best"] == "002-b" and r["metric"]["lower"]
    assert report.build(task, metric="val_mae", lower=False)["best"] == "001-a"


def test_gpu_idle_time(task):
    # GPU 0: busy 30 min, free 60, busy 30. GPU 1: one job of 120 min.
    job(task, "python train.py --name a", "val_acc=0.7\n", gpus=(0,), minutes=30, ago=90)
    job(task, "python train.py --name b --lr 0.2", "val_acc=0.7\n", gpus=(1,), minutes=120, ago=0)
    job(task, "python train.py --name c --lr 0.3", "val_acc=0.7\n", gpus=(0,), minutes=30, ago=0)
    r = report.build(task)
    g0, g1 = r["gpus"]
    assert round(g0["busy"] / 60) == 60 and round(g0["idle"] / 60) == 60 and len(g0["gaps"]) == 1
    assert round(g1["busy"] / 60) == 120 and round(g1["idle"] / 60) == 0 and not g1["gaps"]
    assert "GPU 0 was free for 60 min" in report.text(r)


def test_jobs_sharing_a_log_file_are_read_apart(task):
    # All-PR run: a test run and its retries appended to one log; each read the others' numbers.
    job(task, "python train.py --name a", "val_acc=0.30\n", log="shared.log", ago=60)
    job(task, "python train.py --name a --lr 0.2", "val_acc=0.60\n", log="shared.log", ago=20)
    one, two = report.build(task)["jobs"]
    assert one["metric"]["best"] == 0.30 and two["metric"]["best"] == 0.60
    assert report.build(task, metric="val_acc")["jobs"][0]["metric"]["best"] == 0.30


def test_the_same_setting_twice_and_the_journal_are_noted(task):
    job(task, "python train.py --name a", "val_acc=0.7\n", ago=60)
    job(task, "python train.py --name b --seed 2", "val_acc=0.7\n", ago=20)
    runs.note(task, "idle_gpu_nudge", session="s1")
    runs.note(task, "idle_gpu_nudge", session="s1")
    out = report.text(report.build(task))
    assert "the same setting and scripts ran more than once: 001-a, 002-b" in out
    assert "continued the agent to use a free GPU: 2×" in out


def test_no_records(tmp_path):
    assert "no records" in report.text(report.build(tmp_path))


def test_the_cli(task, capsys):
    from yuzuru.cli import main

    job(task, "python train.py --name a", "val_acc=0.7\n")
    main(["report", str(task)])
    assert "Best  001-a" in capsys.readouterr().out
    main(["report", str(task), "--json"])
    assert json.loads(capsys.readouterr().out)["best"] == "001-a"
