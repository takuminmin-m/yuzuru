"""A job whose epochs will not fit its time limit is stopped early, with the epochs that fit.

Loop run 12: four of five 45-min jobs (200 epochs at 16 s, 130 at 37 s) ran their whole limit and were
stopped with the learning rate still high; the one that fit (110 epochs) was the best of the run.
"""

import os
import subprocess
import sys

import pytest

from .test_core import make
from yuzuru.waits import TIME_LIMIT_RC, Waiter, epochs_done

LIMIT = 45 * 60


@pytest.mark.parametrize("text,total,done", [
    ("Ep 1/200 loss=4.2\nEp 12/200 loss=3.1 lr=0.1", 200, 12),
    ("[epoch] 44/45  loss=0.0143 acc=0.9982", 45, 44),
    ("E3/100 tl=3.9\nE4/100 tl=3.7", 100, 4),
    ("Epoch 7 / 40", 40, 7),
    ("val 0.5/200.5 1/2000", 200, None),  # not this job's epochs
    ("starting", 30, None),
])
def test_epochs_done(text, total, done):
    assert epochs_done(text, total) == done


async def job(cfg, world, http, tmp_path, epochs):
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    log = tmp_path / "exp.log"
    log.write_text("")
    lease = await broker.create_lease(pid=proc.pid, label="train", gpus=[], vram=None, drain=False, session="s1",
                                      cwd=str(tmp_path), resume=True, note="", log=str(log))
    w = waiter.watch_lease(lease)
    waiter.note_job(str(log), f"uv run python train.py --epochs {epochs}", LIMIT)
    return waiter, broker, lease, w, proc, log


def at(waiter, w, log, seconds, line):
    with log.open("a") as f:
        f.write(line + "\n")
    waiter.check_overruns(now=w.created + seconds)


async def test_a_job_that_will_not_fit_is_stopped_early(cfg, world, http, tmp_path):
    cfg.resume.job_max_minutes = 45
    waiter, broker, lease, w, proc, log = await job(cfg, world, http, tmp_path, 200)
    try:
        at(waiter, w, log, 20, "Ep 1/200 loss=4.2")
        at(waiter, w, log, 52, "Ep 3/200 loss=3.9")  # too few epochs yet to trust the speed
        assert proc.poll() is None
        at(waiter, w, log, 84, "Ep 5/200 loss=3.6")  # 16 s per epoch: 200 epochs need about 53 min
        assert proc.wait(timeout=5) == -15
        await broker.release_lease(lease.id, rc=143)
        ev = waiter.events[-1]
        assert ev.rc == TIME_LIMIT_RC and ev.overrun
        text = waiter.message(ev)
        assert "after 5 of 200 epochs" in text and "about 53 min" in text and "--epochs 134 or fewer" in text
        # The same epochs again are refused once, as after a stop at the limit.
        refusal = waiter.refuse_after_stop("s1", "uv run python train.py --epochs 200")
        assert "was stopped early, after 5 epochs" in refusal
        assert waiter.refuse_after_stop("s1", "uv run python train.py --epochs 130") is None
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize("epochs,seconds", [
    (45, 61),  # Loop run 11: 46 min, a little over; near the end the learning rate is low already
    (110, 15),  # fits
])
async def test_a_job_that_about_fits_runs_on(cfg, world, http, tmp_path, epochs, seconds):
    cfg.resume.job_max_minutes = 45
    waiter, broker, lease, w, proc, log = await job(cfg, world, http, tmp_path, epochs)
    try:
        for k in range(1, 12):
            at(waiter, w, log, 10 + k * seconds, f"[epoch] {k}/{epochs}")
        assert proc.poll() is None and not waiter.overruns
    finally:
        proc.kill()
        proc.wait()
        await broker.release_lease(lease.id, rc=0)


async def test_not_checked_late_or_when_off(cfg, world, http, tmp_path):
    cfg.resume.job_max_minutes = 45
    waiter, broker, lease, w, proc, log = await job(cfg, world, http, tmp_path, 200)
    try:
        cfg.resume.overrun_margin = 0
        at(waiter, w, log, 20, "Ep 1/200")
        at(waiter, w, log, 84, "Ep 5/200")
        assert proc.poll() is None
        cfg.resume.overrun_margin = 0.1
        at(waiter, w, log, LIMIT / 2 + 10, "Ep 6/200")  # past half the limit: let it run
        at(waiter, w, log, LIMIT / 2 + 100, "Ep 12/200")
        assert proc.poll() is None and not waiter.overruns
    finally:
        proc.kill()
        proc.wait()
        await broker.release_lease(lease.id, rc=0)


async def test_a_job_without_a_limit_is_left_alone(cfg, world, http, tmp_path):
    waiter, broker, lease, w, proc, log = await job(cfg, world, http, tmp_path, 200)
    waiter.note_job(str(log), "uv run python train.py --epochs 200", None)
    try:
        at(waiter, w, log, 20, "Ep 1/200")
        at(waiter, w, log, 84, "Ep 5/200")
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()
        await broker.release_lease(lease.id, rc=0)
