"""A GPU job that trains the same as one the session already ran is refused once.

Loop run 17: four of the run's jobs trained one setting under four names (r18, cutmix, r18_v2, r18_v3).
"""

import pytest

from yuzuru.waits import TIME_LIMIT_RC, Event, Waiter, run_key

from .test_core import make


@pytest.mark.parametrize("a,b,same", [
    ("uv run python train.py --name r18 --epochs 75 --lr 0.1", "uv run python train.py --name=cutmix --epochs 75  --lr 0.1", True),
    ("python train.py --epochs 75 --out runs/a --log-dir logs/a", "python train.py --epochs 75 --out runs/b --log-dir logs/b", True),
    ("python train.py --name a --epochs 75", "python train.py --name a --epochs 100", False),
    ("python train.py --name a --wd 5e-4", "python train.py --name b --wd 1e-3", False),
    ("python train.py --name a --seed 1", "python train.py --name a --seed 2", False),
    ("python train.py --lr 0.1", "python train.py --lr 0.1 --mixup 0.2", False),
])
def test_run_key(a, b, same):
    assert (run_key(a) == run_key(b)) is same


def ended(session, log, rc):
    return Event(id=0, watch=log, session=session, label="train", rc=rc, duration=600.0, log=log, log_tail="",
                 note="")


async def test_the_same_setting_again_is_refused_once(cfg, world, http, tmp_path):
    catalog, broker, router = await make(cfg, world, http)
    w = Waiter(cfg, broker, router)
    (tmp_path / "train.py").write_text("print('v1')\n")
    (tmp_path / "r18.log").write_text("[best] epoch=98 val_acc=0.7694\n")
    w.note_started("s1", "r18.log", "uv run python train.py --name r18 --epochs 100", tmp_path)

    refusal = w.refuse_repeat("s1", "uv run python train.py --name cutmix --epochs 100", tmp_path)
    assert refusal and "the same as r18.log, which is still running" in refusal and "val_acc=0.7694" in refusal
    # Sent again: the agent means it.
    assert w.refuse_repeat("s1", "uv run python train.py --name cutmix --epochs 100", tmp_path) is None
    # Another session, another setting: not a repeat.
    assert w.refuse_repeat("s2", "uv run python train.py --name x --epochs 100", tmp_path) is None
    assert w.refuse_repeat("s1", "uv run python train.py --name v4 --epochs 100 --wd 1e-3", tmp_path) is None


async def test_not_a_repeat_after_a_fix_or_a_failure(cfg, world, http, tmp_path):
    catalog, broker, router = await make(cfg, world, http)
    w = Waiter(cfg, broker, router)
    script = tmp_path / "train.py"
    script.write_text("print('v1')\n")
    w.note_started("s1", "a.log", "python train.py --name a", tmp_path)
    w.note_started("s1", "b.log", "python train.py --name b --lr 0.05", tmp_path)
    w.events.append(ended("s1", "b.log", 1))
    assert w.refuse_repeat("s1", "python train.py --name b2 --lr 0.05", tmp_path) is None  # b failed
    script.write_text("print('v2: mixup labels fixed')\n")
    assert w.refuse_repeat("s1", "python train.py --name a2", tmp_path) is None  # the script changed


async def test_a_finished_job_is_named_with_how_it_ended(cfg, world, http, tmp_path):
    catalog, broker, router = await make(cfg, world, http)
    w = Waiter(cfg, broker, router)
    (tmp_path / "train.py").write_text("x\n")
    w.note_started("s1", "a.log", "python train.py --name a --epochs 75", tmp_path)
    w.events.append(ended("s1", "a.log", 0))
    assert "which finished after 10 min" in w.refuse_repeat("s1", "python train.py --name c --epochs 75", tmp_path)
    w.note_started("s1", "d.log", "python train.py --name d --epochs 30", tmp_path)
    w.events.append(ended("s1", "d.log", TIME_LIMIT_RC))
    assert w.refuse_repeat("s1", "python train.py --name e --epochs 30", tmp_path) is None  # refuse_after_stop's
