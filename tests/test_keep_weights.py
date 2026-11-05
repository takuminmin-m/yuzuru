"""Weights in a job's directory are copied when it starts: the next job cannot destroy them.

Loop run 9 lost a model with val 73.58% to a job writing the same best.pth; it submitted 71.32%.
"""

import os
from pathlib import Path

from yuzuru.weights import KEPT_DIR, WeightKeeper, weight_files

GB = 1024 ** 3


def write(path: Path, text: str, mtime: int = 1_000_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))
    return path


def copies(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / KEPT_DIR).iterdir()) if (tmp_path / KEPT_DIR).exists() else []


def test_an_overwritten_file_can_be_got_back(tmp_path):
    best = write(tmp_path / "best.pth", "good")
    keeper = WeightKeeper(1)
    keeper.keep(tmp_path)
    assert keeper.news() == ""  # nothing has happened to it yet
    write(best, "worse!", mtime=2_000_000)
    news = keeper.news()
    copy = copies(tmp_path)[0]
    assert copy.read_text() == "good" and "best.pth was overwritten" in news and str(copy) in news
    assert keeper.news() == ""  # told once


def test_a_file_is_copied_again_only_when_it_changed(tmp_path):
    best = write(tmp_path / "best.pth", "a")
    keeper = WeightKeeper(1)
    keeper.keep(tmp_path)
    keeper.keep(tmp_path)
    assert len(copies(tmp_path)) == 1
    write(best, "bb", mtime=2_000_000)
    keeper.keep(tmp_path)
    assert sorted(c.read_text() for c in copies(tmp_path)) == ["a", "bb"]


def test_only_weights_outside_hidden_directories(tmp_path):
    write(tmp_path / "runs/a/model.safetensors", "w")
    write(tmp_path / "train.py", "code")
    write(tmp_path / ".venv/lib/x.pt", "package")
    assert weight_files(tmp_path) == [tmp_path / "runs/a/model.safetensors"]
    keeper = WeightKeeper(1)
    keeper.keep(tmp_path)
    keeper.keep(tmp_path)  # its own copies are not copied again
    assert [c.name.split(".")[0] for c in copies(tmp_path)] == ["runs__a__model"]


def test_the_limit_keeps_the_older_copies(tmp_path):
    best = write(tmp_path / "best.pth", "x" * 800)
    keeper = WeightKeeper(1000 / GB)
    keeper.keep(tmp_path)
    write(best, "y" * 800, mtime=2_000_000)
    keeper.keep(tmp_path)
    assert [c.read_text()[0] for c in copies(tmp_path)] == ["x"]


def test_zero_turns_it_off(tmp_path):
    write(tmp_path / "best.pth", "a")
    keeper = WeightKeeper(0)
    keeper.keep(tmp_path)
    assert not copies(tmp_path)


def test_a_failure_does_not_stop_the_job(tmp_path):
    write(tmp_path / "best.pth", "a")
    (tmp_path / ".yuzuru").write_text("a file where the directory should go")
    WeightKeeper(1).keep(tmp_path)  # does not raise
