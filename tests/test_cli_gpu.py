import pytest

from yuzuru import cli


@pytest.fixture
def seen(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YUZURU_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    got = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args, client: got.update(gpu=args.gpu, cmd=args.cmd))
    return got


def test_run_gpu_takes_commas_and_repeats(seen):
    # Same spelling as yuzuru_train's `gpus`; the 2-GPU accelerate run was refused as "unknown GPU 'a,b'".
    cli.main(["run", "--gpu", "GPU-a, GPU-b", "--gpu", "1", "--", "python", "t.py"])
    assert seen == {"gpu": ["GPU-a", "GPU-b", "1"], "cmd": ["--", "python", "t.py"]}


def test_run_without_gpu_stays_none(seen):
    cli.main(["run", "--", "python", "t.py"])
    assert seen["gpu"] is None


def test_empty_gpu_is_an_error(seen):
    with pytest.raises(SystemExit):
        cli.main(["run", "--gpu", ",", "--", "python", "t.py"])
