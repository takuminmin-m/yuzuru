"""A job started with `verify` evaluates an earlier job's saved weights on the validation data; yuzuru compares what
it reports with that job's log and tells the agent.

The All-PR run's log said va_acc 78.74; its weights, BatchNorm buffers not saved, scored 74.16 on test. HPO run 3's
predict script normalized differently from training: 57.12 in the log, a submission of 7.23.
"""

import json
import os

import pytest

from yuzuru import report, runs

from .test_core import make, resume_capture, wait_for

TRAIN_LOG = "".join(f"epoch {i} loss={1 / i:.3f} va_acc={v}\n" for i, v in enumerate([61.2, 70.5, 76.03, 78.74, 77.9], 1))


def trained(task, name="cutmix50", log_text=TRAIN_LOG):
    log = task / f"{name}.log"
    d = runs.begin(f"python train.py --name {name}", task, "s1", str(log), gpu=True)
    log.write_text(log_text)
    runs.started(d, 999_999_999, str(log))
    runs.ended(task, str(log), 0, "done")
    return d.name


def verify(task, of, out, rc=0, tolerance=0.02):
    log = task / f"verify-{of}.log"
    d = runs.begin(f"python predict.py --split val --weights best_{of}.pth", task, "s1", str(log), gpu=True,
                   verify=runs.verify_target(task, of, tolerance))
    log.write_text(out)
    runs.started(d, 999_999_999, str(log))
    return runs.ended(task, str(log), rc, out.strip().splitlines()[-1] if out.strip() else "")


@pytest.fixture
def task(tmp_path):
    (tmp_path / "train.py").write_text("print('hi')\n")
    return tmp_path


def test_a_job_is_found_by_id_number_or_name(task):
    first, second = trained(task, "a"), trained(task, "b")
    assert runs.find(task, first)[1]["id"] == "001-a"
    assert runs.find(task, "2")[1]["id"] == runs.find(task, "002")[1]["id"] == "002-b"
    assert runs.find(task, "b")[1]["id"] == "002-b" and runs.find(task, "c") is None


def test_what_cannot_be_verified_is_refused(task):
    with pytest.raises(ValueError, match="no job"):
        runs.verify_target(task, "nope", 0.02)
    silent = trained(task, "silent", "epoch 1 loss=0.5\n")
    with pytest.raises(ValueError, match="no validation metric"):
        runs.verify_target(task, silent, 0.02)
    running = runs.begin("python train.py --name r", task, "s1", str(task / "r.log"), gpu=True)
    with pytest.raises(ValueError, match="not ended"):
        runs.verify_target(task, running.name, 0.02)


def test_weights_that_give_the_logs_best_match(task):
    of = trained(task)
    run = verify(task, of, "loaded best_cutmix50.pth\nval accuracy: 78.70%\n")  # another name, same metric
    assert run["verify"]["verdict"] == "match" and run["verify"]["got"] == 78.70
    text = runs.verify_text(run)
    assert "78.70" in text and "78.74" in text and "FAILED" not in text
    assert runs.find(task, of)[1]["verified"][0]["verdict"] == "match"


def test_a_fraction_matches_a_percentage(task):
    of = trained(task, "frac", "".join(f"epoch {i} val_acc={v}\n" for i, v in enumerate([0.61, 0.7874], 1)))
    assert verify(task, of, "val_acc=78.69\n")["verify"]["verdict"] == "match"


def test_weights_that_do_not_give_the_logs_best_are_reported(task):
    of = trained(task)
    run = verify(task, of, "[val] acc=74.16\n")
    assert run["verify"]["verdict"] == "mismatch"
    text = runs.verify_text(run)
    assert "FAILED" in text and "74.16" in text and "78.74" in text and "BatchNorm" in text
    assert [e["kind"] for e in runs.journal(task)] == ["verify"]


def test_the_last_epochs_weights_are_named(task):
    # The file was written at the end of training, not when the metric was best.
    run = verify(task, trained(task), "va_acc=77.90\n", tolerance=0.005)
    assert run["verify"]["like_last"] == 77.9
    assert "from the end of training" in runs.verify_text(run)
    # A value that is only somewhere on the rising curve says nothing about which weights these are.
    assert "like_last" not in verify(task, trained(task, "b"), "va_acc=76.03\n", tolerance=0.005)["verify"]


def test_a_verify_job_that_prints_no_metric_is_told_how(task):
    run = verify(task, trained(task), "wrote predictions\n")
    assert run["verify"]["verdict"] == "unread"
    assert "`va_acc=<value>`" in runs.verify_text(run)


def test_a_failed_verify_job_says_nothing_more(task):
    assert runs.verify_text(verify(task, trained(task), "Traceback\nKeyError: 'bn1.running_mean'\n", rc=1)) == ""


def test_verify_jobs_are_not_repeats(task):
    of = trained(task)
    verify(task, of, "va_acc=74.1\n")
    assert runs.refuse_repeat(f"python predict.py --split val --weights best_{of}.pth", task) is None


def test_the_report_shows_the_check(task):
    of = trained(task)
    r = report.build(task)
    assert any("were not evaluated again" in n for n in r["notes"])
    verify(task, of, "va_acc=74.16\n")
    r = report.build(task)
    assert r["best"] == of
    assert any("gave 74.16" in n for n in r["notes"])
    out = report.text(r)
    assert f"verified by 002-verify-{of}: mismatch 74.16" in out and f"verify of {of}: mismatch (74.16)" in out


async def test_the_agent_is_told_when_the_verify_job_ends(cfg, world, http, task):
    from yuzuru.waits import Waiter

    out = resume_capture(cfg, task)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    of = trained(task)
    log = task / "verify.log"
    d = runs.begin("python predict.py --split val", task, "s1", str(log), gpu=True,
                   verify=runs.verify_target(task, of, cfg.resume.verify_tolerance))
    log.write_text("va_acc=7.23\n")
    runs.started(d, os.getpid(), str(log))
    lease = await broker.create_lease(pid=os.getpid(), label="train", gpus=[], vram="1G", session="s1",
                                      cwd=str(task), log=str(log), resume=True, note="")
    waiter.watch_lease(lease)
    await broker.release_lease(lease.id, rc=0)
    await wait_for(out.exists)
    _, message = json.loads(out.read_text())
    assert f"Verify of {of} FAILED" in message and "7.23" in message


async def test_the_train_tool_starts_a_verify_job(cfg, task, monkeypatch):
    import subprocess

    from yuzuru.mcp import MCPServer
    from yuzuru.server import State

    cfg.isolation.scope = "off"
    of = trained(task)
    monkeypatch.setattr("yuzuru.mcp.PLACEMENT_WAIT", 0)
    monkeypatch.setattr(subprocess, "run", lambda argv, cwd, **kw: subprocess.CompletedProcess(
        argv, 0, json.dumps({"pid": os.getpid(), "log": str(task / "v.log")}), ""))
    train = MCPServer(State(cfg)).tools["train"][1]
    call = {"command": "python predict.py --split val", "cwd": str(task), "vram": "2G", "resume_when_done": False}
    with pytest.raises(ValueError, match="no job"):
        await train({**call, "verify": "999"}, {})
    text = await train({**call, "verify": "cutmix50"}, {})
    assert "(record 002-" in text and f"with {of}'s best in its log (78.74)" in text
    record = runs.find(task, "2")[1]
    assert record["verify"]["of"] == of and record["verify"]["tolerance"] == cfg.resume.verify_tolerance
    # The same check again is not refused as a repeat.
    await train({**call, "verify": of}, {})
    assert runs.find(task, "3")[1]["verify"]["of"] == of


async def test_an_agent_that_finishes_unchecked_is_asked_once(cfg, world, http, task):
    import asyncio

    from yuzuru.waits import Waiter

    from .test_stall_nudge import agent, job_ends, reply, turns

    out, go = agent(cfg, task, nudges=0)  # no reply in the fake agent's next turn: not a stall to continue here
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    of = trained(task)
    await job_ends(broker, waiter, task, rc=0)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router, "tool_calls")  # wrote RESULTS.md...
    reply(router)  # ...and finished, the weights unchecked
    go.touch()
    await wait_for(lambda: len(turns(out)) == 2)
    message = turns(out)[1][1]
    assert of in message and f'verify \\"{of}\\"' not in message and f'verify "{of}"' in message
    assert "have not been checked" in message
    reply(router, "tool_calls")
    reply(router)  # finished again without checking: not asked twice
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 2 and waiter.verify_nudged["s1"][of] == 1
    assert [e["kind"] for e in runs.journal(task)] == ["verify_nudge"]


async def test_a_checked_best_job_needs_no_reminder(cfg, world, http, task):
    import asyncio

    from yuzuru.waits import Waiter

    from .test_stall_nudge import agent, job_ends, reply, turns

    out, go = agent(cfg, task)
    catalog, broker, router = await make(cfg, world, http)
    waiter = Waiter(cfg, broker, router)
    verify(task, trained(task), "va_acc=78.70\n")
    await job_ends(broker, waiter, task, rc=0)
    await wait_for(lambda: len(turns(out)) == 1)
    reply(router, "tool_calls")
    reply(router)
    go.touch()
    await asyncio.sleep(1.5)
    assert len(turns(out)) == 1
