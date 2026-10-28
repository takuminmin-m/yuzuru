"""Keeps a `yuzuru run` job's Python on the job's GPUs.

yuzuru puts this directory first on the job's PYTHONPATH, so every Python the job starts
imports it at startup, and passes the job's CUDA_VISIBLE_DEVICES in YUZURU_CUDA_VISIBLE_DEVICES.
CUDA reads CUDA_VISIBLE_DEVICES once, when it initializes; a script that sets it to other GPUs
before that would quietly use GPUs that another job or the LLM holds, or that `resources.gpus`
excludes. Here that raises instead. Narrowing to some of the job's GPUs, or hiding them all
("-1", ""), is fine. Deleting it would expose every GPU, so a delete leaves it in place:
accelerate's clear_environment() (used by the HF Trainer) clears and restores the whole
environment and must keep working.

`yuzuru agent` uses the same pin with no GPUs (YUZURU_AGENT=1): what the agent runs itself, such
as `python train.py` from its shell, sees no GPU and is told to use yuzuru's train tool instead.

Then the sitecustomize this one shadows, if any (a venv's or the distribution's), runs as usual.
"""

import os
import sys

_KEY = "CUDA_VISIBLE_DEVICES"
_PINNED = os.environ.get("YUZURU_CUDA_VISIBLE_DEVICES")
_AGENT = os.environ.get("YUZURU_AGENT") == "1"
_ADVICE = ("GPUs are hidden from commands the agent runs itself: start GPU work with yuzuru's train "
           "tool (MCP yuzuru_train) and give it the VRAM the job needs")


def _str(s):
    return s.decode() if isinstance(s, bytes) else str(s)


def _entries(value):
    return {e.strip() for e in _str(value).split(",")}


def _allowed(value):
    return _str(value).strip() in ("-1", "") or _entries(value) <= _entries(_PINNED)


def _check(value):
    if not _allowed(value) and _AGENT:
        raise RuntimeError(f"{_KEY} cannot be set to {_str(value)!r} here. {_ADVICE}")
    if not _allowed(value):
        raise RuntimeError(
            f"{_KEY} is limited to this yuzuru job's GPUs ({_PINNED}); setting it to {_str(value)!r} "
            "would use GPUs yuzuru did not give the job")


def _pin():
    if _KEY not in os.environ or not _allowed(os.environ[_KEY]):
        if _KEY in os.environ and _AGENT:  # e.g. `CUDA_VISIBLE_DEVICES=0 python train.py` from the agent's shell
            print(f"[yuzuru] {_KEY}={os.environ[_KEY]} ignored. {_ADVICE}", file=sys.stderr)
        elif _KEY in os.environ:  # e.g. `CUDA_VISIBLE_DEVICES=1 python train.py` inside the job
            print(f"[yuzuru] {_KEY}={os.environ[_KEY]} is outside this job's GPUs; using {_PINNED}",
                  file=sys.stderr)
        os.environ[_KEY] = _PINNED

    environ = type(os.environ)
    setitem, delitem = environ.__setitem__, environ.__delitem__
    putenv, unsetenv = os.putenv, os.unsetenv

    def __setitem__(self, key, value):
        if _str(key) == _KEY:
            _check(value)
        setitem(self, key, value)

    def __delitem__(self, key):
        if _str(key) != _KEY:
            delitem(self, key)

    def popitem(self):
        # MutableMapping.clear() pops until KeyError; skip the pinned key or it would never end.
        for key in self:
            if _str(key) != _KEY:
                value = self[key]
                delitem(self, key)
                return key, value
        raise KeyError

    def checked_putenv(key, value):
        if _str(key) == _KEY:
            _check(value)
        putenv(key, value)

    def checked_unsetenv(key):
        if _str(key) != _KEY:
            unsetenv(key)

    environ.__setitem__, environ.__delitem__, environ.popitem = __setitem__, __delitem__, popitem
    os.putenv, os.unsetenv = checked_putenv, checked_unsetenv


def _hint_on_cuda_errors():
    """A script that wants CUDA fails without saying why there is none; say it after the traceback."""
    previous = sys.excepthook

    def hook(kind, value, tb):
        previous(kind, value, tb)
        if any(w in f"{kind.__name__}: {value}" for w in ("CUDA", "cuda", "NVIDIA", "GPU")):
            print(f"[yuzuru] {_ADVICE}", file=sys.stderr)

    sys.excepthook = hook


def _chain():
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


if _PINNED is not None:
    _pin()
    if _AGENT:
        _hint_on_cuda_errors()
_chain()
