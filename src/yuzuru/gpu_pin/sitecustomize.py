"""Keeps a `yuzuru run` job's Python on the job's GPUs.

yuzuru puts this directory first on the job's PYTHONPATH, so every Python the job starts
imports it at startup, and passes the job's CUDA_VISIBLE_DEVICES in YUZURU_CUDA_VISIBLE_DEVICES.
CUDA reads CUDA_VISIBLE_DEVICES once, when it initializes; a script that sets it to other GPUs
before that would quietly use GPUs that another job or the LLM holds, or that `resources.gpus`
excludes. Here that raises instead. Narrowing to some of the job's GPUs, or hiding them all
("-1", ""), is fine. Deleting it would expose every GPU, so a delete leaves it in place:
accelerate's clear_environment() (used by the HF Trainer) clears and restores the whole
environment and must keep working.

A number is the GPU's nvidia-smi index, whatever CUDA_DEVICE_ORDER says, so
`CUDA_VISIBLE_DEVICES=1` (or `accelerate launch --gpu_ids 1`) picks the same GPU with or without
this pin. An allowed number is replaced by its UUID, which CUDA reads the same in either order.
Indices come from NVML, or from YUZURU_GPU_INDEX ("0=GPU-...,1=GPU-...") when that is set.
Nothing here depends on the rest of yuzuru: set PYTHONPATH to this directory and
YUZURU_CUDA_VISIBLE_DEVICES to the allowed GPUs to use it in any project.

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


_index = None  # nvidia-smi index -> UUID, read when a number first needs it


def _entries(value):
    return [e.strip() for e in _str(value).split(",")]


def _indices():
    global _index
    if _index is None:
        given = os.environ.get("YUZURU_GPU_INDEX")
        if given is not None:
            _index = dict(e.strip().split("=", 1) for e in given.split(",") if "=" in e)
        else:
            _index = _nvml_indices()
    return _index


def _nvml_indices():
    """NVML numbers GPUs as nvidia-smi does (PCI bus order) and ignores CUDA_VISIBLE_DEVICES."""
    import ctypes

    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return {}
    if nvml.nvmlInit_v2() != 0:
        return {}
    try:
        count, found = ctypes.c_uint(), {}
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return {}
        for i in range(count.value):
            handle, uuid = ctypes.c_void_p(), ctypes.create_string_buffer(96)
            if (nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) == 0
                    and nvml.nvmlDeviceGetUUID(handle, uuid, len(uuid)) == 0):
                found[str(i)] = uuid.value.decode()
        return found
    finally:
        nvml.nvmlShutdown()


def _uuid(entry):
    return _indices().get(entry) if entry.isdigit() else entry


def _allowed(value):
    """`value` as the job may keep it (numbers turned into UUIDs), or None if it leaves the job's GPUs."""
    s = _str(value).strip()
    if s in ("-1", ""):
        return s
    entries = _entries(s)
    pinned = set(_entries(_PINNED)) - {"", "-1"}
    if set(entries) <= pinned:  # the common case, and no NVML needed
        return s
    if not pinned:
        return None
    uuids = [_uuid(e) for e in entries]
    if None in uuids or not set(uuids) <= {_uuid(p) for p in pinned}:
        return None
    return ",".join(uuids)


def _check(value):
    kept = _allowed(value)
    if kept is None and _AGENT:
        raise RuntimeError(f"{_KEY} cannot be set to {_str(value)!r} here. {_ADVICE}")
    if kept is None:
        raise RuntimeError(
            f"{_KEY} is limited to this yuzuru job's GPUs ({_PINNED}); setting it to {_str(value)!r} "
            "would use GPUs yuzuru did not give the job")
    return kept


def _pin():
    kept = _allowed(os.environ[_KEY]) if _KEY in os.environ else None
    if kept is None:
        if _KEY in os.environ and _AGENT:  # e.g. `CUDA_VISIBLE_DEVICES=0 python train.py` from the agent's shell
            print(f"[yuzuru] {_KEY}={os.environ[_KEY]} ignored. {_ADVICE}", file=sys.stderr)
        elif _KEY in os.environ:  # e.g. `CUDA_VISIBLE_DEVICES=2 python train.py` inside the job
            print(f"[yuzuru] {_KEY}={os.environ[_KEY]} is outside this job's GPUs; using {_PINNED}",
                  file=sys.stderr)
        kept = _PINNED
    os.environ[_KEY] = kept  # e.g. `accelerate launch --gpu_ids 1` hands its children "1": the UUID now

    environ = type(os.environ)
    setitem, delitem = environ.__setitem__, environ.__delitem__
    putenv, unsetenv = os.putenv, os.unsetenv

    def __setitem__(self, key, value):
        if _str(key) == _KEY:
            value = _check(value)
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
            value = _check(value)
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
