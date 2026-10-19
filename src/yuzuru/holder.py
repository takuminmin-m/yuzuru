"""VRAM holder: physically occupy GPU memory so nothing else (Ollama included) can take it.

Run as `python -m yuzuru.holder --bytes N1[,N2...] [--ttl SECONDS]` with
CUDA_VISIBLE_DEVICES set to the target GPUs (one byte count per device). Talks to the driver API via
ctypes (libcuda ships with the NVIDIA driver), so no CUDA toolkit or torch is
needed. Prints one JSON line when ready, then holds until SIGTERM or the TTL.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import signal
import sys
import time

CHUNK = 256 * 1024 * 1024


class HolderError(RuntimeError):
    pass


def _libcuda() -> ctypes.CDLL:
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise HolderError("libcuda not found (NVIDIA driver required for a hard reservation)")


def hold(sizes: list[int]) -> tuple[ctypes.CDLL, list[ctypes.c_uint64]]:
    """Allocate sizes[i] bytes on visible device i."""
    cu = _libcuda()

    def check(rc: int, what: str) -> None:
        if rc != 0:
            raise HolderError(f"{what} failed with CUDA error {rc}")

    check(cu.cuInit(0), "cuInit")
    ptrs: list[ctypes.c_uint64] = []
    for index, nbytes in enumerate(sizes):
        dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        check(cu.cuDeviceGet(ctypes.byref(dev), index), "cuDeviceGet")
        check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")  # becomes current
        left = nbytes
        while left > 0:
            size = min(CHUNK, left)
            p = ctypes.c_uint64()
            rc = cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(size))
            if rc != 0:
                got = (nbytes - left) / 2**30
                raise HolderError(f"device {index}: could only reserve {got:.1f} of {nbytes / 2**30:.1f} GB (CUDA error {rc})")
            ptrs.append(p)
            left -= size
    return cu, ptrs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yuzuru.holder")
    ap.add_argument("--bytes", required=True, help="comma-separated byte counts, one per visible device")
    ap.add_argument("--ttl", type=float, default=0, help="seconds; 0 = until terminated")
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.bytes.split(",")]
    try:
        _keep = hold(sizes)
    except HolderError as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
        sys.exit(2)
    print(json.dumps({"ok": True, "bytes": sizes}), flush=True)

    stop = False

    def on_term(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    deadline = time.time() + args.ttl if args.ttl > 0 else float("inf")
    while not stop and time.time() < deadline:
        time.sleep(0.5)
    # Exiting releases the context and every allocation with it.


if __name__ == "__main__":
    main()
