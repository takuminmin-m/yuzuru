"""The validation metric of a job, read from its log.

yuzuru does not know what a job trains; its log is what there is. Agents print one line per epoch with
`name=value` pairs, under names of their own (`va_acc=45.48%`, `val_acc=0.5474`, `val=0.6464`,
`[val] accuracy=74.94%`). The pair that looks most like a validation metric is taken: the report compares
jobs by it, and `verify` compares saved weights with it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PAIR = re.compile(r"([A-Za-z][\w./-]*)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w.])\s*(%?)")
TAG = re.compile(r"^\s*\[([A-Za-z_]+)\]")
VAL = re.compile(r"(?:^|[^a-z])(?:val|valid|validation|va|eval|dev)(?:[^a-z]|$)")
LOWER = re.compile(r"loss|err|mae|mse|rmse|mape|ppl|perplexity|wer|cer")
BEST = re.compile(r"(?:^|[^a-z])best(?:[^a-z]|$)")
COUNT = re.compile(r"(?:^|[^a-z])(?:n|num|len|size|samples|count|batches|steps|time|sec|seconds)(?:[^a-z]|$)")  # n_val=5000


def lower_is_better(name: str) -> bool:
    return bool(LOWER.search(name.lower()))


def series(text: str) -> dict[str, list[float]]:
    """Every `name=value` of `text`, in order, by name. A pair on a line tagged `[val]` is `val_<name>`."""
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        tag = TAG.match(line)
        prefix = tag.group(1).lower() if tag and VAL.search(tag.group(1).lower()) else ""
        for name, value, _ in PAIR.findall(line):
            name = name.lower()
            if prefix and not VAL.search(name):
                name = f"{prefix}_{name}"
            out.setdefault(name, []).append(float(value))
    return out


def choose(found: dict[str, list[float]]) -> str | None:
    """The name that is the validation metric: a score before a loss, one value per epoch before a running
    best, then the one printed most often."""
    names = [n for n in found if VAL.search(n) and not COUNT.search(n)]
    if not names:
        return None
    return min(names, key=lambda n: (lower_is_better(n), bool(BEST.search(n)), -len(found[n])))


def summarize(name: str, values: list[float], lower: bool | None = None) -> dict[str, Any]:
    lower = lower_is_better(name) if lower is None else lower
    best = min(values) if lower else max(values)
    return {"name": name, "lower": lower, "best": best, "last": values[-1], "n": len(values),
            "best_at": values.index(best) + 1}


def read_text(path: str | Path | None, offset: int = 0, end: int | None = None) -> str:
    """What was written to the log from `offset` on, up to `end` (jobs given the same log file append to it)."""
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return (f.read() if end is None else f.read(max(end - offset, 0))).decode(errors="replace")
    except OSError:
        return ""


def read(path: str | Path | None, offset: int = 0, name: str | None = None,
         lower: bool | None = None, end: int | None = None) -> dict[str, Any] | None:
    """The validation metric in a log: {"name", "lower", "best", "last", "n", "best_at"}, or None.

    `name` picks the pair to read instead of guessing it.
    """
    found = series(read_text(path, offset, end))
    name = name.lower() if name else choose(found)
    if not name or name not in found:
        return None
    return summarize(name, found[name], lower)


def same_scale(a: float, b: float) -> tuple[float, float]:
    """`a` and `b` on one scale: 0.7874 next to 78.74 is a fraction next to a percentage."""
    if 0 <= a <= 1 < b <= 100:
        return a * 100, b
    if 0 <= b <= 1 < a <= 100:
        return a, b * 100
    return a, b
