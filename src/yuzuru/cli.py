"""Command line: `yuzuru serve`, `yuzuru run -- python train.py`, `yuzuru status`, ..."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from .config import GB, MB, load_config

UNITS = {"": GB, "b": 1, "k": 1024, "m": MB, "g": GB, "t": 1024 * GB}


def parse_size(text: str | None) -> int | None:
    """'10G', '512M', '12gb', '8' (GiB) -> bytes."""
    if not text:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*([bkmgt]?)i?b?\s*", text.lower())
    if not m:
        raise SystemExit(f"bad size: {text!r}")
    return int(float(m.group(1)) * UNITS[m.group(2)])


def gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


class Client:
    def __init__(self, url: str, token: str | None) -> None:
        self.url = url.rstrip("/")
        self.headers = {"authorization": f"Bearer {token}"} if token else {}

    def call(self, method: str, path: str, payload: Any = None, timeout: float = 30) -> Any:
        r = httpx.request(method, self.url + path, json=payload, headers=self.headers, timeout=timeout)
        if r.status_code >= 400:
            raise SystemExit(f"yuzuru: {r.status_code} {r.text}")
        return r.json()


# ---------------------------------------------------------------------- serve


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    cfg = load_config(args.config)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if cfg.host not in ("127.0.0.1", "localhost", "::1") and not cfg.token:
        print("yuzuru: warning: listening beyond localhost without a token (set YUZURU_TOKEN)", file=sys.stderr)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level=args.log_level)


# ------------------------------------------------------------------------ run


def cmd_run(args: argparse.Namespace, client: Client) -> None:
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        raise SystemExit("usage: yuzuru run [options] -- command ...")

    if args.detach:
        log = Path(args.log or f".yuzuru/logs/run-{os.getpid()}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = [a for a in sys.argv[1:] if a != "--detach"]
        with log.open("ab") as f:
            proc = subprocess.Popen(
                [sys.executable, "-m", "yuzuru.cli", *argv],
                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            )
        print(json.dumps({"pid": proc.pid, "log": str(log.resolve())}))
        return

    env = os.environ.copy()
    lease_id = None
    try:
        lease = client.call(
            "POST", "/yuzuru/leases",
            {"pid": os.getpid(), "label": args.label or Path(cmd[0]).name, "gpus": args.gpu or [],
             "vram": parse_size(args.vram), "session": args.session or os.environ.get("YUZURU_SESSION")},
            timeout=300,
        )
        lease_id = lease["id"]
        if lease["gpus"]:
            # UUIDs are unambiguous regardless of CUDA_DEVICE_ORDER.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(lease["gpus"])
        env["YUZURU_LEASE"] = lease_id
        print(f"[yuzuru] lease {lease_id}: LLM moved out of GPU memory", file=sys.stderr)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(f"[yuzuru] server unreachable ({e.__class__.__name__}); running without a lease", file=sys.stderr)

    out = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.log, "ab")
    proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None)

    def forward(signum: int, _frame: Any) -> None:
        proc.send_signal(signum)

    # Ctrl-C reaches the child through the terminal's process group already.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, forward)
    try:
        rc = proc.wait()
    finally:
        if lease_id:
            try:
                client.call("DELETE", f"/yuzuru/leases/{lease_id}")
            except (httpx.HTTPError, SystemExit):
                pass  # the server also reaps leases whose pid is gone
        if out:
            out.close()
    sys.exit(rc)


# --------------------------------------------------------------------- status


def print_status(s: dict[str, Any]) -> None:
    for w in s.get("warnings", []):
        print(f"! {w}")
    for g in s["gpus"]:
        print(f"GPU{g['index']} {g['name']:<28} used {gb(g['used']):>8} / {gb(g['total'])}")
    print(f"RAM {'':<28} avail {gb(s['ram']['available']):>7} / {gb(s['ram']['total'])}")
    b = s["budget"]
    pool = f"{gb(b['fast'])} (unified memory)" if b["unified"] else f"GPU {gb(b['fast'])} + RAM {gb(b['slow'])}"
    print(f"LLM may use: {pool}")
    print("\nTiers:")
    for name, t in s["tiers"].items():
        est = gb(t["estimated_size"]) + ("" if t["measured"] else "*")
        print(f"  {name:<7} {t['model']:<28} ctx {t['num_ctx']:>6}  {est:>9}  -> {t['placement']}")
    if s.get("classifier"):
        print(f"  router  {s['classifier']}")
    print("  (* = estimate until first load)")
    if s["loaded"]:
        print("\nLoaded:")
        for m in s["loaded"]:
            print(f"  {m['name']:<44} {gb(m['size_vram'])} GPU + {gb(m['size'] - m['size_vram'])} RAM"
                  + (f"  [{m['inflight']} in flight]" if m["inflight"] else ""))
    if s["leases"]:
        print("\nLeases:")
        for l in s["leases"]:
            where = ",".join(g[:12] for g in l["gpus"]) or "any GPU"
            print(f"  {l['id']} {l['label']:<20} pid {l['pid']:<8} {l['state']:<9} +{gb(l['grown'])} on {where}")


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yuzuru", description="resource-aware local LLM orchestrator")
    ap.add_argument("--config", help="config file (default: ./yuzuru.yaml or ~/.config/yuzuru/config.yaml)")
    ap.add_argument("--url", default=os.environ.get("YUZURU_URL"), help="server URL for client commands")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the proxy")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--log-level", default="info")

    p = sub.add_parser("run", help="run a training job under a GPU lease")
    p.add_argument("--gpu", action="append", help="GPU index or UUID to dedicate (repeatable)")
    p.add_argument("--vram", help="VRAM to reserve, e.g. 10G (default: measure)")
    p.add_argument("--label")
    p.add_argument("--log", help="write the job's output to this file")
    p.add_argument("--detach", action="store_true", help="run in the background; print pid and log path")
    p.add_argument("--session", help=argparse.SUPPRESS)  # OpenCode session that started the job
    p.add_argument("cmd", nargs=argparse.REMAINDER)

    p = sub.add_parser("status", help="show memory, tiers, models, leases")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("route", help="dry-run the router on a message")
    p.add_argument("text")

    p = sub.add_parser("warm", help="load a tier now")
    p.add_argument("tier")
    p.add_argument("--force", action="store_true", help="load even if it does not fit the budget")

    p = sub.add_parser("unload", help="unload a tier (default: all)")
    p.add_argument("tier", nargs="?", default="all")

    p = sub.add_parser("pin", help="pin an OpenCode session to a tier ('auto' to unpin)")
    p.add_argument("session")
    p.add_argument("tier")

    p = sub.add_parser("opencode", help="print or install the OpenCode provider + plugin")
    p.add_argument("--install", action="store_true", help="write into the OpenCode config directory")
    p.add_argument("--project", help="install into this project instead of the global config")
    p.add_argument("--set-default", action="store_true", help="make yuzuru/auto the default model")

    args = ap.parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)

    cfg = load_config(args.config)
    client = Client(args.url or cfg.url, cfg.token)

    if args.command == "run":
        return cmd_run(args, client)
    if args.command == "opencode":
        from .opencode import cmd_opencode

        return cmd_opencode(args, cfg)

    try:
        if args.command == "status":
            s = client.call("GET", "/yuzuru/status")
            print(json.dumps(s, indent=1)) if args.json else print_status(s)
        elif args.command == "route":
            print(json.dumps(client.call("POST", "/yuzuru/route", {"text": args.text}, timeout=120), indent=1, ensure_ascii=False))
        elif args.command == "warm":
            print(json.dumps(client.call("POST", "/yuzuru/warm", {"tier": args.tier, "force": args.force}, timeout=600)))
        elif args.command == "unload":
            print(json.dumps(client.call("POST", "/yuzuru/unload", {"tier": args.tier}, timeout=120)))
        elif args.command == "pin":
            print(json.dumps(client.call("POST", f"/yuzuru/sessions/{args.session}/pin", {"tier": args.tier})))
    except httpx.ConnectError:
        raise SystemExit(f"yuzuru: server not running at {client.url} (start it with `yuzuru serve`)")


if __name__ == "__main__":
    main()
