"""Minimal MCP server (Streamable HTTP, JSON responses only) exposing yuzuru tools.

Tool names are unprefixed; clients namespace them by server (OpenCode shows
`yuzuru_status`, `yuzuru_train`, ...).

Works with any MCP client; OpenCode picks it up from the `mcp` section that
`yuzuru opencode --install` writes. Tool results are plain text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from . import __version__, scope
from .waits import outside_env

log = logging.getLogger("yuzuru.mcp")

PROTOCOL_VERSION = "2025-06-18"

INSTRUCTIONS = (
    "yuzuru serves this assistant from local models sized to the free GPU memory. "
    "Call this server's `model` tool with 'large' when a task needs deeper reasoning than the current model gives. "
    "Start GPU training / fine-tuning with its `train` tool rather than bash: it moves the LLM out of VRAM first "
    "and runs the job in the background with a log file. For a job that uses no GPU (LightGBM, XGBoost on CPU, "
    "scikit-learn, data preparation) pass device 'cpu': the LLM then stays loaded."
)

Handler = Callable[[dict[str, Any], dict[str, str]], Awaitable[str]]


def _device(args: dict[str, Any]) -> str:
    device = (args.get("device") or "gpu").lower()
    if device not in ("gpu", "cpu"):
        raise ValueError(f"device must be 'gpu' or 'cpu', not {device!r}")
    return device


def gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


def format_status(s: dict[str, Any]) -> str:
    lines = [f"warning: {w}" for w in s.get("warnings", [])]
    lines += [f"GPU{g['index']} {g['name']}: {gb(g['used'])} / {gb(g['total'])} used" for g in s["gpus"]]
    lines.append(f"RAM: {gb(s['ram']['available'])} free of {gb(s['ram']['total'])}")
    b = s["budget"]
    lines.append(f"LLM budget: {gb(b['fast'])} (unified memory)" if b["unified"] else
                 f"LLM budget: GPU {gb(b['fast'])} + RAM {gb(b['slow'])}")
    lines.append("Tiers (placement if loaded now):")
    for name, t in s["tiers"].items():
        lines.append(f"  {name}: {t['model']}, ctx {t['num_ctx']}, ~{gb(t['estimated_size'])} -> {t['placement']}")
    if s["loaded"]:
        lines.append("Loaded: " + ", ".join(f"{m['name']} ({gb(m['size_vram'])} GPU)" for m in s["loaded"]))
    for l in s["leases"]:
        use = "CPU only" if l.get("cpu") else f"using +{gb(l['grown'])}"
        lines.append(f"Lease {l['id']}: {l['label']} pid {l['pid']} {l['state']}, {use}")
    return "\n".join(lines)


class MCPServer:
    def __init__(self, state: Any) -> None:
        self.st = state
        self.tools: dict[str, tuple[dict[str, Any], Handler]] = {}
        self._register()

    def tool(self, name: str, description: str, properties: dict[str, Any], required: list[str] | None = None):
        def deco(fn: Handler) -> Handler:
            schema = {"type": "object", "properties": properties, "required": required or []}
            self.tools[name] = ({"name": name, "description": description, "inputSchema": schema}, fn)
            return fn

        return deco

    def _register(self) -> None:
        st = self.st

        @self.tool("status", "Show GPU/RAM usage, which local model sizes fit right now, loaded models and "
                   "running training jobs (leases).", {})
        async def status(args: dict[str, Any], headers: dict[str, str]) -> str:
            await st.broker.refresh()
            return format_status(st.broker.status())

        tiers_desc = ("auto | " + " | ".join(t.name for t in st.cfg.tiers)
                      + (" | cloud (paid API, for problems local models cannot solve)" if st.cfg.cloud.usable else "")
                      + " | an installed model name")

        @self.tool("models", "List the local models installed in Ollama, which tier each serves, and whether "
                   "each is enabled.", {})
        async def models(args: dict[str, Any], headers: dict[str, str]) -> str:
            lines = []
            for m in st.catalog.models_info():
                if m["embedding"]:
                    continue
                flags = [f"tier={m['tier']}" if m["tier"] else "", "disabled" if not m["enabled"] else "",
                         "moe" if m["moe"] else ""]
                lines.append(f"{m['name']} ({m['params_b']:.1f}B, {gb(m['size'])}) " + " ".join(f for f in flags if f))
            return "\n".join(lines) or "no models installed"

        @self.tool(
            "model",
            "Choose the local model for the rest of this session, from the next model call on: a size tier "
            "('large' before hard multi-step work such as architecture, subtle debugging or math; the smallest tier "
            "for trivial chores), a specific installed model name (see the `models` tool), or 'auto' to let the router "
            "decide per message again.",
            {"tier": {"type": "string", "description": tiers_desc},
             "session": {"type": "string", "description": "session id (optional; defaults to the active session)"}},
            ["tier"],
        )
        async def model(args: dict[str, Any], headers: dict[str, str]) -> str:
            session = args.get("session") or st.router.header_session(headers) or st.router.last_session
            if not session:
                raise ValueError("no active session to pin")
            tier = args["tier"]
            st.router.pin(session, None if tier in ("", "auto") else tier)
            return f"model for session {session}: {st.router.sessions[session].pinned or 'auto'}"

        @self.tool(
            "train",
            "Start a training / fine-tuning job in the background. For a GPU job (the default) yuzuru first moves "
            "the local LLM out of GPU memory, waits for the job's memory use to settle, then lets the LLM use what "
            "remains. With device 'cpu' the job sees no GPU, the LLM is left alone and the job yields CPU to it. "
            "Returns the PID and log file; read the log to follow progress.",
            {
                "command": {"type": "string", "description": "shell command, e.g. 'python train.py --epochs 10'"},
                "cwd": {"type": "string", "description": "absolute working directory (the project root)"},
                "device": {"type": "string", "enum": ["gpu", "cpu"],
                           "description": "'cpu' for a job that uses no GPU (default 'gpu')"},
                "gpus": {"type": "string", "description": "comma-separated GPU indices/UUIDs to dedicate (optional)"},
                "vram": {"type": "string", "description": "VRAM to reserve, e.g. '10G' (optional; default measured)"},
                "log": {"type": "string", "description": "log file (optional; default .yuzuru/logs/train-<pid>.log)"},
                "resume_when_done": {"type": "boolean", "description": "free the LLM while the job runs and "
                                     "resume this session when it ends (default true); end your turn after starting"},
                "note": {"type": "string", "description": "what you will do when the job ends (handed back to you)"},
            },
            ["command", "cwd"],
        )
        async def train(args: dict[str, Any], headers: dict[str, str]) -> str:
            cwd = Path(args["cwd"]).expanduser()
            if not cwd.is_dir():
                raise ValueError(f"cwd does not exist: {cwd}")
            cpu = _device(args) == "cpu"
            if cpu and (args.get("gpus") or args.get("vram")):
                raise ValueError("device 'cpu' takes no gpus or vram")
            # The job runs in `cwd`: a relative config path (YUZURU_CONFIG, ./yuzuru.yaml) would not resolve there.
            argv = [sys.executable, "-m", "yuzuru.cli", *(["--config", str(st.cfg.source)] if st.cfg.source else []),
                    "--url", st.cfg.url, "run", "--detach", "--label", "train"]
            if args.get("log"):
                argv += ["--log", args["log"]]
            if cpu:
                argv.append("--cpu")
            for g in filter(None, (x.strip() for x in (args.get("gpus") or "").split(","))):
                argv += ["--gpu", g]
            if args.get("vram"):
                argv += ["--vram", args["vram"]]
            session = st.router.header_session(headers) or st.router.last_session
            resume = args.get("resume_when_done", True) and session is not None
            if session:
                argv += ["--session", session]
            if resume:
                argv += ["--resume", "--note", args.get("note") or ""]
            argv += ["--", "bash", "-c", args["command"]]
            if await asyncio.to_thread(scope.use, st.cfg.isolation.scope):
                # Not in yuzuru's own unit: a job that runs out of memory is killed alone (its lease
                # ends with 137 and the agent is resumed with that) instead of systemd stopping yuzuru.
                argv = scope.wrap(argv, scope.unit_name("job"), st.cfg.isolation.memory_max, oom_continue=True,
                                  cpu_weight=st.cfg.isolation.cpu_job_weight if cpu else None)
            r = await asyncio.to_thread(subprocess.run, argv, cwd=cwd, env=outside_env(), capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise RuntimeError(r.stderr or r.stdout)
            info = json.loads(r.stdout)
            text = f"started pid {info['pid']}\nlog: {info['log']}\nfollow with: tail -n 50 {info['log']}"
            if resume:
                text += ("\n" + ("The local LLM stays loaded (CPU-only job)." if cpu else
                                 "The local LLM is released while the job runs.") +
                         " End your turn now: this session is resumed automatically with the exit status and "
                         "the log tail when the job ends.")
            return text

        @self.tool(
            "wait",
            "Wait for a running process to finish without holding GPU/RAM: yuzuru unloads the local LLM, then "
            "resumes this session with the outcome (exit status, log tail, your note) when the process exits. "
            "Use it instead of polling or sleeping. End your turn right after calling it.",
            {
                "pid": {"type": "integer", "description": "process id to wait for"},
                "lease": {"type": "string", "description": "or a yuzuru lease id (from the `status` tool)"},
                "note": {"type": "string", "description": "what you will do when it ends (handed back to you)"},
                "log": {"type": "string", "description": "log file whose last lines to hand back (optional)"},
                "cwd": {"type": "string", "description": "working directory for resolving the log path (optional; default the process's own)"},
                "device": {"type": "string", "enum": ["gpu", "cpu"],
                           "description": "'cpu' if the process uses no GPU: the LLM is not unloaded (default 'gpu')"},
            },
        )
        async def wait(args: dict[str, Any], headers: dict[str, str]) -> str:
            session = st.router.header_session(headers) or st.router.last_session
            if not (args.get("pid") or args.get("lease")):
                raise ValueError("give pid or lease")
            w = await st.waiter.add(
                session=session, pid=int(args["pid"]) if args.get("pid") else None, lease=args.get("lease"),
                label=f"pid {args['pid']}" if args.get("pid") else f"lease {args['lease']}",
                note=args.get("note") or "", log=args.get("log"), cwd=args.get("cwd"),
                suspend=_device(args) != "cpu",
            )
            return (f"waiting ({w.id}); the local LLM is {'unloaded' if w.suspend else 'left loaded'}. End your "
                    f"turn now: session {session} is resumed automatically when the process exits.")

    async def handle(self, request: Request) -> Response:
        try:
            msg = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(_error(None, -32700, "parse error"), status_code=400)
        headers = {k.lower(): v for k, v in request.headers.items()}
        if isinstance(msg, list):
            out = [r for m in msg if (r := await self._dispatch(m, headers)) is not None]
            return JSONResponse(out) if out else Response(status_code=202)
        result = await self._dispatch(msg, headers)
        return JSONResponse(result) if result is not None else Response(status_code=202)

    async def _dispatch(self, msg: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if mid is None:  # notification
            return None
        if method == "initialize":
            return _ok(mid, {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "yuzuru", "version": __version__},
                "instructions": INSTRUCTIONS,
            })
        if method == "ping":
            return _ok(mid, {})
        if method == "tools/list":
            return _ok(mid, {"tools": [spec for spec, _ in self.tools.values()]})
        if method == "tools/call":
            name = params.get("name")
            if name not in self.tools:
                return _error(mid, -32602, f"unknown tool {name!r}")
            try:
                text = await self.tools[name][1](params.get("arguments") or {}, headers)
                return _ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})
            except Exception as e:  # noqa: BLE001 - report tool errors to the model
                log.warning("tool %s failed: %s", name, e)
                return _ok(mid, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        return _error(mid, -32601, f"method not found: {method}")


def _ok(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}
