"""OpenAI-compatible proxy in front of Ollama, plus the control API.

  /v1/chat/completions   model = auto | <tier>  -> routed, memory-managed
  /v1/*                  passed through to Ollama
  /yuzuru/*              status, leases, pins, warm/unload
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import logging
import secrets
import uuid
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .broker import Broker, LoadFailed, Placement
from .catalog import Catalog
from .mcp import MCPServer
from .config import Config
from .ollama import Ollama
from .managed import ManagedOllama
from .resources import Probe, make_probe
from .router import Router
from .waits import Waiter

log = logging.getLogger("yuzuru.server")

KEEPALIVE_SECONDS = 10.0  # SSE comment interval while waiting for the first token

HOP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}


class State:
    def __init__(self, cfg: Config, probe: Probe | None = None, http: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.http = http or httpx.AsyncClient(base_url=cfg.ollama_url, timeout=httpx.Timeout(None, connect=5))
        self.ollama = Ollama(cfg.ollama_url, self.http)
        self.catalog = Catalog(cfg, self.ollama)
        self.managed: ManagedOllama | None = None
        if probe is None:
            probe = make_probe(cfg.resources.probe, cfg.resources.ollama_cgroup, cfg.resources.gpus)
            if cfg.ollama.manage:
                probe.host.pid = lambda: self.managed.pid if self.managed else None
        if cfg.ollama.manage:
            self.managed = ManagedOllama(cfg, lambda: [g.uuid for g in probe.snapshot().gpus], self._ollama_moved)
            self.catalog.create_variant = functools.partial(self.managed.create_variant, self.ollama)
        self.broker = Broker(cfg, probe, self.ollama, self.catalog)
        self.router = Router(cfg, self.catalog, self.broker, self.ollama)
        self.cloud_http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10))
        self.waiter = Waiter(cfg, self.broker, self.router)
        self.tasks: list[asyncio.Task] = []

    def _ollama_moved(self, url: str) -> None:
        """The managed server picked its port (`ollama.port: auto`) only now."""
        self.http.base_url = url
        self.ollama.base_url = url

    async def start(self) -> None:
        if self.managed:
            await self.managed.start()
        # Come up even if Ollama (or the GPU driver) is not ready yet, as at boot
        # when both are services; the loops below keep retrying.
        try:
            await self.catalog.refresh()
        except Exception as e:  # noqa: BLE001
            log.warning("Ollama not reachable yet (%r); retrying in the background", e)
        try:
            await self.broker.refresh()
        except Exception as e:  # noqa: BLE001
            log.warning("resource probe failed (%r); retrying in the background", e)
        self.tasks.append(asyncio.create_task(self.broker.monitor()))
        self.tasks.append(asyncio.create_task(self._refresh_catalog()))
        self.tasks.append(asyncio.create_task(self._poll_watches()))

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
        await self.broker.shutdown()
        if self.managed:
            await self.managed.stop()
        await self.http.aclose()
        await self.cloud_http.aclose()

    async def _poll_watches(self) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                await self.waiter.poll()
            except Exception as e:  # noqa: BLE001
                log.warning("watch poll: %r", e)

    async def _refresh_catalog(self) -> None:
        """Pick up models pulled/removed while running."""
        while True:
            await asyncio.sleep(60 if self.catalog.specs else 5)
            try:
                before = {t: s.base for t, s in self.catalog.specs.items()}
                await self.catalog.refresh()
                if before != {t: s.base for t, s in self.catalog.specs.items()}:
                    log.info("tiers changed: %s", {t: s.base for t, s in self.catalog.specs.items()})
            except Exception as e:  # noqa: BLE001
                log.warning("catalog refresh: %r", e)


def create_app(cfg: Config, state: State | None = None) -> FastAPI:
    st = state or State(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await st.start()
        yield
        await st.stop()

    app = FastAPI(title="yuzuru", lifespan=lifespan)
    app.state.yuzuru = st

    def auth(request: Request) -> None:
        if not cfg.token:
            return
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given, cfg.token):
            raise HTTPException(401, "invalid token")

    guarded = [Depends(auth)]

    local_hosts = {"127.0.0.1", "localhost", "::1", cfg.host}

    @app.middleware("http")
    async def same_machine_clients_only(request: Request, call_next: Any) -> Response:
        """Keep web pages out: yuzuru can start processes (MCP train tool, leases).

        - Browsers attach Origin to cross-site requests; agent/CLI clients do not.
        - A Host other than ours means DNS rebinding (unless a token guards the API).
        - JSON-only POST bodies force a CORS preflight, which is never granted.
        """
        origin = request.headers.get("origin")
        if origin and origin not in cfg.allowed_origins:
            return JSONResponse({"error": "cross-origin requests are not allowed"}, status_code=403)
        host = request.headers.get("host", "")
        hostname = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
        if not cfg.token and hostname not in local_hosts:
            return JSONResponse({"error": f"unexpected Host {host!r}"}, status_code=403)
        if request.method in ("POST", "PUT", "PATCH") and request.headers.get("content-length", "0") != "0":
            if "application/json" not in request.headers.get("content-type", ""):
                return JSONResponse({"error": "Content-Type must be application/json"}, status_code=415)
        return await call_next(request)

    # ------------------------------------------------------------- OpenAI API

    @app.get("/v1/models", dependencies=guarded)
    async def models() -> dict[str, Any]:
        ids = ["auto"] + st.catalog.tiers + (["cloud"] if cfg.cloud.usable else []) + st.catalog.enabled
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "yuzuru"} for i in ids]}

    async def pick(body: dict[str, Any], headers: dict[str, str]) -> tuple[Any, Placement]:
        """Route and load the model (none for the cloud). Raises HTTPException.

        A load can fail despite the estimate; the failed model is then excluded and
        routing runs again, so the request falls back to a model that does load.
        """
        for _ in range(len(cfg.tiers) + 1):
            try:
                decision = await st.router.route(body, headers)
            except LookupError as e:
                raise HTTPException(404, str(e)) from e
            except RuntimeError as e:
                raise HTTPException(503, str(e)) from e
            if decision.cloud:
                return decision, Placement.NONE
            assert decision.spec is not None
            try:
                placement = await st.broker.acquire(decision.spec, force=True)
            except LoadFailed:
                continue
            if placement is not Placement.NONE:
                decision.placement = placement
            return decision, placement
        raise HTTPException(503, "no local model could be loaded")

    @app.post("/v1/chat/completions", dependencies=guarded)
    async def chat(request: Request) -> Response:
        body = await request.json()
        headers = {k.lower(): v for k, v in request.headers.items()}
        if os.environ.get("YUZURU_DUMP"):  # debugging aid: keep the last request as sent by the client
            (cfg.state_dir / f"last_request_{str(body.get('model', '')).split('/')[-1]}.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
        if body.get("stream"):
            return StreamingResponse(stream_chat(body, headers), media_type="text/event-stream",
                                     headers={"cache-control": "no-cache"})

        decision, _ = await pick(body, headers)
        if decision.cloud:
            return await forward_cloud(body, decision)
        spec = decision.spec
        try:
            upstream = await st.http.post("/v1/chat/completions", json={**body, "model": spec.variant})
        finally:
            await st.broker.release(spec)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=decision.headers(),
        )

    def sse_error(message: str) -> bytes:
        return b"data: " + json.dumps({"error": {"message": message, "type": "server_error"}}).encode() + b"\n\n"

    async def stream_chat(body: dict[str, Any], headers: dict[str, str]):
        """Stream a completion, sending headers at once.

        Loading a model or processing a long prompt partly on CPU can take minutes
        before the first token, and clients (OpenCode) give up when the response
        headers take that long. So headers go out immediately, SSE comments keep the
        connection alive meanwhile, and errors arrive as an SSE error event.
        """
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def produce() -> None:
            spec = None
            upstream = None
            try:
                decision, _ = await pick(body, headers)
                await queue.put(f": yuzuru {decision.tier} ({decision.reason})\n\n".encode())
                if decision.cloud:
                    c = cfg.cloud
                    log.info("cloud: %s (%s)", c.model, decision.reason)
                    req = st.cloud_http.build_request(
                        "POST", c.base_url.rstrip("/") + "/chat/completions", json={**body, "model": c.model},
                        headers={"authorization": f"Bearer {c.api_key}"},
                    )
                    upstream = await st.cloud_http.send(req, stream=True)
                else:
                    spec = decision.spec
                    req = st.http.build_request("POST", "/v1/chat/completions", json={**body, "model": spec.variant})
                    upstream = await st.http.send(req, stream=True)
                if upstream.status_code >= 400:
                    text = (await upstream.aread()).decode(errors="replace")
                    await queue.put(sse_error(f"upstream returned {upstream.status_code}: {text[:500]}"))
                    return
                async for chunk in upstream.aiter_bytes():
                    await queue.put(chunk)
            except HTTPException as e:
                await queue.put(sse_error(str(e.detail)))
            except Exception as e:  # noqa: BLE001 - report to the client instead of dropping the stream
                log.warning("stream failed: %r", e)
                await queue.put(sse_error(f"yuzuru: {e!r}"))
            finally:
                if upstream is not None:
                    await upstream.aclose()
                if spec is not None:
                    await st.broker.release(spec)
                await queue.put(None)

        task = asyncio.create_task(produce())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield item
        finally:
            if not task.done():  # client went away: stop generating and free the model
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def forward_cloud(body: dict[str, Any], decision: Any) -> Response:
        """Non-streaming cloud call (streaming ones go through stream_chat)."""
        c = cfg.cloud
        log.info("cloud: %s (%s)", c.model, decision.reason)
        upstream = await st.cloud_http.post(
            c.base_url.rstrip("/") + "/chat/completions", json={**body, "model": c.model},
            headers={"authorization": f"Bearer {c.api_key}"},
        )
        return Response(upstream.content, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type"), headers=decision.headers())

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"], dependencies=guarded)
    async def passthrough(path: str, request: Request) -> Response:
        upstream = await st.http.request(
            request.method, f"/v1/{path}", content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")},
        )
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS},
        )

    # -------------------------------------------------------------------- MCP

    mcp = MCPServer(st)

    @app.post("/mcp", dependencies=guarded)
    async def mcp_post(request: Request) -> Response:
        return await mcp.handle(request)

    @app.get("/mcp", dependencies=guarded)
    async def mcp_get() -> Response:
        return Response(status_code=405)  # no server-initiated stream

    # ----------------------------------------------------------- control API

    @app.get("/yuzuru/status", dependencies=guarded)
    async def status() -> dict[str, Any]:
        await st.broker.refresh()
        c = cfg.cloud
        cloud = {"enabled": c.enabled, "usable": c.usable, "model": c.model, "auto": c.auto,
                 "calls_last_hour": len(st.router.cloud_calls), "limit_per_hour": c.max_requests_per_hour}
        return st.broker.status() | {"sessions": st.router.sessions_status(), "cloud": cloud,
                                     "waits": st.waiter.status(),
                                     "ollama": st.managed.status() if st.managed else {"managed": False},
                                     "gpu_allowlist": cfg.resources.gpus}

    @app.post("/yuzuru/leases", dependencies=guarded)
    async def create_lease(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            lease = await st.broker.create_lease(
                pid=int(payload["pid"]),
                label=str(payload.get("label", "job")),
                gpus=[str(g) for g in payload.get("gpus") or []],
                vram=int(payload["vram"]) if payload.get("vram") else None,
                session=payload.get("session"),
                **job_fields(payload),
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        watch = st.waiter.watch_lease(lease)
        return {"id": lease.id, "gpus": lease.gpus, "visible": st.broker.visible_gpus(lease), "state": lease.state,
                "watch": watch.id if watch else None}

    def job_fields(payload: dict[str, Any]) -> dict[str, Any]:
        return {"cwd": payload.get("cwd"), "log": payload.get("log"), "resume": bool(payload.get("resume")),
                "note": str(payload.get("note") or "")}

    @app.post("/yuzuru/reservations", dependencies=guarded)
    async def reserve(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            lease = await st.broker.create_reservation(
                gpus=[str(g) for g in payload.get("gpus") or []],
                vram=int(payload["vram"]) if payload.get("vram") else None,
                ttl=float(payload["ttl"]) if payload.get("ttl") else None,
                label=str(payload.get("label", "reserved")),
                hard=bool(payload.get("hard", True)),
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"id": lease.id, "gpus": lease.gpus, "vram": lease.vram, "hard": lease.hard, "expires": lease.expires}

    @app.post("/yuzuru/leases/{lease_id}/handover", dependencies=guarded)
    async def handover(lease_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            lease = await st.broker.handover(lease_id, int(payload["pid"]), payload.get("session"))
        except KeyError as e:
            raise HTTPException(404, f"no reservation {lease_id}") from e
        for k, v in job_fields(payload).items():
            setattr(lease, k, v)
        watch = st.waiter.watch_lease(lease)
        return {"id": lease.id, "gpus": lease.gpus, "visible": st.broker.visible_gpus(lease), "state": lease.state,
                "watch": watch.id if watch else None}

    @app.get("/yuzuru/leases", dependencies=guarded)
    async def list_leases() -> list[dict[str, Any]]:
        return st.broker.status()["leases"]

    @app.delete("/yuzuru/leases/{lease_id}", dependencies=guarded)
    async def release_lease(lease_id: str, rc: int | None = None) -> dict[str, Any]:
        lease = await st.broker.release_lease(lease_id, rc)
        if lease is None:
            raise HTTPException(404, "no such lease")
        return {"released": lease_id}

    @app.post("/yuzuru/watches", dependencies=guarded)
    async def add_watch(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        headers = {k.lower(): v for k, v in request.headers.items()}
        session = payload.get("session") or st.router.header_session(headers) or st.router.last_session
        try:
            w = await st.waiter.add(
                session=session, pid=int(payload["pid"]) if payload.get("pid") else None, lease=payload.get("lease"),
                label=str(payload.get("label") or "process"), note=str(payload.get("note") or ""),
                cwd=payload.get("cwd"), log=payload.get("log"),
                suspend=bool(payload.get("suspend", True)), resume=bool(payload.get("resume", True)),
            )
        except KeyError as e:
            raise HTTPException(404, f"no lease {e}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if w.pid is None and w.lease is None:
            st.waiter.watches.pop(w.id, None)
            raise HTTPException(400, "give pid or lease")
        return {"id": w.id, "session": w.session}

    @app.get("/yuzuru/events", dependencies=guarded)
    async def events(after: int = 0, wait: float = 0) -> list[dict[str, Any]]:
        from dataclasses import asdict

        return [asdict(e) for e in await st.waiter.wait_events(after, min(wait, 300))]

    @app.post("/yuzuru/unload", dependencies=guarded)
    async def unload(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        tier = (payload or {}).get("tier", "all")
        variants = None
        if tier != "all":
            spec = st.catalog.specs.get(tier)
            if spec is None:
                raise HTTPException(404, f"no tier {tier!r}")
            variants = [spec.variant]
        return {"unloaded": await st.broker.unload(variants)}

    @app.post("/yuzuru/warm", dependencies=guarded)
    async def warm(payload: dict[str, Any]) -> dict[str, Any]:
        spec = st.catalog.specs.get(payload.get("tier", ""))
        if spec is None:
            raise HTTPException(404, "unknown tier")
        placement = await st.broker.acquire(spec, force=bool(payload.get("force")))
        if placement is not Placement.NONE:
            await st.broker.release(spec)
        return {"tier": spec.tier, "placement": placement.value}

    @app.post("/yuzuru/route", dependencies=guarded)
    async def dry_route(payload: dict[str, Any]) -> dict[str, Any]:
        """Show what `auto` would pick for a message, without generating."""
        sid = f"dry-{uuid.uuid4().hex[:6]}"
        body = {"model": payload.get("model", "auto"), "messages": payload.get("messages") or [
            {"role": "user", "content": payload.get("text", "")}
        ]}
        d = await st.router.route(body, {"x-yuzuru-session": sid})
        st.router.sessions.pop(sid, None)
        if d.cloud:
            st.router.cloud_calls.pop()  # a dry run is not a cloud call
        return {"tier": d.tier, "wanted": d.wanted, "model": d.model,
                "placement": "cloud" if d.cloud else d.placement.value,
                "reason": d.reason, "ctx_need": d.ctx_need}

    @app.post("/yuzuru/sessions/{session}/pin", dependencies=guarded)
    async def pin(session: str, payload: dict[str, Any]) -> dict[str, Any]:
        tier = payload.get("tier")
        try:
            st.router.pin(session, tier if tier not in (None, "", "auto") else None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"session": session, "pinned": st.router.sessions[session].pinned}

    @app.get("/yuzuru/models", dependencies=guarded)
    async def list_models() -> dict[str, Any]:
        return {"models": st.catalog.models_info(), "tiers": {t: s.base for t, s in st.catalog.specs.items()}}

    @app.post("/yuzuru/models/assign", dependencies=guarded)
    async def assign_model(payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        try:
            st.catalog.assign(str(payload.get("tier", "")), None if model in (None, "", "auto") else model)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        await st.catalog.refresh()
        return {"tiers": {t: s.base for t, s in st.catalog.specs.items()}}

    @app.post("/yuzuru/models/enable", dependencies=guarded)
    async def enable_model(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            st.catalog.set_enabled(str(payload.get("model", "")), bool(payload.get("enabled", True)))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        await st.catalog.refresh()
        return {"tiers": {t: s.base for t, s in st.catalog.specs.items()}, "enabled": st.catalog.enabled}

    @app.post("/yuzuru/refresh", dependencies=guarded)
    async def refresh() -> dict[str, Any]:
        await st.catalog.refresh()
        return {"tiers": {t: s.base for t, s in st.catalog.specs.items()}}

    @app.exception_handler(httpx.ConnectError)
    async def ollama_down(request: Request, exc: httpx.ConnectError) -> JSONResponse:
        return JSONResponse({"error": f"cannot reach Ollama at {cfg.ollama_url}"}, status_code=502)

    return app
