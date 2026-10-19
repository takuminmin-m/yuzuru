"""OpenAI-compatible proxy in front of Ollama, plus the control API.

  /v1/chat/completions   model = auto | <tier>  -> routed, memory-managed
  /v1/*                  passed through to Ollama
  /yuzuru/*              status, leases, pins, warm/unload
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import uuid
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .broker import Broker, Placement
from .catalog import Catalog
from .mcp import MCPServer
from .config import Config
from .ollama import Ollama
from .resources import Probe, make_probe
from .router import Router

log = logging.getLogger("yuzuru.server")

HOP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}


class State:
    def __init__(self, cfg: Config, probe: Probe | None = None, http: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.http = http or httpx.AsyncClient(base_url=cfg.ollama_url, timeout=httpx.Timeout(None, connect=5))
        self.ollama = Ollama(cfg.ollama_url, self.http)
        self.catalog = Catalog(cfg, self.ollama)
        self.broker = Broker(cfg, probe or make_probe(cfg.resources.probe, cfg.resources.ollama_cgroup), self.ollama, self.catalog)
        self.router = Router(cfg, self.catalog, self.broker, self.ollama)
        self.tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.catalog.refresh()
        await self.broker.refresh()
        self.tasks.append(asyncio.create_task(self.broker.monitor()))
        self.tasks.append(asyncio.create_task(self._refresh_catalog()))

    async def stop(self) -> None:
        for t in self.tasks:
            t.cancel()
        await self.http.aclose()

    async def _refresh_catalog(self) -> None:
        """Pick up models pulled/removed while running."""
        while True:
            await asyncio.sleep(60)
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

    # ------------------------------------------------------------- OpenAI API

    @app.get("/v1/models", dependencies=guarded)
    async def models() -> dict[str, Any]:
        ids = ["auto"] + st.catalog.tiers
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "yuzuru"} for i in ids]}

    @app.post("/v1/chat/completions", dependencies=guarded)
    async def chat(request: Request) -> Response:
        body = await request.json()
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            decision = await st.router.route(body, headers)
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        spec = decision.spec
        placement = await st.broker.acquire(spec, force=True)
        if placement is not Placement.NONE:
            decision.placement = placement
        body["model"] = spec.variant
        out_headers = decision.headers()

        if body.get("stream"):
            try:
                upstream = await st.http.send(
                    st.http.build_request("POST", "/v1/chat/completions", json=body), stream=True
                )
            except BaseException:
                await st.broker.release(spec)
                raise

            async def relay():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()
                    await st.broker.release(spec)

            return StreamingResponse(
                relay(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
                headers=out_headers,
            )

        try:
            upstream = await st.http.post("/v1/chat/completions", json=body)
        finally:
            await st.broker.release(spec)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=out_headers,
        )

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
        return st.broker.status() | {"sessions": st.router.sessions_status()}

    @app.post("/yuzuru/leases", dependencies=guarded)
    async def create_lease(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            lease = await st.broker.create_lease(
                pid=int(payload["pid"]),
                label=str(payload.get("label", "job")),
                gpus=[str(g) for g in payload.get("gpus") or []],
                vram=int(payload["vram"]) if payload.get("vram") else None,
                session=payload.get("session"),
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        return {"id": lease.id, "gpus": lease.gpus, "state": lease.state}

    @app.get("/yuzuru/leases", dependencies=guarded)
    async def list_leases() -> list[dict[str, Any]]:
        return st.broker.status()["leases"]

    @app.delete("/yuzuru/leases/{lease_id}", dependencies=guarded)
    async def release_lease(lease_id: str) -> dict[str, Any]:
        lease = await st.broker.release_lease(lease_id)
        if lease is None:
            raise HTTPException(404, "no such lease")
        return {"released": lease_id}

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
        return {"tier": d.tier, "wanted": d.wanted, "model": d.spec.base, "placement": d.placement.value,
                "reason": d.reason, "ctx_need": d.ctx_need}

    @app.post("/yuzuru/sessions/{session}/pin", dependencies=guarded)
    async def pin(session: str, payload: dict[str, Any]) -> dict[str, Any]:
        tier = payload.get("tier")
        try:
            st.router.pin(session, tier if tier not in (None, "", "auto") else None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"session": session, "pinned": st.router.sessions[session].pinned}

    @app.post("/yuzuru/refresh", dependencies=guarded)
    async def refresh() -> dict[str, Any]:
        await st.catalog.refresh()
        return {"tiers": {t: s.base for t, s in st.catalog.specs.items()}}

    @app.exception_handler(httpx.ConnectError)
    async def ollama_down(request: Request, exc: httpx.ConnectError) -> JSONResponse:
        return JSONResponse({"error": f"cannot reach Ollama at {cfg.ollama_url}"}, status_code=502)

    return app
