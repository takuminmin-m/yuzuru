"""Thin async client for the parts of the Ollama API yuzuru needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LoadedModel:
    name: str
    size: int  # total bytes (VRAM + RAM)
    size_vram: int
    context_length: int | None = None

    @property
    def size_ram(self) -> int:
        return max(0, self.size - self.size_vram)


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = client or httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(600, connect=5))

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        r = await self.http.post(path, json=payload, timeout=timeout or httpx.USE_CLIENT_DEFAULT)
        if r.status_code >= 400:
            raise OllamaError(f"{path}: {r.status_code} {r.text[:300]}")
        return r.json() if r.content else {}

    async def tags(self) -> list[dict[str, Any]]:
        r = await self.http.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    async def ps(self) -> list[LoadedModel]:
        r = await self.http.get("/api/ps")
        r.raise_for_status()
        return [
            LoadedModel(m["name"], m.get("size", 0), m.get("size_vram", 0), m.get("context_length"))
            for m in r.json().get("models", [])
        ]

    async def show(self, model: str, verbose: bool = False) -> dict[str, Any]:
        return await self._post("/api/show", {"model": model, "verbose": verbose})

    async def create_variant(self, name: str, base: str, parameters: dict[str, Any]) -> None:
        await self._post("/api/create", {"model": name, "from": base, "parameters": parameters, "stream": False})

    async def load(self, model: str, keep_alive: str | int) -> None:
        """Load (or refresh keep_alive of) a model without generating."""
        await self._post("/api/generate", {"model": model, "keep_alive": keep_alive})

    async def unload(self, model: str) -> None:
        await self._post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)

    async def chat(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Native (non-OpenAI) chat; used for the classifier's structured output."""
        return await self._post("/api/chat", {**payload, "stream": False}, timeout=timeout)
