"""Model catalog: which Ollama model serves each tier, and how much memory it needs.

Ollama's OpenAI-compatible endpoint ignores per-request `num_ctx`, so each
(tier, base model) gets a derived Ollama model ("variant") with the tier's
context length baked in. Variants share weights with the base model, so
creating one is instant and costs no disk.

Memory needs start as an estimate from GGUF metadata and are replaced by the
real number the first time Ollama loads the variant.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, TierConfig
from .ollama import Ollama, OllamaError

log = logging.getLogger("yuzuru.catalog")

FALLBACK_KV_BYTES_PER_TOKEN = 128 * 1024  # ~ a 7-8B GQA model at f16


@dataclass
class ModelSpec:
    tier: str
    base: str
    variant: str
    num_ctx: int
    file_size: int
    params_b: float
    kv_bytes_per_token: float
    measured_size: int | None = None
    cpu_only: bool = False
    moe: bool = False  # MoE models tolerate CPU offload far better than dense ones
    # Host RAM Ollama actually took when loading this variant, by placement.
    measured_host: dict[str, int] = field(default_factory=dict)

    @property
    def estimated_size(self) -> int:
        if self.measured_size:
            return self.measured_size
        overhead = int(0.04 * self.file_size) + 384 * 1024 * 1024
        return int(self.file_size + self.kv_bytes_per_token * self.num_ctx + overhead)

    def host_need(self, placement: str) -> int:
        """Host RAM a load with this placement takes ("gpu" / "offload" / "cpu").

        Until measured, assume the worst for partial offload: with CUDA,
        llama.cpp has been seen keeping the *entire* model in host memory.
        """
        if placement in self.measured_host:
            return self.measured_host[placement]
        if placement == "gpu":
            return int(0.1 * self.file_size)
        return self.estimated_size


def variant_name(tier: str, base: str, num_ctx: int, cpu: bool = False) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", base.replace(":latest", "")).strip("-")
    suffix = "-cpu" if cpu else ""
    return f"yuzuru-{tier}:{safe}-{num_ctx // 1024}k{suffix}"


def parse_params_b(details: dict[str, Any]) -> float:
    m = re.match(r"([\d.]+)\s*([BMK]?)", str(details.get("parameter_size", "")))
    if not m:
        return 0.0
    scale = {"B": 1.0, "M": 1e-3, "K": 1e-6, "": 1e-9}[m.group(2)]
    return float(m.group(1)) * scale


def kv_bytes_per_token(model_info: dict[str, Any]) -> float:
    """f16 KV-cache bytes per context token, from GGUF metadata."""
    arch = model_info.get("general.architecture")
    if not arch:
        return FALLBACK_KV_BYTES_PER_TOKEN

    def get(key: str) -> Any:
        return model_info.get(f"{arch}.{key}")

    n_layer = get("block_count")
    n_head = get("attention.head_count")
    n_kv = get("attention.head_count_kv")
    if isinstance(n_head, list):
        n_head = max(n_head)
    if isinstance(n_kv, list):
        # Per-layer list; layers with 0 KV heads have no attention cache.
        n_kv_layers = sum(1 for x in n_kv if x)
        n_kv = max(n_kv) if n_kv else None
    else:
        n_kv_layers = None
    if not n_layer or not (n_kv or n_head):
        return FALLBACK_KV_BYTES_PER_TOKEN
    n_kv = n_kv or n_head
    emb = get("embedding_length") or 0
    k_len = get("attention.key_length") or (emb // n_head if n_head else 128)
    v_len = get("attention.value_length") or k_len
    attn_layers = n_kv_layers or n_layer
    if interval := get("full_attention_interval"):
        attn_layers = min(attn_layers, n_layer // interval)
    return float(attn_layers * n_kv * (k_len + v_len) * 2)


class ProfileStore:
    """Persists measured memory sizes and metadata so estimates improve over time."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                log.warning("ignoring corrupt profile store %s", path)

    def get(self, key: str) -> dict[str, Any]:
        return self.data.get(key, {})

    def update(self, key: str, **values: Any) -> None:
        self.data.setdefault(key, {}).update(values)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))


class Selection:
    """The user's model choices, persisted across restarts.

    `assign` pins a model to a tier (beats config and auto-assignment);
    `disabled` models are never auto-assigned or offered.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.assign: dict[str, str] = {}
        self.disabled: set[str] = set()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.assign = dict(data.get("assign", {}))
                self.disabled = set(data.get("disabled", []))
            except json.JSONDecodeError:
                log.warning("ignoring corrupt selection file %s", path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"assign": self.assign, "disabled": sorted(self.disabled)}, indent=1))


class Catalog:
    def __init__(self, cfg: Config, ollama: Ollama) -> None:
        self.cfg = cfg
        self.ollama = ollama
        self.profiles = ProfileStore(cfg.state_dir / "profiles.json")
        self.selection = Selection(cfg.state_dir / "selection.json")
        self.specs: dict[str, ModelSpec] = {}  # tier -> spec
        self.direct: dict[str, ModelSpec] = {}  # model name -> spec, for explicitly chosen models
        self.router_spec: ModelSpec | None = None
        self.installed: dict[str, dict[str, Any]] = {}

    @property
    def tiers(self) -> list[str]:
        """Tier names that have a model, smallest first."""
        return [t.name for t in self.cfg.tiers if t.name in self.specs]

    def by_variant(self, variant: str) -> ModelSpec | None:
        specs = list(self.specs.values()) + list(self.direct.values())
        specs += [self.router_spec] if self.router_spec else []
        return next((s for s in specs if s.variant == variant), None)

    def resolve(self, name: str) -> str | None:
        """Installed model name for `name` (accepts a missing ':latest')."""
        for key in (name, f"{name}:latest"):
            if key in self.installed:
                return key
        return None

    @property
    def enabled(self) -> list[str]:
        return sorted(n for n, m in self.installed.items() if n not in self.selection.disabled and not _is_embedding(m))

    async def direct_spec(self, model: str) -> ModelSpec:
        """Spec for a model chosen by name, outside the tier mapping."""
        name = self.resolve(model)
        if name is None:
            raise KeyError(model)
        if name not in self.direct:
            top = self.cfg.tiers[-1]
            self.direct[name] = await self._spec(top, name, tier="direct")
        return self.direct[name]

    def models_info(self) -> list[dict[str, Any]]:
        tier_of = {s.base: t for t, s in self.specs.items()}
        out = []
        for name, m in sorted(self.installed.items()):
            d = m.get("details", {})
            out.append({
                "name": name,
                "params_b": parse_params_b(d),
                "size": m.get("size", 0),
                "quantization": d.get("quantization_level"),
                "family": d.get("family"),
                "moe": self.profiles.get(name).get("moe"),
                "tier": tier_of.get(name),
                "pinned_to": next((t for t, n in self.selection.assign.items() if self.resolve(n) == name), None),
                "enabled": name not in self.selection.disabled,
                "embedding": _is_embedding(m),
            })
        return out

    def assign(self, tier: str, model: str | None) -> None:
        if self.cfg.tier(tier) is None:
            raise ValueError(f"unknown tier {tier!r}")
        if model is None:
            self.selection.assign.pop(tier, None)
        else:
            name = self.resolve(model)
            if name is None:
                raise ValueError(f"model {model!r} is not installed in Ollama")
            self.selection.assign[tier] = name
            self.selection.disabled.discard(name)
        self.selection.save()

    def set_enabled(self, model: str, enabled: bool) -> None:
        name = self.resolve(model)
        if name is None:
            raise ValueError(f"model {model!r} is not installed in Ollama")
        if enabled:
            self.selection.disabled.discard(name)
        else:
            self.selection.disabled.add(name)
            self.selection.assign = {t: n for t, n in self.selection.assign.items() if n != name}
        self.selection.save()

    def nearest_tier(self, wanted: str) -> str | None:
        """`wanted` if it has a model, else the closest larger tier, else the closest smaller."""
        names = [t.name for t in self.cfg.tiers]
        if wanted not in names:
            return None
        i = names.index(wanted)
        for name in names[i:] + names[:i][::-1]:
            if name in self.specs:
                return name
        return None

    async def refresh(self) -> None:
        models = await self.ollama.tags()
        self.installed = {
            m["name"]: m for m in models if not m["name"].startswith("yuzuru-")
        }
        self.specs = {}
        self.direct = {k: v for k, v in self.direct.items() if k in self.installed}
        assigned = self._assign()
        for tier_cfg in self.cfg.tiers:
            base = assigned.get(tier_cfg.name)
            if base:
                self.specs[tier_cfg.name] = await self._spec(tier_cfg, base)
        self.router_spec = None
        # The classifier only makes sense if a genuinely small model exists.
        if self.cfg.router.classifier == "llm" and len(self.tiers) > 1 and self.specs[self.tiers[0]].params_b <= 9:
            small_cfg = self.cfg.tier(self.tiers[0])
            assert small_cfg is not None
            base = assigned[small_cfg.name]
            self.router_spec = await self._spec(
                small_cfg, base, num_ctx=4096, cpu=self.cfg.router.classifier_on_cpu, tier="router"
            )
        log.debug("tiers: %s", {t: s.base for t, s in self.specs.items()})

    def _assign(self) -> dict[str, str]:
        """Pick a base model per tier: user selection, then config candidates, then by size."""
        out: dict[str, str] = {}
        taken: set[str] = set()
        disabled = self.selection.disabled

        def resolve(name: str) -> str | None:
            found = self.resolve(name)
            return found if found and found not in disabled else None

        for t in self.cfg.tiers:
            if (name := resolve(self.selection.assign.get(t.name, ""))) is not None:
                out[t.name] = name
                taken.add(name)
        for t in self.cfg.tiers:
            if t.name in out:
                continue
            for cand in t.models:
                if (name := resolve(cand)) is not None:
                    out[t.name] = name
                    taken.add(name)
                    break

        auto = [t for t in self.cfg.tiers if not t.models and t.name not in out]
        if not auto:
            return out
        pool = sorted(
            (
                (parse_params_b(m.get("details", {})), name)
                for name, m in self.installed.items()
                if name not in taken and name not in disabled and not _is_embedding(m)
            ),
            reverse=True,
        )
        # Largest model under each tier's ceiling.
        lower = 0.0
        for t in auto:
            ceiling = t.auto_params_max_b if t.auto_params_max_b is not None else float("inf")
            fits = [(p, n) for p, n in pool if lower <= p < ceiling and n not in taken]
            if fits:
                out[t.name] = fits[0][1]
                taken.add(fits[0][1])
            lower = ceiling if t.auto_params_max_b is not None else lower
        return out

    async def _spec(
        self, tier_cfg: TierConfig, base: str, num_ctx: int | None = None, cpu: bool = False, tier: str | None = None
    ) -> ModelSpec:
        num_ctx = num_ctx or tier_cfg.num_ctx
        tier = tier or tier_cfg.name
        variant = variant_name(tier, base, num_ctx, cpu)
        meta = self.installed[base]
        digest = meta.get("digest", "")

        prof = self.profiles.get(base)
        if prof.get("digest") != digest or "kv_bytes_per_token" not in prof:
            try:
                info = (await self.ollama.show(base, verbose=True)).get("model_info", {})
            except OllamaError:
                info = {}
            arch = info.get("general.architecture", "")
            moe = "moe" in arch or bool(info.get(f"{arch}.expert_count"))
            self.profiles.update(base, digest=digest, kv_bytes_per_token=kv_bytes_per_token(info), moe=moe)
            prof = self.profiles.get(base)

        params: dict[str, Any] = {"num_ctx": num_ctx}
        if cpu:
            params["num_gpu"] = 0
        vprof = self.profiles.get(variant)
        if vprof.get("digest") != digest or vprof.get("params") != params:
            await self.ollama.create_variant(variant, base, params)
            self.profiles.update(variant, digest=digest, params=params, measured_size=None, measured_host={})
            vprof = self.profiles.get(variant)

        return ModelSpec(
            tier=tier,
            base=base,
            variant=variant,
            num_ctx=num_ctx,
            file_size=int(meta.get("size", 0)),
            params_b=parse_params_b(meta.get("details", {})),
            kv_bytes_per_token=float(prof["kv_bytes_per_token"]),
            measured_size=vprof.get("measured_size"),
            measured_host=dict(vprof.get("measured_host") or {}),
            cpu_only=cpu,
            moe=bool(prof.get("moe")),
        )

    def record_loaded(self, variant: str, size: int) -> None:
        spec = self.by_variant(variant)
        if spec and size and spec.measured_size != size:
            spec.measured_size = size
            self.profiles.update(variant, measured_size=size)

    def record_host(self, variant: str, placement: str, host: int) -> None:
        spec = self.by_variant(variant)
        if spec and host > 0:
            spec.measured_host[placement] = host
            self.profiles.update(variant, measured_host=dict(spec.measured_host))


def _is_embedding(model: dict[str, Any]) -> bool:
    fam = " ".join(model.get("details", {}).get("families") or [model.get("details", {}).get("family", "")])
    return "bert" in fam.lower() or "embed" in model.get("name", "").lower()
