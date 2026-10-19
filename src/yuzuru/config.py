"""Configuration.

Everything has a hardware-independent default: with no config file, tiers are
filled from whatever models are installed in Ollama, and memory budgets are
derived from what the resource probe reports at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

MB = 1024 * 1024
GB = 1024 * MB

CONFIG_CANDIDATES = (
    Path("yuzuru.yaml"),
    Path.home() / ".config" / "yuzuru" / "config.yaml",
)


@dataclass
class TierConfig:
    name: str
    # Candidate Ollama models in preference order. Empty = auto-assign from
    # installed models by parameter count (see auto_params_max_b).
    models: list[str] = field(default_factory=list)
    num_ctx: int = 32768
    # Upper bound (billions of params) used only for auto-assignment.
    auto_params_max_b: float | None = None
    # Allow running partly in system RAM when VRAM is short (good for MoE).
    allow_cpu_offload: bool = True
    # Max fraction of the model that may live in system RAM when offloading.
    # None = by architecture (MoE tolerates much more offload than dense).
    max_cpu_fraction: float | None = None


@dataclass
class RouterConfig:
    # "llm": ask the small model to grade difficulty, "heuristic": rules only.
    classifier: str = "llm"
    # Run the classifier on CPU so it never competes for VRAM.
    classifier_on_cpu: bool = True
    classifier_timeout: float = 20.0
    # If a model at or above the wanted tier is already resident, use it
    # instead of swapping (swaps cost seconds to minutes).
    prefer_loaded: bool = True
    # Escalate one tier after this many failing tool results within a turn.
    error_escalation: int = 3
    # OpenCode agents that are always cheap work.
    small_agents: list[str] = field(default_factory=lambda: ["title", "summary"])


@dataclass
class CloudConfig:
    """An OpenAI-compatible cloud API used when local models are not enough.

    Any /chat/completions provider works (OpenAI, OpenRouter, Groq, a remote
    vLLM, ...). The key is read from the environment, never from this file.
    """

    enabled: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = ""
    context: int = 200000
    # When to go to the cloud without being asked:
    #   off        - only when requested (yuzuru/cloud, @cloud, pin)
    #   escalation - when a turn needs to escalate past the largest local tier
    #   hard       - also whenever the router grades a new turn "hard"
    auto: str = "escalation"
    # Also use the cloud when the wanted local tier cannot be loaded right now
    # (e.g. a training job holds the GPU) instead of a smaller local model.
    when_no_memory: bool = False
    # Also use the cloud when the prompt exceeds every local tier's context.
    when_context: bool = True
    max_requests_per_hour: int = 60

    @property
    def api_key(self) -> str | None:
        import os

        return os.environ.get(self.api_key_env) or None

    @property
    def usable(self) -> bool:
        return self.enabled and bool(self.model) and self.api_key is not None


@dataclass
class ResourceConfig:
    probe: str = "auto"  # auto | nvidia | none
    # cgroup dir of the Ollama service for true host-RAM usage: auto | none | <path>
    ollama_cgroup: str = "auto"
    vram_margin_mb: int = 512  # per GPU, never handed to the LLM
    ram_reserve_mb: int = 6144  # kept free for the OS, dataloaders, etc.


@dataclass
class LeaseConfig:
    # A lease is "settled" once VRAM use stops moving for this long.
    settle_seconds: float = 15.0
    settle_timeout: float = 180.0
    settle_tolerance_mb: int = 256
    # Extra VRAM kept free for a settled training job to grow into.
    headroom_mb: int = 1024
    # Wait this long for in-flight generations before unloading for a lease.
    drain_timeout: float = 30.0
    # Re-warm the models that were resident before the lease once it ends.
    restore_after: bool = True


@dataclass
class ResumeConfig:
    """Free the LLM while an agent waits for a process, and resume it afterwards."""

    enabled: bool = True
    # Command that continues an OpenCode session; {session}, {message} and {cwd}
    # are substituted. `opencode run --session` works for OpenCode 1.x and 2.x.
    command: list[str] = field(default_factory=lambda: ["opencode", "run", "--session", "{session}", "{message}"])
    # While waiting, unload every model (CPU classifier and RAM-offloaded ones too).
    suspend_all: bool = True
    # Reload the session's model before resuming so the prompt does not wait on a load.
    warm_before_resume: bool = True
    log_tail_lines: int = 40


@dataclass
class Config:
    ollama_url: str = "http://127.0.0.1:11434"
    host: str = "127.0.0.1"
    port: int = 11435
    # Bearer token required from clients when set (use when not on localhost).
    token: str | None = None
    keep_alive: str = "30m"
    state_dir: Path = field(default_factory=lambda: Path.home() / ".local" / "state" / "yuzuru")
    tiers: list[TierConfig] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    lease: LeaseConfig = field(default_factory=LeaseConfig)

    def __post_init__(self) -> None:
        if not self.tiers:
            self.tiers = default_tiers()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def tier(self, name: str) -> TierConfig | None:
        return next((t for t in self.tiers if t.name == name), None)

    def tier_index(self, name: str) -> int:
        return [t.name for t in self.tiers].index(name)


def default_tiers() -> list[TierConfig]:
    return [
        # 32k everywhere: a coding agent's system prompt + tool schemas alone is ~20k tokens.
        TierConfig("small", num_ctx=32768, auto_params_max_b=6, allow_cpu_offload=True, max_cpu_fraction=1.0),
        TierConfig("medium", num_ctx=32768, auto_params_max_b=20),
        TierConfig("large", num_ctx=32768, auto_params_max_b=None),
    ]


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> Config:
    path = path or os.environ.get("YUZURU_CONFIG")
    if path is None:
        path = next((p for p in CONFIG_CANDIDATES if p.exists()), None)
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "tiers":
            kwargs["tiers"] = [_build(TierConfig, t) for t in value]
        elif key == "router":
            kwargs["router"] = _build(RouterConfig, value)
        elif key == "resources":
            kwargs["resources"] = _build(ResourceConfig, value)
        elif key == "lease":
            kwargs["lease"] = _build(LeaseConfig, value)
        elif key == "cloud":
            kwargs["cloud"] = _build(CloudConfig, value)
        elif key == "resume":
            kwargs["resume"] = _build(ResumeConfig, value)
        elif key == "state_dir":
            kwargs["state_dir"] = Path(value).expanduser()
        else:
            kwargs[key] = value
    cfg = _build(Config, kwargs)

    if env_url := os.environ.get("OLLAMA_HOST_URL"):
        cfg.ollama_url = env_url
    if env_token := os.environ.get("YUZURU_TOKEN"):
        cfg.token = env_token
    return cfg
