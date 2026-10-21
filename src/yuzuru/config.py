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
    # GPUs yuzuru may use, by index or UUID. Empty = all. The others are
    # invisible to budgets, leases, reservations, status and started jobs.
    gpus: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gpus = [str(g).strip() for g in self.gpus if str(g).strip()]


@dataclass
class OllamaConfig:
    """Optionally run a private `ollama serve` instead of using the system one.

    With `manage: true`, yuzuru starts it with CUDA_VISIBLE_DEVICES limited to
    `resources.gpus` (a hard reservation alone cannot stop Ollama placing
    compute buffers on an excluded GPU), restarts it if it dies and stops it
    on shutdown. `ollama_url` then points at it.
    """

    manage: bool = False
    binary: str = "ollama"
    host: str = "127.0.0.1"
    # A fixed port, or "auto": the first free port of `port_range` (inclusive),
    # so several managed servers (or a hand-started one) never collide.
    port: int | str = "auto"
    port_range: list[int] = field(default_factory=lambda: [11436, 11499])
    # OLLAMA_MODELS of the managed server. Blobs of `mirror_from` are symlinked
    # in (no copies), manifests are copied, so models pulled for the system
    # Ollama show up and variants yuzuru creates stay in this directory.
    models: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "yuzuru-ollama" / "models")
    # auto: $OLLAMA_MODELS, /usr/share/ollama/.ollama/models or ~/.ollama/models; none: no mirroring.
    mirror_from: str = "auto"
    # Run it in its own systemd user scope so its cgroup measures exactly its
    # host RAM: auto | on | off.
    scope: str = "auto"
    # Extra environment, e.g. OLLAMA_FLASH_ATTENTION: "1" as set for ollama.service.
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout: float = 60.0

    def __post_init__(self) -> None:
        self.models = Path(self.models).expanduser()
        self.env = {str(k): str(v) for k, v in self.env.items()}
        if str(self.port).isdigit():
            self.port = int(self.port)
        elif self.port != "auto":
            raise ValueError(f"ollama.port must be a number or 'auto', not {self.port!r}")
        lo, hi = (int(p) for p in self.port_range)
        if not 0 < lo <= hi < 65536:
            raise ValueError(f"ollama.port_range must be [low, high], not {self.port_range!r}")
        self.port_range = [lo, hi]

    @property
    def url(self) -> str:
        """Where it listens; with port "auto" only a guess until it is started."""
        return f"http://{self.host}:{self.port if isinstance(self.port, int) else self.port_range[0]}"


@dataclass
class LeaseConfig:
    # A lease is "settled" once VRAM use stops moving for this long.
    settle_seconds: float = 15.0
    settle_timeout: float = 180.0
    settle_tolerance_mb: int = 256
    # Extra VRAM kept free for a settled training job to grow into.
    headroom_mb: int = 1024
    # Wait this long for in-flight generations before unloading for a lease. One
    # reply of a model running partly on CPU can take minutes, and the agent that
    # started the job is usually still finishing its turn.
    drain_timeout: float = 300.0
    # Then wait this long for the unloaded models to actually leave the GPU.
    unload_timeout: float = 60.0
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
    # Unload only once the waiting session has been quiet this long: right after
    # starting a job the agent still sends one more request to wrap up its turn.
    suspend_idle_seconds: float = 20.0
    # A job can end while the agent that started it is still in that turn (it failed at
    # once, or the agent kept working). Resuming then would run a second client on the
    # same session, so the resume waits for the turn to end. A turn whose last reply
    # called tools counts as running for this long without a new request (a tool such
    # as a test run takes a while; a client killed mid-turn never comes back).
    turn_idle_seconds: float = 600.0
    # Reload the session's model before resuming so the prompt does not wait on a load.
    warm_before_resume: bool = True
    log_tail_lines: int = 40


@dataclass
class IsolationConfig:
    """Own systemd user scopes for what runs for an agent: the agent itself (resumed turns,
    `yuzuru agent`) and its training jobs (MCP `yuzuru_train`).

    A process's memory counts against the unit it was started from, and when the kernel
    OOM-kills one, systemd stops that whole unit (yuzuru's service, a terminal, an IDE).
    In a scope of its own with a memory cap, a runaway script or job is killed alone.
    """

    # auto (when `systemd-run --user --scope` works) | on | off
    scope: str = "auto"
    # Each scope's memory cap (systemd MemoryMax: "80%", "24G"; null: none).
    memory_max: str | None = "80%"

    def __post_init__(self) -> None:
        if self.scope not in ("auto", "on", "off"):
            raise ValueError(f"isolation.scope must be auto, on or off, not {self.scope!r}")


@dataclass
class Config:
    ollama_url: str = "http://127.0.0.1:11434"
    host: str = "127.0.0.1"
    port: int = 11435
    # Bearer token required from clients when set (use when not on localhost).
    token: str | None = None
    # Browser origins allowed to call yuzuru. Empty: any request carrying an Origin
    # header is refused, so a web page cannot drive the API (it can start jobs).
    allowed_origins: list[str] = field(default_factory=list)
    keep_alive: str = "30m"
    state_dir: Path = field(default_factory=lambda: Path.home() / ".local" / "state" / "yuzuru")
    tiers: list[TierConfig] = field(default_factory=list)
    router: RouterConfig = field(default_factory=RouterConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    lease: LeaseConfig = field(default_factory=LeaseConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    # The file this was loaded from (absolute), handed on to the CLI processes yuzuru starts.
    source: Path | None = None

    def __post_init__(self) -> None:
        if not self.tiers:
            self.tiers = default_tiers()
        if self.ollama.manage:
            self.ollama_url = self.ollama.url

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
        elif key == "isolation":
            kwargs["isolation"] = _build(IsolationConfig, value)
        elif key == "ollama":
            kwargs["ollama"] = _build(OllamaConfig, value)
        elif key == "state_dir":
            kwargs["state_dir"] = Path(value).expanduser()
        else:
            kwargs[key] = value
    cfg = _build(Config, kwargs)
    cfg.source = Path(path).expanduser().resolve() if path is not None else None

    if (env_url := os.environ.get("OLLAMA_HOST_URL")) and not cfg.ollama.manage:
        cfg.ollama_url = env_url
    if env_token := os.environ.get("YUZURU_TOKEN"):
        cfg.token = env_token
    return cfg
