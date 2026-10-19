"""Difficulty routing.

Per request, decide the tier the task *wants* (explicit request > session pin
> rules/classifier > context-length floor), then the tier it *gets* given the
memory the broker can offer right now.

New user turns are classified; agent-loop continuations (tool results) stick
to the turn's tier so a model is never swapped mid-task, except for one
escalation when tool calls keep failing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .broker import Broker, Placement
from .catalog import Catalog, ModelSpec
from .config import Config
from .ollama import Ollama

log = logging.getLogger("yuzuru.router")

LEVELS = ("easy", "medium", "hard")

HARD_RE = re.compile(
    r"設計|アーキテクチャ|リファクタ|デバッグ|原因|なぜ|最適化|証明|導出|数式|論文|再現|全体|調査|"
    r"refactor|architect|design|debug|root cause|\bwhy\b|optimi[sz]|prove|derive|reproduce|investigate",
    re.I,
)
EASY_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks?|thank you|ok|ありがとう|こんにちは|おはよう|了解)\b|typo|誤字|rename|リネーム|翻訳|translate",
    re.I,
)
TRACE_RE = re.compile(r"Traceback \(most recent call last\)|^\s*at .+:\d+|error\[E\d+\]", re.M)
TOOL_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|\b(Error|Exception):|\bFAILED\b|exit code [1-9]|command not found|No such file",
)
OVERRIDE_RE = re.compile(r"(?:^|\s)@(small|medium|large|auto)\b", re.I)

CLASSIFIER_SYSTEM = """You route requests for a local coding assistant used for machine-learning research.
Grade how capable a model the assistant's NEXT reply needs.
- easy: greetings, short factual questions, trivial edits, renames, formatting, simple shell commands
- medium: ordinary coding: write or explain a function, a test, a small bug fix, a plot, reading a log
- hard: multi-file design or refactoring, subtle debugging, math or derivations, research-level reasoning,
  performance tuning of training code, anything needing careful planning
Answer with JSON only."""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {"difficulty": {"enum": list(LEVELS)}, "reason": {"type": "string"}},
    "required": ["difficulty"],
}


@dataclass
class SessionState:
    tier: str | None = None
    pinned: str | None = None
    escalated: bool = False
    updated: float = field(default_factory=time.time)


@dataclass
class Decision:
    tier: str
    spec: ModelSpec
    placement: Placement
    wanted: str
    reason: str
    session: str
    ctx_need: int

    def headers(self) -> dict[str, str]:
        return {
            "x-yuzuru-tier": self.tier,
            "x-yuzuru-model": self.spec.base,
            "x-yuzuru-placement": self.placement.value,
            "x-yuzuru-reason": self.reason.encode("ascii", "replace").decode()[:200],
        }


def message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def estimate_tokens(body: dict[str, Any]) -> int:
    # Conservative: ~3.5 bytes/token for code/English; Japanese is ~3 bytes/char
    # and ~1 char/token, so bytes/3.5 slightly over-counts it — the safe side.
    raw = json.dumps(body.get("messages", []), ensure_ascii=False) + json.dumps(body.get("tools", []))
    return int(len(raw.encode()) / 3.5)


def heuristic_level(text: str) -> tuple[int, str]:
    if HARD_RE.search(text):
        return 2, "keyword:hard"
    if len(text) > 2000 or TRACE_RE.search(text):
        return 1, "long-or-trace"
    if EASY_RE.search(text) and len(text) < 300:
        return 0, "keyword:easy"
    return 1, "default"


class Router:
    def __init__(self, cfg: Config, catalog: Catalog, broker: Broker, ollama: Ollama) -> None:
        self.cfg = cfg
        self.catalog = catalog
        self.broker = broker
        self.ollama = ollama
        self.sessions: dict[str, SessionState] = {}
        self.last_session: str | None = None
        self.log_path = cfg.state_dir / "decisions.jsonl"

    # ---------------------------------------------------------------- helpers

    def level_to_tier(self, level: int) -> str:
        names = [t.name for t in self.cfg.tiers]
        return names[round(level / (len(LEVELS) - 1) * (len(names) - 1))]

    @staticmethod
    def header_session(headers: dict[str, str]) -> str | None:
        # OpenCode v2 sends x-opencode-session natively; the v1 plugin sends x-yuzuru-session.
        for key in ("x-yuzuru-session", "x-opencode-session", "x-session-affinity"):
            if sid := headers.get(key):
                return sid
        return None

    def session_key(self, body: dict[str, Any], headers: dict[str, str]) -> str:
        if sid := self.header_session(headers):
            return sid
        msgs = body.get("messages", [])
        first_user = next((message_text(m) for m in msgs if m.get("role") == "user"), "")
        system = next((message_text(m) for m in msgs if m.get("role") == "system"), "")
        return "h:" + hashlib.sha1((system[:2000] + "\0" + first_user[:2000]).encode()).hexdigest()[:12]

    def pin(self, session: str, target: str | None) -> None:
        """Pin a session to a tier or to a specific installed model (None = auto)."""
        if target is not None and self.cfg.tier(target) is None:
            name = self.catalog.resolve(target)
            if name is None:
                raise ValueError(f"{target!r} is neither a tier nor an installed model")
            target = name
        self.sessions.setdefault(session, SessionState()).pinned = target

    def _gc(self) -> None:
        cutoff = time.time() - 6 * 3600
        for k in [k for k, s in self.sessions.items() if s.updated < cutoff]:
            del self.sessions[k]

    async def classify(self, text: str, n_messages: int) -> tuple[int, str]:
        h_level, h_reason = heuristic_level(text)
        spec = self.catalog.router_spec
        if self.cfg.router.classifier != "llm" or spec is None:
            return h_level, h_reason
        prompt = f"Conversation length so far: {n_messages} messages.\nNext user message:\n<<<\n{text[:3000]}\n>>>"
        placement = Placement.NONE
        try:
            placement = await self.broker.acquire(spec)
            if placement is Placement.NONE:
                return h_level, h_reason + " (classifier: no memory)"
            resp = await self.ollama.chat(
                {
                    "model": spec.variant,
                    "messages": [{"role": "system", "content": CLASSIFIER_SYSTEM}, {"role": "user", "content": prompt}],
                    "format": CLASSIFIER_SCHEMA,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 80},
                    "keep_alive": self.cfg.keep_alive,
                },
                timeout=self.cfg.router.classifier_timeout,
            )
            out = json.loads(resp["message"]["content"])
            level = LEVELS.index(out["difficulty"])
            return level, f"classifier:{out['difficulty']}" + (f" ({out['reason'][:80]})" if out.get("reason") else "")
        except Exception as e:  # noqa: BLE001 - routing must never fail on the classifier
            log.warning("classifier failed (%s); using heuristics", e)
            return h_level, h_reason
        finally:
            if placement is not Placement.NONE:
                await self.broker.release(spec, touch=False)

    # ------------------------------------------------------------------ route

    async def route(self, body: dict[str, Any], headers: dict[str, str]) -> Decision:
        await self.broker.wait_settled()
        self._gc()
        names = [t.name for t in self.cfg.tiers]
        requested = str(body.get("model", "auto")).split("/")[-1]
        agent = headers.get("x-yuzuru-agent", "")
        msgs = body.get("messages", [])
        key = self.session_key(body, headers)
        sess = self.sessions.setdefault(key, SessionState())
        sess.updated = time.time()
        if not key.startswith("dry-"):
            self.last_session = key
        last = msgs[-1] if msgs else {}
        last_user = next((message_text(m) for m in reversed(msgs) if m.get("role") == "user"), "")
        new_turn = last.get("role") == "user" or sess.tier is None

        escalate = False
        if new_turn:
            sess.escalated = False
        override = OVERRIDE_RE.search(last_user) if new_turn else None

        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
        ctx_need = estimate_tokens(body) + min(int(max_tokens), 8192)

        # A model chosen by name (request or session pin) bypasses tiers entirely.
        direct = None
        if requested not in names and requested != "auto":
            direct, reason = self.catalog.resolve(requested), "requested model"
            if direct is None:
                raise LookupError(f"unknown model {requested!r}")
        elif requested == "auto" and sess.pinned and sess.pinned not in names:
            direct, reason = self.catalog.resolve(sess.pinned), "session pin (model)"
        if direct:
            spec = await self.catalog.direct_spec(direct)
            placement, _ = self.broker.plan(spec, True)
            why = "fits" if placement is Placement.GPU else placement.value
            if ctx_need > spec.num_ctx:
                why += f"; context needs {ctx_need} > {spec.num_ctx}"
            decision = Decision(direct, spec, placement, direct, f"{reason}; {why}", key, ctx_need)
            self._log(decision, requested, agent, last_user)
            return decision

        if requested in names:
            wanted, reason = requested, "requested"
        elif override and override.group(1).lower() in names:
            wanted, reason = override.group(1).lower(), "inline @override"
        elif sess.pinned:
            wanted, reason = sess.pinned, "session pin"
        elif agent in self.cfg.router.small_agents:
            wanted, reason = names[0], f"agent:{agent}"
        elif new_turn:
            level, reason = await self.classify(last_user, len(msgs))
            wanted = self.level_to_tier(level)
        else:
            wanted, reason = sess.tier or names[-1], "continuation"
            errors = sum(
                1 for m in _since_last_user(msgs) if m.get("role") == "tool" and TOOL_ERROR_RE.search(message_text(m))
            )
            if errors >= self.cfg.router.error_escalation and not sess.escalated:
                i = names.index(wanted)
                if i + 1 < len(names):
                    wanted, reason = names[i + 1], f"escalated after {errors} tool errors"
                    escalate = True
                sess.escalated = True

        floor = next((t.name for t in self.cfg.tiers if t.num_ctx >= ctx_need), names[-1])
        if names.index(floor) > names.index(wanted):
            wanted, reason = floor, reason + f"; context needs {ctx_need} tokens"

        tier, placement, why = self._fit(wanted, ctx_need)
        if new_turn or escalate:
            sess.tier = wanted
        decision = Decision(
            tier=tier,
            spec=self.catalog.specs[tier],
            placement=placement,
            wanted=wanted,
            reason=f"{reason}; {why}",
            session=key,
            ctx_need=ctx_need,
        )
        self._log(decision, requested, agent, last_user)
        return decision

    def _fit(self, wanted: str, ctx_need: int) -> tuple[str, Placement, str]:
        available = self.catalog.tiers
        if not available:
            raise RuntimeError("no models available in Ollama")
        tier = self.catalog.nearest_tier(wanted) or available[-1]
        order = [t.name for t in self.cfg.tiers]
        idx = order.index(tier)
        ok_ctx = {t.name for t in self.cfg.tiers if t.num_ctx >= ctx_need} or {order[-1]}

        if self.cfg.router.prefer_loaded:
            for name in order[idx:]:
                spec = self.catalog.specs.get(name)
                if spec and name in ok_ctx and self.broker.is_loaded(spec.variant):
                    return name, self.broker.place(spec, True), "already loaded" if name == tier else f"reusing loaded {name}"

        placement, _ = self.broker.plan(self.catalog.specs[tier], True)
        if placement is not Placement.NONE:
            return tier, placement, "fits" if placement is Placement.GPU else "fits with CPU offload"

        for name in reversed(order[:idx]):
            spec = self.catalog.specs.get(name)
            if spec and name in ok_ctx:
                p, _ = self.broker.plan(spec, True)
                if p is not Placement.NONE:
                    return name, p, f"downgraded: {tier} does not fit now"
        return tier, Placement.NONE, "nothing fits; letting Ollama place it"

    def _log(self, d: Decision, requested: str, agent: str, text: str) -> None:
        rec = {
            "ts": time.time(),
            "session": d.session,
            "agent": agent,
            "requested": requested,
            "wanted": d.wanted,
            "tier": d.tier,
            "model": d.spec.base,
            "placement": d.placement.value,
            "reason": d.reason,
            "ctx_need": d.ctx_need,
            "text": text[:200],
        }
        log.info("route %s -> %s (%s)", d.session, d.tier, d.reason)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def sessions_status(self) -> dict[str, Any]:
        return {k: asdict(v) for k, v in self.sessions.items()}


def _since_last_user(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            return msgs[i + 1 :]
    return msgs
