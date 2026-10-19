"""OpenCode integration: provider + MCP config, and the 1.x plugin."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

from .config import Config


def provider_config(cfg: Config, base_url: str | None = None) -> dict[str, Any]:
    url = (base_url or cfg.url).rstrip("/") + "/v1"
    max_ctx = max(t.num_ctx for t in cfg.tiers)
    models: dict[str, Any] = {
        "auto": {
            "name": "Auto (yuzuru picks the size)",
            "tool_call": True,
            "limit": {"context": max_ctx, "output": min(16384, max_ctx // 4)},
        }
    }
    for t in cfg.tiers:
        models[t.name] = {
            "name": f"{t.name.capitalize()} (yuzuru)",
            "tool_call": True,
            "limit": {"context": t.num_ctx, "output": min(16384, t.num_ctx // 4)},
        }
    options: dict[str, Any] = {"baseURL": url}
    if cfg.token:
        options["apiKey"] = "{env:YUZURU_TOKEN}"
    return {"npm": "@ai-sdk/openai-compatible", "name": "yuzuru (local)", "options": options, "models": models}


def mcp_config(cfg: Config, base_url: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "remote", "url": (base_url or cfg.url).rstrip("/") + "/mcp"}
    if cfg.token:
        entry["headers"] = {"Authorization": "Bearer {env:YUZURU_TOKEN}"}
    return entry


def add_mcp(conf: dict[str, Any], entry: dict[str, Any], major: int | None) -> None:
    """OpenCode 2.x nests servers under mcp.servers; 1.x puts them directly under mcp."""
    mcp = conf.setdefault("mcp", {})
    if major is None or major >= 2:
        mcp.setdefault("servers", {})["yuzuru"] = entry
    else:
        mcp["yuzuru"] = {**entry, "enabled": True}


def opencode_major() -> int | None:
    if not shutil.which("opencode"):
        return None
    try:
        out = subprocess.run(["opencode", "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"(\d+)\.\d+", out)
    return int(m.group(1)) if m else None


def plugin_source() -> str:
    return resources.files("yuzuru").joinpath("opencode/yuzuru.ts").read_text()


def install(cfg: Config, project: Path | None, set_default: bool) -> list[str]:
    """Write provider + MCP config (and the plugin for OpenCode 1.x). Returns what changed."""
    if project:
        plugin_dir, config_file = project / ".opencode" / "plugins", project / "opencode.json"
    else:
        base = Path.home() / ".config" / "opencode"
        plugin_dir, config_file = base / "plugins", base / "opencode.json"
    changed: list[str] = []

    major = opencode_major()
    plugin_file = plugin_dir / "yuzuru.ts"
    if major is not None and major < 2:
        plugin_dir.mkdir(parents=True, exist_ok=True)
        plugin_file.write_text(plugin_source())
        changed.append(str(plugin_file))
    elif plugin_file.exists():
        # OpenCode 2.x rejects the classic plugin format; remove a stale copy.
        plugin_file.unlink()
        changed.append(f"removed {plugin_file}")

    if config_file.with_suffix(".jsonc").exists() and not config_file.exists():
        raise SystemExit(f"{config_file.with_suffix('.jsonc')} exists; add the config by hand (see `yuzuru opencode`)")
    conf: dict[str, Any] = {"$schema": "https://opencode.ai/config.json"}
    if config_file.exists():
        try:
            conf = json.loads(config_file.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"{config_file} is not plain JSON ({e}); add the config by hand (see `yuzuru opencode`)")
        shutil.copy(config_file, config_file.with_suffix(".json.bak"))
    conf.setdefault("provider", {})["yuzuru"] = provider_config(cfg)
    add_mcp(conf, mcp_config(cfg), major)
    if set_default or "model" not in conf:
        conf["model"] = "yuzuru/auto"
    if set_default or "small_model" not in conf:
        conf["small_model"] = f"yuzuru/{cfg.tiers[0].name}"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(conf, indent=2, ensure_ascii=False) + "\n")
    changed.append(str(config_file))
    return changed


def cmd_opencode(args: argparse.Namespace, cfg: Config) -> None:
    if not args.install:
        snippet = {
            "$schema": "https://opencode.ai/config.json",
            "model": "yuzuru/auto",
            "small_model": f"yuzuru/{cfg.tiers[0].name}",
            "provider": {"yuzuru": provider_config(cfg)},
        }
        add_mcp(snippet, mcp_config(cfg), opencode_major())
        print(json.dumps(snippet, indent=2))
        print("\n# OpenCode 1.x only: `yuzuru opencode --install` also adds plugins/yuzuru.ts")
        return
    for path in install(cfg, Path(args.project).resolve() if args.project else None, args.set_default):
        print(f"wrote {path}" if not path.startswith("removed") else path)
