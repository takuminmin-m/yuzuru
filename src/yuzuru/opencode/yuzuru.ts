// yuzuru plugin for OpenCode 1.x (classic hook API). Dependency-free: the only
// import is type-only, so no package install is needed.
//
// OpenCode 2.x cannot load this format and does not need it: it sends
// x-opencode-session to providers by itself, and the yuzuru tools come from
// the MCP server configured next to the provider.
//
// - Sends the session id / agent name so routing sticks per session and cheap
//   agents go to the small model.
// - Runs GPU training commands under a yuzuru lease, so the local LLM gets out
//   of VRAM before the job allocates.
//
// Environment: YUZURU_BIN (default "yuzuru"), YUZURU_PROVIDER (default "yuzuru"),
// YUZURU_TRAIN_PATTERN (regex), YUZURU_WRAP_TRAINING=0 to disable wrapping.
import type { Plugin } from "@opencode-ai/plugin"

const BIN = process.env.YUZURU_BIN ?? "yuzuru"
const PROVIDER = process.env.YUZURU_PROVIDER ?? "yuzuru"
const TRAIN_RE = new RegExp(
  process.env.YUZURU_TRAIN_PATTERN ??
    String.raw`\b(torchrun|accelerate\s+launch|deepspeed|lightning\s+run)\b|\bpython3?\s+(-m\s+)?\S*(train|finetune|fine_tune|pretrain|fit)\S*`,
)

const quote = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`

export const YuzuruPlugin: Plugin = async () => ({
  "chat.headers": async (input, output) => {
    if (input.provider?.info?.id !== PROVIDER) return
    output.headers["x-yuzuru-session"] = input.sessionID
    output.headers["x-yuzuru-agent"] = input.agent
  },

  "tool.execute.before": async (input, output) => {
    if (input.tool !== "bash" || process.env.YUZURU_WRAP_TRAINING === "0") return
    const cmd = output.args?.command
    if (typeof cmd !== "string" || cmd.includes(`${BIN} run`) || !TRAIN_RE.test(cmd)) return
    // A backgrounded command would end the lease immediately; yuzuru_train handles that case.
    if (/&\s*$|\bnohup\b|\bsetsid\b/.test(cmd)) return
    output.args.command = `${BIN} run --label opencode -- bash -c ${quote(cmd)}`
  },
})
