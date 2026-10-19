# yuzuru

ローカルLLM（Ollama）と機械学習の学習ジョブで、限られたGPU/RAMを取り合わないようにするオーケストレーター。
OpenCode から `yuzuru/auto` を選ぶだけで、タスクの難易度と空きメモリを見てモデルサイズを自動で切り替え、
学習ジョブを起動するとLLMがVRAMから退き、終わると戻ってくる。

```
 OpenCode ──┬─ OpenAI互換 API  model = auto | small | medium | large   (x-opencode-session でセッション識別)
            └─ MCP  /mcp     yuzuru_status / yuzuru_model / yuzuru_train
    ▼
 yuzuru serve (:11435)
    ├─ Router  : 難易度判定（ルール + CPU上の小モデル）→ 欲しい tier
    ├─ Broker  : GPU/RAM 監視・予算計算・ロード/アンロード・lease
    └─ Catalog : tier ↔ Ollama モデル、num_ctx 付き variant、実測メモリ
    ▼
 Ollama (:11434)  ← 複数GPUへの分割・RAMへの退避は Ollama に任せる
```

## ハードウェア非依存の考え方

- **tier は抽象的な大きさ**（small / medium / large、増減も可）。設定がなければ Ollama にあるモデルを
  パラメータ数で自動的に割り当てる。GPUが何枚でも、GPUなし（CPU / Apple のユニファイドメモリ）でも同じ設定で動く。
- **配置は Ollama に任せ、yuzuru は「どの大きさなら載るか」だけを決める**。予算 =
  GPU総量 − 他プロセス（学習など）の使用量 − lease の予約 − マージン。足りない分は RAM に逃がす
  （MoE は 70%、dense は 25% まで）か、小さい tier に落とす。
- **メモリ量は最初はGGUFメタデータからの推定、一度ロードした後は実測値**（`~/.local/state/yuzuru/profiles.json`）。
- Ollama の OpenAI 互換APIはリクエストごとの `num_ctx` を無視するので、tier ごとに `num_ctx` を焼き込んだ
  派生モデル（`yuzuru-large:qwen3.6-32k` など。重みは共有なのでディスクは消費しない）を自動で作る。

## セットアップ

```bash
uv tool install -e /home/takumi/Documents/yuzuru   # `yuzuru` コマンドを PATH に
yuzuru serve                                       # 常駐（systemd 化は下記）
yuzuru opencode --install                          # ~/.config/opencode にプロバイダとプラグインを追加
```

`yuzuru opencode --install` は次を行う:
- `opencode.json` に `yuzuru` プロバイダ（`auto` と各 tier）と MCP サーバー（`http://127.0.0.1:11435/mcp`）を追加し、
  `model` / `small_model` が未設定なら `yuzuru/auto` / `yuzuru/small` にする（既存ファイルは `.json.bak` に退避。
  `--set-default` で上書き）。MCP の書式は OpenCode のバージョン（1.x: `mcp.<name>`、2.x: `mcp.servers.<name>`）に合わせる
- OpenCode 1.x のときだけ `plugins/yuzuru.ts`（依存なし）を置く。2.x はプロバイダへ `x-opencode-session` を
  自前で送るのでプラグイン不要（2.x のプラグインAPIにはツール実行フックが無いため、学習コマンドの自動ラップは 1.x のみ）
- プロジェクト単位にしたい場合は `--project <dir>`

小さいモデルも入れておくと、ルーターがそれで難易度を判定し、軽い作業はそちらに回す:

```bash
ollama pull qwen3:4b      # 例: small 兼 分類器
ollama pull qwen3:14b     # 例: medium（学習中でも 8GB 級GPUに載る）
```

## 使い方

### 学習ジョブ

```bash
yuzuru run -- python train.py              # LLMをVRAMから退かして実行、終わったら戻す
yuzuru run --gpu 1 -- python train.py      # GPU1 を専有（CUDA_VISIBLE_DEVICES を UUID で設定）
yuzuru run --vram 10G -- python train.py   # 10GB を予約し続ける（学習途中の増加に備える）
```

流れ: lease 作成 → GPU上のモデルをアンロード → ジョブ起動 → VRAM使用量が `settle_seconds` 安定したら
「確定」→ 残り（− headroom）を LLM が使える → 終了すると lease を解放し、元のモデルを再ロード。
ラッパーが kill されても、サーバーがPIDの消滅を検知して lease を回収する。

OpenCode 上では MCP の `yuzuru_train` ツール（バックグラウンド実行 + ログファイル）を使うよう、MCP の
instructions とツール説明でモデルに指示している（bash ツールはタイムアウトがあるので長い学習に向かない）。
OpenCode 1.x ではさらにプラグインが `python train.py` / `torchrun` / `accelerate launch` などの bash 実行を
自動で `yuzuru run` に包む。

### ルーティング

| 何で決まるか | 例 |
|---|---|
| OpenCode でモデルを明示 | `yuzuru/large` |
| メッセージ内の指定 | `この設計を見直して @large` |
| セッション固定（モデル自身も呼べる） | MCP `yuzuru_model` ツール / `yuzuru pin <session> large` |
| 軽い作業 | タイトル生成などは `small_model`（= `yuzuru/small`）、1.x プラグインならエージェント名でも判定 |
| 新しいユーザーターン | 分類器（CPU上の small）が easy / medium / hard を判定（なければキーワードルール） |
| エージェントループの続き | そのターンの tier を維持（途中でモデルを入れ替えない） |
| ツールエラーが続く | 1段上の tier に昇格 |
| コンテキスト長 | 入らない tier は使わない |

そのうえで空きメモリと照合する: 既に載っている同等以上のモデルがあればそれを使い（入れ替えコストを避ける）、
載らなければ RAM 退避 → 小さい tier へ降格、の順に試す。判定はすべて `~/.local/state/yuzuru/decisions.jsonl`
に記録されるので、後でルーターの評価や学習に使える。

### 確認

```bash
yuzuru status                     # GPU/RAM、LLM予算、tierごとの配置見込み、ロード中モデル、lease
yuzuru route "このlossが発散する原因を調べて"   # ルーターのドライラン
yuzuru warm large / yuzuru unload
```

## 常駐させる（systemd user unit）

```ini
# ~/.config/systemd/user/yuzuru.service
[Unit]
Description=yuzuru LLM orchestrator
After=network.target

[Service]
ExecStart=%h/.local/bin/yuzuru serve
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now yuzuru
```

## リモート操作（検討メモ）

OpenCode v2 は「バックグラウンドサーバー + クライアント」構成で、`opencode --server <URL>` で任意のサーバーに繋げる。

1. **推奨: GPU マシンの OpenCode サーバーに SSH トンネルで繋ぐ**
   ```bash
   # GPUマシン: opencode service status で URL/port を確認（固定したい場合は opencode service set）
   ssh -N -L 4096:127.0.0.1:<port> gpu-box
   opencode --server http://127.0.0.1:4096      # 手元の OpenCode TUI
   ```
   ファイル操作・bash・学習ジョブ・yuzuru のツールはすべて GPU マシン上で動くので、そのまま使える。
2. **SSH して GPU マシン上で `opencode` を起動する** — 一番単純。tmux と組み合わせれば切断にも強い。
3. **手元の OpenCode から yuzuru だけを使う**（`ssh -L 11435:127.0.0.1:11435`）— LLM の推論は GPU マシンで
   行われるが、bash やファイル操作は手元で実行されるので、学習ジョブを GPU マシンで動かす用途には向かない。

yuzuru 自体は既定で `127.0.0.1` にしか bind しない。外部に公開する場合は `YUZURU_TOKEN` を設定すると
Bearer トークン必須になる（OpenCode 側は `{env:YUZURU_TOKEN}` を apiKey として送る）。

## 開発

```bash
uv sync && uv run pytest
```
