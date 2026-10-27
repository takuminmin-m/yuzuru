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
- OpenCode 2.x では MCP サーバーに `codemode: false` を付ける。2.x の既定では MCP ツールが JavaScript を書いて呼ぶ
  `execute` ツールの奥に隠れ、ローカルの小さなモデルではまず呼べないため
- `--local-only` を付けると OpenCode 組み込みの無料クラウドモデル（`opencode` プロバイダ）を無効にする。yuzuru が
  落ちているとき、OpenCode はそれらに黙って切り替えてコードを外部に送ってしまうので、ローカル専用で使うなら推奨
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
yuzuru run --cpu -- python lgbm.py         # GPU を使わないジョブ（LLM はそのまま、GPU は見せない）
```

流れ: lease 作成 → 生成中の応答を待つ（`drain_timeout`）→ GPU上のモデルをアンロードし、実際に VRAM から消えるまで待つ（`unload_timeout`、
消えなければジョブのログに警告）→ ジョブ起動 → VRAM使用量が `settle_seconds` 安定したら
「確定」→ 残り（− headroom）を LLM が使える → 終了すると lease を解放し、元のモデルを再ロード。
ラッパーが kill されても、サーバーがPIDの消滅を検知して lease を回収する。

`--gpu`（MCP `yuzuru_train` の `gpus`）で GPU を専有させたジョブ（と `yuzuru reserve --gpu` で量を指定せずに確保した GPU）の
間は、その GPU を LLM に一切使わせない。`--vram` で量を指定した場合は、その量だけを差し引き、残りは LLM が使える。
予算はほかの GPU だけで計算し、`ollama.manage: true` なら管理下の Ollama をその GPU を隠して再起動する
（Ollama は見えている GPU すべてに空き容量に応じて層を置くので、予算だけでは止められない。学習がまだ
メモリを増やしている最中に LLM の層が隣に載り、学習側が OOM する）。ジョブが終わると全 GPU で再起動する。
再起動の間はリクエストを待たせ、生成中のものは `drain_timeout` まで終わるのを待つ。`manage` なしでは
予算から外すだけで、Ollama がその GPU を使う可能性は残る（警告を出す）。

`--cpu`（MCP `yuzuru_train` / `yuzuru_wait` の `device: "cpu"`）は GPU を使わないジョブ（LightGBM、CPU の XGBoost、
scikit-learn、前処理など）用。LLM のアンロードも VRAM の安定待ちもせず、待っている間も LLM を降ろさない
（lease は作るので、終了後の再開や `status` の表示はそのまま）。ジョブには `CUDA_VISIBLE_DEVICES=-1` を渡すので、
うっかり GPU を使うコードは GPU を取らずに失敗するか CPU で動く。MCP から起動した場合は scope に
`CPUWeight=isolation.cpu_job_weight`（既定 20、Ollama は既定の 100）を付ける。空いているコアはすべて使えるが、
LLM が CPU 側の層を計算している間は LLM に譲る（GPU に載りきらないモデルでは、生成速度が CPU に左右される）。

### VRAM / GPU を強制的に確保する

ジョブを始める前に「この GPU（この量）は空けておいて」と押さえておける。

```bash
yuzuru reserve --gpu 1                 # GPU1 の空きを丸ごと確保（LLM は即座に退く）
yuzuru reserve --gpu 1 --vram 10G --for 2h
yuzuru reserve --vram 8G --soft        # 予算から差し引くだけ（GPUの種類を問わない）
yuzuru run --use <id> -- python train.py   # 確保した枠でジョブを実行（そのまま lease に引き継ぐ）
yuzuru release <id>
```

既定は **hard**: 小さなホルダープロセスが libcuda（NVIDIA ドライバ同梱。torch 不要）経由で実際に VRAM を
確保するので、Ollama も他のプロセスもその分を使えない。`run --use` するとホルダーが解放した直後に
ジョブへ引き継ぐ（間に LLM が割り込まない）。`--soft` は確保量を LLM の予算から引くだけで、NVIDIA 以外でも使える。
`--for` を過ぎるか、ホルダーが落ちると自動で解放され、元のモデルが戻る。

OpenCode 上では MCP の `yuzuru_train` ツール（バックグラウンド実行 + ログファイル）を使うよう、MCP の
instructions とツール説明でモデルに指示している（bash ツールはタイムアウトがあるので長い学習に向かない）。
OpenCode 1.x ではさらにプラグインが `python train.py` / `torchrun` / `accelerate launch` などの bash 実行を
自動で `yuzuru run` に包む。

### 使う GPU を限定する（GPU 許可リスト）

「この GPU（と CPU）だけを使う」は設定 1 か所で済む。

```yaml
# ~/.config/yuzuru/config.yaml
resources:
  gpus: [0]            # index か UUID（GPU-a0d7...）。空 = 全部（既定）
ollama:
  manage: true         # yuzuru 専用の ollama serve を起動・監視する（下記）
  # port: auto         # 既定。port_range の空きポートを自動で選ぶ。固定するなら 11437 など
  # port_range: [11436, 11499]
  # env: {OLLAMA_FLASH_ATTENTION: "1"}   # ollama.service に設定していた環境変数があればここへ
```

- **`resources.gpus`**: リソース probe は許可した GPU だけを報告する。予算、lease（`yuzuru run`）、
  `yuzuru reserve`（許可した GPU が 1 枚なら `--gpu` 省略可）、`yuzuru status` はそれ以外の GPU を一切見ない。
  `--gpu` に許可外の GPU を指定するとエラーになる
- `yuzuru run` / MCP の `yuzuru_train` で起動したジョブには、`--gpu` を付けなくても `CUDA_VISIBLE_DEVICES` が
  許可した GPU の UUID に設定される（呼び出し側の `CUDA_VISIBLE_DEVICES` は上書きする）。サーバーに繋がらない
  ときも設定ファイルの値をそのまま使う
  - ジョブの Python はこの値を自分の GPU 以外に変えられない。yuzuru はジョブの `PYTHONPATH` の先頭に
    `sitecustomize.py`（`src/yuzuru/gpu_pin/`）を置き、スクリプトが lease 外の GPU（番号での指定を含む）を
    設定すると `RuntimeError` にする。lease の GPU の一部に絞る、`-1` で全部隠すのは可。削除は無視する
    （HF Trainer の中で `accelerate` が環境変数を一時的に全部消して戻すため）。Python の起動前に別の値が
    入っていたら、置き換えて警告を出す。ジョブの venv などにある `sitecustomize` はその後に通常どおり動く
  - Python の中での制限なので、`python -S` / `-E` / `-I` や Python 以外のプログラム、`subprocess` に `env=` で
    直接渡した値には効かない
- **`ollama.manage: true`**: `resources.gpus` だけでは、システムの Ollama（`ollama.service`）は全 GPU を見たまま。
  除外した GPU を hard reserve で埋めても、Ollama はそこに計算バッファを置こうとして OOM する。`manage` を
  有効にすると、yuzuru が自前の `ollama serve` を `CUDA_VISIBLE_DEVICES=<許可した GPU の UUID>` 付きで起動し、
  `ollama_url` をそちらに向ける
  - ポートは既定で自動選択（`port: auto`）。`port_range`（既定 11436〜11499）の中から、起動時に空いている最初の
    ポートを使う。手で立てた Ollama や別の yuzuru の管理 Ollama がいても衝突せず、複数同時に動かせる。
    yuzuru 自身のポート（`port`）と 11434 は選ばない。再起動時は前回のポートが空いていればそのまま使い、
    埋まっていれば別のポートに移る（yuzuru 内部の接続先も追従する）。実際のポートは `yuzuru status` と
    `~/.local/state/yuzuru/ollama-<yuzuru の port>.json` に出る（yuzuru ごとに別ファイルなので、同じ `state_dir` の
    yuzuru を複数動かしても互いの管理 Ollama を止めない）
  - `port: 11437` のように固定すると、そのポートが埋まっている間は起動せずに警告を出して待ち、空いたら起動する
  - `OLLAMA_MODELS` は `ollama.models`（既定 `~/.local/share/yuzuru-ollama/models`）。システムのモデル置き場
    （`$OLLAMA_MODELS`、`/usr/share/ollama/.ollama/models`、`~/.ollama/models` の順に探す。`ollama.mirror_from` で指定可、
    `none` で無効）の blob をシンボリックリンクで、manifest をコピーで取り込むので、重みは複製しない。
    システム側で `ollama pull` したモデルは 1 分以内に取り込まれる。yuzuru が作る派生モデルはこちらの置き場に書かれる。
    ただし Ollama 0.40 以降は派生モデルを作るときに元の blob の更新時刻を書き換えるため、blob が別ユーザー（`ollama`）の
    ものだと作れない。その場合はシステムの Ollama（:11434）で作って取り込む（システム側の `ollama list` にも出る）
  - 可能なら `systemd-run --user --scope` で専用の cgroup に入れて起動するので、ホスト RAM の実測
    （`memory.current`）はシステムの `ollama.service` ではなくこのプロセスが対象になる（`ollama.scope: off` で無効。
    scope が使えない環境ではプロセスツリーの RSS で測る）
  - 落ちたら自動で再起動し、yuzuru の終了時に止める。yuzuru が強制終了されて残ったものは、次の起動時に止めてから
    起動し直す。ログは `~/.local/state/yuzuru/ollama-<yuzuru の port>.log`、状態は `yuzuru status` の `Ollama (managed)` 行
  - システムの `ollama.service` は 11434 で動いたままでよい（止めるなら `sudo systemctl disable --now ollama`）。
    OpenCode などは yuzuru 経由で使う（11434 を直接使うと、その分は yuzuru から見て他プロセスの使用量になる）

### 待っている間は LLM を止め、終わったら自動で再開する

エージェントが学習や前処理の終了を待つ間、LLM はメモリを抱えたまま何もしていない。yuzuru はその間、
ルーター用の小モデルや RAM に退避したモデルも含めて全モデルをアンロードし、プロセスが終わると
セッションの tier のモデルを先に読み込んでから、結果（終了コード、ログの末尾、待つ前にモデルが残したメモ）を
添えてセッションを再開する。OpenCode は再開のターンで会話の履歴をすべて送り直すので、文脈はそのまま戻る。

- MCP `yuzuru_train`: 既定で `resume_when_done: true`。ジョブを起動したらモデルはターンを終える
- MCP `yuzuru_wait`: 自分で起動したわけではないプロセス（PID または lease）を待つ。`device: "cpu"`（または
  CPU ジョブの lease / PID）なら LLM は降ろさない
- CLI: `yuzuru run --resume --session <id> --note "..." --log train.log -- python train.py`、
  `yuzuru wait --pid 1234 --session <id>`、`yuzuru events -f`

再開は `opencode run --session <id> "<結果>"` で行う（OpenCode 1.x / 2.x 共通。`resume.command` で変更可）。
ジョブを起動したターンがまだ続いている間（すぐ落ちた、エージェントがターンを終えずに作業を続けた、など）は、
そのターンが終わるまで再開しない。同じセッションに `opencode run` が 2 つ並ぶと、2 つのエージェントが同じ
ファイルを編集しジョブを起動してしまうため。ターンの続き具合は、最後の応答がツール呼び出しで終わったか
（`finish_reason: tool_calls`）で判断する（応答が来ないまま `resume.turn_idle_seconds`、既定 600 秒たてば終わったとみなす）。
そのターンの中でエージェントが次のジョブを起動していれば、前のジョブの結果では再開せず、次のジョブの終了を待つ。
OpenCode のバックグラウンドサービスにつながっている TUI なら、再開したターンがそのまま表示される。

再開したエージェントは、それぞれ専用の systemd user scope（`yuzuru-agent-*.scope`）で動く。MCP `yuzuru_train` の学習ジョブも同じく `yuzuru-job-*.scope` で動く。エージェントが
`uv run python debug.py` のように直接実行したスクリプトもその scope に入るので、メモリを食い尽くしても
`isolation.memory_max`（既定 `80%`、systemd の `MemoryMax`）で止まり、OOM で殺されるのはそのスクリプトだけになる（ジョブなら終了コード 137 で lease が終わり、それを添えて再開する）。
scope がないと、スクリプトのメモリは起動元（yuzuru のサービスや端末、IDE、Claude Code など）の unit に数えられ、
カーネルの OOM killer がスクリプトを殺したあとに systemd がその unit ごと止めてしまう（`OOMPolicy=stop` が既定）。
最初のターンも同じように隔離するには `yuzuru agent` から起動する（systemd がなければそのまま実行する。`isolation.scope: off` で無効）:

```bash
yuzuru agent -- opencode run "README.md を読んで課題に取り組んで"
```

OpenCode 2.x の `opencode run` はターンをバックグラウンドサービス（`opencode serve --service`）に渡すので、
ツールはサービスの cgroup で動く。サービスは、それを最初に必要としたコマンドを実行した場所（Claude Code の
シェルなど）で起動される。サービスがまだ動いていなければ上のコマンドが scope の中で起動するが、すでに scope の外で
動いていれば `yuzuru agent` と再開時に警告を出す。そのときは scope の中で起動し直す:

```bash
opencode service stop && yuzuru agent -- opencode service start
```

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

### モデルを選ぶ

```bash
yuzuru models                       # 入っているモデル、担当 tier、MoE かどうか、無効化状態
yuzuru models use medium qwen3:14b  # medium をこのモデルに固定（auto で自動割り当てに戻す）
yuzuru models disable llama3:70b    # 自動割り当て・一覧から外す / enable で戻す
```

選択は `~/.local/state/yuzuru/selection.json` に保存され、設定ファイルの `models` や自動割り当てより優先される。
OpenCode からは tier だけでなく、有効なモデルを名前で直接選べる（`yuzuru/qwen3:14b`。メモリ管理は同じく
yuzuru が行い、tier の自動切り替えはしない）。モデル自身も MCP の `yuzuru_models` で一覧を見て、
`yuzuru_model` に名前を渡してセッションを固定できる。

### ローカルで手に負えないときはクラウドへ

`yuzuru.yaml` の `cloud:` に OpenAI 互換 API（OpenAI / OpenRouter / Groq / 別マシンの vLLM など）を設定すると、
最上位の `cloud` tier として使える。API キーは設定ファイルに書かず、`api_key_env` で指定した環境変数から読む。

| いつクラウドに行くか | 設定 |
|---|---|
| 明示したとき: `yuzuru/cloud`、メッセージ内の `@cloud`、`yuzuru_model cloud` | 常に（`enabled` なら） |
| 最大のローカル tier でもツールエラーが続くとき | `auto: escalation`（既定） |
| ルーターが新しいターンを "hard" と判定したとき | `auto: hard` |
| どのローカル tier のコンテキストにも収まらないとき | `when_context: true`（既定） |
| 学習ジョブが GPU を使っていて、欲しい tier が載らないとき | `when_no_memory: true` |

クラウドに移ったターンは、ツールループの間ずっとクラウドに留まり、次のユーザーターンで改めて判定される。
自動で使う回数は `max_requests_per_hour` で頭打ちにできる（明示した場合は制限しない）。
`yuzuru status` に使用回数が出る。

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

yuzuru はジョブを起動できる（MCP の `train` ツールや lease）ので、Webページから叩かれないようにしている。
`Origin` ヘッダー付きのリクエスト（ブラウザ）は拒否する（許可するなら `allowed_origins`）。トークン未設定時は
`Host` が localhost 以外のリクエストも拒否する（DNS rebinding 対策）。POST の本文は `application/json` のみ受け付ける。
yuzuru 自体は既定で `127.0.0.1` にしか bind しない。外部に公開する場合は `YUZURU_TOKEN` を設定すると
Bearer トークン必須になる（OpenCode 側は `{env:YUZURU_TOKEN}` を apiKey として送る）。

## トラブルシューティング

- `yuzuru status` の先頭に `!` 付きの警告が出る:
  - **GPU が応答しない**（カーネルログに `Xid 79, GPU has fallen off the bus`）: yuzuru は残りの GPU だけで動き続ける。
    再起動か `sudo nvidia-smi -r` で戻ることが多い。繰り返す場合は電源容量・PCIe の挿さり・ライザーを疑う
  - **Ollama がモデルに対応しない RAM を抱えている**: 強制終了されたランナーの分が残ることがある。`sudo systemctl restart ollama`
- OpenCode から送られている実際のリクエストを見たいとき: `YUZURU_DUMP=1 yuzuru serve` で、要求モデルごとの最新リクエストを
  `~/.local/state/yuzuru/last_request_<model>.json` に保存する

## 開発

```bash
uv sync && uv run pytest
```
