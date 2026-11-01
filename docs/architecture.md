# yuzuru の仕組みと構造

このドキュメントは、yuzuru の内部の構成と動きを説明します。
使い方は [README](../README.md)、エージェントを動かす通しの手順は [チュートリアル](tutorial.md) を見てください。
用語（tier、lease、scope など）は [README の用語](../README.md#用語) で説明しています。

## 全体像

yuzuru は 1 つの常駐プロセス（`yuzuru serve`）と、それに HTTP で話しかける CLI でできています。

```
 OpenCode（エージェント）
   │  /v1（LLM API）        /mcp（yuzuru_train など）
   ▼                        ▼
 yuzuru serve（:11435）
   ├─ server.py   HTTP の入口（/v1、/mcp、/yuzuru/*）
   ├─ router.py   リクエストごとに tier を選ぶ
   ├─ broker.py   メモリの予算、モデルの読み込み・解放、lease
   │    └─ resources.py  GPU / RAM の測定（nvidia-smi、/proc/meminfo、cgroup）
   ├─ catalog.py  tier とモデルの対応、必要メモリ
   ├─ waits.py    ジョブの終了を見張り、会話を再開する
   └─ managed.py  専用の ollama serve を起動・監視する（任意）
   │
   ▼
 Ollama（システムの ollama.service か、管理下の Ollama）

 yuzuru CLI ──（/yuzuru/* 制御 API）──▶ yuzuru serve

 yuzuru_train ──▶ yuzuru run --detach（yuzuru-job-*.scope）──▶ 学習ジョブ
 ジョブ終了 ──▶ resume.command（yuzuru-agent-*.scope）──▶ エージェントの会話を再開
```

## コンポーネント

### CLI（`cli.py`）

`yuzuru` コマンドの入口です。`serve`、`agent`、`opencode` 以外のサブコマンドは、HTTP で `yuzuru serve` の制御 API を呼びます。
接続先は `--url`、環境変数 `YUZURU_URL`、設定の `host` / `port` の順に決まります。

| サブコマンド | 役割 | 呼ぶ API |
|---|---|---|
| `serve [--host] [--port] [--log-level]` | サーバーを起動する | — |
| `run [options] -- <cmd>` | lease を取ってジョブを実行し、終わったら返す | `POST /yuzuru/leases`、`DELETE /yuzuru/leases/{id}` |
| `agent -- <cmd>` | エージェントを scope の中で、GPU を隠して起動する | — |
| `reserve [--gpu] [--vram] [--for] [--soft] [--label]` | GPU メモリを予約する | `POST /yuzuru/reservations` |
| `release <id>` | 予約か lease を終える | `DELETE /yuzuru/leases/{id}` |
| `wait [--pid\|--lease] [--session] [--note] [--log] [--no-suspend] [--no-resume]` | プロセスの終了を待ち、会話を再開する | `POST /yuzuru/watches` |
| `events [-f]` | ジョブの終了と再開の履歴 | `GET /yuzuru/events` |
| `status [--json]` | 状態を表示する | `GET /yuzuru/status` |
| `route <text>` | ルーターの判定だけを試す | `POST /yuzuru/route` |
| `warm <tier> [--force]` / `unload [tier]` | モデルを読み込む / 解放する | `POST /yuzuru/warm`、`POST /yuzuru/unload` |
| `pin <session> <tier>` | セッションを tier に固定する | `POST /yuzuru/sessions/{session}/pin` |
| `models [list\|use\|disable\|enable]` | モデルと tier の対応を見る・変える | `GET /yuzuru/models`、`POST /yuzuru/models/*` |
| `opencode [--install] [--project] [--set-default] [--local-only]` | OpenCode の設定を出力する / 書き込む | — |

`yuzuru run` の中身:

1. `POST /yuzuru/leases` で lease を作る。サーバーは LLM を GPU から降ろしてから応答する（[学習ジョブの流れ](#学習ジョブの流れ)）。
2. 応答に含まれる GPU の UUID を、ジョブの `CUDA_VISIBLE_DEVICES` に設定する。`PYTHONPATH` の先頭に `gpu_pin/` を足す（[ガードレール](#ガードレール)）。
3. ジョブを子プロセスとして起動し、終了を待つ。`SIGTERM` / `SIGHUP` はジョブに転送する。
4. 終わったら `DELETE /yuzuru/leases/{id}?rc=<終了コード>` で lease を返す。

サーバーに繋がらないときは、lease なしでジョブを実行します。`resources.gpus` があれば、その GPU だけを見せます。

### サーバー（`server.py`）

FastAPI のアプリです。起動時に次を作り、バックグラウンドのループを 3 つ回します。

- `Broker.monitor()`: 2 秒ごとに GPU / RAM を測り、予算を超えていれば使っていないモデルを解放する
- `_refresh_catalog()`: 60 秒ごと（モデルがないうちは 5 秒ごと）に Ollama のモデル一覧を読み直す
- `_poll_watches()`: 2 秒ごとに、PID で見張っているプロセスの終了を確かめる

公開するエンドポイント:

| パス | 中身 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 互換。`model` が `auto` / tier 名 / `cloud` / モデル名なら、Router で選んだモデルに転送する |
| `GET /v1/models` | `auto`、tier、`cloud`（使えるとき）、有効なモデル名の一覧 |
| `/v1/*`（そのほか） | Ollama にそのまま転送する |
| `POST /mcp` | MCP サーバー（Streamable HTTP、JSON 応答のみ） |
| `/yuzuru/*` | 制御 API（status、leases、reservations、watches、events、warm、unload、route、pin、models、refresh） |

ストリーミングの応答では、ヘッダーをすぐに返し、最初のトークンが出るまで 10 秒ごとに SSE のコメントを送ります。
モデルの読み込みに数分かかっても、クライアントが接続を切らないようにするためです。

### Router: tier の選び方

`router.py` は、リクエストごとに「欲しい tier」を決め、次に「今載る tier」を決めます。

欲しい tier は、次の優先順で決まります。

1. リクエストの `model` で tier を指定した（`yuzuru/large`）
2. メッセージ内の `@small` / `@medium` / `@large` / `@cloud`
3. セッションの固定（MCP `yuzuru_model`、`yuzuru pin`）
4. 軽い作業のエージェント（`router.small_agents`、既定 `title` / `summary`。OpenCode 1.x のプラグインが `x-yuzuru-agent` ヘッダーで送る）→ 最小の tier
5. 新しいユーザーターン → 難易度を判定する。`router.classifier: llm` なら最小 tier のモデルを CPU で動かして easy / medium / hard を判定し、失敗したらキーワードのルールを使う。`heuristic` ならルールだけ
6. エージェントループの続き（ツールの結果）→ そのターンで答えたモデルを使い続ける。ツールのエラーが `router.error_escalation` 回（既定 3）続くと、一度だけ 1 段上の tier に上げる

さらに、プロンプトが入らない tier は使いません（トークン数はリクエストから推定します）。

今載る tier は、Broker の予算と照らして決めます。

- `router.prefer_loaded: true` なら、欲しい tier 以上ですでに載っているモデルを使う（入れ替えには時間がかかるため）
- 載らなければ、RAM に一部を逃がして載せる（offload）
- それでも載らなければ、小さい tier に落とす

モデルを名前で指定した場合（`yuzuru/qwen3:14b`）は tier を通さず、そのモデルを使います。
セッションは `x-yuzuru-session`、`x-opencode-session`、`x-session-affinity` のいずれかのヘッダーで見分けます。ヘッダーがなければ、システムプロンプトと最初のユーザーメッセージのハッシュで見分けます。
判定はすべて `~/.local/state/yuzuru/decisions.jsonl` に 1 行ずつ記録します。

### Broker（メモリの予算と lease）

`broker.py` は「LLM が今どれだけメモリを使ってよいか」を決め、Ollama のモデルを読み込み・解放してそれを守ります。
モデルを複数の GPU にどう分けるか、RAM にどれだけ逃がすかは Ollama に任せます。Broker が決めるのは「どの大きさなら載せてよいか」だけです。

**予算**（`Broker.budget()`）:

- GPU（fast）= 共有している GPU の総量 − ほかのプロセスの使用量 − lease の予約分 − `resources.vram_margin_mb` × GPU 数
- RAM（slow）= 空き RAM + Ollama が今使っている RAM − `resources.ram_reserve_mb`
- GPU がない環境では、RAM を 1 つの予算として扱う（ユニファイドメモリ）
- lease が GPU を丸ごと取った場合、その GPU は予算の計算から外す

**配置**（`place()`）: モデルが GPU に全部載るなら `gpu`、一部を RAM に逃がせば載るなら `offload`、載らなければ `none` です。
RAM に逃がしてよい割合は tier の `max_cpu_fraction` で決まり、未設定なら MoE モデルは 70%、dense モデルは 25% です。

**lease の状態**:

| 状態 | 意味 | 予約する量 |
|---|---|---|
| `draining` | LLM を GPU から降ろしている | 対象 GPU の空き全部 |
| `settling` | ジョブがメモリを確保している最中 | 対象 GPU の空き全部（または `vram`） |
| `active` | 使用量が `lease.settle_seconds` の間安定した | `vram` か、これまでのピークの大きい方 + `lease.headroom_mb` |
| `reserved` | `yuzuru reserve` の予約 | 指定した量 |

lease が `draining` / `settling` の間、新しいリクエストのルーティングは待たされます（最大 `lease.settle_timeout` 秒ほど）。
ジョブの PID が消えると、Broker は lease を自動で回収します（`yuzuru run` が kill された場合も）。

**hard 予約**: `holder.py` を別プロセスで起動し、libcuda（NVIDIA ドライバに付属）で実際に VRAM を確保します。torch は要りません。
`yuzuru run --use <id>` でジョブに引き継ぐときは、holder を止めた直後に lease に切り替えるので、間に LLM が入り込みません。

### Catalog（モデルと必要メモリ）

`catalog.py` は、tier ごとに Ollama のモデルを割り当て、必要なメモリを見積もります。

- 割り当ての優先順: `yuzuru models use` の選択（`selection.json`）→ 設定の `tiers[].models` → 自動（パラメータ数が `auto_params_max_b` 未満で最大のモデル）
- Ollama の OpenAI 互換 API はリクエストごとの `num_ctx` を無視します。そこで tier ごとに `num_ctx` を焼き込んだ派生モデルを作ります（例 `yuzuru-large:qwen3-14b-32k`）。重みは元のモデルと共有するので、ディスクは増えません
- 必要メモリは、最初は GGUF のメタデータ（ファイルサイズと KV キャッシュの大きさ）から推定します。一度読み込むと実測値に置き換え、`profiles.json` に保存します
- 難易度判定用のモデル: tier が 2 つ以上あり、最小 tier のモデルが 9B 以下なら、そのモデルを `num_ctx` 4096・CPU 専用で使います

### Probe（GPU と RAM の測定）

`resources.py` は、GPU（`nvidia-smi`）と RAM（`/proc/meminfo`）を測ります。
Ollama が実際に使っているホスト RAM は、Ollama の cgroup の `memory.current` から測ります（測れなければ Ollama の報告値を使います）。
`resources.gpus` を設定すると、Probe はその GPU だけを報告します。以降、予算・lease・予約・`status` はほかの GPU を見ません。
`nvidia-smi` の一括取得が失敗したときは GPU を 1 枚ずつ問い合わせ、応答しない GPU を `status` の警告に出します。

### 管理下の Ollama（`managed.py`）

`ollama.manage: true` のとき、yuzuru は専用の `ollama serve` を起動します。システムの Ollama（`ollama.service`、:11434）は動いたままで構いません。

- `CUDA_VISIBLE_DEVICES` を `resources.gpus` の UUID に限定して起動します。システムの Ollama は全 GPU を見ているので、予算だけでは除外した GPU を使わせないようにできません
- lease が GPU を丸ごと取ると、その GPU を隠して再起動します。ジョブが終わると、全 GPU で再起動します。再起動の間、リクエストは待たされます
- ポートは既定で `ollama.port_range`（11436〜11499）の空きから選びます。yuzuru 自身のポートと 11434 は使いません。実際のポートは `yuzuru status` と `~/.local/state/yuzuru/ollama-<yuzuru のポート>.json` に出ます
- モデル置き場は `ollama.models`（既定 `~/.local/share/yuzuru-ollama/models`）です。システムの置き場（`ollama.mirror_from`、既定は `$OLLAMA_MODELS`、`/usr/share/ollama/.ollama/models`、`~/.ollama/models` の順に探す）から、重み（blob）はシンボリックリンクで、manifest はコピーで取り込みます。1 分ごとに取り込み直すので、システム側で `ollama pull` したモデルも使えます
- 共有した blob に書き込めず派生モデルを作れないときは、システムの Ollama（:11434）で作ってから取り込みます
- 可能なら専用の scope で起動し、ホスト RAM を正確に測ります（`ollama.scope`）
- 落ちたら自動で再起動し、yuzuru の終了時に止めます。前回の yuzuru が強制終了されて残ったものは、次の起動時に止めます。ログは `~/.local/state/yuzuru/ollama-<yuzuru のポート>.log` です

### MCP サーバー（`mcp.py`）

エージェントが呼べるツールを `/mcp` で公開します。OpenCode では `yuzuru_` を前に付けた名前で見えます。

| ツール | 引数 | 動き |
|---|---|---|
| `train` | `command`、`cwd`（必須）、`device`、`gpus`、`vram`、`log`、`resume_when_done`、`note` | `yuzuru run --detach` を `cwd` で起動する。GPU ジョブは `vram` か `gpus` がないとエラーを返す。`resume_when_done`（既定 true）なら終了時に会話を再開する |
| `wait` | `pid` か `lease`、`note`、`log`、`cwd`、`device` | そのプロセスの終了を待ち、会話を再開する |
| `status` | — | GPU / RAM、予算、tier の配置見込み、lease |
| `models` | — | 入っているモデルと担当 tier |
| `model` | `tier`（必須）、`session` | セッションの tier かモデルを固定する（`auto` で解除） |

`train` が起動するジョブのログは、`log` を省くと `.yuzuru/logs/` の下に作られます。
MCP の `instructions` とツールの説明で、学習は bash ではなく `train` で起動するようモデルに伝えています。

### Waiter（待機と自動再開）

`waits.py` は、プロセス（lease か PID）とエージェントのセッションを結びつけます。この結びつきを watch と呼びます。

- `yuzuru_train` で起動したジョブには、自動で watch が付きます
- watch がある間、待っているセッションが `resume.suspend_idle_seconds`（既定 20 秒）静かになったら、全モデルを解放します（`resume.suspend_all`）。CPU ジョブの watch では解放しません
- プロセスが終わると、イベントを記録します（`yuzuru events` で見えます。サーバーのメモリ上に直近 200 件）
- 再開する前に、そのセッションの tier のモデルを読み込みます（`resume.warm_before_resume`）
- `resume.command` を、エージェント用の scope と環境で起動します。メッセージには、終了コード、経過時間、ログの末尾（`resume.log_tail_lines` 行）、エージェントのメモが入ります

### OpenCode 連携（`opencode.py`、`opencode/yuzuru.ts`）

`yuzuru opencode --install` は、OpenCode の設定（既定 `~/.config/opencode/opencode.json`、`--project <dir>` ならそのディレクトリ）に次を書き込みます。既存のファイルは `.json.bak` に残します。

- `provider.yuzuru`: `auto`、各 tier、`cloud`（設定したとき）、有効なモデル名
- MCP サーバー: OpenCode 2.x は `mcp.servers.yuzuru`（`codemode: false` 付き）、1.x は `mcp.yuzuru`。2.x の既定では MCP ツールが JavaScript を書いて呼ぶ `execute` ツールの奥に隠れ、小さいローカルモデルではまず呼べないためです
- `model` / `small_model` が未設定なら `yuzuru/auto` / `yuzuru/<最小の tier>`（`--set-default` で上書き）
- `--local-only`: `disabled_providers` に `opencode` を足す。yuzuru に繋がらないとき、OpenCode が組み込みの無料クラウドモデルに黙って切り替えるのを防ぎます
- OpenCode 1.x のときだけ、プラグイン `plugins/yuzuru.ts` を置きます。セッション ID とエージェント名をヘッダーで送り、`python train.py`、`torchrun`、`accelerate launch` などの bash コマンドを `yuzuru run` で包みます。2.x はセッション ID を自分で送るので、プラグインは要りません

`--install` を付けなければ、設定の断片を表示するだけです。

### 課題ディレクトリの準備（`scaffold.py`、`templates/`）

`yuzuru init <dir>` は、エージェントに渡す課題ディレクトリに次を作ります。すでにあるファイルは残し、`--force` のときだけ `.bak` を残して作り直します。

- `yuzuru.yaml`（`templates/yuzuru.yaml` から）: `--gpu` の番号を `nvidia-smi` で UUID にして `resources.gpus` に書きます。モデルは `--model`、なければ yuzuru が前回 `large` に使ったもの（派生モデル `yuzuru-large:*` の digest から元のモデルを引く）、それもなければ一番大きいモデルです。`ollama.manage: true`、`isolation.memory_max`、`port` に加え、`resume.command` は出力を `<dir>/.yuzuru/resume.log` に追記する形にします。カレントディレクトリの `yuzuru.yaml` は `~/.config/yuzuru/config.yaml` より先に読まれるので、課題ディレクトリで動かした yuzuru はこの設定を使います
- `opencode.json`: `opencode.py` と同じプロバイダと MCP の設定に、`disabled_providers: ["opencode"]` と、無人実行向けの権限の許可リスト（`scaffold.py` の `PERMISSION`）を足したもの
- `AGENTS.md`（`templates/AGENTS.md` から）: `uv run python` を使う、GPU の処理は `yuzuru_train`（`vram` 必須）で動かしてターンを終える、学習のたびに設定と検証データの指標を `RESULTS.md` に 1 行足す、課題の `README.md` の終わる条件を確かめてから終える、など。採点のある課題を前提にしていません
- `.gitignore` への追記、`uv venv` と `--packages` の導入（`pyproject.toml` があれば `uv sync`、`requirements.txt` があればそれも入れる）、リポジトリでなければ `git init` と最初の commit

## 学習ジョブの流れ

エージェントが `yuzuru_train` を呼んでから、会話が再開するまでの流れです。

1. MCP `train` が `python -m yuzuru.cli run --detach ...` を、`yuzuru-job-*.scope` の中で起動する（`isolation.scope`）。
2. `yuzuru run` が `POST /yuzuru/leases` を呼ぶ。Broker は lease を `draining` で作る。
3. Broker は、生成中の応答が終わるのを最大 `lease.drain_timeout`（既定 300 秒）待つ。次に GPU 上のモデルを解放し、実際に VRAM から消えるのを最大 `lease.unload_timeout`（既定 60 秒）待つ。消えなければ、ジョブのログに警告を出す。
4. lease が GPU を丸ごと取るなら、管理下の Ollama をその GPU を隠して再起動する。
5. lease が `settling` になり、`yuzuru run` がジョブを起動する。ジョブには lease の GPU だけが見える。
6. Waiter が lease の watch を作る。セッションが静かになったら、全モデルを解放する。
7. ジョブの VRAM 使用量が `lease.settle_seconds` 安定すると、lease は `active` になる。予約する量が決まり、残りは LLM が使える。
8. ジョブが終わると、`yuzuru run` が lease を返す。Broker は watch に終了を知らせる。
   GPU を丸ごと取っていた lease なら、Ollama を全 GPU が見える状態で再起動する。このとき、最初のトークンをまだ返していない応答は待たない。
   たとえば、ジョブが GPU を全部取っている間に LLM が CPU で長いプロンプトを処理していた場合、`drain_timeout` まで GPU が遊んでしまうからです。
   再起動で切れた応答は、クライアントにはまだ何も届いていないので、yuzuru が再起動後の Ollama（GPU）に送り直す。トークンを返している途中の応答は、これまでどおり `drain_timeout` まで待つ。
9. Waiter は、そのセッションのターンが終わっているのを確かめる（[ターンが続いている間は再開しない](#ターンが続いている間は再開しない)）。
10. セッションのモデルを読み込み、`resume.command`（既定 `opencode run --session {session} {message}`）を `yuzuru-agent-*.scope` の中で起動する。OpenCode は会話の履歴をすべて送り直すので、文脈はそのまま戻る。
11. ほかに lease がなければ、Broker は lease の前に載っていたモデルを読み込み直す（`lease.restore_after`）。

`yuzuru run --cpu`（`device: "cpu"`）のジョブでは、手順 3、4、7 を行わず、手順 6 でもモデルを解放しません。lease は作るので、終了の検知と再開は同じです。
ジョブには `CUDA_VISIBLE_DEVICES=-1` を渡し、scope に `CPUWeight=isolation.cpu_job_weight`（既定 20）を付けます。LLM が CPU で計算している間は、LLM に CPU を譲ります。

### ターンが続いている間は再開しない

ジョブが終わったとき、そのジョブを起動したエージェントのターンがまだ続いていることがあります（ジョブがすぐ落ちた、エージェントがターンを終えずに作業を続けた、など）。
そこで `opencode run --session` を起動すると、同じ会話でエージェントが 2 つ並んで動いてしまいます。

そのため Waiter は、ターンが終わるまで再開を待ちます。ターンが続いているかどうかは、Router が記録した次の状態で判断します。

- そのセッションの応答を生成中である
- 最後の応答がツール呼び出しで終わっていて（`finish_reason: tool_calls`）、そこから `resume.turn_idle_seconds`（既定 600 秒）たっていない

待っている間にエージェントが次のジョブを起動したら、前のジョブの結果では再開しません。次のジョブの終了で再開します。

### 止まったターンをもう一度動かす

yuzuru が再開したターンが、応答 1 回だけ・ツール呼び出しなしで終わることがあります。そのセッションのジョブも動いていなければ、エージェントを起こすものが何も残りません。
yuzuru はこれを検知すると、「何もせずにターンが終わった。続きから進め、終わっているなら結果をまとめよ」というメッセージで、もう一度再開します。
連続で `resume.stall_nudges` 回（既定 1、0 で無効）までです。ジョブを起動すると回数は戻ります。人が自分のクライアントで進めているターンは対象にしません。

## エージェントの隔離と GPU を隠す仕組み

### systemd scope によるメモリの隔離（`scope.py`）

プロセスのメモリは、そのプロセスを起動した systemd の unit に数えられます。カーネルの OOM killer がプロセスを殺すと、systemd はその unit ごと止めます（`OOMPolicy=stop` が既定）。
たとえばエージェントのスクリプトがメモリを食い尽くすと、yuzuru のサービスや端末、IDE まで止まることがあります。

yuzuru は、次のものをそれぞれ専用の scope に入れます。

| scope | 中身 | 作るところ |
|---|---|---|
| `yuzuru-agent-*.scope` | `yuzuru agent` で起動したエージェント、自動再開したターン | `cli.py`（`agent`）、`waits.py` |
| `yuzuru-job-*.scope` | MCP `yuzuru_train` の学習ジョブ | `mcp.py` |
| `yuzuru-ollama-*.scope` | 管理下の Ollama | `managed.py` |

エージェントとジョブの scope には `MemoryMax=isolation.memory_max`（既定 `80%`）と `OOMPolicy=continue` を付けます。上限を超えたプロセスだけが殺され、ジョブなら終了コード 137 で lease が終わり、それを添えて会話が再開します。
scope は `systemd-run --user --scope` で作ります。使えない環境（コンテナ、systemd のない SSH セッションなど）では、scope なしで実行します。`isolation.scope: off` で無効にできます。

OpenCode 2.x の `opencode run` は、ターンの実行をバックグラウンドサービス（`opencode serve --service`）に渡します。エージェントのツールはサービスの cgroup で動くので、サービスも scope の中で起動する必要があります。

```bash
opencode service stop && yuzuru agent -- opencode service start
```

サービスが scope の外で動いていると、`yuzuru agent` と自動再開のたびに警告を出します。

### GPU のピン留め（`gpu_pin/sitecustomize.py`）

yuzuru が起動するプロセスの `PYTHONPATH` の先頭に、`gpu_pin/` を置きます。その中の `sitecustomize.py` は、Python の起動時に読み込まれ、`CUDA_VISIBLE_DEVICES` を固定します。

- **学習ジョブ**: lease の GPU に固定します。スクリプトが lease の外の GPU に変えようとすると `RuntimeError` にします。lease の GPU の一部に絞る、`-1` や空で全部隠す、は許します。削除は無視します（HF Trainer の中で `accelerate` が環境変数を一時的に全部消して戻すため）
- **エージェント**（`yuzuru agent` と自動再開したターン）: GPU を見せません（`CUDA_VISIBLE_DEVICES=""`、`YUZURU_AGENT=1`）。CUDA 関係のエラーで落ちると、トレースバックの後に「`yuzuru_train` を使う」よう案内を出します。`isolation.hide_gpus_from_agent: false` で無効にできます
- ジョブの venv などにある元の `sitecustomize` は、その後に通常どおり実行します

これはうっかりした直接実行を防ぐためのもので、サンドボックスではありません。`python -S` / `-E` / `-I`、`env -u PYTHONPATH`、Python 以外のプログラム、`subprocess` に `env=` で直接渡した値には効きません。

## ガードレール

エージェントが人の見ていないところで動くことを前提に、yuzuru は次の仕組みを持っています。

| 仕組み | 防ぐこと | 場所 |
|---|---|---|
| エージェントから GPU を隠す | シェルから直接学習を始め、lease なしで LLM の隣に載る。シェルツールのタイムアウトで途中で殺される | `gpu_pin/`、`waits.agent_env()` |
| ジョブを lease の GPU に固定する | スクリプトがほかのジョブや LLM の GPU を使う | `gpu_pin/` |
| `yuzuru_train` で `vram` を必須にする | 使用量の測り始めが早すぎて予約が小さくなり、あとで OOM する | `mcp.py` |
| メモリ上限付きの scope | 暴走したスクリプトが yuzuru や端末ごと落とす | `scope.py` |
| ターン中は再開しない | 同じ会話でエージェントが 2 つ同時に動く | `waits.py` |
| 止まったターンの再開 | 再開したエージェントが何もせずに止まったままになる | `waits.py`（`resume.stall_nudges`） |
| ブラウザからのリクエストを拒否する | Web ページから API を叩かれ、ジョブを起動される | `server.py` |

`server.py` の拒否の内容:

- `Origin` ヘッダー付きのリクエスト（ブラウザ）は拒否する。許可するなら `allowed_origins` に書く
- トークンを設定していないとき、`Host` が localhost 以外のリクエストは拒否する（DNS rebinding 対策）
- 本文のある POST / PUT / PATCH は `application/json` だけ受け付ける
- `YUZURU_TOKEN`（または設定の `token`）を設定すると、すべての API に Bearer トークンが必要になる

コマンドの許可リストのように、エージェント自体に何をさせるかは OpenCode の `permission` 設定で決めます。`yuzuru init` が作る `opencode.json` の許可リスト（`scaffold.py` の `PERMISSION`）と、その説明が [チュートリアルの手順 7](tutorial.md#7-エージェントの準備をする) にあります。

## 採点（grader、任意）

yuzuru 本体には採点の機能はなく、採点は必須でもありません。ハイパーパラメータやモデル構造のチューニングでは、エージェントは検証データの指標で設定を選び、結果を `RESULTS.md` に残して終わります（[チュートリアルの採点なしで使う](tutorial.md#採点なしで使うハイパーパラメータモデル構造のチューニング)）。

隠したテストデータで採点したいときは、課題ごとにエージェントから見えない場所で行います。
[`examples/cifar10/`](../examples/cifar10/) では、次のように分けています。

- `prepare.py`: データを課題ディレクトリ（ラベルなしのテストデータ）と採点用ディレクトリ（テストの正解ラベル `test_labels.npy`）に分ける。テストデータの順番はシャッフルし、配布元の並びから答えを引けないようにする
- `score.py`: エージェントが作った `submission.csv` を、採点用ディレクトリの正解ラベルで採点する
- 採点用ディレクトリは課題ディレクトリの外に置き、OpenCode の `external_directory` を拒否して読ませない

## ディレクトリ構成

```
yuzuru/
├── src/yuzuru/
│   ├── cli.py            CLI（yuzuru コマンド）
│   ├── server.py         FastAPI アプリ（/v1、/mcp、/yuzuru/*）
│   ├── router.py         tier の選択、セッションとターンの状態
│   ├── broker.py         メモリの予算、モデルの読み込み・解放、lease と予約
│   ├── catalog.py        tier とモデルの対応、派生モデル、必要メモリの推定と実測
│   ├── resources.py      GPU / RAM / Ollama のホスト RAM の測定
│   ├── waits.py          watch、自動再開、エージェントとジョブの環境変数
│   ├── mcp.py            MCP サーバーとツール
│   ├── scope.py          systemd user scope の作成
│   ├── managed.py        管理下の Ollama
│   ├── holder.py         hard 予約で VRAM を確保するプロセス
│   ├── ollama.py         Ollama API のクライアント
│   ├── opencode.py       OpenCode の設定の生成・書き込み
│   ├── scaffold.py       yuzuru init（課題ディレクトリの準備）
│   ├── templates/        yuzuru init が書く yuzuru.yaml と AGENTS.md のひな形
│   ├── opencode/yuzuru.ts  OpenCode 1.x 用プラグイン
│   └── gpu_pin/sitecustomize.py  CUDA_VISIBLE_DEVICES の固定
├── examples/cifar10/     チュートリアルの課題（データ準備、採点、課題ディレクトリのひな形）
├── docs/                 チュートリアルとこのドキュメント
├── tests/                pytest
└── yuzuru.example.yaml   全設定項目の例
```

yuzuru が書き込むファイル（`state_dir`、既定 `~/.local/state/yuzuru/`）:

| ファイル | 中身 |
|---|---|
| `profiles.json` | モデルごとの必要メモリの実測値とメタデータ |
| `selection.json` | `yuzuru models use` / `disable` の選択 |
| `decisions.jsonl` | ルーターの判定の記録 |
| `ollama-<port>.json` / `ollama-<port>.log` | 管理下の Ollama の PID と URL / ログ（`<port>` は yuzuru のポート） |
| `last_request_<model>.json` | `YUZURU_DUMP=1` のときの最新のリクエスト |

## 主な設定項目

設定ファイルの探し方は [README の設定](../README.md#設定) にあります。すべてのキーは省略でき、知らないキーがあるとエラーになります。全項目の例は [`yuzuru.example.yaml`](../yuzuru.example.yaml) にあります。

### トップレベル

| キー | 既定 | 意味 |
|---|---|---|
| `ollama_url` | `http://127.0.0.1:11434` | Ollama の URL（`ollama.manage: true` なら無視） |
| `host` / `port` | `127.0.0.1` / `11435` | yuzuru が listen する場所 |
| `token` | なし | Bearer トークン（環境変数 `YUZURU_TOKEN` が優先） |
| `allowed_origins` | `[]` | 受け付けるブラウザの Origin |
| `keep_alive` | `30m` | 使っていないモデルを載せておく時間 |
| `state_dir` | `~/.local/state/yuzuru` | 状態ファイルの置き場 |

### `tiers`（小さい順に並べる）

| キー | 既定 | 意味 |
|---|---|---|
| `name` | — | tier の名前 |
| `models` | `[]` | 使うモデルの候補（先頭が優先）。空なら自動で割り当てる |
| `num_ctx` | `32768` | コンテキスト長。エージェントはシステムプロンプトとツール定義だけで 2 万トークン近く使うので、32k 以上を勧める |
| `auto_params_max_b` | なし | 自動割り当てのときのパラメータ数の上限（B） |
| `allow_cpu_offload` | `true` | VRAM に入りきらない分を RAM に逃がしてよいか |
| `max_cpu_fraction` | なし（MoE 0.7、dense 0.25） | RAM に逃がしてよい割合 |

`tiers` がなければ、`small`（6B 未満）、`medium`（20B 未満）、`large`（それ以上）の 3 つを使います。

### `router`

| キー | 既定 | 意味 |
|---|---|---|
| `classifier` | `llm` | `llm`（小さいモデルで難易度を判定）か `heuristic`（ルールだけ） |
| `classifier_on_cpu` | `true` | 判定用モデルを CPU で動かす |
| `classifier_timeout` | `20` | 判定のタイムアウト（秒） |
| `prefer_loaded` | `true` | 欲しい tier 以上ですでに載っているモデルを使う |
| `error_escalation` | `3` | 1 段上の tier に上げるまでの、1 ターン内のツールエラーの回数 |
| `small_agents` | `[title, summary]` | 常に最小の tier を使う OpenCode のエージェント名 |

### `resources`

| キー | 既定 | 意味 |
|---|---|---|
| `probe` | `auto` | `auto` / `nvidia` / `none`（CPU・ユニファイドメモリ） |
| `ollama_cgroup` | `auto` | Ollama のホスト RAM を測る cgroup（`auto` / `none` / パス） |
| `vram_margin_mb` | `512` | GPU ごとに LLM に渡さない量 |
| `ram_reserve_mb` | `6144` | OS やデータローダーのために空けておく RAM |
| `gpus` | `[]` | 使う GPU（番号か UUID）。空なら全部 |

### `ollama`

| キー | 既定 | 意味 |
|---|---|---|
| `manage` | `false` | 管理下の Ollama を使う |
| `binary` | `ollama` | 起動する実行ファイル |
| `host` | `127.0.0.1` | 管理下の Ollama の listen アドレス |
| `port` | `auto` | 固定するならポート番号 |
| `port_range` | `[11436, 11499]` | `port: auto` のときに選ぶ範囲 |
| `models` | `~/.local/share/yuzuru-ollama/models` | 管理下の Ollama のモデル置き場 |
| `mirror_from` | `auto` | 取り込むシステムのモデル置き場（`auto` / `none` / パス） |
| `scope` | `auto` | 専用の scope で起動する（`auto` / `on` / `off`） |
| `env` | `{}` | 追加の環境変数（例 `OLLAMA_FLASH_ATTENTION: "1"`） |
| `startup_timeout` | `60` | 起動を待つ秒数 |

### `lease`

| キー | 既定 | 意味 |
|---|---|---|
| `settle_seconds` | `15` | 使用量がこの秒数安定したら lease を `active` にする |
| `settle_timeout` | `180` | 安定しなくても `active` にするまでの秒数 |
| `settle_tolerance_mb` | `256` | 「安定」とみなす変動の幅 |
| `headroom_mb` | `1024` | `active` のジョブが増える分として空けておく量 |
| `drain_timeout` | `300` | Ollama を止める前（lease の前、GPU を返したあとの再起動の前）に、生成中の応答を待つ秒数。GPU を返すときは、最初のトークンをまだ返していない応答は待たずに送り直す |
| `unload_timeout` | `60` | 解放したモデルが VRAM から消えるのを待つ秒数 |
| `restore_after` | `true` | ジョブの後に、前に載っていたモデルを読み込み直す |

### `resume`

| キー | 既定 | 意味 |
|---|---|---|
| `enabled` | `true` | 自動再開する |
| `command` | `["opencode", "run", "--session", "{session}", "{message}"]` | 再開に使うコマンド。`{session}`、`{message}`、`{cwd}` を置き換える |
| `suspend_all` | `true` | 待っている間、全モデル（判定用モデルを含む）を解放する |
| `suspend_idle_seconds` | `20` | セッションがこの秒数静かになってから解放する |
| `turn_idle_seconds` | `600` | ツール呼び出しで終わったターンを「続いている」とみなす秒数 |
| `warm_before_resume` | `true` | 再開の前にモデルを読み込む |
| `log_tail_lines` | `40` | 再開のメッセージに入れるログの行数 |
| `stall_nudges` | `1` | 止まったターンを続けて再開する回数（0 で無効） |

### `isolation`

| キー | 既定 | 意味 |
|---|---|---|
| `scope` | `auto` | エージェントとジョブを scope に入れる（`auto` / `on` / `off`） |
| `memory_max` | `80%` | scope ごとのメモリ上限（systemd の `MemoryMax`。`24G` なども可、`null` で上限なし） |
| `cpu_job_weight` | `20` | CPU ジョブの scope の `CPUWeight`（1〜10000） |
| `hide_gpus_from_agent` | `true` | エージェントが自分で実行するコマンドから GPU を隠す |

### `cloud`

| キー | 既定 | 意味 |
|---|---|---|
| `enabled` | `false` | クラウドの tier を使う |
| `base_url` | `https://openrouter.ai/api/v1` | OpenAI 互換 API の URL |
| `api_key_env` | `OPENROUTER_API_KEY` | API キーを読む環境変数 |
| `model` | `""` | プロバイダのモデル ID |
| `context` | `200000` | コンテキスト長 |
| `auto` | `escalation` | 自動で使う条件（`off` / `escalation` / `hard`） |
| `when_no_memory` | `false` | 欲しい tier が載らないときにクラウドを使う |
| `when_context` | `true` | どのローカル tier にも入らないときにクラウドを使う |
| `max_requests_per_hour` | `60` | 自動で使う回数の上限（1 時間あたり） |
