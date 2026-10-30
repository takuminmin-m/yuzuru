# yuzuru

yuzuru は、1 台のマシンの GPU を「ローカル LLM」と「機械学習の学習ジョブ」で取り合わないように仲裁するツールです。
ローカル LLM で動くコーディングエージェント（主に [OpenCode](https://opencode.ai)）に、
学習コードを書かせ、GPU で学習させ、結果を見て改善させる、という作業を人の手を借りずに回させることができます。

- [チュートリアル: ローカル LLM エージェントに CIFAR-10 を解かせる](docs/tutorial.md)
- [仕組みと構造（アーキテクチャ）](docs/architecture.md)

## yuzuru でできること

- **学習中は LLM を GPU から降ろす。** 学習ジョブを始めると、yuzuru は LLM を VRAM から退かします。ジョブが終わると LLM を読み込み直します。
- **待っている間は LLM を止め、終わったら会話を再開する。** エージェントが学習を起動してターンを終えると、yuzuru は学習の終了を見張ります。終わると、終了コードとログの末尾を添えてエージェントの会話を自動で再開させます。
- **空きメモリに合わせてモデルの大きさを選ぶ。** 「small / medium / large」のような大きさの区分（tier）ごとにモデルを割り当てます。リクエストごとに、作業の難しさと空きメモリを見て tier を選びます。
- **エージェントの暴走から守る。** エージェントと学習ジョブを、メモリ上限付きの systemd scope に入れます。エージェントが自分のシェルから GPU を使うことも防ぎます。
- **使う GPU を限定する。** 設定 1 か所で、LLM と学習ジョブが使う GPU を決められます。

## 用語

このドキュメント全体で、次の言葉を使います。

| 用語 | 意味 |
|---|---|
| エージェント | LLM を使ってファイル編集やコマンド実行を自律的に行うプログラム。yuzuru は OpenCode を想定している |
| tier | LLM の大きさの区分（`small` / `medium` / `large` など）。tier ごとに Ollama のモデルを 1 つ割り当てる |
| 学習ジョブ | GPU（または CPU）で動かす学習・前処理のプロセス |
| lease（リース） | 学習ジョブのために確保した GPU メモリの枠。lease がある間、その分を LLM に使わせない |
| 予約（reservation） | ジョブを始める前に、手で押さえておく GPU メモリの枠。あとから lease に引き継げる |
| scope | systemd の一時的な cgroup。メモリ上限を超えたプロセスは、その scope の中だけで止められる |
| 自動再開 | 学習ジョブが終わったとき、yuzuru がエージェントの会話を続きから動かすこと |
| 管理下の Ollama | yuzuru が自分で起動・監視する専用の `ollama serve`（`ollama.manage: true` のとき） |

## 全体の構成

```
 OpenCode ──┬─ OpenAI 互換 API（/v1）  model = auto | small | medium | large | <モデル名>
            └─ MCP（/mcp）            yuzuru_train / yuzuru_wait / yuzuru_status / yuzuru_model / yuzuru_models
    ▼
 yuzuru serve（:11435）
    ├─ Router : リクエストごとに tier を選ぶ
    ├─ Broker : GPU / RAM を監視し、モデルの読み込み・解放と lease を管理する
    ├─ Waiter : 学習ジョブの終了を見張り、エージェントの会話を再開する
    └─ Catalog: tier とモデルの対応、必要メモリの推定と実測
    ▼
 Ollama（:11434、または管理下の Ollama）
```

各部の詳しい動きは [docs/architecture.md](docs/architecture.md) にあります。

## 必要なもの

- Linux（NVIDIA GPU で確認しています。GPU がなくても、CPU / ユニファイドメモリで動きます）
- Python 3.11 以上と [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com)
- エージェントとして使うなら [OpenCode](https://opencode.ai)（1.x / 2.x に対応）
- あると良いもの: systemd のユーザーセッション（`systemd-run --user --scope` が動くこと）。scope による隔離に使います。なくても動きます

## インストール

1. リポジトリを取得し、`yuzuru` コマンドを入れます。`~/.local/bin` に入ります。

   ```bash
   git clone https://github.com/takuminmin-m/yuzuru.git
   uv tool install -e ./yuzuru
   yuzuru --help
   ```

2. Ollama にモデルを入れます。モデルは 1 つでも動きます。小さいモデルもあると、軽い作業をそちらに回せます。

   ```bash
   ollama pull qwen3:4b      # 例: small（難易度の判定にも使う）
   ollama pull qwen3:14b     # 例: medium
   ```

3. 必要なら設定ファイルを書きます（[設定](#設定)）。設定ファイルがなくても、入っているモデルをパラメータ数で tier に自動で割り当てて動きます。

### 課題ごとに設定一式を作る（`yuzuru init`）

エージェントに課題を任せるなら、課題ディレクトリごとに `yuzuru init` で設定一式を作れます。

```bash
yuzuru init ~/work/my-task --gpu 1 --packages torch numpy   # --model / --port / --memory-max も指定できる
cd ~/work/my-task && yuzuru serve                          # このディレクトリでは ./yuzuru.yaml が使われる
```

| 作るもの | 中身 |
|---|---|
| `yuzuru.yaml` | 使う GPU（`--gpu` の番号を UUID にして書く。既定は全部）、モデル（既定は前回 `large` に使ったもの、初めてなら一番大きいもの）、管理下の Ollama、RAM の上限、再開したターンの出力先（`.yuzuru/resume.log`） |
| `opencode.json` | yuzuru のプロバイダと MCP、クラウドへの切り替えの無効化、無人で動かすための権限の許可リスト |
| `AGENTS.md` | エージェントへの作業ルール（`uv run python` を使う、学習は `yuzuru_train` で動かしてターンを終える、試した設定と指標を `RESULTS.md` に残す、など） |
| `.gitignore`、`.venv`、git | `.gitignore` への追記、`uv venv`（`--packages`、`requirements.txt`、`pyproject.toml` なら `uv sync`）、`git init` と最初の commit |

すでにあるファイルはそのまま残します（`--force` を付けると `.bak` を残して作り直します）。課題を並行して動かすときは、課題ごとに `--port` を変えてください。

課題は、採点のあるコンペ形式でなくても構いません。ハイパーパラメータやモデル構造のチューニングなら、課題の `README.md` に検証データの指標、変えてよい範囲、終わる条件を書けば十分です。エージェントは試した設定と指標を `RESULTS.md` に残して終わります（[チュートリアルの採点なしで使う](docs/tutorial.md#採点なしで使うハイパーパラメータモデル構造のチューニング)）。

## 最初の一回を動かす

1. yuzuru を起動します。この端末にログが流れます。

   ```bash
   yuzuru serve
   ```

2. 別の端末で状態を見ます。GPU と RAM の使用量、LLM が使ってよい量、tier ごとの割り当てが出ます。

   ```bash
   yuzuru status
   ```

3. 学習ジョブを yuzuru 経由で動かします。LLM が GPU から退き、ジョブが終わると戻ります。

   ```bash
   yuzuru run --vram 4G -- python train.py
   ```

4. OpenCode から使えるようにします。`~/.config/opencode/opencode.json` に yuzuru のプロバイダと MCP サーバーが追加されます。

   ```bash
   yuzuru opencode --install --local-only
   ```

5. エージェントを scope の中で起動します。

   ```bash
   yuzuru agent -- opencode run "README.md を読んで課題に取り組んでください"
   ```

エージェントに学習を任せる一連の流れは、[チュートリアル](docs/tutorial.md)で手順を追って説明しています。

## 使い方

### 学習ジョブを動かす（`yuzuru run`）

```bash
yuzuru run --vram 10G -- python train.py   # 10 GB を予約して実行する（推奨）
yuzuru run -- python train.py              # 使用量を測って予約する
yuzuru run --gpu 1 -- python train.py      # GPU 1 をジョブ専用にする
yuzuru run --cpu -- python lgbm.py         # GPU を使わないジョブ
yuzuru run --log train.log -- python train.py
```

| オプション | 意味 |
|---|---|
| `--vram 10G` | ジョブが最大で使う VRAM を申告する。その分は LLM に使わせない。残りは LLM が使える |
| `--gpu <番号か UUID>` | その GPU をジョブ専用にする（複数指定可）。ジョブの間、LLM はその GPU を一切使わない |
| `--cpu` | GPU を使わないジョブ。LLM は降ろさない。ジョブには GPU を見せない |
| `--log <file>` | ジョブの出力をファイルに書く |
| `--detach` | バックグラウンドで実行し、PID とログのパスを表示する |
| `--use <id>` | `yuzuru reserve` で確保した枠の中で実行する |
| `--label <名前>` | `yuzuru status` に出る名前 |

`--vram` は付けることを勧めます。省略すると、yuzuru はジョブの VRAM 使用量が落ち着いた時点の量で予約します。
ただし、その後の最初のステップや評価で使用量が増えると、LLM がすでに隣に載っていて OOM になることがあります。

ジョブには `CUDA_VISIBLE_DEVICES` が lease の GPU の UUID で設定されます。
`yuzuru serve` に繋がらないときも、ジョブはそのまま実行されます（lease なし）。

### GPU メモリを先に確保する（`yuzuru reserve`）

ジョブを始める前に、GPU メモリを押さえておけます。

```bash
yuzuru reserve --gpu 1                     # GPU 1 の空きを丸ごと確保する
yuzuru reserve --gpu 1 --vram 10G --for 2h # 10 GB を 2 時間確保する
yuzuru reserve --vram 8G --soft            # LLM の予算から差し引くだけ
yuzuru run --use <id> -- python train.py   # 確保した枠でジョブを実行する
yuzuru release <id>                        # 枠を返す
```

- 既定は **hard** です。小さなプロセスが実際に VRAM を確保するので、Ollama もほかのプロセスもその分を使えません。NVIDIA のドライバ（libcuda）が必要です。
- `--soft` は LLM の予算から差し引くだけです。GPU の種類を問いません。
- `--for` の期限が過ぎると自動で解放され、元のモデルが戻ります。

### エージェントに学習を任せる

1. エージェントを `yuzuru agent` で起動します。OpenCode 2.x では、バックグラウンドサービスもこの形で起動します（理由は [architecture.md](docs/architecture.md#エージェントの隔離と-gpu-を隠す仕組み)）。

   ```bash
   opencode service stop && yuzuru agent -- opencode service start   # OpenCode 2.x
   yuzuru agent -- opencode run "課題に取り組んでください"
   ```

2. エージェントは MCP の `yuzuru_train` ツールで学習を起動し、ターンを終えます。
3. yuzuru は LLM を降ろし、学習の終了を待ちます。
4. 学習が終わると、yuzuru は LLM を読み込み直し、終了コード・ログの末尾・エージェントのメモを添えて会話を再開させます。

エージェントが使える MCP ツール:

| ツール | 用途 |
|---|---|
| `yuzuru_train` | 学習ジョブをバックグラウンドで起動する。GPU ジョブには `vram`（または `gpus`）が必須。CPU ジョブは `device: "cpu"` |
| `yuzuru_wait` | 自分で起動したのではないプロセス（PID か lease）の終了を待つ |
| `yuzuru_status` | GPU / RAM、載せられる tier、学習ジョブを見る |
| `yuzuru_models` | 入っているモデルの一覧 |
| `yuzuru_model` | このセッションで使う tier かモデルを固定する（`auto` で戻す） |

エージェントが自分のシェルで実行するコマンドからは、GPU が見えません。GPU を使う処理は `yuzuru_train` を通させるためです。

CLI からも同じことができます。

```bash
yuzuru wait --pid 1234 --session <id> --log train.log   # PID の終了を待って再開する
yuzuru events -f                                       # ジョブの終了と再開の履歴を流し続ける
```

### モデルを選ぶ

```bash
yuzuru models                        # 入っているモデル、担当 tier、MoE かどうか
yuzuru models use medium qwen3:14b   # medium をこのモデルに固定する（auto で自動に戻す）
yuzuru models disable llama3:70b     # 自動割り当てと一覧から外す
yuzuru models enable llama3:70b      # 元に戻す
yuzuru pin <session> large           # セッションを tier に固定する（auto で解除）
```

OpenCode では、モデル名に `yuzuru/auto`（自動）、`yuzuru/large`（tier を指定）、`yuzuru/qwen3:14b`（モデルを名前で指定）を選べます。
メッセージに `@large` と書くと、そのターンだけ tier を指定できます。

tier の選び方の詳細は [architecture.md の Router](docs/architecture.md#router-tier-の選び方) にあります。

### ローカルで足りないときにクラウドを使う

設定ファイルの `cloud:` に OpenAI 互換 API（OpenAI、OpenRouter、Groq、別マシンの vLLM など）を書くと、`cloud` という最上位の tier として使えます。
API キーは設定ファイルに書かず、`api_key_env` で指定した環境変数から読みます。

| いつクラウドに行くか | 設定 |
|---|---|
| 明示したとき（`yuzuru/cloud`、`@cloud`、`yuzuru_model cloud`） | `enabled: true` なら常に |
| 最大のローカル tier でもツールエラーが続くとき | `auto: escalation`（既定） |
| ルーターが新しいターンを「hard」と判定したとき | `auto: hard` |
| どのローカル tier のコンテキストにも入らないとき | `when_context: true`（既定） |
| 学習ジョブが GPU を使っていて、欲しい tier が載らないとき | `when_no_memory: true` |

自動で使う回数は `max_requests_per_hour` で抑えられます。

### 状態を確かめる

```bash
yuzuru status                              # GPU / RAM、LLM の予算、tier ごとの配置見込み、lease
yuzuru status --json                       # 同じ内容を JSON で
yuzuru route "この loss が発散する原因を調べて"   # ルーターの判定だけを試す
yuzuru warm large                          # tier を今読み込む（--force で予算を超えても読み込む）
yuzuru unload                              # 全モデルを解放する（tier 名を付けるとその tier だけ）
```

## 設定

設定ファイルは次の順に探します。

1. `--config <file>` で指定したファイル
2. 環境変数 `YUZURU_CONFIG`
3. カレントディレクトリの `yuzuru.yaml`
4. `~/.config/yuzuru/config.yaml`

すべてのキーは省略できます。全項目の例は [`yuzuru.example.yaml`](yuzuru.example.yaml)、全項目の説明は [architecture.md の設定項目](docs/architecture.md#主な設定項目) にあります。

よく使う設定の例です。

```yaml
# ~/.config/yuzuru/config.yaml
resources:
  gpus: ["GPU-xxxxxxxx-..."]   # 使う GPU（UUID か番号）。空なら全部
ollama:
  manage: true                 # 上の GPU だけが見える専用の ollama serve を yuzuru が起動する
tiers:
  - name: large
    models: ["qwen3:14b"]      # `ollama list` の名前
    num_ctx: 32768
isolation:
  memory_max: 80%              # エージェントと学習ジョブ 1 つあたりの RAM 上限
```

- GPU は番号より UUID（`nvidia-smi -L` で分かる）で書くことを勧めます。`nvidia-smi` の番号と CUDA の番号は一致しないことがあります。
- `resources.gpus` で GPU を限定するときは、`ollama.manage: true` も付けてください。システムの Ollama は全 GPU を見たままなので、除外した GPU にも計算用のメモリを置くことがあります。

環境変数:

| 変数 | 意味 |
|---|---|
| `YUZURU_CONFIG` | 設定ファイルのパス |
| `YUZURU_URL` | CLI が接続する yuzuru の URL（既定 `http://127.0.0.1:11435`） |
| `YUZURU_TOKEN` | API に Bearer トークンを必須にする（[別のマシンから使う](#別のマシンから使う)） |
| `OLLAMA_HOST_URL` | Ollama の URL（`ollama.manage` が無効のとき） |
| `YUZURU_DUMP=1` | `yuzuru serve` が、受け取ったリクエストを `~/.local/state/yuzuru/last_request_<model>.json` に保存する |

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

自動再開では、yuzuru が `opencode` を起動します。unit から `opencode` が見つからないときは、設定の `resume.command` に `opencode` の絶対パスを書いてください。

## 別のマシンから使う

yuzuru は既定で `127.0.0.1` にしか bind しません。手元のマシンから GPU マシンを使うには、SSH を使います。

- **GPU マシンの OpenCode サービスに繋ぐ（推奨）。** ファイル操作、コマンド、学習ジョブがすべて GPU マシン上で動きます。

  ```bash
  # GPU マシンで `opencode service status` を実行し、ポートを確かめておく
  ssh -N -L 4096:127.0.0.1:<port> gpu-box
  opencode --server http://127.0.0.1:4096
  ```

- **SSH して GPU マシン上で `opencode` を起動する。** 一番単純です。tmux と組み合わせると切断に強くなります。
- **手元の OpenCode から yuzuru だけを使う**（`ssh -L 11435:127.0.0.1:11435 gpu-box`）。推論は GPU マシンで行われますが、コマンドやファイル操作は手元で動きます。学習ジョブを GPU マシンで動かす用途には向きません。

yuzuru を localhost 以外に公開する場合は、`YUZURU_TOKEN` を設定してください。すべての API に Bearer トークンが必要になります。
`yuzuru opencode --install` は、トークンが設定されていれば OpenCode 側にも `{env:YUZURU_TOKEN}` を書き込みます。

## 困ったとき

| 症状 | 対処 |
|---|---|
| `yuzuru: server not running at ...` | `yuzuru serve` が動いていない。起動する |
| `yuzuru status` の先頭に「GPU ... is not responding」 | GPU がバスから外れている（カーネルログに `Xid 79`）。yuzuru は残りの GPU で動き続ける。再起動か `sudo nvidia-smi -r` で戻ることが多い。繰り返すなら電源容量、PCIe の挿さり、ライザーを確かめる |
| `yuzuru status` の先頭に「Ollama holds ... GB of RAM」 | 強制終了された Ollama のランナーのメモリが残っている。Ollama を再起動する（例 `sudo systemctl restart ollama`） |
| tier の配置見込みが `none` で、モデルが読み込まれない | 初回は必要メモリを大きめに見積もる。`yuzuru warm <tier> --force` で一度読み込むと実測値に置き換わる |
| 学習が OOM する | `--vram` を付ける。GPU を限定しているなら `ollama.manage: true` にする |
| 違う GPU が使われる | GPU を番号で書いている。UUID で書く |
| OpenCode がクラウドのモデルで応答する | yuzuru に繋がらず、OpenCode が組み込みの無料モデルに切り替えている。`yuzuru opencode --install --local-only` で無効にする |
| `... not in an agent scope` の警告 | OpenCode のサービスが scope の外で動いている。`opencode service stop && yuzuru agent -- opencode service start` |
| OpenCode が実際に送っているリクエストを見たい | `YUZURU_DUMP=1 yuzuru serve` で起動する |

エージェントを動かすときの問題は、[チュートリアルの「うまくいかないとき」](docs/tutorial.md#うまくいかないとき)にまとめています。

## 開発

```bash
uv sync && uv run pytest
```
