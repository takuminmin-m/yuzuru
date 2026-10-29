# チュートリアル: ローカル LLM エージェントに CIFAR-10 を解かせる

GPU 1 枚のマシンで、ローカルの LLM エージェント（OpenCode）に「課題を読む → 学習コードを書く → GPU で学習 →
結果を見て改善 → 提出」を自律的に回させる。LLM と学習ジョブは同じ GPU を使うので、その取り合いを yuzuru が仲裁する。

- 前半（手順 1〜6、30 分ほど）: yuzuru を入れ、手で学習ジョブを流して「学習中は LLM が GPU から退き、終わると戻る」のを見る
- 後半（手順 7〜10、エージェント次第で 1〜2 時間）: エージェントに課題を渡して放っておき、終わったら採点・検証する

参考: 作者の環境（RTX 3060 12 GB + Qwen3.6-35B-A3B の 3 ビット量子化）では、エージェントがベースライン（val 66%）を
ResNet-18 + CutMix/Mixup に書き換え、約 90 分の学習でテスト正解率 95.5% に達した。人が打ったのは最初の 1 行だけ。

使うファイルは [`examples/cifar10/`](../examples/cifar10/) にある。

| ファイル | 中身 |
|---|---|
| `config.yaml` | yuzuru の設定のひな形（GPU 1 枚、モデル 1 つ、専用 Ollama） |
| `prepare.py` | CIFAR-10 を取得し、ラベルなしのテストセットと隠し正解ラベルに分ける |
| `score.py` | `submission.csv` を隠しラベルで採点する |
| `task/` | エージェントに渡す課題ディレクトリのひな形（`README.md`、`AGENTS.md`、ベースライン、`opencode.json`） |

## 全体像

```
あなた ── yuzuru agent -- opencode run "課題に取り組んで"
              │
              ▼
          OpenCode ──(LLM API / MCP)──▶ yuzuru (:11435) ──▶ 専用 Ollama ──▶ GPU
              │
     エージェントが MCP の yuzuru_train で学習を起動し、ターンを終える
              ▼
  yuzuru: LLM を GPU から降ろす → 学習を実行 → 終わるまで待つ（その間 LLM はメモリを使わない）
              ▼
  学習が終わる → LLM を読み込み直す → 終了コードとログ末尾を添えて会話を自動で再開
              ▼
  エージェントが結果を見て次の手を打つ（目標に届くまで繰り返す）
```

人が何もしなくても会話が続くのは、yuzuru が学習の終了を見張っていて、`opencode run --session <id> "<結果>"` で
会話を再開させるから。

## 0. 前提

- Linux + NVIDIA GPU（`nvidia-smi` が動く）。VRAM 12 GB / RAM 32 GB で確認
- systemd のユーザーセッション（普通のデスクトップ Linux ならある）。エージェントと学習ジョブをメモリ上限付きの
  scope に入れるのに使う。無くても動くが、暴走したスクリプトが端末ごと落とすことがある
- [Ollama](https://ollama.com)（0.40 で確認）
- [OpenCode](https://opencode.ai) 2.x（2.0.32 で確認。この手順の `opencode.json` は 2.x の書式）
- [uv](https://docs.astral.sh/uv/)

## 1. yuzuru を入れる

```bash
uv tool install -e /path/to/yuzuru
yuzuru --help
```

`yuzuru` コマンドが `~/.local/bin` に入る。

## 2. エージェント用のモデルを用意する

エージェントに使うモデルは **ツール呼び出しができて、それなりに大きいもの**にする。4B〜14B 級では、ファイル編集や
ツールの使い方を間違え続けて先に進まないことが多かった。VRAM に入りきらない分は RAM に逃がせるので、
MoE（一度に動くパラメータが少ない）モデルの低ビット量子化が扱いやすい。

```bash
ollama pull <モデル名>
ollama list          # 名前を確認（例: qwen3.6-iq3:latest）
```

> Hugging Face から GGUF を取ってきた場合、同梱のチャットテンプレートが壊れていることがある
> （作者の Qwen3.6 では `<think>` が停止語になっていて、考え始めた瞬間に応答が切れた）。その場合は
> `Modelfile` で `RENDERER` / `PARSER` を指定して作り直す（`FROM <blob>` + `RENDERER qwen3.5` + `PARSER qwen3.5`）。

## 3. yuzuru を設定する

まず GPU の UUID を調べる。

```bash
nvidia-smi -L
```

```
GPU 0: NVIDIA GeForce RTX 3050 (UUID: GPU-6a1e...)
GPU 1: NVIDIA GeForce RTX 3060 (UUID: GPU-403d...)
```

**GPU は番号ではなく UUID で指定する**。`nvidia-smi` の番号（PCI バス順）と CUDA / PyTorch の番号（速い順）は
一致しないことがある（上の例では CUDA の 0 番は 3060）。

設定のひな形をコピーし、`resources.gpus` を使う GPU の UUID に、`tiers[0].models` を手順 2 のモデル名に書き換える。

```bash
mkdir -p ~/.config/yuzuru
cp /path/to/yuzuru/examples/cifar10/config.yaml ~/.config/yuzuru/config.yaml
$EDITOR ~/.config/yuzuru/config.yaml
```

ひな形で効いている設定:

| 設定 | 意味 |
|---|---|
| `resources.gpus` | ここに書いた GPU だけを LLM にも学習にも使う。学習ジョブには `CUDA_VISIBLE_DEVICES` が自動で設定される |
| `ollama.manage: true` | yuzuru が、上の GPU だけが見える専用の `ollama serve` を起動する。システムの `ollama.service`（:11434）は動いたままでよい。重みはシンボリックリンクで共有するのでディスクは増えない |
| `tiers` | LLM の大きさの区分。ここではモデル 1 つだけの `large`。`num_ctx` は 32k 以上（OpenCode はシステムプロンプトとツール定義だけで 2 万トークン近く送る） |
| `isolation.memory_max` | エージェントと学習ジョブ 1 つあたりの RAM 上限。超えたプロセスだけが殺され、yuzuru や端末は無事 |
| `resume.command` | 学習後に会話を再開するコマンド。ひな形は出力を `~/.local/state/yuzuru/resume.log` に残す形にしてある |

## 4. yuzuru を起動して状態を見る

端末を 1 つ yuzuru 用にして起動する（ログがここに流れる）。

```bash
yuzuru serve
```

別の端末で状態を見る。

```bash
yuzuru status
```

```
GPU1 NVIDIA GeForce RTX 3060      used   0.2 GB / 12.0 GB
RAM                              avail 18.7 GB / 31.3 GB
LLM may use: GPU 11.3 GB + RAM 12.7 GB
GPUs limited to: GPU-403d... (resources.gpus)
Ollama (managed): http://127.0.0.1:11436 pid 82589, CUDA_VISIBLE_DEVICES=GPU-403d...

Tiers:
  large   qwen3.6-iq3:latest           ctx  65536   14.4 GB*  -> none
  (* = estimate until first load)
```

- `LLM may use` は、学習ジョブなどを差し引いたうえで LLM に使わせてよい量
- `Tiers` の右端は「今読み込んだらどこに載るか」の見込み。`gpu`（全部 GPU）、`offload`（一部 RAM）、`none`（載らない）
- `*` 付きのサイズは GGUF からの推定値。一度読み込むと実測値に置き換わる

先頭に `!` 付きの警告が出ていたら先に片付ける（[README のトラブルシューティング](../README.md#トラブルシューティング)）。

上の例では `none` になっている。実測値がないうちは、GPU からはみ出す場合に「重み全体がホスト RAM にも載る」という
最悪の見積もりをするので、空き RAM が少ないと予算（`RAM 12.7 GB`）を超えてしまう。この状態でエージェントを
動かしても、モデルが読み込まれない。一度だけ強制的に読み込んで実測値を作る。

```bash
yuzuru warm large --force
```

`{"tier": "large", "placement": "gpu"}` のように、実際に載った場所が返る。yuzuru のログには
`took 10.2 GB VRAM + 0.4 GB host RAM` のような実測値が出て、以後の見込みはこれを使う（この例では 14.4 GB の推定が
10.4 GB になり、GPU に全部載った）。`status` の見込みが最初から `gpu` か `offload` なら `--force` は要らない。
どちらの場合も、先に読み込んでおくと最初の応答が速い。

## 5. 課題ディレクトリを作る

課題ディレクトリ（エージェントが作業する場所）と、採点用ディレクトリ（エージェントから見えない場所）を分ける。

```bash
YUZURU=/path/to/yuzuru
TASK=~/work/cifar10-task
GRADER=~/work/.cifar10-grader

cp -r "$YUZURU/examples/cifar10/task" "$TASK"
cd "$TASK"
uv venv .venv
uv pip install torch torchvision numpy pillow
uv run python -c "import torch; print(torch.cuda.is_available())"   # True になること
uv run python "$YUZURU/examples/cifar10/prepare.py" "$TASK" "$GRADER"  # 約 170 MB をダウンロード
git init -q && git add -A && git commit -qm "task + baseline"
```

CIFAR-10 の Python 版（`cifar-10-batches-py/`）が手元にあれば `--root <その親ディレクトリ>` でダウンロードを省ける。

できあがるもの:

```
cifar10-task/
├── README.md        課題（目標: テスト正解率 85%、ルール）
├── AGENTS.md        作業ルール（後述）
├── model.py / train.py / predict.py   ベースライン（val 約 66%）
├── opencode.json    OpenCode の設定（yuzuru への接続 + 権限）
└── data/            train.npz（ラベル付き 50000 枚）、test.npz（ラベルなし 10000 枚）など
```

`git init` しておくと、エージェントが何を変えたかを後で `git diff` で追える。

## 6. 手で学習ジョブを流してみる

エージェントに渡す前に、yuzuru 経由の学習を自分で動かして仕組みを見ておく。別の端末で
`watch -n1 yuzuru status` を開いておくと様子がわかりやすい。

```bash
cd "$TASK"
yuzuru run --vram 2G --log /tmp/baseline.log -- uv run python train.py
```

```
[yuzuru] lease cc071634: LLM moved out of GPU memory
```

起きていること:

1. yuzuru が学習用の枠（**lease**）を作り、GPU 上の LLM を降ろす
2. 学習を起動する。`status` には `Leases:` の行が出て、`LLM may use: GPU 0.0 GB` になる
3. 学習が終わる（RTX 3060 で 15 秒ほど、`val_acc` 約 0.66）と lease が消え、LLM が読み込み直される

yuzuru の端末にはこう出る:

```
lease cc071634: unloading yuzuru-large:qwen3.6-iq3-64k
lease cc071634 (uv, pid 83240) created
lease cc071634 released (rc=0)
loading yuzuru-large:qwen3.6-iq3-64k (gpu, ~10.4 GB)
```

`--vram 2G` は「この学習は VRAM を 2 GB まで使う」という申告。学習の間、yuzuru はその分を LLM に使わせない。
この例では LLM ごと降ろすので差は出ないが、GPU に余裕があるときは残りを LLM が使い続けられる。

採点の流れも確かめておく。

```bash
uv run python predict.py
uv run python "$YUZURU/examples/cifar10/score.py" "$GRADER/test_labels.npy" submission.csv   # accuracy 0.66 前後
rm model.pt submission.csv
```

## 7. エージェントの準備

### `AGENTS.md`（エージェントへの作業ルール）

OpenCode は `AGENTS.md` を毎ターン読み込む。ローカルの LLM は細かい作法を外しやすいので、ここで明示しておく。
ひな形で指示していること:

- Python は `uv run python` で動かす（素の `python` だとシステムの Python を使ってしまう）
- **GPU を使う処理は MCP の `yuzuru_train` ツールで起動し、呼んだらターンを終える**。終わると会話が自動で再開する
- `yuzuru_train` には必ず `vram`（使う VRAM 量）を渡す。渡さないとツールがエラーを返す
- `sleep` で待たない、実行中の学習を止めようとしない、良い重みを上書きしない
- 判断は `val_acc` で。提出物は重み + `predict.py` + `RESULTS.md`、`submission.csv` は `predict.py` の出力

課題に合わせて足すとよいもの: RAM の上限（`isolation.memory_max` に合わせる）、CPU だけで済む学習
（LightGBM など）は `yuzuru_train` の `device: "cpu"` で動かすこと、など。

### `opencode.json`（接続と権限）

課題ディレクトリの `opencode.json` が、このディレクトリで動く OpenCode の設定になる。

- `provider.yuzuru` と `mcp.servers.yuzuru`: yuzuru（:11435）への接続。`models` のキー（`large`）は yuzuru の
  tier 名と合わせる。yuzuru のポートや tier を変えたらここも直す
  （`yuzuru opencode --install --project "$TASK" --local-only` で生成し直してから `permission` を足してもよい）
- `disabled_providers: ["opencode"]`: yuzuru が落ちていると、OpenCode は組み込みの無料クラウドモデルに黙って
  切り替えてコードを外に送る。それを防ぐ
- `permission`: 人が見ていない間に走らせるので、全部許可ではなく許可リストにしている
  - `bash` は既定で拒否し、`uv run *`、`ls`、`cat`、`tail`、`git diff` など必要なものだけ許可。許可した中でも
    `rm -rf`、`--break-system-packages` は拒否（`sudo` や `kill` は許可リストにないので拒否される）
  - `external_directory` は拒否（課題ディレクトリの外 = 採点用ディレクトリを読ませない）。例外として
    `~/.local/share/opencode/shell/*` だけ許可する。OpenCode 2.x は長いコマンド出力をここに保存して
    エージェントに読ませるため、拒否すると自分のコマンドの出力を読めなくなる
  - `execute` は拒否（OpenCode 2.x の `execute` ツールは JavaScript でコードを実行でき、bash の許可リストを素通りする）
  - `question` は拒否（人への質問で止まらないように）

### OpenCode のバックグラウンドサービスを yuzuru の scope で起動する

OpenCode 2.x の `opencode run` や TUI は、ターンの実行をバックグラウンドサービス（`opencode service`）に任せる。
エージェントが実行するコマンドはサービスのプロセスの子として動くので、**サービスをどこで起動したか**が効いてくる。

```bash
cd "$TASK"
opencode service stop                          # 動いていなければ何もしない
yuzuru agent -- opencode service start
```

`yuzuru agent` で起動すると、サービス（とエージェントが実行するすべてのコマンド）は次の状態になる:

- メモリ上限付きの systemd scope（`yuzuru-agent-*.scope`、上限は `isolation.memory_max`）に入る。
  エージェントの書いたスクリプトが RAM を食い尽くしても、殺されるのはそのスクリプトだけ
- **GPU が見えない**。シェルから直接 `python train.py` を走らせると、LLM の隣で lease なしに GPU を使い、
  シェルツールのタイムアウトで途中で殺される。GPU が見えなければ CUDA のエラーですぐ落ち、`yuzuru_train` を
  使うよう案内が出る。学習ジョブには yuzuru が lease の GPU を渡すので、`yuzuru_train` 経由なら GPU が使える

サービスが scope の外（ふつうの端末や IDE から先に `opencode` を起動した、など）で動いていると、
`yuzuru agent` の出力と、自動再開のたびに yuzuru のログへ
`the OpenCode service (pid ...) runs in ..., not in an agent scope` という警告が出る。そのときは上の 2 行をやり直す。

## 8. エージェントを走らせる

```bash
cd "$TASK"
yuzuru agent -- opencode run "README.md と AGENTS.md を読んで、課題を最後までやり遂げてください。"
```

（TUI で `opencode` を開いて同じことを打ってもよい。サービスにつながった TUI には、自動で再開したターンも表示される。）

この後に起きること:

1. エージェントが課題とベースラインを読み、`train.py` などを書き換える
2. `yuzuru_train` で学習を起動し、ターンを終える。`opencode run` もここで戻ってくる
3. yuzuru が LLM を全部降ろし（GPU 全体が学習に使える）、学習の終了を待つ
4. 学習が終わると LLM を読み込み直し、終了コード・ログ末尾・エージェントが残したメモを添えて会話を再開する
5. エージェントが結果を読み、次の改善を書いて 2 に戻る。目標に届けば `predict.py` → `RESULTS.md` を書いて終わる

人は何もしなくてよい。動作確認で実際に出たやりとりはこんな感じ（ベースラインを 1 回だけ回させたもの）:

```
> build · large
ではベースラインの train.py を vram=2G で実行します。
✗ yuzuru_train {"command":"uv run python train.py","log":".yuzuru/logs/train-baseline.log","vram":"2G",...} failed
Error: Invalid arguments for tool "yuzuru_train":
- cwd: Missing key
⚙ yuzuru_train {"command":"uv run python train.py","cwd":"/home/you/work/cifar10-task","log":...,"vram":"2G",...}
学習ジョブ（pid 84113）を開始しました。このターンは終了し、完了後に自動的に再開します。
```

ツールの引数を間違えてもエラーを読んで直すので、1 回の失敗は気にしなくてよい。15 秒後、`resume.log` に
再開したターンが現れる:

```
ベースラインの学習が完了しました。結果は以下の通りです：
| 5/5 | 0.9250 | 0.6642 |
```

補足:

- 再開したターンが、ツールを使わずに返事 1 回だけで終わり、学習も動いていない場合、yuzuru は「止まっている」とみなし、
  「続きを進めるか、終わっているなら結果をまとめよ」というメッセージで 1 回だけ追加で再開する
  （`resume.stall_nudges`）。本当に終わっていても 1 回は追加のターンが走る（「タスク完了です」と繰り返して終わる）
- エージェントが学習を起動したのにターンを続けている間は、そのターンが終わるまで再開を待つ。同じ会話で
  エージェントが 2 つ同時に動くことはない

## 9. 見守る・止める

| 見たいもの | コマンド / 場所 |
|---|---|
| GPU・LLM の予算・読み込み中のモデル・学習ジョブ（lease） | `yuzuru status` |
| 学習の終了と再開の履歴 | `yuzuru events`（`-f` で流し続ける） |
| 学習ログ | `yuzuru_train` の `log` に渡したパス（課題ディレクトリからの相対） |
| 自動で再開したターンの出力 | `~/.local/state/yuzuru/resume.log` |
| エージェントが変えたもの | `cd $TASK && git status && git diff` |
| yuzuru のログ | `yuzuru serve` の端末 |
| 管理下の Ollama のログ | `~/.local/state/yuzuru/ollama-11435.log`（11435 は yuzuru のポート） |
| 会話の全体 | `opencode` の TUI でセッションを開く |

- **学習だけ止める**: `yuzuru status` の `Leases:` に出ている PID を kill する。lease は自動で回収され、
  エージェントには「異常終了した」として会話が再開する
- **エージェントごと止める**: `yuzuru serve` を止める（Ctrl-C）。再開が起きなくなり、管理下の Ollama も止まる。
  実行中のターンが残っていれば `opencode service stop`
- **止まった会話を手で動かす**: セッション ID（`yuzuru events` や `resume.log` に出る `ses_...`）を指定して続きを頼む
  ```bash
  cd "$TASK"
  yuzuru agent -- opencode run --session ses_xxxx "前回の続きから進めてください"
  ```

## 10. 採点と検証

エージェントが終わったら採点する。

```bash
cd "$TASK"
uv run python "$YUZURU/examples/cifar10/score.py" "$GRADER/test_labels.npy" submission.csv
```

自律実行の結果をそのまま信じず、次も確かめる。実際の実行でも、良い重みを後の実験で上書きして `RESULTS.md` の数字と
提出物が食い違った、提出形式を間違えた（ラベルをクラス名で書いた）、といったことが起きている。

- `submission.csv` が本当に提出された重みから出たものか: `predict.py` を走らせ直し、差分がないことを確かめる
  ```bash
  cp submission.csv /tmp/submitted.csv && uv run python predict.py && cmp submission.csv /tmp/submitted.csv
  ```
- `RESULTS.md` の `val_acc` が学習ログと一致するか、その重みが今のファイルか（ファイルの更新時刻とログの時刻）
- テストの答えに触っていないか: `git diff` と書かれたコードで、`data/train.npz` 以外を学習に使っていないか、
  配布元からテストセットを取りに行っていないか、`$GRADER` を読もうとしていないか
- 検証データが学習データと重なっていないか（分け方のコード）

## うまくいかないとき（実際にあったもの）

| 症状 | 原因と対処 |
|---|---|
| `yuzuru warm` が `"placement": "none"` を返す | 初回の最悪見積もりが RAM 予算を超えている（手順 4）。`yuzuru warm large --force` で一度読み込む |
| 学習が OOM する、LLM の層が学習中の GPU に載っている | `ollama.manage: true` か確認（システムの Ollama を直接使うと GPU を隠せない）。Vulkan バックエンドは `CUDA_VISIBLE_DEVICES` を無視するので `OLLAMA_VULKAN: "0"` |
| 違う GPU を使っている | GPU を番号で書いている。UUID で書く |
| エージェントのシェルで `torch.cuda.is_available()` が False、CUDA エラー | 意図どおり（手順 7）。学習は `yuzuru_train` で動かす。エージェントが気づかなければ `AGENTS.md` に書く |
| `... not in an agent scope` の警告 | 手順 7 の `opencode service stop` → `yuzuru agent -- opencode service start` |
| エージェントが `sleep` や `yuzuru_wait` で待ち続ける | `AGENTS.md` に「`yuzuru_train` を呼んだらターンを終える」と書く。応答がないまま `resume.turn_idle_seconds`（既定 600 秒）たてば終わったとみなして再開する |
| 学習が全く進まない（val 10% のまま） | ハイパーパラメータの誤り（AdamW に SGD 用の LR 0.1 を渡す、など）。ローカル LLM は同じ間違いを繰り返すことがある。`AGENTS.md` にヒントを足すか、大きいモデルにする |
| OpenCode がクラウドのモデルで応答している | yuzuru が落ちている。`disabled_providers: ["opencode"]` を入れる |
| 応答が考え始めた所で切れる | モデルのチャットテンプレートの問題（手順 2 の注） |
| 許可リストをすり抜けてファイルを書かれた | `echo ... > file` のようなリダイレクトは `echo *` の許可で通る。気になるなら `echo` を外す |
| エージェントが自分のコマンドの出力を読めない（`external_directory` で拒否） | `opencode.json` に `~/.local/share/opencode/shell/*` の許可があるか |
| 再開したはずなのに何も起きない | `resume.log` を見る。`opencode` が見つからない場合は、`yuzuru serve` を `opencode` が PATH にある端末から起動する |

## 応用

- **GPU が 2 枚以上**: `resources.gpus` に両方の UUID を書く。`yuzuru_train` に `gpus`（UUID）を渡すと、その GPU は
  学習が終わるまで LLM に一切使われない（管理下の Ollama がその GPU を隠して再起動する）。`vram` を渡した場合は
  その量だけを差し引き、残りは LLM が使える
- **CPU だけの学習**（LightGBM、scikit-learn、前処理など）: `yuzuru run --cpu` / `yuzuru_train` の `device: "cpu"`。
  LLM を降ろさずに走り、ジョブには GPU を見せない
- **モデルを複数**: tier を増やし（`small` / `medium` / `large`）、`router.classifier: llm` にすると、軽い作業は
  小さいモデルに回る。`opencode.json` の `models` にも tier を足す
- **常駐させる**: README の [systemd user unit](../README.md#常駐させるsystemd-user-unit)。自動再開で `opencode` を
  起動するので、unit の環境から `opencode` が見えることを確かめる
- **他の課題**: `task/` の `README.md`（課題と評価）、`AGENTS.md`（作法）、`prepare.py`（データと隠しラベルの分離）を
  差し替えれば同じ流れで使える
