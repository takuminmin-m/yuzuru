# チュートリアル: ローカル LLM エージェントに CIFAR-10 を解かせる

このチュートリアルでは、GPU 1 枚のマシンで、ローカル LLM のエージェント（OpenCode）に画像分類の課題を最後までやらせます。
エージェントは「課題を読む → 学習コードを書く → GPU で学習する → 結果を見て改善する → 提出する」を自分で繰り返します。
LLM と学習ジョブは同じ GPU を使うので、その取り合いを yuzuru が仲裁します。

- 手順 1〜6（30 分ほど）: yuzuru を入れ、手で学習ジョブを動かします。「学習中は LLM が GPU から退き、終わると戻る」ことを確かめます。
- 手順 7〜10（モデルによって 1〜2 時間）: エージェントに課題を渡して任せ、終わったら結果を確かめます。

この例はコンペ形式で、隠したテストラベルでの採点まで行います。**採点は必須ではありません。**
ハイパーパラメータやモデル構造のチューニングを任せるなら、検証データの指標と終わる条件を課題の `README.md` に書くだけで済みます
（[採点なしで使う](#採点なしで使うハイパーパラメータモデル構造のチューニング)）。採点に関わる手順には「（任意）」と付けています。

用語（tier、lease、scope など）は [README の用語](../README.md#用語) で説明しています。仕組みの詳細は [architecture.md](architecture.md) にあります。

## 使うファイル

使うファイルは [`examples/cifar10/`](../examples/cifar10/) にあります。

| ファイル | 中身 |
|---|---|
| `prepare.py` | CIFAR-10 を取得し、ラベルなしのテストデータと、隠した正解ラベルに分ける |
| `score.py` | `submission.csv` を隠した正解ラベルで採点する（任意） |
| `task/` | エージェントに渡す課題ディレクトリのひな形（`README.md`、`AGENTS.md`、ベースライン）。yuzuru と OpenCode の設定は `yuzuru init` で足す |

## 全体の流れ

```
あなた ── yuzuru agent -- opencode run "課題に取り組んで"
              │
              ▼
          OpenCode ──(LLM API / MCP)──▶ yuzuru (:11435) ──▶ 管理下の Ollama ──▶ GPU
              │
     エージェントが MCP の yuzuru_train で学習を起動し、ターンを終える
              ▼
  yuzuru: LLM を GPU から降ろす → 学習を実行 → 終わるまで待つ（その間 LLM はメモリを使わない）
              ▼
  学習が終わる → LLM を読み込み直す → 終了コードとログ末尾を添えて会話を自動で再開する
              ▼
  エージェントが結果を見て次の手を打つ（目標に届くまで繰り返す）
```

人が何もしなくても会話が続くのは、yuzuru が学習の終了を見張っていて、`opencode run --session <id> "<結果>"` で会話を再開させるからです。

## 0. 必要なものを確かめる

- Linux と NVIDIA GPU（`nvidia-smi` が動くこと）。VRAM 12 GB、RAM 32 GB で動作を確認しています
- systemd のユーザーセッション（普通のデスクトップ Linux にはあります）。エージェントと学習ジョブを、メモリ上限付きの scope に入れるのに使います。なくても動きますが、暴走したスクリプトが端末ごと落とすことがあります
- [Ollama](https://ollama.com)（0.40 で確認）
- [OpenCode](https://opencode.ai) 2.x（2.0.32 で確認）。`yuzuru init` は、入っている OpenCode の版に合わせた `opencode.json` を作ります
- [uv](https://docs.astral.sh/uv/)

## 1. yuzuru を入れる

```bash
git clone https://github.com/takuminmin-m/yuzuru.git
uv tool install -e ./yuzuru
yuzuru --help
```

`yuzuru` コマンドが `~/.local/bin` に入ります。以降、リポジトリの場所を `$YUZURU` で表します。

```bash
YUZURU=$PWD/yuzuru
```

## 2. エージェント用のモデルを用意する

エージェントには、**ツール呼び出しができて、ある程度大きいモデル**を使ってください。
小さいモデル（数 B 程度）は、ファイル編集やツールの使い方を間違えて先に進まないことが多くなります。
VRAM に入りきらない分は RAM に逃がせます。そのため、MoE モデル（一度に動くパラメータが少ないモデル）の低ビット量子化が扱いやすいです。

```bash
ollama pull <モデル名>
ollama list          # 名前を確かめる（例: qwen3.6-iq3:latest）
```

> Hugging Face から取ってきた GGUF は、同梱のチャットテンプレートが壊れていることがあります。
> たとえば `<think>` が停止語になっていると、考え始めた所で応答が切れます。
> その場合は `Modelfile` で `RENDERER` と `PARSER` を指定して作り直してください（例 `FROM <blob>`、`RENDERER qwen3.5`、`PARSER qwen3.5`）。

## 3. 課題ディレクトリを作る

課題ディレクトリ（エージェントが作業する場所）を作ります。採点する場合は、正解ラベルを置く採点用ディレクトリを、エージェントから見えない場所に分けておきます（この例の `prepare.py` は両方を作ります）。

```bash
TASK=~/work/cifar10-task
GRADER=~/work/.cifar10-grader

cp -r "$YUZURU/examples/cifar10/task" "$TASK"
```

## 4. `yuzuru init` で設定する

yuzuru、OpenCode、エージェント、uv の設定は `yuzuru init` がまとめて作ります。

1. GPU の番号を調べます。

   ```bash
   nvidia-smi -L
   ```

   ```
   GPU 0: NVIDIA GeForce RTX 3050 (UUID: GPU-6a1e...)
   GPU 1: NVIDIA GeForce RTX 3060 (UUID: GPU-403d...)
   ```

2. 課題ディレクトリに設定一式を作ります。

   ```bash
   yuzuru init "$TASK" --gpu 1 --packages torch torchvision numpy pillow
   ```

   | 作るもの | 中身 |
   |---|---|
   | `yuzuru.yaml` | yuzuru の設定。この課題ディレクトリで実行した yuzuru コマンドは、`~/.config/yuzuru/config.yaml` よりこちらを使う |
   | `opencode.json` | OpenCode の設定（yuzuru への接続と権限）。手順 7 |
   | `AGENTS.md` | エージェントへの作業ルール（手順 7）。すでにあれば残す（この例では課題用のものが入っている） |
   | `.gitignore` | `.venv/`、`.yuzuru/`、`data/`、重みファイルなどを足す |
   | `.venv` | `uv venv` を作り、`--packages` を入れる（`pyproject.toml` があれば `uv sync`、`requirements.txt` があればそれも入れる） |
   | git | リポジトリでなければ `git init` して課題を commit する。エージェントが何を変えたかを、あとで `git diff` で追える |

   すでにあるファイルは書き換えません（`--force` を付けると `.bak` を残して作り直します）。

3. データを用意します。

   ```bash
   cd "$TASK"
   uv run python -c "import torch; print(torch.cuda.is_available())"   # True になること
   uv run python "$YUZURU/examples/cifar10/prepare.py" "$TASK" "$GRADER"  # 約 170 MB をダウンロードする
   ```

   CIFAR-10 の Python 版（`cifar-10-batches-py/`）が手元にあれば、`--root <その親ディレクトリ>` を付けるとダウンロードを省けます。

`yuzuru.yaml` の設定と、対応する `init` の引数:

| 設定 | `init` の引数 | 意味 |
|---|---|---|
| `resources.gpus` | `--gpu`（番号か UUID、複数可。既定は全部） | ここに書いた GPU だけを、LLM にも学習にも使う。学習ジョブには `CUDA_VISIBLE_DEVICES` が自動で設定される |
| `tiers` | `--model`、`--ctx`（既定 65536） | モデル 1 つだけの `large`。モデルの既定は、yuzuru が前回 `large` に使ったもの（初めてなら `ollama list` で一番大きいもの）。`num_ctx` は 32k 以上にする（OpenCode はシステムプロンプトとツール定義だけで 2 万トークン近く送る） |
| `ollama.manage: true` | | yuzuru が、上の GPU だけが見える専用の `ollama serve` を起動する。システムの `ollama.service`（:11434）は動いたままでよい。重みはシンボリックリンクで共有するので、ディスクは増えない |
| `isolation.memory_max` | `--memory-max`（既定 80%） | エージェントと学習ジョブ 1 つあたりの RAM 上限。超えたプロセスだけが殺され、yuzuru や端末は無事 |
| `port` | `--port`（既定 11435） | yuzuru のポート。課題を並行して動かすときは課題ごとに変える |
| `resume.command` | | 学習後に会話を再開するコマンド。出力を課題ディレクトリの `.yuzuru/resume.log` に残す |

**`yuzuru.yaml` には GPU が UUID で書かれます**（`--gpu 1` は `nvidia-smi` の番号）。`nvidia-smi` の番号（PCI バス順）と、CUDA や PyTorch の番号（速い順）は一致しないことがあります。上の例では、CUDA の 0 番は 3060 です。

設定項目の一覧は [architecture.md の主な設定項目](architecture.md#主な設定項目) にあります。

できあがるもの:

```
cifar10-task/
├── README.md        課題（目標: テスト正解率 85%、ルール）
├── AGENTS.md        エージェントへの作業ルール（手順 7）
├── model.py / train.py / predict.py   ベースライン（val_acc 約 0.66）
├── yuzuru.yaml      yuzuru の設定
├── opencode.json    OpenCode の設定（yuzuru への接続と権限）
└── data/            train.npz（ラベル付き 50000 枚）、test.npz（ラベルなし 10000 枚）など
```

## 5. yuzuru を起動して状態を見る

1. 端末を 1 つ yuzuru 用にして、**課題ディレクトリで**起動します（`yuzuru.yaml` を読みます）。ログはこの端末に流れます。

   ```bash
   cd "$TASK"
   yuzuru serve
   ```

2. 別の端末で（同じく課題ディレクトリで）状態を見ます。

   ```bash
   cd "$TASK"
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

   - `LLM may use`: 学習ジョブなどを差し引いたうえで、LLM に使わせてよい量
   - `Tiers` の右端: 今読み込んだらどこに載るかの見込み。`gpu`（全部 GPU）、`offload`（一部 RAM）、`none`（載らない）
   - `*` 付きのサイズ: GGUF からの推定値。一度読み込むと実測値に置き換わる

   先頭に `!` 付きの警告が出ていたら、先に片付けてください（[README の困ったとき](../README.md#困ったとき)）。

3. 見込みが `none` なら、一度だけ強制的に読み込みます。

   ```bash
   yuzuru warm large --force
   ```

   実測値がないうちは、yuzuru は「GPU からはみ出す分だけでなく、重み全体がホスト RAM にも載る」という最悪の見積もりをします。空き RAM が少ないと予算を超えて `none` になり、このままではエージェントを動かしてもモデルが読み込まれません。

   `{"tier": "large", "placement": "gpu"}` のように、実際に載った場所が返ります。yuzuru のログには `took 10.2 GB VRAM + 0.4 GB host RAM` のような実測値が出て、以後の見込みはこれを使います。
   見込みが最初から `gpu` か `offload` なら、`--force` は要りません。どちらの場合も、先に読み込んでおくと最初の応答が速くなります。

## 6. 手で学習ジョブを動かしてみる

エージェントに渡す前に、yuzuru 経由の学習を自分で動かして、仕組みを確かめます。
別の端末で `watch -n1 yuzuru status` を開いておくと、様子が分かりやすくなります。

1. 学習を動かします。

   ```bash
   cd "$TASK"
   yuzuru run --vram 2G --log /tmp/baseline.log -- uv run python train.py
   ```

   ```
   [yuzuru] lease cc071634: LLM moved out of GPU memory
   ```

   `--vram 2G` は「この学習は VRAM を 2 GB まで使う」という申告です。学習の間、yuzuru はその分を LLM に使わせません。GPU に余裕があれば、残りは LLM が使い続けられます。

2. 起きていることを確かめます。
   1. yuzuru が学習用の lease を作り、GPU 上の LLM を降ろす
   2. 学習が始まる。`yuzuru status` に `Leases:` の行が出る
   3. 学習が終わる（RTX 3060 で 15 秒ほど、`val_acc` 約 0.66）と lease が消え、LLM が読み込み直される

   yuzuru の端末には、次のように出ます。

   ```
   lease cc071634: unloading yuzuru-large:qwen3.6-iq3-64k
   lease cc071634 (uv, pid 83240) created
   lease cc071634 released (rc=0)
   loading yuzuru-large:qwen3.6-iq3-64k (gpu, ~10.4 GB)
   ```

3. （任意）採点の流れも確かめておきます。

   ```bash
   uv run python predict.py
   uv run python "$YUZURU/examples/cifar10/score.py" "$GRADER/test_labels.npy" submission.csv   # accuracy 0.66 前後
   rm model.pt submission.csv
   ```

## 7. エージェントの準備をする

### `AGENTS.md`: エージェントへの作業ルール

OpenCode は `AGENTS.md` を毎ターン読み込みます。ローカルの LLM は細かい作法を外しやすいので、ここで明示しておきます。
ひな形で指示していること:

- Python は `uv run python` で動かす（素の `python` だとシステムの Python を使ってしまう）
- **GPU を使う処理は MCP の `yuzuru_train` ツールで起動し、呼んだらターンを終える。** 学習が終わると会話が自動で再開する
- `yuzuru_train` には必ず `vram`（使う VRAM 量）を渡す。渡さないとツールがエラーを返す
- `sleep` で待たない。実行中の学習を止めようとしない。良い重みを上書きしない
- 判断は `val_acc` で行う。学習のたびに、設定と `val_acc` を `RESULTS.md` の表に 1 行足す
- 提出物は重み、`predict.py`、`RESULTS.md`。`submission.csv` は `predict.py` の出力にする

課題に合わせて足すとよいもの:

- RAM の上限（`isolation.memory_max` に合わせる）
- CPU だけで済む学習（LightGBM など）は、`yuzuru_train` の `device: "cpu"` で動かすこと

### `opencode.json`: 接続と権限

課題ディレクトリの `opencode.json`（`yuzuru init` が作ります）が、このディレクトリで動く OpenCode の設定になります。

- `provider.yuzuru` と `mcp.servers.yuzuru`: yuzuru（:11435）への接続です。`models` のキー（`large`）は yuzuru の tier 名と合わせます。`yuzuru.yaml` のポートや tier を変えたら、`yuzuru init --force` で作り直してください（`AGENTS.md` なども作り直されるので、手を入れていたら `.bak` から戻します）
- `disabled_providers: ["opencode"]`: yuzuru に繋がらないとき、OpenCode は組み込みの無料クラウドモデルに黙って切り替え、コードを外部に送ります。それを防ぎます
- `permission`: 人が見ていない間に動かすので、全部許可ではなく許可リストにしています
  - `bash` は既定で拒否し、`uv run *`、`ls`、`cat`、`tail`、`git diff` など必要なものだけ許可します。許可した中でも `rm -r*` / `rm -f*`、`--break-system-packages` は拒否します。`sudo` は拒否し、`kill` は許可リストにないので拒否されます
  - `external_directory` は拒否します（課題ディレクトリの外、たとえば採点用ディレクトリを読ませない）。例外として `~/.local/share/opencode/shell/*` だけ許可します。OpenCode 2.x は長いコマンド出力をここに保存してエージェントに読ませるので、拒否すると自分のコマンドの出力を読めなくなります
  - `execute` は拒否します。OpenCode 2.x の `execute` ツールは JavaScript でコードを実行でき、bash の許可リストを素通りするためです
  - `question` は拒否します。人への質問で止まらないようにするためです

### OpenCode のサービスを yuzuru の scope で起動する

OpenCode 2.x の `opencode run` や TUI は、ターンの実行をバックグラウンドサービス（`opencode service`）に任せます。
エージェントが実行するコマンドは、サービスのプロセスの子として動きます。そのため、**サービスをどこで起動したか**が効いてきます。

```bash
cd "$TASK"
opencode service stop                          # 動いていなければ何もしない
yuzuru agent -- opencode service start
```

`yuzuru agent` で起動すると、サービスと、エージェントが実行するすべてのコマンドは次の状態になります。

- メモリ上限付きの scope（`yuzuru-agent-*.scope`、上限は `isolation.memory_max`）に入ります。エージェントの書いたスクリプトが RAM を食い尽くしても、殺されるのはそのスクリプトだけです
- **GPU が見えません。** シェルから直接 `python train.py` を動かすと、lease なしで LLM の隣に載り、シェルツールのタイムアウトで途中で殺されます。GPU が見えなければ CUDA のエラーですぐ落ち、`yuzuru_train` を使うよう案内が出ます。`yuzuru_train` で起動したジョブには、yuzuru が lease の GPU を渡します

サービスが scope の外（普通の端末や IDE から先に `opencode` を起動した、など）で動いていると、`yuzuru agent` の出力と yuzuru のログに次の警告が出ます。そのときは上の 2 行をやり直してください。

```
the OpenCode service (pid ...) runs in ..., not in an agent scope
```

## 8. エージェントを動かす

```bash
cd "$TASK"
yuzuru agent -- opencode run "README.md と AGENTS.md を読んで、課題を最後までやり遂げてください。"
```

TUI で `opencode` を開いて同じことを打っても構いません。サービスに繋がった TUI には、自動で再開したターンも表示されます。

このあと起きること:

1. エージェントが課題とベースラインを読み、`train.py` などを書き換える
2. `yuzuru_train` で学習を起動し、ターンを終える。`opencode run` もここで戻ってくる
3. yuzuru が LLM をすべて降ろし（GPU 全体が学習に使える）、学習の終了を待つ
4. 学習が終わると、yuzuru が LLM を読み込み直し、終了コード・ログの末尾・エージェントが残したメモを添えて会話を再開する
5. エージェントが結果を読み、次の改善を書いて 2 に戻る。目標に届けば `predict.py` を実行し、`RESULTS.md` を書いて終わる

人は何もしなくて構いません。ベースラインを 1 回だけ回させたときのやりとりは、次のようになります。

```
> build · large
ではベースラインの train.py を vram=2G で実行します。
✗ yuzuru_train {"command":"uv run python train.py","log":".yuzuru/logs/train-baseline.log","vram":"2G",...} failed
Error: Invalid arguments for tool "yuzuru_train":
- cwd: Missing key
⚙ yuzuru_train {"command":"uv run python train.py","cwd":"/home/you/work/cifar10-task","log":...,"vram":"2G",...}
学習ジョブ（pid 84113）を開始しました。このターンは終了し、完了後に自動的に再開します。
```

ツールの引数を間違えても、エージェントはエラーを読んで直します。1 回の失敗は気にしなくて構いません。
学習が終わると、課題ディレクトリの `.yuzuru/resume.log` に再開したターンが現れます。

```
ベースラインの学習が完了しました。結果は以下の通りです：
| 5/5 | 0.9250 | 0.6642 |
```

補足:

- 再開したターンが、ツールを使わずに応答 1 回だけで終わり、学習も動いていない場合、yuzuru は「止まっている」とみなします。そして「続きを進めるか、終わっているなら結果をまとめよ」というメッセージで、1 回だけ追加で再開します（`resume.stall_nudges`）。本当に終わっていても、追加のターンが 1 回走ります
- エージェントが学習を起動したあともターンを続けている間は、そのターンが終わるまで再開を待ちます。同じ会話でエージェントが 2 つ同時に動くことはありません

## 9. 見守る・止める

| 見たいもの | コマンド / 場所 |
|---|---|
| GPU、LLM の予算、読み込み中のモデル、学習ジョブ（lease） | `yuzuru status` |
| 学習の終了と再開の履歴 | `yuzuru events`（`-f` で流し続ける） |
| 学習ログ | `yuzuru_train` の `log` に渡したパス（課題ディレクトリからの相対パス） |
| 自動で再開したターンの出力 | `$TASK/.yuzuru/resume.log` |
| エージェントが変えたもの | `cd "$TASK" && git status && git diff` |
| yuzuru のログ | `yuzuru serve` の端末 |
| 管理下の Ollama のログ | `~/.local/state/yuzuru/ollama-11435.log`（11435 は yuzuru のポート） |
| 会話の全体 | `opencode` の TUI でセッションを開く |

止め方:

- **学習だけ止める**: `yuzuru status` の `Leases:` に出ている PID を kill します。lease は自動で回収され、エージェントには「異常終了した」として会話が再開します
- **エージェントごと止める**: `yuzuru serve` を止めます（Ctrl-C）。自動再開が起きなくなり、管理下の Ollama も止まります。実行中のターンが残っていれば `opencode service stop` も実行します
- **止まった会話を手で動かす**: セッション ID（`yuzuru events` や `resume.log` に出る `ses_...`）を指定して、続きを頼みます

  ```bash
  cd "$TASK"
  yuzuru agent -- opencode run --session ses_xxxx "前回の続きから進めてください"
  ```

## 10. 結果を確かめる

エージェントは `RESULTS.md` に、試した設定と検証データの指標の表、一番良かった設定を残します。まずこれを読みます。

自律実行の結果は、そのまま信じずに確かめてください。良い重みをあとの実験で上書きして `RESULTS.md` の数字と重みが食い違う、提出形式を間違える（ラベルをクラス名で書く）、といったことが起こりえます。

- `RESULTS.md` の指標が学習ログと一致するか。その重みが今のファイルか（ファイルの更新時刻とログの時刻を比べる）
- 検証データが学習データと重なっていないか（分け方のコード）。重なっていると検証の指標が高く出て、チューニングの判断そのものが狂います
- `git diff` で、指示していない所（データの読み込み、評価のコード）を変えていないか

（任意）採点する場合は、さらに次を行います。

1. 採点します。

   ```bash
   cd "$TASK"
   uv run python "$YUZURU/examples/cifar10/score.py" "$GRADER/test_labels.npy" submission.csv
   ```

2. `submission.csv` が、本当に提出された重みから出たものかを確かめます。`predict.py` を実行し直し、差分がないことを確かめます。

   ```bash
   cp submission.csv /tmp/submitted.csv && uv run python predict.py && cmp submission.csv /tmp/submitted.csv
   ```

3. テストの答えに触れていないかを確かめます。`git diff` と書かれたコードで、`data/train.npz` 以外を学習に使っていないか、配布元からテストセットを取りに行っていないか、`$GRADER` を読もうとしていないかを見ます。

## 採点なしで使う（ハイパーパラメータ・モデル構造のチューニング）

手元の学習コードの学習率やバッチサイズ、層の数や幅などを試させる使い方では、隠した正解ラベルも `score.py` も要りません。
エージェントは検証データの指標を見て次の設定を決め、結果を `RESULTS.md` に残して終わります。手順 1、2、5〜9 はそのままで、違うのは課題ディレクトリの中身だけです。

```bash
cd ~/work/my-model                    # 自分の学習コードがあるディレクトリ（git 管理ならそのまま）
yuzuru init . --gpu 1 --packages torch numpy
```

課題の `README.md` に次を書きます。エージェントが迷うのは、たいていこのどれかが抜けているときです。

- **指標**: 何をどのデータで測り、大きい方と小さい方のどちらが良いか（例: `train.py` が最後に出す `val_loss`、小さいほど良い）
- **変えてよい範囲**: 変えてよい設定やファイルと、変えてはいけないもの（データの分け方、評価のコード、エポック数の上限など）
- **終わる条件**: 試す回数、かける時間、指標の目標、「3 回続けて改善しなければ終わる」など。ないとエージェントは自分で区切りをつけ、早く終わりがちです
- **1 回の学習の重さ**: 使う VRAM、かかる時間の目安。`yuzuru_train` の `vram` の手がかりになります

例:

```markdown
# 画像分類モデルのチューニング

`train.py` のハイパーパラメータとモデル構造（`model.py`）を調整して、`val_acc` を上げてください。

- 指標: `uv run python train.py --out <name>` が最後に出す `val_acc`（大きいほど良い）
- 変えてよいもの: `train.py` の引数と既定値、`model.py`。データの読み込み（`data.py`）と検証データの分け方は変えない
- 1 回の学習は VRAM 3 GB、10 分ほど。エポック数は 30 まで
- 終わる条件: 学習を 15 回回すか、`val_acc` 0.93 を超えたら終わり
- 結果: `RESULTS.md` に設定と `val_acc` の表、一番良かった設定、その重みのパス
```

`yuzuru init` が作る `AGENTS.md` は、この形の課題を前提にしています（学習のたびに `RESULTS.md` に 1 行足す、`README.md` の終わる条件を確かめてから終える、重みを上書きしない）。
テストデータがある場合も、設定を選ぶのには使わせず、終わったあとに人が一番良かった重みで測ってください。

## うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| `yuzuru warm` が `"placement": "none"` を返す | 初回の最悪の見積もりが RAM の予算を超えている（手順 5）。`yuzuru warm large --force` で一度読み込む |
| 学習が OOM する。LLM の層が学習中の GPU に載っている | `ollama.manage: true` か確かめる（システムの Ollama を直接使うと GPU を隠せない）。Vulkan バックエンドは `CUDA_VISIBLE_DEVICES` を無視するので `OLLAMA_VULKAN: "0"` にする |
| 違う GPU が使われる | GPU を番号で書いている。UUID で書く |
| エージェントのシェルで `torch.cuda.is_available()` が False になる。CUDA のエラーが出る | 意図どおり（手順 7）。学習は `yuzuru_train` で動かす。エージェントが気づかなければ `AGENTS.md` に書く |
| `... not in an agent scope` の警告が出る | 手順 7 の `opencode service stop` → `yuzuru agent -- opencode service start` をやり直す |
| エージェントが `sleep` や `yuzuru_wait` で待ち続ける | `AGENTS.md` に「`yuzuru_train` を呼んだらターンを終える」と書く。応答がないまま `resume.turn_idle_seconds`（既定 600 秒）たてば、ターンが終わったとみなして再開する |
| 学習がまったく進まない（val 10% のまま） | ハイパーパラメータの誤り（AdamW に SGD 用の学習率 0.1 を渡す、など）。ローカル LLM は同じ間違いを繰り返すことがある。`AGENTS.md` にヒントを足すか、大きいモデルにする |
| OpenCode がクラウドのモデルで応答している | yuzuru に繋がっていない。`disabled_providers: ["opencode"]` を入れる |
| 応答が考え始めた所で切れる | モデルのチャットテンプレートの問題（手順 2 の注） |
| 許可リストをすり抜けてファイルを書かれた | `echo ... > file` のようなリダイレクトは `echo *` の許可で通る。気になるなら `echo` を外す |
| エージェントが自分のコマンドの出力を読めない（`external_directory` で拒否される） | `opencode.json` に `~/.local/share/opencode/shell/*` の許可があるか確かめる |
| 再開するはずなのに何も起きない | `resume.log` を見る。`opencode` が見つからないなら、`opencode` が PATH にある端末から `yuzuru serve` を起動するか、`yuzuru.yaml` の `resume.command` に `opencode` の絶対パスを書く |

yuzuru 自体の問題は [README の困ったとき](../README.md#困ったとき) も見てください。

## 応用

- **GPU が 2 枚以上ある**: `resources.gpus` に両方の UUID を書きます。`yuzuru_train` に `gpus`（UUID）を渡すと、その GPU は学習が終わるまで LLM に一切使われません（管理下の Ollama がその GPU を隠して再起動します）。`vram` を渡した場合は、その量だけを差し引き、残りは LLM が使えます
- **CPU だけの学習**（LightGBM、scikit-learn、前処理など）: `yuzuru run --cpu` か、`yuzuru_train` の `device: "cpu"` を使います。LLM を降ろさずに動き、ジョブには GPU を見せません
- **モデルを複数使う**: tier を増やし（`small` / `medium` / `large`）、`router.classifier: llm` にすると、軽い作業は小さいモデルに回ります。`opencode.json` の `models` にも tier を足してください
- **常駐させる**: README の [systemd user unit](../README.md#常駐させるsystemd-user-unit) を使います。自動再開で `opencode` を起動するので、unit から `opencode` が見つかることを確かめてください
- **ほかの課題**: `task/` の `README.md`（課題と評価）と `AGENTS.md`（作法）を差し替えれば、同じ流れで使えます。採点したいときだけ、`prepare.py` のようにテストの正解を課題ディレクトリの外に分けます
