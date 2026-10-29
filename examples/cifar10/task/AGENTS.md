# 作業ルール

課題は `README.md` を読むこと。テスト正解率 85% 以上が目標。学習・推論のコードは自分で書く。

## Python と パッケージ

- Python は必ず `uv run python ...` で実行する（このディレクトリの `.venv` が使われる）。素の `python` / `python3` は使わない
- パッケージが足りないときは `uv pip install <pkg>`（`.venv` に入る）。`pip install`、`--user`、`--break-system-packages` は禁止
- `.venv` には torch / torchvision / numpy / pillow が入っている

## 学習の実行

- **GPU を使う処理は、必ず MCP の `yuzuru_train` ツールで実行する**
  （`command: "uv run python train.py"`、`cwd` はこのディレクトリ、`log` にログファイル名）。
  シェルからは GPU が見えない（`torch.cuda.is_available()` が False になる）
- ツールを呼んだらそのターンは終えること。学習が終わると、終了コードとログの末尾付きで会話が自動で再開する
- **GPU はあなた自身（LLM）と共有している**。`yuzuru_train` では必ず `vram` に学習で使う VRAM 量を
  指定する（例 `"4G"`）。指定した量を超えて使うと、LLM とぶつかって学習が OOM で落ちることがある
- 学習スクリプトの `print` には `flush=True` を付ける（付けないとログに出るのが遅れる）
- 実行中の学習の様子は `tail -n 50 <log>` で見る。`sleep` で待たない。実行中の学習を止めようとしない
- 改善の判断は検証精度（`val_acc`）で行う。テストデータのラベルはどこにもない
- 新しい学習で、前の良い重みを上書きしないこと（実験ごとに別のファイル名で保存する）

## 提出

- 最終的に次の 3 つをこのディレクトリに残す
  - 学習済みの重み
  - その重みから `submission.csv` を作り直せる `predict.py`
  - `RESULTS.md`: 最終的な手法、`val_acc`、学習に使ったコマンドとログファイル名
- `submission.csv` は必ず `predict.py` の出力であること（形式は `data/sample_submission.csv` と同じ `id,label`、label は整数）
