# CIFAR-10 example

ローカル LLM エージェントに CIFAR-10 を解かせる例。手順は [docs/tutorial.md](../../docs/tutorial.md)。

- `prepare.py` — CIFAR-10 を課題用データ（ラベルなしテスト）と隠し正解ラベルに分ける
- `score.py` — `submission.csv` を採点する
- `task/` — エージェントに渡す課題ディレクトリのひな形（yuzuru・OpenCode の設定は `yuzuru init` で足す）
