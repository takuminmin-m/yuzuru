# CIFAR-10 example

ローカル LLM エージェントに CIFAR-10 を解かせる例。手順は [docs/tutorial.md](../../docs/tutorial.md)。

- `config.yaml` — yuzuru の設定（`~/.config/yuzuru/config.yaml` にコピーして GPU の UUID とモデル名を直す）
- `prepare.py` — CIFAR-10 を課題用データ（ラベルなしテスト）と隠し正解ラベルに分ける
- `score.py` — `submission.csv` を採点する
- `task/` — エージェントに渡す課題ディレクトリのひな形
