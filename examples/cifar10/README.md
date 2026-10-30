# CIFAR-10 の例

ローカル LLM のエージェントに、CIFAR-10 の画像分類を解かせる例です。
使い方は [チュートリアル](../../docs/tutorial.md) を、手順 1 から順に見てください。

| ファイル | 中身 | 使う手順 |
|---|---|---|
| `prepare.py` | CIFAR-10 を、課題用のデータ（ラベルなしのテストデータ）と隠した正解ラベルに分ける | 手順 4 |
| `score.py` | `submission.csv` を隠した正解ラベルで採点する（任意） | 手順 6、10 |
| `task/` | エージェントに渡す課題ディレクトリのひな形（課題文、作業ルール、ベースライン）。yuzuru と OpenCode の設定は `yuzuru init` で足す | 手順 3、4 |

`prepare.py` と `score.py` の使い方（`prepare.py` は torchvision の入った環境で実行します）:

```bash
uv run python prepare.py <課題ディレクトリ> <採点用ディレクトリ> [--root <CIFAR-10 のダウンロード先>]
uv run python score.py <採点用ディレクトリ>/test_labels.npy <課題ディレクトリ>/submission.csv
```

採点用ディレクトリは、課題ディレクトリの外（エージェントから見えない場所）に置いてください。採点しないなら、`score.py` と採点用ディレクトリは使いません。
