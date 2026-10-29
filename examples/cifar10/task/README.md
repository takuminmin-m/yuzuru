# CIFAR-10 Image Classification

32×32 のカラー画像を 10 クラスに分類するコンペです。**目標はテスト正解率 85% 以上**。

## データ（`data/`）

| ファイル | 中身 |
|---|---|
| `train.npz` | `images` (50000, 32, 32, 3) uint8、`labels` (50000,) int64 |
| `test.npz` | `images` (10000, 32, 32, 3) uint8、`ids` (10000,) — ラベルなし |
| `classes.txt` | ラベル番号 → クラス名 |
| `sample_submission.csv` | 提出形式の例 |

## ベースライン

| ファイル | 中身 |
|---|---|
| `model.py` | 小さな CNN |
| `train.py` | 学習（train の最後の 5000 枚を検証用に取り分け、エポックごとに `val_acc` を表示）→ `model.pt` |
| `predict.py` | `model.pt` でテスト画像を推論 → `submission.csv` |

```bash
uv run python train.py     # 5 エポック、val_acc 約 0.66
uv run python predict.py
```

## 提出

`submission.csv`（`id,label` の 2 列、テスト全 10000 行）をこのディレクトリ直下に置いてください。
学習したモデルの重みと、その重みから `submission.csv` を作り直せる `predict.py` もセットで残すこと。

## 評価

テストデータの正解率（accuracy）。

## ルール

- 学習に使えるのはこのディレクトリのデータだけ。外部データ・事前学習済みの重みは使用禁止
- テストデータの正解ラベルを、どんな形であれ参照・入手しないこと（CIFAR-10 の配布元からテストセットを取得する、など）
- 手法を調べるためにネットを見るのは自由
- 使える計算資源は GPU 1 枚と CPU
