"""CIFAR-10 を「コンペ形式」に変換する。

    uv run python prepare.py <課題ディレクトリ> <採点用ディレクトリ>   # torchvision の入った環境で

課題ディレクトリには data/{train,test}.npz、classes.txt、sample_submission.csv を書く。
test.npz にはラベルを入れず、テスト画像の順番もシャッフルする（配布元の並びから答えを引けないように）。
正解ラベルは採点用ディレクトリの test_labels.npy にだけ置く。採点用ディレクトリはエージェントから
見えない場所（課題ディレクトリの外）にすること。
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("grader_dir", type=Path)
    ap.add_argument("--root", type=Path, default=Path.home() / ".cache" / "cifar10",
                    help="download cache (keep it outside the task dir)")
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    train = CIFAR10(args.root, train=True, download=True)
    test = CIFAR10(args.root, train=False, download=True)
    rng = np.random.default_rng(args.seed)

    data = args.task_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(train.data))
    np.savez(data / "train.npz", images=train.data[order],
             labels=np.asarray(train.targets, dtype=np.int64)[order])

    order = rng.permutation(len(test.data))
    ids = np.arange(len(order), dtype=np.int64)
    np.savez(data / "test.npz", images=test.data[order], ids=ids)
    (data / "classes.txt").write_text("".join(f"{i} {c}\n" for i, c in enumerate(train.classes)))
    with open(data / "sample_submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows((i, 0) for i in ids.tolist())

    args.grader_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.grader_dir / "test_labels.npy", np.asarray(test.targets, dtype=np.int64)[order])
    print(f"wrote {data} and {args.grader_dir / 'test_labels.npy'}")


if __name__ == "__main__":
    main()
