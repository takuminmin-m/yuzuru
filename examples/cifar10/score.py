"""submission.csv（id,label）を隠しラベルで採点する。

    python score.py <採点用ディレクトリ>/test_labels.npy <課題ディレクトリ>/submission.csv
"""
import csv
import sys

import numpy as np

labels = np.load(sys.argv[1])
pred = {}
with open(sys.argv[2]) as f:
    for row in csv.DictReader(f):
        pred[int(row["id"])] = int(row["label"])
missing = len(labels) - len(pred)
acc = np.mean([pred.get(i, -1) == labels[i] for i in range(len(labels))])
print(f"accuracy {acc:.4f} ({len(pred)} rows, {missing} missing)")
