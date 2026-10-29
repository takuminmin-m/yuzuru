"""model.pt でテスト画像を推論し、submission.csv を書き出す。

    uv run python predict.py
"""
import csv

import numpy as np
import torch

from model import SimpleCNN
from train import to_tensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    data = np.load("data/test.npz")
    x, ids = to_tensor(data["images"]), data["ids"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            preds.append(model(x[i:i + 512].to(device)).argmax(1).cpu())
    preds = torch.cat(preds).numpy()

    with open("submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(ids.tolist(), preds.tolist()))
    print(f"wrote submission.csv ({len(preds)} rows)")


if __name__ == "__main__":
    main()
