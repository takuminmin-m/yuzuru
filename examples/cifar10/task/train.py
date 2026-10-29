"""ベースライン: 小さなCNNを数エポック学習して model.pt を保存する。

    uv run python train.py

train.npz の最後の 5000 枚を検証用に取り分け、エポックごとに検証精度を表示する。
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleCNN

EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
VAL_SIZE = 5000
SEED = 0


def to_tensor(images: np.ndarray) -> torch.Tensor:
    # (N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float32 in [0, 1]
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    data = np.load("data/train.npz")
    x, y = to_tensor(data["images"]), torch.from_numpy(data["labels"]).long()
    x_train, y_train = x[:-VAL_SIZE], y[:-VAL_SIZE]
    x_val, y_val = x[-VAL_SIZE:], y[-VAL_SIZE:]
    print(f"train {len(x_train)}, val {len(x_val)}")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=512)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, running = time.time(), 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{EPOCHS}  loss {running / len(x_train):.4f}  "
              f"val_acc {val_acc:.4f}  ({time.time() - start:.1f}s)", flush=True)

    torch.save(model.state_dict(), "model.pt")
    print("saved model.pt")


if __name__ == "__main__":
    main()
