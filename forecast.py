"""Train an LSTM to forecast next-day log returns, and compare against baselines.

Usage:
    python forecast.py --ticker AAPL
    python forecast.py --csv prices.csv --lookback 30 --epochs 40
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import FEATURES, add_features, chronological_split, load_prices

torch.manual_seed(42)
np.random.seed(42)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # Directional accuracy: did we predict up/down correctly? Undefined for a model that never picks a side.
    direction = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if np.any(y_pred) else float("nan")
    return {"RMSE": rmse, "MAE": mae, "Direction acc": direction}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--csv")
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    args = p.parse_args()

    df = add_features(load_prices(args.ticker, args.csv))
    s = chronological_split(df, args.lookback)
    (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, dates, closes) = s["train"], s["val"], s["test"]
    print(f"windows  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    # Targets are tiny log returns; scale them for stable training.
    y_scale = float(ytr.std())
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr / y_scale)), batch_size=64, shuffle=True)
    model = LSTMForecaster(len(FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()

    best_val, bad_epochs, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(torch.tensor(Xva)), torch.tensor(yva / y_scale)).item()
        print(f"epoch {epoch:02d}  val_loss {val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).numpy() * y_scale

    results = {
        "LSTM": metrics(yte, pred),
        "Zero-return (naive)": metrics(yte, np.zeros_like(yte)),
        # Momentum baseline: tomorrow's return equals today's return.
        "Yesterday's return": metrics(yte, np.r_[0.0, yte[:-1]]),
        # Direction baseline: always predict a small up move (markets drift upward on average).
        "Always up": metrics(yte, np.full_like(yte, abs(float(ytr.mean())) or 1e-6)),
    }
    print("\nTest-set comparison")
    print(f"{'model':<22}{'RMSE':>10}{'MAE':>10}{'Dir acc':>10}")
    for name, m in results.items():
        print(f"{name:<22}{m['RMSE']:>10.5f}{m['MAE']:>10.5f}{m['Direction acc']:>10.3f}")

    # Convert return predictions to next-day price predictions for plotting.
    pred_price = closes * np.exp(pred)
    actual_next = closes * np.exp(yte)
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, actual_next, label="Actual next-day close", lw=1.2)
    ax.plot(dates, pred_price, label="LSTM prediction", lw=1.2, alpha=0.8)
    ax.set_title(f"{args.csv or args.ticker}: next-day close on the held-out test period")
    ax.legend()
    fig.savefig("results/forecast.png", dpi=130, bbox_inches="tight")
    print("\nSaved results/forecast.png")


if __name__ == "__main__":
    main()
