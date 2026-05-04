# Stock Return Forecasting with LSTM

An LSTM model that forecasts the **next-day log return** of a stock from engineered technical features, evaluated honestly against naive baselines on a held-out time period.

Most LSTM stock tutorials predict raw prices and show a chart that looks impressive. That's misleading: predicting "tomorrow ≈ today" already tracks the price curve almost perfectly. This project avoids that trap.

## Design choices

**Predict returns, not prices.** Prices are non-stationary, while log returns are closer to stationary and are what actually matters for decisions.

**No look-ahead leakage.**
- Chronological 70 / 15 / 15 train/validation/test split, never shuffled across time
- The feature scaler is fitted on training rows only
- All indicators use rolling windows over past data only

**Compare against baselines.** The model is only interesting if it beats:
- *Zero-return*: predict no change every day (best RMSE baseline)
- *Momentum*: tomorrow's return equals today's
- *Always up*: predict a small positive return every day (the directional-accuracy baseline)

**Report directional accuracy**, the share of days the up/down direction was right, alongside RMSE and MAE.

## Features

| Feature | Description |
|---------|-------------|
| `log_ret` | Daily log return |
| `hl_range` | (High − Low) / Close, intraday volatility |
| `vol_z` | Volume z-score vs 20-day window |
| `ma10_gap`, `ma50_gap` | Distance of price from its 10- and 50-day moving averages |
| `volatility20` | 20-day rolling std of returns |
| `rsi14` | 14-day Relative Strength Index |

## Model

- 2-layer LSTM (hidden size 64, dropout 0.2) over a 30-day lookback window
- MLP head → one output (next-day return)
- Huber loss, AdamW, gradient clipping, early stopping on validation loss

## Usage

```bash
pip install -r requirements.txt

python forecast.py --ticker AAPL
python forecast.py --ticker RELIANCE.NS --lookback 45
python forecast.py --csv my_prices.csv   # columns: Date, Open, High, Low, Close, Volume
```

The script prints a comparison table for the test period and saves `results/forecast.png`.

## Interpreting results

Daily returns are extremely noisy, and financial markets are close to efficient. A directional accuracy of **52–55%** that beats the baselines is a genuinely meaningful result. If the LSTM does *not* beat the baselines on a given stock, that is an honest and expected outcome, not a bug.

This project is for learning and is not financial advice.

## Project structure

```
data.py       # loading (yfinance or CSV), features, leakage-safe scaling and windowing
forecast.py   # LSTM model, training with early stopping, evaluation vs baselines, plot
```

## Tech stack

Python, PyTorch, pandas, NumPy, yfinance, matplotlib
