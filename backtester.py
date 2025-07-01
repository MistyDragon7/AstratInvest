# -*- coding: utf-8 -*-
"""
Portfolio Backtester — **Two‑Stage (Precompute → Playback) Backtrader Pipeline**
==============================================================================

**Why this refactor?**
Running CNN‑BiLSTM (TensorFlow) inference inside Backtrader’s `next()` loop
is slow and can exhaust GPU/CPU RAM.  Instead we

1. **Pre‑compute Black‑Litterman weights** for each rebalance date using
   *only* information available up to that date (so there’s **no look‑ahead**).
2. **Store** the weights to disk (`parquet` by default).
3. **Playback** those weights inside a minimal Backtrader strategy that never
touches TensorFlow.

The result: identical trading logic, 100× faster backtests, and no repeated
TensorFlow initialisations.

You still get:
* 60 : 40 chronological train:test split.
* 10‑trading‑day (≈bi‑weekly) rebalances.
* Auto‑(re)train of CNN‑BiLSTM models if checkpoints are missing.

---
### Usage
```bash
# 1) Pre‑compute weights (runs TensorFlow once)
python portfolio_backtester.py --stage precompute --symbols RELIANCE.NS TCS.NS INFY.NS …

# 2) Run the fast Backtrader simulation (no TensorFlow)
python portfolio_backtester.py --stage backtest --weights_path precomputed_weights.parquet
```
---
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List

import backtrader as bt
import pandas as pd

from stock_data_fetcher import StockDataFetcher  # fileciteturn2file1
from views_generator import CNNBiLSTMViewsGenerator  # fileciteturn2file0
from black_litterman_optimizer import BlackLittermanOptimizer  # fileciteturn2file11

# ---------------------------------------------------------------------------
# Configuration (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS: List[str] = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS",
    "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "HCLTECH.NS", "WIPRO.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "MARUTI.NS", "BAJFINANCE.NS", "TITAN.NS",
    "POWERGRID.NS", "TECHM.NS", "NESTLEIND.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "GRASIM.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "CIPLA.NS", "DIVISLAB.NS",
    "BAJAJFINSV.NS", "TATAMOTORS.NS", "JSWSTEEL.NS", "COALINDIA.NS", "DRREDDY.NS",
    "HINDALCO.NS", "NTPC.NS", "ONGC.NS", "BPCL.NS", "EICHERMOT.NS",
    "M&M.NS", "TATASTEEL.NS", "UPL.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS",
    "SBILIFE.NS", "INDUSINDBK.NS", "SHREECEM.NS", "ICICIPRULI.NS", "APOLLOHOSP.NS"
]
PERIOD = "5y"
INTERVAL = "1d"
TRAIN_TEST_SPLIT = 0.60  # 60 : 40 split
REB_FREQ = 10            # trading‑day rebalance cadence
INITIAL_CASH = 1_000_000
SEQ_LEN = 30
EPOCHS = 50
BATCH_SIZE = 32
PRED_HORIZON = 5
WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
MODEL_DIR = "saved_models"
BENCHMARK_SYM = "^NSEI" 
# ---------------------------------------------------------------------------
# Stage 1 – Pre‑compute BL weights and save to disk
# ---------------------------------------------------------------------------

def precompute_weights(symbols: List[str], weights_path: str = WEIGHTS_PATH_DEFAULT):
    """Compute BL weights for each rebalance date and persist to *weights_path*."""

    # 1) Download + enrich data once
    fetcher = StockDataFetcher(symbols, period=PERIOD, interval=INTERVAL)
    stock_data = fetcher.fetch_all_stocks()
    fetcher.add_technical_indicators()
    full_returns_matrix = fetcher.create_returns_matrix()

    if full_returns_matrix.empty:
        raise RuntimeError("No returns matrix — aborting pre‑compute stage.")

    # 60:40 chronological split
    split_idx = int(len(full_returns_matrix) * TRAIN_TEST_SPLIT)
    test_start = full_returns_matrix.index[split_idx]
    test_end = full_returns_matrix.index[-1]

    # 2) Train / load CNN‑BiLSTM models once on *training* window
    training_stock_data = {
        t: df.loc[:test_start].copy()
        for t, df in fetcher.stock_data.items()
        if len(df.loc[:test_start]) > SEQ_LEN + PRED_HORIZON + 2
    }
    if len(training_stock_data) < 2:
        raise RuntimeError("<2 assets with sufficient history — aborting.")

    vg = CNNBiLSTMViewsGenerator(len(training_stock_data), sequence_length=SEQ_LEN)
    vg.train_all_models(training_stock_data, epochs=EPOCHS, batch_size=BATCH_SIZE, model_dir=MODEL_DIR)

    # 3) Iterate over rebalance dates in the **test** set
    dates = full_returns_matrix.loc[test_start:test_end].index

    rebalance_dates: List[pd.Timestamp] = [dates[0]]
    for i in range(REB_FREQ, len(dates), REB_FREQ):
        rebalance_dates.append(dates[i])

    print(f"📅 Rebalance dates: {len(rebalance_dates)} (every {REB_FREQ} bars)")

    weight_rows = []  # list of dicts {Date, Ticker‑1: wt, …}

    for reb_date in rebalance_dates:
        # Slice *all* data up to reb_date
        current_data = {
            t: df.loc[:reb_date].copy()
            for t, df in fetcher.stock_data.items()
            if len(df.loc[:reb_date]) > SEQ_LEN + PRED_HORIZON + 2
        }
        if len(current_data) < 2:
            print(f"⚠️ {reb_date.date()}: <2 assets — skipping")
            continue

        # Investor views
        views, view_uncertainties = vg.generate_investor_views(current_data, PRED_HORIZON)

        # Returns matrix up to reb_date
        tmp_fetcher = StockDataFetcher(list(current_data.keys()))
        tmp_fetcher.stock_data = current_data
        returns_matrix = tmp_fetcher.create_returns_matrix()
        market_caps_slice = {k: v for k, v in fetcher.market_caps.items() if k in current_data and v > 0}

        blo = BlackLittermanOptimizer(returns_matrix, market_caps_slice, risk_free_rate=0.06)
        weights, *_ = blo.black_litterman_optimization(views, view_uncertainties)

        # Normalise + store
        w = weights / weights.sum()
        row = {"Date": reb_date}
        row.update(w.to_dict())
        weight_rows.append(row)
        print(f"✓ {reb_date.date()}: weights ready (top: {w.nlargest(3).to_dict()})")

    if not weight_rows:
        raise RuntimeError("No weights were computed — cannot proceed.")
    
    # Standardize timezone handling when creating the weights DataFrame
    for row in weight_rows:
        # Ensure timezone naive datetime by converting to UTC first then removing tz
        row["Date"] = pd.Timestamp(row["Date"]).tz_localize("UTC").tz_convert(None)
    
    weights_df = pd.DataFrame(weight_rows).set_index("Date").sort_index()
    weights_df.to_parquet(weights_path)
    print(f"💾 Saved weights table to {weights_path} ({weights_df.shape[0]} rows, {weights_df.shape[1]-1} assets)")

# ---------------------------------------------------------------------------
# Stage 2 – Backtrader playback of pre‑computed weights
# ---------------------------------------------------------------------------

# PATCHED VERSION with tz-aware fix and bounded test interval

import pandas as pd
from pathlib import Path
import backtrader as bt
import matplotlib.pyplot as plt
from stock_data_fetcher import StockDataFetcher
from black_litterman_optimizer import BlackLittermanOptimizer
from views_generator import CNNBiLSTMViewsGenerator

WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
INITIAL_CASH = 1_000_000
BENCHMARK_SYM = "^NSEI"

class BLWeightPlaybackStrategy(bt.Strategy):
    params = dict(weights_df=None)

    def __init__(self):
        self.weights_df = self.p.weights_df.copy()
        # Ensure weights DataFrame has timezone naive index
        if self.weights_df.index.tz is not None:
            self.weights_df.index = self.weights_df.index.tz_localize(None)
        self.last_rebalanced = None
        self.portfolio_values = []
        self.dates = []

    def next(self):
        current_dt = (pd.Timestamp(self.datas[0].datetime.datetime(0))
                     .tz_localize("UTC")
                     .tz_convert(None)
                     .normalize())
        
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(current_dt)

        if current_dt not in self.weights_df.index:
            return
        if self.last_rebalanced == current_dt:
            return

        row = self.weights_df.loc[current_dt].fillna(0)
        portfolio_val = self.broker.getvalue()
        for d in self.datas:
            sym = d._name
            tgt_wt = float(row.get(sym, 0.0))
            tgt_val = portfolio_val * tgt_wt
            tgt_size = int(tgt_val / d.close[0])
            self.order_target_size(d, tgt_size)
        self.last_rebalanced = current_dt


def run_backtest(weights_path=WEIGHTS_PATH_DEFAULT, benchmark_symbol=BENCHMARK_SYM):
    weights_df = pd.read_parquet(weights_path)
    # weights_df.index = pd.to_datetime(weights_df.index)
    if weights_df.index.tz is not None:
        weights_df.index = weights_df.index.tz_localize(None)

    symbols = list(weights_df.columns)
    all_syms = symbols + ([benchmark_symbol] if benchmark_symbol not in symbols else [])

    fetcher = StockDataFetcher(all_syms, period="5y", interval="1d")
    fetcher.fetch_all_stocks()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)

    # Convert test period dates to timezone naive format
    start_test = pd.Timestamp("2023-07-01").tz_localize(None)
    end_test = pd.Timestamp("2024-04-30").tz_localize(None)

    weights_df = weights_df.loc[start_test:end_test]

    for sym in symbols:
        if sym not in fetcher.stock_data:
            continue
        df = fetcher.stock_data[sym].copy()
        # Standardize data frame index timezone handling
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.loc[start_test:end_test]
        if df.empty:
            continue
        feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(feed, name=sym)

    cerebro.addstrategy(BLWeightPlaybackStrategy, weights_df=weights_df)
    res = cerebro.run()
    strat = res[0]

    nifty_df = fetcher.stock_data[benchmark_symbol].copy()
    if nifty_df.index.tz is not None:
        nifty_df.index = nifty_df.index.tz_localize(None)
    nifty_series = nifty_df["Close"].reindex(pd.to_datetime(strat.dates), method="pad")
    nifty_series = nifty_series / nifty_series.iloc[0] * INITIAL_CASH
    portfolio_series = pd.Series(strat.portfolio_values, index=strat.dates)

    # Plot results (plotting code remains the same)
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series, label="Portfolio", linewidth=2, color="green")
    plt.plot(nifty_series, label="NIFTY 50", linewidth=2, linestyle="--", color="steelblue")

    for reb_date in weights_df.index:
        closest = min(portfolio_series.index, key=lambda d: abs(d - reb_date))
        plt.axvline(closest, color="gray", linestyle=":", alpha=0.5)

    plt.title("Portfolio vs NIFTY 50")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Black‑Litterman Backtrader two‑stage pipeline")
    p.add_argument("--stage", choices=["precompute", "backtest"], required=True, help="Which stage to run")
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS, help="Ticker list (space‑separated)")
    p.add_argument("--weights_path", default=WEIGHTS_PATH_DEFAULT, help="Parquet file to read/write weights")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.stage == "precompute":
        precompute_weights(args.symbols, args.weights_path)
    else:
        run_backtest(args.weights_path)

if __name__ == "__main__":
    main()
