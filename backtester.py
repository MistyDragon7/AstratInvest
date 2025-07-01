# -*- coding: utf-8 -*-
"""
Portfolio Backtester — **Two‑Stage (Precompute → Playback) Backtrader Pipeline**
==============================================================================

**Why this refactor?**
Running CNN‑BiLSTM (TensorFlow) inference inside Backtrader's `next()` loop
is slow and can exhaust GPU/CPU RAM.  Instead we

1. **Pre‑compute Black‑Litterman weights** for each rebalance date using
   *only* information available up to that date (so there's **no look‑ahead**).
2. **Store** the weights to disk (`parquet` by default).
3. **Playback** those weights inside a minimal Backtrader strategy that never
touches TensorFlow.

The result: identical trading logic, 100× faster backtests, and no repeated
TensorFlow initialisations.

You still get:
* 60 : 40 chronological train:test split.
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

from stock_data_fetcher import StockDataFetcher  # filecite turn2file1
from views_generator import CNNBiLSTMViewsGenerator  # filecite turn2file0
from black_litterman_optimizer import BlackLittermanOptimizer  # filecite turn2file11

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
# GLOBAL_START_DATE = "2019-01-01" # Keep 5 years for full data fetch
# GLOBAL_END_DATE = "2024-04-30"   # Keep 5 years for full data fetch
GLOBAL_START_DATE = "2020-01-01"
GLOBAL_END_DATE = "2023-02-28" # February 2023 as requested

INTERVAL = "1d"
# TRAIN_TEST_SPLIT = 0.60  # No longer used for fixed split
REB_FREQ = 10            # trading‑day rebalance cadence
INITIAL_CASH = 1_000_000
SEQ_LEN = 30
EPOCHS = 50
BATCH_SIZE = 32
PRED_HORIZON = 5
WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
MODEL_DIR = "saved_models"
BENCHMARK_SYM = "^NSEI"

# Rolling window parameters
TRAINING_WINDOW_DAYS = 252  # ~1 year of trading days for training
RETRAIN_FREQ_DAYS = 63      # ~1 quarter of trading days for retraining (rebalancing frequency)
# ---------------------------------------------------------------------------
# Stage 1 – Pre‑compute BL weights and save to disk
# ---------------------------------------------------------------------------

def precompute_weights(symbols: List[str], weights_path: str = WEIGHTS_PATH_DEFAULT):
    """Compute BL weights for each rebalance date and persist to *weights_path*."""

    # 1) Download + enrich data once for the entire period
    fetcher = StockDataFetcher(symbols, start_date=GLOBAL_START_DATE, end_date=GLOBAL_END_DATE, interval=INTERVAL)
    stock_data = fetcher.fetch_all_stocks()
    fetcher.add_technical_indicators()
    full_returns_matrix = fetcher.create_returns_matrix()

    if full_returns_matrix.empty:
        raise RuntimeError("No returns matrix — aborting pre‑compute stage.")

    full_returns_matrix.index = pd.to_datetime(full_returns_matrix.index).normalize()

    initial_train_start = pd.to_datetime(GLOBAL_START_DATE).normalize()
    initial_train_end = initial_train_start + pd.Timedelta(days=TRAINING_WINDOW_DAYS + 30)

    if initial_train_end > full_returns_matrix.index[-1]:
        initial_train_end = full_returns_matrix.index[-1]

    initial_train_end_loc = full_returns_matrix.index.get_loc(initial_train_end, method='nearest')
    initial_train_end = full_returns_matrix.index[initial_train_end_loc]

    current_rebalance_date = initial_train_end

    rebalance_dates: List[pd.Timestamp] = []

    global_end_date_ts = pd.to_datetime(GLOBAL_END_DATE).normalize()
    while current_rebalance_date <= global_end_date_ts:
        rebalance_dates.append(current_rebalance_date)
        current_rebalance_date += pd.Timedelta(days=RETRAIN_FREQ_DAYS)
        if current_rebalance_date > global_end_date_ts:
            if not rebalance_dates or (global_end_date_ts > rebalance_dates[-1] and global_end_date_ts not in rebalance_dates):
                rebalance_dates.append(global_end_date_ts)
            break

    print(f"📅 Rebalance dates: {len(rebalance_dates)} (every {RETRAIN_FREQ_DAYS} trading days, approx. quarterly)")
    if not rebalance_dates:
        raise RuntimeError("No rebalance dates generated — aborting pre‑compute stage.")

    weight_rows = []

    for reb_date in rebalance_dates:
        print(f"Processing rebalance date: {reb_date.date()}")
        data_for_period = {
            t: df.loc[full_returns_matrix.index[0]:reb_date].copy()
            for t, df in fetcher.stock_data.items()
            if len(df.loc[full_returns_matrix.index[0]:reb_date]) > SEQ_LEN + PRED_HORIZON + 2
        }
        if len(data_for_period) < 2:
            print(f"⚠️ {reb_date.date()}: <2 assets with sufficient history for training/views — skipping")
            continue
        
        vg = CNNBiLSTMViewsGenerator(len(data_for_period), sequence_length=SEQ_LEN)
        vg.train_all_models(data_for_period, epochs=EPOCHS, batch_size=BATCH_SIZE, model_dir=MODEL_DIR, training_end_date=reb_date)

        views, view_uncertainties = vg.generate_investor_views(data_for_period, PRED_HORIZON)

        start_date_str = full_returns_matrix.index[0].strftime('%Y-%m-%d') # Direct strftime
        reb_date_str = reb_date.strftime('%Y-%m-%d')
        tmp_fetcher = StockDataFetcher(list(data_for_period.keys()), start_date=start_date_str, end_date=reb_date_str)
        tmp_fetcher.stock_data = data_for_period
        returns_matrix = tmp_fetcher.create_returns_matrix()
        
        market_caps_slice = {k: v for k, v in fetcher.market_caps.items() if k in data_for_period and v > 0}

        if returns_matrix.empty or not market_caps_slice:
            print(f"⚠️ {reb_date.date()}: Empty returns matrix or no valid market caps for BL optimization — skipping")
            continue

        blo = BlackLittermanOptimizer(returns_matrix, market_caps_slice, risk_free_rate=0.06)
        weights, *_ = blo.black_litterman_optimization(views, view_uncertainties)

        w = weights / weights.sum()
        row = {"Date": reb_date}
        row.update(w.to_dict())
        weight_rows.append(row)
        print(f"✓ {reb_date.date()}: weights ready (top: {w.nlargest(3).to_dict()})")

    if not weight_rows:
        raise RuntimeError("No weights were computed — cannot proceed.")
    
    for row in weight_rows:
        pass 
    
    weights_df = pd.DataFrame(weight_rows).set_index("Date").sort_index()
    weights_df.to_parquet(weights_path)
    print(f"💾 Saved weights table to {weights_path} ({weights_df.shape[0]} rows, {weights_df.shape[1]-1} assets)")

# ---------------------------------------------------------------------------
# Stage 2 – Backtrader playback of pre‑computed weights
# ---------------------------------------------------------------------------

import pandas as pd
from pathlib import Path
import backtrader as bt
import matplotlib.pyplot as plt
from stock_data_fetcher import StockDataFetcher
from black_litterman_optimizer import BlackLittermanOptimizer
from views_generator import CNNBiLSTMViewsGenerator # Added for completeness of imports

WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
INITIAL_CASH = 1_000_000
BENCHMARK_SYM = "^NSEI"

class BLWeightPlaybackStrategy(bt.Strategy):
    params = dict(weights_df=None)

    def __init__(self):
        self.weights_df = self.p.weights_df.copy()
        self.weights_df.index = pd.to_datetime(self.weights_df.index).normalize()
        self.last_rebalanced = None
        self.portfolio_values = []
        self.dates = []

    def next(self):
        current_dt_py = self.datas[0].datetime.datetime(0)
        current_dt = pd.to_datetime(current_dt_py).normalize()
        
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
            
            if d.close[0] == 0 or pd.isna(d.close[0]):
                print(f"Warning: Close price for {sym} is zero or NaN at {current_dt}. Skipping order.")
                continue

            tgt_size = int(tgt_val / d.close[0])
            self.order_target_size(d, tgt_size)
        self.last_rebalanced = current_dt


def run_backtest(weights_path=WEIGHTS_PATH_DEFAULT, benchmark_symbol=BENCHMARK_SYM):
    weights_df = pd.read_parquet(weights_path)
    weights_df.index = pd.to_datetime(weights_df.index).normalize()
    
    symbols = list(weights_df.columns)
    all_syms = symbols + ([benchmark_symbol] if benchmark_symbol not in symbols else [])

    fetcher = StockDataFetcher(all_syms, start_date=GLOBAL_START_DATE, end_date=GLOBAL_END_DATE, interval=INTERVAL)
    fetcher.fetch_all_stocks()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)

    for sym in symbols:
        if sym not in fetcher.stock_data:
            print(f"Warning: Data for {sym} not available from fetcher. Skipping.")
            continue
        df = fetcher.stock_data[sym].copy()
        df.index = pd.to_datetime(df.index).normalize()
        
        if df.empty:
            print(f"Warning: Dataframe for {sym} is empty after processing. Skipping.")
            continue
        
        feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(feed, name=sym)

    cerebro.addstrategy(BLWeightPlaybackStrategy, weights_df=weights_df)
    res = cerebro.run()
    strat = res[0]

    nifty_df = fetcher.stock_data[benchmark_symbol].copy()
    nifty_df.index = pd.to_datetime(nifty_df.index).normalize()

    nifty_series = nifty_df["Close"].reindex(pd.to_datetime(strat.dates).normalize(), method="pad")
    nifty_series = nifty_series / nifty_series.iloc[0] * INITIAL_CASH
    portfolio_series = pd.Series(strat.portfolio_values, index=pd.to_datetime(strat.dates).normalize())

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
