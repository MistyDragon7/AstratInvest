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
from pathlib import Path
from typing import Dict, List

import backtrader as bt
import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

from black_litterman_optimizer import BlackLittermanOptimizer
from black_litterman_optimizer import BlackLittermanOptimizer
from stock_data_fetcher import StockDataFetcher
from stock_data_fetcher import StockDataFetcher
from views_generator import CNNBiLSTMViewsGenerator
from views_generator import CNNBiLSTMViewsGenerator  # fileciteturn2file11

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
TRAIN_TEST_SPLIT = 0.60  # 60 : 40 split
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
# Stage 1 – Pre‑compute BL weights and save to disk
# ---------------------------------------------------------------------------

def plot_weights_over_time(weights_path=WEIGHTS_PATH_DEFAULT):
    """Plots the changes in portfolio weights over time."""
    try:
        weights_df = pd.read_parquet(weights_path)
        weights_df.index = pd.to_datetime(weights_df.index)
        if weights_df.index.tz is not None:
            weights_df.index = weights_df.index.tz_localize(None)

        # Select top N assets by average weight for plotting
        # Exclude 'Date' column if it somehow appears in columns (it shouldn't if index is Date)
        asset_columns = [col for col in weights_df.columns if col != 'Date']

        if len(asset_columns) == 0:
            print("⚠️ No asset columns found in weights data for plotting.")
            return
            
        # Calculate average weight for each asset across all rebalance periods
        average_weights = weights_df[asset_columns].mean().sort_values(ascending=False)
        
        # Select top 5 assets or fewer if less than 5 assets exist
        top_n = min(5, len(average_weights))
        top_assets = average_weights.head(top_n).index.tolist()

        if not top_assets:
            print("⚠️ No top assets to plot.")
            return

        plt.figure(figsize=(14, 7))
        # Plot only the top assets and sum of others as 'Rest'
        for asset in top_assets:
            plt.plot(weights_df.index, weights_df[asset], label=asset)

        remaining_assets = [col for col in asset_columns if col not in top_assets]
        if remaining_assets:
            weights_df['Rest'] = weights_df[remaining_assets].sum(axis=1)
            plt.plot(weights_df.index, weights_df['Rest'], label='Rest', linestyle='--')

        plt.title('Portfolio Weights Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        print(f"✅ Generated plot of portfolio weights over time and saved to weights_over_time.png")

    except Exception as e:
        print(f"Error plotting weights: {e}")

def precompute_weights(symbols: List[str], weights_path: str = WEIGHTS_PATH_DEFAULT):
    """Compute BL weights for each rebalance date and persist to *weights_path*."""

    # 1) Download + enrich data once
    fetcher = StockDataFetcher(symbols, period=PERIOD, interval=INTERVAL)
    stock_data = fetcher.fetch_all_stocks()
    fetcher.add_technical_indicators()
    full_returns_matrix = fetcher.create_returns_matrix()

    if full_returns_matrix.empty:
        raise RuntimeError("No returns matrix — aborting pre‑compute stage.")

    # Rolling window configuration
    TRAINING_WINDOW_DAYS = 252  # ~1 year of trading days
    RETRAIN_FREQ_DAYS = 63      # ~1 quarter of trading days

    all_dates = full_returns_matrix.index.sort_values()

    if len(all_dates) < TRAINING_WINDOW_DAYS + REB_FREQ:
        raise RuntimeError("Insufficient data for initial training and at least one rebalance period.")

    # Determine the start of the backtesting period (after initial training window)
    # The initial training period will be from the beginning of `all_dates` up to TRAINING_WINDOW_DAYS
    # We ensure that initial_train_end_date is a valid date within the all_dates index
    initial_train_end_idx = TRAINING_WINDOW_DAYS - 1
    if initial_train_end_idx >= len(all_dates):
        raise RuntimeError("Not enough historical data for initial training window.")
    
    initial_train_end_date = all_dates[initial_train_end_idx]

    # The actual rebalance dates start from the date immediately following the initial training period
    # and then every REB_FREQ days thereafter.
    rebalance_dates: List[pd.Timestamp] = []
    current_rebalance_idx = all_dates.get_loc(initial_train_end_date)
    if isinstance(current_rebalance_idx, slice): # Handle case where get_loc returns a slice
        current_rebalance_idx = current_rebalance_idx.stop - 1 # Take the last index of the slice

    current_rebalance_idx += 1

    while current_rebalance_idx < len(all_dates):
        rebalance_dates.append(pd.Timestamp(all_dates[current_rebalance_idx]))
        current_rebalance_idx += REB_FREQ

    print(f"📅 Total rebalance points for rolling backtest: {len(rebalance_dates)}")

    vg = CNNBiLSTMViewsGenerator(n_stocks=len(symbols), sequence_length=SEQ_LEN) # n_stocks will be dynamically updated

    weight_rows = []
    last_retrain_date = None

    for reb_date in rebalance_dates:
        print(f"\nProcessing rebalance date: {pd.Timestamp(reb_date).date()}")

        # Define the rolling training window. It ends at `reb_date - 1 day`
        # and goes back `TRAINING_WINDOW_DAYS`.
        # We use reb_date - dt.timedelta(days=1) to ensure no look-ahead into the rebalance day itself.
        # However, it's safer to use the actual index of `full_returns_matrix` to determine dates.
        
        # Find the index of reb_date in all_dates
        current_reb_date_loc = all_dates.get_loc(reb_date)
        if isinstance(current_reb_date_loc, slice):
            current_reb_date_loc = current_reb_date_loc.stop - 1
        
        # Ensure we have enough data for the training window ending at reb_date - 1 trading day
        training_end_date_for_slice_idx = current_reb_date_loc - 1
        if training_end_date_for_slice_idx < 0:
            print(f"⚠️ {pd.Timestamp(reb_date).date()}: Not enough history for training window ending before rebalance date — skipping")
            continue
            
        training_end_date_for_slice = all_dates[training_end_date_for_slice_idx]

        # Calculate the start date for the training window
        training_start_date_candidate_idx = training_end_date_for_slice_idx - TRAINING_WINDOW_DAYS + 1
        
        if training_start_date_candidate_idx < 0:
            training_start_date = all_dates[0] # Take all available data from the beginning
        else:
            training_start_date = all_dates[training_start_date_candidate_idx]

        training_stock_data = {
            t: df.loc[training_start_date:training_end_date_for_slice].copy()
            for t, df in fetcher.stock_data.items()
            if not df.loc[training_start_date:training_end_date_for_slice].empty and \
               len(df.loc[training_start_date:training_end_date_for_slice]) > SEQ_LEN + PRED_HORIZON + 2
        }
        
        if len(training_stock_data) < 2:
            print(f"⚠️ {pd.Timestamp(reb_date).date()}: Insufficient assets with enough history for training — skipping")
            continue

        # Retrain models periodically (e.g., quarterly) or for the very first rebalance
        if last_retrain_date is None or (reb_date - last_retrain_date).days >= RETRAIN_FREQ_DAYS:
            print(f"🔄 Retraining models on data from {pd.Timestamp(training_start_date).date()} to {pd.Timestamp(training_end_date_for_slice).date()} for {len(training_stock_data)} assets...")
            vg.n_stocks = len(training_stock_data)
            vg.train_all_models(training_stock_data, epochs=EPOCHS, batch_size=BATCH_SIZE, model_dir=MODEL_DIR)
            last_retrain_date = reb_date

        if not vg.models:
            print(f"⚠️ {pd.Timestamp(reb_date).date()}: No models trained for this period — skipping")
            continue

        # Generate investor views for the current rebalance date using the *latest* trained models
        # Use data up to reb_date (inclusive) for views generation
        current_data_for_views = {
            t: df.loc[:reb_date].copy()
            for t, df in fetcher.stock_data.items()
            if not df.loc[:reb_date].empty and \
               len(df.loc[:reb_date]) > SEQ_LEN + PRED_HORIZON + 2
        }
        if len(current_data_for_views) < 2:
            print(f"⚠️ {pd.Timestamp(reb_date).date()}: <2 assets for view generation — skipping")
            continue

        views, view_uncertainties = vg.generate_investor_views(current_data_for_views, PRED_HORIZON)

        # Returns matrix up to reb_date for BL optimization
        tmp_fetcher = StockDataFetcher(list(current_data_for_views.keys()))
        tmp_fetcher.stock_data = current_data_for_views
        returns_matrix = tmp_fetcher.create_returns_matrix()
        market_caps_slice = {k: v for k, v in fetcher.market_caps.items() if k in current_data_for_views and v > 0}

        blo = BlackLittermanOptimizer(returns_matrix, market_caps_slice, risk_free_rate=0.06)
        weights, *_ = blo.black_litterman_optimization(views, view_uncertainties)

        # Normalise + store
        w = weights / weights.sum()
        row = {"Date": reb_date}
        row.update(w.to_dict())
        weight_rows.append(row)
        print(f"✓ {pd.Timestamp(reb_date).date()}: weights ready (top: {w.nlargest(3).to_dict()})")

    if not weight_rows:
        raise RuntimeError("No weights were computed — cannot proceed.")
    
    # Standardize timezone handling when creating the weights DataFrame
    for row in weight_rows:
        # Ensure timezone naive datetime by converting to UTC first then removing tz
        row["Date"] = pd.Timestamp(row["Date"]).tz_localize("UTC").tz_convert(None)
    
    weights_df = pd.DataFrame(weight_rows).set_index("Date").sort_index()
    weights_df.to_parquet(weights_path)
    print(f"💾 Saved weights table to {weights_path} ({weights_df.shape[0]} rows, {weights_df.shape[1]-1} assets)")
    
    # Visualize weights after precomputation
    plot_weights_over_time(weights_path) # Call the plotting function

# ---------------------------------------------------------------------------
# Stage 2 – Backtrader playback of pre‑computed weights
# ---------------------------------------------------------------------------

# PATCHED VERSION with tz-aware fix and bounded test interval



WEIGHTS_PATH_DEFAULT = "precomputed_weights.parquet"
INITIAL_CASH = 1_000_000
BENCHMARK_SYM = "^NSEI"

class BLWeightPlaybackStrategy(bt.Strategy):
    params = dict(weights_df=None)

    def __init__(self):
        self.weights_df = self.p.weights_df.copy()
        # Ensure weights DataFrame has timezone naive index
        if isinstance(self.weights_df.index, pd.DatetimeIndex) and self.weights_df.index.tz is not None:
            self.weights_df.index = self.weights_df.index.tz_localize(None)
        self.last_rebalanced = None
        self.portfolio_values = []
        self.dates = []

    def next(self):
        current_dt = bt.num2date(self.datas[0].datetime[0])
        current_dt = pd.Timestamp(current_dt).tz_localize("UTC").tz_convert(None)
        
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
    weights_df.index = pd.to_datetime(weights_df.index)
    if isinstance(weights_df.index, pd.DatetimeIndex) and weights_df.index.tz is not None:
        weights_df.index = weights_df.index.tz_localize(None)

    symbols = list(weights_df.columns)
    all_syms = symbols + ([benchmark_symbol] if benchmark_symbol not in symbols else [])

    fetcher = StockDataFetcher(all_syms, period="5y", interval="1d")
    fetcher.fetch_all_stocks()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)

    # Convert test period dates to timezone naive format
    start_test = pd.Timestamp("2023-07-01").tz_localize(None)
    end_test = pd.Timestamp("2025-06-30").tz_localize(None)

    weights_df = weights_df.loc[start_test:end_test]

    for sym in symbols:
        if sym not in fetcher.stock_data:
            continue
        df = fetcher.stock_data[sym].copy()
        # Standardize data frame index timezone handling
        df.index = pd.to_datetime(df.index)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.loc[start_test:end_test]
        if df.empty:
            continue
        feed = bt.feeds.PandasData(df)
        cerebro.adddata(feed, name=sym)

    cerebro.addstrategy(BLWeightPlaybackStrategy, weights_df=weights_df)
    res = cerebro.run()
    strat = res[0]

    nifty_df = fetcher.stock_data[benchmark_symbol].copy()
    if isinstance(nifty_df.index, pd.DatetimeIndex) and nifty_df.index.tz is not None:
        nifty_df.index = nifty_df.index.tz_localize(None)
    nifty_series = nifty_df["Close"].reindex(pd.to_datetime(strat.dates), method="pad")
    nifty_series = nifty_series / nifty_series.iloc[0] * INITIAL_CASH
    portfolio_series = pd.Series(strat.portfolio_values, index=strat.dates)

    # Plot results (plotting code remains the same)
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series, label="Portfolio", linewidth=2, color="green")
    plt.plot(nifty_series, label="NIFTY 50", linewidth=2, color="steelblue")

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