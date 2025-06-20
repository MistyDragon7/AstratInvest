from portfolio_backtester import PortfolioBacktester
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
nifty_50_stocks = [
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

backtester = PortfolioBacktester(stock_list=nifty_50_stocks)

results = backtester.run_comprehensive_backtest(
    sequence_length=30,
    epochs=25,
    batch_size=32,
    prediction_horizon=5,
    tau=0.025,
    output_dir="results",
    use_frozen_data=True,
    frozen_data_path="data/frozen_data.pkl"
)

# Save performance plot
print("Saving performance comparison chart...")
plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
print("Saved as performance_comparison.png")

# Save summary CSV
import pandas as pd
summary = []
for backtest_type, res in results.items():
    if backtest_type not in res:
        continue

    label = {
        "type_1": "Full Training",
        "type_2": "Out-of-Sample (Bi-weekly)",
        "rolling": "Rolling Rebalance"
    }.get(backtest_type, backtest_type)

    p = res.get('portfolio_performance')
    n = res.get('nifty_performance')

    if p and n:
        summary.append({
            "Backtest": label,
            "Portfolio Return": p['annualized_return'],
            "Portfolio Sharpe": p['sharpe_ratio'],
            "Portfolio Volatility": p['volatility'],
            "Nifty Return": n['annualized_return'],
            "Nifty Sharpe": n['sharpe_ratio'],
            "Nifty Volatility": n['volatility'],
            "Excess Return": p['annualized_return'] - n['annualized_return']
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("backtest_summary.csv", index=False)
print("Saved summary metrics to backtest_summary.csv")

# Zip output
import shutil, os
output_dir = "results"
zip_path = f"{output_dir}.zip"

if os.path.exists(zip_path):
    os.remove(zip_path)
shutil.make_archive(output_dir, 'zip', output_dir)

print(f"✅ Zipped all output to: {zip_path}")
