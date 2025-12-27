#!/usr/bin/env python3
"""
oil_price_forecast.py
Clean, robust oil direction model (Brent + WTI)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ------------------
# CONFIG
# ------------------
START_DATE = "2010-01-01"
BRENT = "BZ=F"
WTI = "CL=F"

PROB_TRADE_THRESHOLD = 0.57
OUTPUT_FILE = "oil_forecast.txt"

# ------------------
# LOAD DATA
# ------------------
def load_prices():
    brent = yf.download(BRENT, start=START_DATE, progress=False)
    wti = yf.download(WTI, start=START_DATE, progress=False)

    df = pd.DataFrame({
        "Brent": brent["Close"],
        "WTI": wti["Close"]
    }).dropna()

    return df

# ------------------
# FEATURES
# ------------------
def build_features(df):
    df["Brent_ret"] = df["Brent"].pct_change()
    df["WTI_ret"] = df["WTI"].pct_change()

    df["Momentum_5"] = df["Brent"].pct_change(5)
    df["Momentum_20"] = df["Brent"].pct_change(20)

    df["Trend_50"] = df["Brent"] > df["Brent"].rolling(50).mean()
    df["Trend_200"] = df["Brent"] > df["Brent"].rolling(200).mean()

    df["Target"] = (df["Brent_ret"].shift(-1) > 0).astype(int)

    return df.dropna()

# ------------------
# PROBABILITY ENGINE
# ------------------
def compute_probability(row):
    score = 0.5

    # momentum
    if row["Momentum_5"] > 0:
        score += 0.05
    if row["Momentum_20"] > 0:
        score += 0.05

    # trend filter
    if row["Trend_50"]:
        score += 0.05
    if row["Trend_200"]:
        score += 0.05

    return min(max(score, 0.0), 1.0)

# ------------------
# MAIN
# ------------------
def main():
    df = load_prices()
    df = build_features(df)

    last = df.iloc[-1]
    prob_up = compute_probability(last)
    prob_down = 1 - prob_up

    trend_ok = last["Trend_50"] and last["Trend_200"]

    signal = "NO_TRADE"
    if prob_up >= PROB_TRADE_THRESHOLD and trend_ok:
        signal = "LONG"
    elif prob_down >= PROB_TRADE_THRESHOLD and not trend_ok:
        signal = "SHORT"

    lines = []
    lines.append("=" * 40)
    lines.append("OIL PRICE FORECAST (BRENT + WTI)")
    lines.append("=" * 40)
    lines.append(f"Run time (UTC): {datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append(f"Brent Close : {last['Brent']:.2f}")
    lines.append(f"WTI Close   : {last['WTI']:.2f}")
    lines.append("")
    lines.append(f"Prob UP   : {prob_up:.2%}")
    lines.append(f"Prob DOWN : {prob_down:.2%}")
    lines.append("")
    lines.append(f"Trend 50/200 OK: {trend_ok}")
    lines.append(f"Signal: {signal}")
    lines.append("=" * 40)

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

if __name__ == "__main__":
    main()
