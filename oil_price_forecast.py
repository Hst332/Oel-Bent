#!/usr/bin/env python3
"""
oil_price_forecast.py
CODE A – robust, professional, no ML, no sklearn.

Brent + WTI + Spread
TXT output only (always overwritten)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
START_DATE = "2015-01-01"
SYMBOL_BRENT = "BZ=F"
SYMBOL_WTI = "CL=F"

OUTPUT_TXT = "oil_forecast_output.txt"

# =========================
# LOAD DATA
# =========================
def load_prices():
    brent = yf.download(SYMBOL_BRENT, start=START_DATE, progress=False)
    wti = yf.download(SYMBOL_WTI, start=START_DATE, progress=False)

    if brent.empty or wti.empty:
        raise RuntimeError("Yahoo data download failed")

    df = pd.DataFrame(index=brent.index)
    df["Brent_Close"] = brent["Close"]
    df["WTI_Close"] = wti["Close"]

    df = df.dropna()
    return df

# =========================
# SIGNAL LOGIC (CODE A)
# =========================
def build_signal(df: pd.DataFrame):
    df = df.copy()

    # Returns
    df["Brent_Return"] = df["Brent_Close"].pct_change()
    df["WTI_Return"] = df["WTI_Close"].pct_change()

    # Trend (20d)
    df["Brent_Trend"] = df["Brent_Close"] > df["Brent_Close"].rolling(20).mean()
    df["WTI_Trend"] = df["WTI_Close"] > df["WTI_Close"].rolling(20).mean()

    # Spread
    df["Brent_WTI_Spread"] = df["Brent_Close"] - df["WTI_Close"]
    df["Spread_Z"] = (
        (df["Brent_WTI_Spread"] - df["Brent_WTI_Spread"].rolling(60).mean())
        / df["Brent_WTI_Spread"].rolling(60).std()
    )

    df = df.dropna()
    last = df.iloc[-1]

    # --- Probability model (simple & robust) ---
    prob_up = 0.50

    if last["Brent_Trend"] and last["WTI_Trend"]:
        prob_up += 0.07

    if last["Spread_Z"] > 0.5:
        prob_up += 0.03
    elif last["Spread_Z"] < -0.5:
        prob_up -= 0.03

    prob_up = max(0.0, min(1.0, prob_up))
    prob_down = 1.0 - prob_up

    # Signal rules
    if prob_up >= 0.57:
        signal = "UP"
    elif prob_up <= 0.43:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    return {
        "prob_up": prob_up,
        "prob_down": prob_down,
        "signal": signal,
        "brent": float(last["Brent_Close"]),
        "wti": float(last["WTI_Close"]),
        "spread": float(last["Brent_WTI_Spread"]),
        "date": last.name.date().isoformat(),
    }

# =========================
# OUTPUT
# =========================
def write_txt(res: dict):
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "===================================",
        "   OIL FORECAST – CODE A",
        "===================================",
        f"Run time (UTC): {now_utc}",
        f"Data date     : {res['date']}",
        "",
        f"Brent Close   : {res['brent']:.2f}",
        f"WTI Close     : {res['wti']:.2f}",
        f"Brent-WTI Spr.: {res['spread']:.2f}",
        "",
        f"Prob UP       : {res['prob_up']*100:.2f}%",
        f"Prob DOWN     : {res['prob_down']*100:.2f}%",
        f"Signal        : {res['signal']}",
        "===================================",
    ]

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =========================
# MAIN
# =========================
def main():
    df = load_prices()
    res = build_signal(df)
    write_txt(res)
    print("[OK] oil_forecast_output.txt written")

if __name__ == "__main__":
    main()
