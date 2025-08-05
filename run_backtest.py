# run_backtest.py
import os
import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta
from datetime import datetime, timedelta

HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_FILE = "model.pkl"
START_CAPITAL = 1000000
RISK_PER_TRADE = 0.02  # 2% per trade
DRAWNDOWN_EXIT = 7  # 7% portfolio drawdown

# Load model
model = joblib.load(MODEL_FILE)

ENTRY_FEATURES = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]

indicator_stats = {f: {"pass": 0, "fail": 0} for f in ENTRY_FEATURES + ["Supertrend", "AI_Score"]}

def calculate_features(df):
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["VolumeChange"] = df["volume"].pct_change()
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = st["SUPERT_10_3.0"]
    return df

def apply_ai_score(df):
    X = df[ENTRY_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        df["ai_score"] = np.nan
        return df
    df.loc[X.index, "ai_score"] = model.predict_proba(X)[:, 1]
    return df

def run_backtest():
    capital = START_CAPITAL
    equity_peak = capital
    trades = []
    active_trades = []
    
    files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".csv")]

    for file in files:
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        if len(df) < 200:
            continue

        df.columns = [c.lower() for c in df.columns]
        df = calculate_features(df)
        df = apply_ai_score(df)

        for i in range(200, len(df)):
            row = df.iloc[i]
            if pd.isna(row["ai_score"]):
                continue

            # Portfolio drawdown check
            equity_value = capital + sum(t["qty"] * row["close"] for t in active_trades if t["symbol"] == symbol)
            equity_peak = max(equity_peak, equity_value)
            if (equity_peak - equity_value) / equity_peak * 100 >= DRAWNDOWN_EXIT:
                for t in active_trades:
                    trades.append({**t, "exit_reason": "Portfolio Drawdown", "exit_price": row["close"], "exit_date": row["date"], "pnl": (row["close"] - t["entry_price"]) * t["qty"]})
                active_trades.clear()
                break

            # Entry check
            conditions = {
                "RSI": 35 < row["RSI"] < 65,
                "EMA": row["EMA10"] > row["EMA21"],
                "Supertrend": row["close"] > row["Supertrend"],
                "AI_Score": row["ai_score"] > 0.25
            }
            for ind, passed in conditions.items():
                indicator_stats[ind if ind != "EMA" else "EMA10"]["pass" if passed else "fail"] += 1

            if all(conditions.values()):
                risk_amount = capital * RISK_PER_TRADE
                qty = int(risk_amount / row["close"])
                if qty > 0:
                    active_trades.append({
                        "symbol": symbol,
                        "entry_price": row["close"],
                        "qty": qty,
                        "entry_date": row["date"],
                        "highest_price": row["close"]
                    })
                    capital -= qty * row["close"]

            # Exit check
            for t in active_trades.copy():
                if t["symbol"] != symbol:
                    continue
                t["highest_price"] = max(t["highest_price"], row["close"])
                atr_sl = row["close"] - 1.5 * row["ATR"]
                recent_low_sl = df["low"].iloc[max(0, i-7):i].min()
                trailing_sl = max(atr_sl, recent_low_sl)

                exit_reason = None
                if row["close"] < t["entry_price"] * 0.98:
                    exit_reason = "Fixed SL -2%"
                elif row["close"] < trailing_sl:
                    exit_reason = "Trailing SL"
                elif row["close"] >= t["entry_price"] * 1.12:
                    exit_reason = "Profit 12%"

                if exit_reason:
                    pnl = (row["close"] - t["entry_price"]) * t["qty"]
                    capital += row["close"] * t["qty"]
                    trades.append({**t, "exit_reason": exit_reason, "exit_price": row["close"], "exit_date": row["date"], "pnl": pnl})
                    active_trades.remove(t)

    # Summary
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv("backtest_trades.csv", index=False)
    profitable = df_trades[df_trades["pnl"] > 0]
    print("\n===== BACKTEST SUMMARY =====")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Profitable Trades: {len(profitable)} ({len(profitable)/len(df_trades)*100:.2f}%)")
    print(f"Total PnL: ₹{df_trades['pnl'].sum():,.2f}")
    print(f"Final Capital: ₹{capital:,.2f}")
    print("\nIndicator Pass/Fail:")
    for ind, counts in indicator_stats.items():
        print(f"{ind}: Pass {counts['pass']}, Fail {counts['fail']}")
    print("\nExit Reasons:")
    print(df_trades["exit_reason"].value_counts())

if __name__ == "__main__":
    run_backtest()
