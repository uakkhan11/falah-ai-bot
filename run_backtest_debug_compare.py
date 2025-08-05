import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import joblib
import pandas_ta as ta

# === CONFIG ===
HISTORICAL_PATH = "/root/falah-ai-bot/historical_data"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
PERIOD_YEARS = 2
TARGET_PROFIT_PCT = 0.08   # 8% target
STOP_LOSS_PCT = 0.03       # 3% fixed SL
TRAILING_SL_MULTIPLIER = 1.5
CSV_TRADE_LOG = "backtest_trades.csv"

# Load AI model
model = joblib.load(MODEL_PATH)

# Summary stats
primary_trades = 0
primary_wins = 0
bb_trades = 0
bb_wins = 0
total_pnl = 0
indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {"AI score fail": 0, "EMA fail": 0, "RSI fail": 0, "Supertrend fail": 0, "Insufficient data": 0}
exit_reasons = {
    "Fixed SL breach (-3%)": 0,
    "Profit >=8% hit": 0,
    "Trailing SL breached": 0,
    "BB SL": 0,
    "BB Target": 0
}
trade_log = []

def calculate_indicators(df):
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["supertrend"] = st["SUPERTd_10_3.0"]
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["VolumeChange"] = df["volume"].pct_change().fillna(0)
    macd = ta.macd(df["close"])
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2)
    df["BB_lower"] = bb["BBL_20_2.0"]
    df["BB_upper"] = bb["BBU_20_2.0"]

    return df

def run_backtest():
    global primary_trades, primary_wins, bb_trades, bb_wins, total_pnl

    cutoff_date = datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(days=PERIOD_YEARS * 365)

    for file in os.listdir(HISTORICAL_PATH):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(HISTORICAL_PATH, file))
        except Exception:
            skip_reasons["Insufficient data"] += 1
            continue

        if "date" not in df.columns:
            skip_reasons["Insufficient data"] += 1
            continue

        # Fix date parsing
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            skip_reasons["Insufficient data"] += 1
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # Primary strategy
            if 35 <= row["RSI"] <= 70 and row["EMA10"] > row["EMA21"] and row["supertrend"] == 1:
                indicator_pass_counts["RSI"] += 1
                indicator_pass_counts["EMA"] += 1
                indicator_pass_counts["Supertrend"] += 1

                features_df = pd.DataFrame([[
                    row["RSI"], row["ATR"], row["ADX"] if "ADX" in row else 0,
                    row["EMA10"], row["EMA21"], row["VolumeChange"]
                ]], columns=["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"])

                try:
                    ai_score = model.predict_proba(features_df)[0][1]
                except:
                    skip_reasons["AI score fail"] += 1
                    continue

                if ai_score < 0.25:
                    skip_reasons["AI score fail"] += 1
                    continue
                indicator_pass_counts["AI_Score"] += 1

                entry_price = row["close"]
                atr_value = row["ATR"]
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                target_price = entry_price * (1 + TARGET_PROFIT_PCT)
                trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        primary_trades += 1
                        exit_reasons["Fixed SL breach (-3%)"] += 1
                        if future_price > entry_price:
                            primary_wins += 1
                        break

                    if future_price >= target_price:
                        primary_trades += 1
                        exit_reasons["Profit >=8% hit"] += 1
                        primary_wins += 1
                        break

                    trailing_sl = max(trailing_sl, future_price - TRAILING_SL_MULTIPLIER * atr_value)
                    if future_price <= trailing_sl:
                        primary_trades += 1
                        exit_reasons["Trailing SL breached"] += 1
                        if future_price > entry_price:
                            primary_wins += 1
                        break

            # Fallback BB strategy
            elif row["close"] <= row["BB_lower"]:
                bb_trades += 1
                entry_price = row["close"]
                stop_loss_price = row["BB_lower"] * 0.98
                target_price = row["BB_upper"]

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        exit_reasons["BB SL"] += 1
                        break

                    if future_price >= target_price:
                        bb_wins += 1
                        exit_reasons["BB Target"] += 1
                        break

    # === Safe division to avoid ZeroDivisionError ===
    total_trades = primary_trades + bb_trades
    total_wins = primary_wins + bb_wins

    primary_win_pct = (primary_wins / primary_trades * 100) if primary_trades > 0 else 0
    bb_win_pct = (bb_wins / bb_trades * 100) if bb_trades > 0 else 0
    overall_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"Period: Last {PERIOD_YEARS} years")
    print(f"Total Trades: {total_trades}")
    print(f"Primary Trades: {primary_trades} | Wins: {primary_wins} ({primary_win_pct:.2f}%)")
    print(f"BB Trades: {bb_trades} | Wins: {bb_wins} ({bb_win_pct:.2f}%)")
    print(f"Overall Win %: {overall_win_pct:.2f}%")
    print(f"Total PnL: ₹{total_pnl:,.2f}\n")

    print("Indicator Pass Counts:")
    for k, v in indicator_pass_counts.items():
        print(f"{k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reasons.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reasons.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_backtest()
