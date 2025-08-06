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
RISK_PER_TRADE = 0.005     # 0.5% of capital risk per trade
CAPITAL_STOP_THRESHOLD = 0.8
CSV_TRADE_LOG = "backtest_trades_debug.csv"

# Load AI model
model = joblib.load(MODEL_PATH)

# Stats
start_capital = 100000
capital = start_capital
primary_trades = 0
primary_wins = 0
bb_trades = 0
bb_wins = 0
total_pnl = 0
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
    bb = ta.bbands(df["close"], length=20, std=2)
    df["BB_lower"] = bb["BBL_20_2.0"]
    df["BB_upper"] = bb["BBU_20_2.0"]
    return df

def run_backtest():
    global capital, primary_trades, primary_wins, bb_trades, bb_wins, total_pnl

    cutoff_date = datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(days=PERIOD_YEARS * 365)

    for file in os.listdir(HISTORICAL_PATH):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_PATH, file))

        if "date" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        df = df[df["date"] >= cutoff_date]
        if len(df) < 50:
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df) - 1):
            row = df.iloc[i]

            # === PRIMARY STRATEGY ===
            if (30 <= row["RSI"] <= 75 and 
                row["EMA10"] > row["EMA21"] and 
                row["supertrend"] == 1):
                
                features_df = pd.DataFrame([[
                    row["RSI"], row["ATR"], row.get("ADX", 0),
                    row["EMA10"], row["EMA21"], row["VolumeChange"]
                ]], columns=["rsi", "atr", "adx", "ema10", "ema21", "volumechange"])

                
                ai_score = model.predict_proba(features_df)[0][1]
                if ai_score < 0.25:
                    continue

                entry_price = row["close"]
                atr_value = row["ATR"]
                qty = (capital * RISK_PER_TRADE) / (STOP_LOSS_PCT * entry_price)
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                target_price = entry_price * (1 + TARGET_PROFIT_PCT)
                trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

                for j in range(i+1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        total_pnl += pnl
                        primary_trades += 1
                        exit_reasons["Fixed SL breach (-3%)"] += 1
                        break

                    if future_price >= target_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        total_pnl += pnl
                        primary_trades += 1
                        primary_wins += 1
                        exit_reasons["Profit >=8% hit"] += 1
                        break

                    trailing_sl = max(trailing_sl, future_price - TRAILING_SL_MULTIPLIER * atr_value)
                    if future_price <= trailing_sl:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        total_pnl += pnl
                        primary_trades += 1
                        break

            # === BB STRATEGY ===
            elif (row["close"] <= row["BB_lower"] and 
                  i+1 < len(df) and 
                  df.iloc[i+1]["close"] > row["BB_lower"] and 
                  row["MACD_Hist"] > 0 and 
                  capital >= start_capital * CAPITAL_STOP_THRESHOLD):

                entry_price = row["close"]
                qty = (capital * RISK_PER_TRADE) / (STOP_LOSS_PCT * entry_price)
                stop_loss_price = row["BB_lower"] * 0.98
                target_price = row["BB_upper"]

                for j in range(i+1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        total_pnl += pnl
                        bb_trades += 1
                        exit_reasons["BB SL"] += 1
                        break

                    if future_price >= target_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        total_pnl += pnl
                        bb_trades += 1
                        bb_wins += 1
                        exit_reasons["BB Target"] += 1
                        break

    total_trades = primary_trades + bb_trades
    total_wins = primary_wins + bb_wins
    print(f"Start Capital: ₹{start_capital:,.2f}")
    print(f"End Capital: ₹{capital:,.2f}")
    print(f"Net Profit: ₹{total_pnl:,.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Primary Trades: {primary_trades} | Wins: {primary_wins}")
    print(f"BB Trades: {bb_trades} | Wins: {bb_wins}")
    print(f"Exit Reasons: {exit_reasons}")

if __name__ == "__main__":
    run_backtest()
