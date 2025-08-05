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
STOP_LOSS_PCT = 0.02       # 2% fixed SL
TARGET_PROFIT_PCT = 0.06   # 6% target
TRAILING_SL_MULTIPLIER = 1.5
CSV_TRADE_LOG = "backtest_trades.csv"

# Load AI model
model = joblib.load(MODEL_PATH)

# Stats
primary_trades = 0
primary_wins = 0
bb_trades = 0
bb_wins = 0
total_pnl = 0
trade_log = []

def calculate_indicators(df):
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = st["SUPERTd_10_3.0"]
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["VolumeChange"] = df["volume"].pct_change().fillna(0)
    macd = ta.macd(df["close"])
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2)
    df["BB_upper"] = bb["BBU_20_2.0"]
    df["BB_lower"] = bb["BBL_20_2.0"]

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
        except:
            continue

        if "date" not in df.columns:
            continue

        # Date fix
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        # Filter last 2 years
        df = df[df["date"] >= cutoff_date]
        if len(df) < 50:
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # === Primary AI Strategy ===
            if (35 <= row["RSI"] <= 70) and \
               (row["EMA10"] > row["EMA21"]) and \
               (row["Supertrend"] == 1):
                
                feature_names = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]
                if not all(f in df.columns for f in feature_names):
                    continue

                # AI prediction
                try:
                    features_df = pd.DataFrame([[row["RSI"], row["ATR"], row.get("ADX", 20), 
                                                 row["EMA10"], row["EMA21"], row["VolumeChange"]]],
                                               columns=feature_names)
                    ai_score = model.predict_proba(features_df)[0][1]
                except:
                    continue

                if ai_score >= 0.25:
                    if execute_trade(df, i, symbol, "Primary"):
                        continue  # Skip fallback if primary trade taken

            # === BB Fallback Strategy ===
            if (row["EMA10"] > row["EMA21"]) and (row["close"] > row["BB_upper"]):
                execute_trade(df, i, symbol, "BB")

    # === Summary ===
    total_trades = primary_trades + bb_trades
    total_wins = primary_wins + bb_wins
    win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"Period: Last {PERIOD_YEARS} years")
    print(f"Total Trades: {total_trades}")
    print(f"Primary Trades: {primary_trades} | Wins: {primary_wins} ({primary_wins/primary_trades*100:.2f}%)")
    print(f"BB Trades: {bb_trades} | Wins: {bb_wins} ({bb_wins/bb_trades*100:.2f}%)")
    print(f"Overall Win %: {win_pct:.2f}%")
    print(f"Total PnL: ₹{total_pnl:,.2f}")

    # Save CSV
    pd.DataFrame(trade_log, columns=["Symbol", "Strategy", "Entry", "Exit", "PnL", "Exit Reason"]) \
      .to_csv(CSV_TRADE_LOG, index=False)
    print(f"✅ {CSV_TRADE_LOG} saved.")

def execute_trade(df, entry_index, symbol, strategy):
    global primary_trades, primary_wins, bb_trades, bb_wins, total_pnl

    entry_price = df.iloc[entry_index]["close"]
    atr_value = df.iloc[entry_index]["ATR"]
    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
    target_price = entry_price * (1 + TARGET_PROFIT_PCT)
    trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

    for j in range(entry_index + 1, len(df)):
        ltp = df.iloc[j]["close"]

        # SL hit
        if ltp <= stop_loss_price:
            pnl = ltp - entry_price
            log_trade(symbol, strategy, entry_price, ltp, pnl, "SL hit")
            update_stats(strategy, pnl)
            return True

        # Target hit
        if ltp >= target_price:
            pnl = ltp - entry_price
            log_trade(symbol, strategy, entry_price, ltp, pnl, "Target hit")
            update_stats(strategy, pnl)
            return True

        # Trailing SL
        trailing_sl = max(trailing_sl, ltp - TRAILING_SL_MULTIPLIER * atr_value)
        if ltp <= trailing_sl:
            pnl = ltp - entry_price
            log_trade(symbol, strategy, entry_price, ltp, pnl, "Trailing SL")
            update_stats(strategy, pnl)
            return True
    return False

def log_trade(symbol, strategy, entry, exit, pnl, reason):
    trade_log.append([symbol, strategy, entry, exit, pnl, reason])

def update_stats(strategy, pnl):
    global primary_trades, primary_wins, bb_trades, bb_wins, total_pnl
    if strategy == "Primary":
        primary_trades += 1
        if pnl > 0:
            primary_wins += 1
    elif strategy == "BB":
        bb_trades += 1
        if pnl > 0:
            bb_wins += 1
    total_pnl += pnl

if __name__ == "__main__":
    run_backtest()
