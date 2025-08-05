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
TARGET_PROFIT_PCT = 0.08   # 8% target profit
STOP_LOSS_PCT = 0.03       # 3% fixed stop loss
TRAILING_SL_MULTIPLIER = 1.5
CSV_TRADE_LOG = "backtest_trades.csv"

# Load AI model
model = joblib.load(MODEL_PATH)

# Summary stats
total_trades = 0
profitable_trades = 0
total_pnl = 0
indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {"AI score fail": 0, "EMA fail": 0, "RSI fail": 0, "Supertrend fail": 0, "Insufficient data": 0, "NaN in features": 0}
exit_reasons = {"Fixed SL breach (-3%)": 0, "Profit >=8% hit": 0, "Trailing SL breached": 0}
trade_log = []

def calculate_indicators(df):
    # Calculate indicators with uppercase column names as AI expects
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    supertrend = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = supertrend["SUPERTd_10_3.0"]
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # Volume change and MACD hist (keep feature names consistent with model if needed)
    df["VolumeChange"] = df["volume"].pct_change().fillna(0)

    return df

def run_backtest():
    global total_trades, profitable_trades, total_pnl

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

        # Fix date parsing with tz-aware handling
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        # Filter data for last PERIOD_YEARS years
        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            skip_reasons["Insufficient data"] += 1
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # Indicator checks
            if not (35 <= row["RSI"] <= 70):
                skip_reasons["RSI fail"] += 1
                continue
            indicator_pass_counts["RSI"] += 1

            if row["EMA10"] <= row["EMA21"]:
                skip_reasons["EMA fail"] += 1
                continue
            indicator_pass_counts["EMA"] += 1

            if row["Supertrend"] != 1:
                skip_reasons["Supertrend fail"] += 1
                continue
            indicator_pass_counts["Supertrend"] += 1

            # Prepare features for AI model prediction
            features = ["RSI", "EMA10", "EMA21", "ATR", "VolumeChange", "ADX"]

            # Check if any NaNs in features, skip if yes
            if any(pd.isna(row[feat]) for feat in features):
                skip_reasons["NaN in features"] += 1
                continue

            features_df = pd.DataFrame([[row[feat] for feat in features]], columns=features)
            ai_score = model.predict_proba(features_df)[0][1]

            if ai_score < 0.25:
                skip_reasons["AI score fail"] += 1
                continue
            indicator_pass_counts["AI_Score"] += 1

            # Entry trade details
            entry_price = row["close"]
            atr_value = row["ATR"]
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PROFIT_PCT)
            trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

            for j in range(i + 1, len(df)):
                future_row = df.iloc[j]
                ltp = future_row["close"]

                # Fixed stop loss hit
                if ltp <= stop_loss_price:
                    pnl = (ltp - entry_price)
                    exit_reasons["Fixed SL breach (-3%)"] += 1
                    total_trades += 1
                    total_pnl += pnl
                    if pnl > 0:
                        profitable_trades += 1
                    trade_log.append([symbol, entry_price, ltp, pnl, "Fixed SL breach (-3%)"])
                    break

                # Profit target hit
                if ltp >= target_price:
                    pnl = (ltp - entry_price)
                    exit_reasons["Profit >=8% hit"] += 1
                    total_trades += 1
                    total_pnl += pnl
                    profitable_trades += 1
                    trade_log.append([symbol, entry_price, ltp, pnl, "Profit >=8% hit"])
                    break

                # Trailing stop loss update and check
                trailing_sl = max(trailing_sl, ltp - TRAILING_SL_MULTIPLIER * atr_value)
                if ltp <= trailing_sl:
                    pnl = (ltp - entry_price)
                    exit_reasons["Trailing SL breached"] += 1
                    total_trades += 1
                    total_pnl += pnl
                    if pnl > 0:
                        profitable_trades += 1
                    trade_log.append([symbol, entry_price, ltp, pnl, "Trailing SL breached"])
                    break

    # === SUMMARY ===
    print(f"Period: Last {PERIOD_YEARS} years")
    print(f"Total Trades: {total_trades}")
    if total_trades > 0:
        print(f"Profitable Trades: {profitable_trades} ({(profitable_trades / total_trades * 100):.2f}%)")
    else:
        print(f"Profitable Trades: {profitable_trades} (0.00%)")
    print(f"Total PnL: ₹{total_pnl:,.2f}")
    print(f"Final Capital: ₹{total_pnl:,.2f}\n")

    print("Indicator Pass Counts:")
    for k, v in indicator_pass_counts.items():
        print(f"{k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reasons.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reasons.items():
        print(f"{k}: {v}")

    # Save trades to CSV
    pd.DataFrame(trade_log, columns=["Symbol", "Entry Price", "Exit Price", "PnL", "Exit Reason"]).to_csv(CSV_TRADE_LOG, index=False)
    print(f"\n✅ {CSV_TRADE_LOG} saved.")

if __name__ == "__main__":
    run_backtest()
