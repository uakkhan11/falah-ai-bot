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

# BB fallback settings
BB_TARGET_PCT = 0.05
BB_SL_PCT = 0.02

# Load AI model
model = joblib.load(MODEL_PATH)

# Summary stats
total_trades = 0
profitable_trades = 0
total_pnl = 0
primary_trades = {"total": 0, "wins": 0}
fallback_trades = {"total": 0, "wins": 0}

indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {"AI score fail": 0, "EMA fail": 0, "RSI fail": 0, "Supertrend fail": 0, "Insufficient data": 0}
exit_reasons = {"Fixed SL breach (-3%)": 0, "Profit >=8% hit": 0, "Trailing SL breached": 0,
                "BB SL": 0, "BB Target": 0}
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
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["BB_Lower"] = bbands["BBL_20_2.0"]
    df["BB_Upper"] = bbands["BBU_20_2.0"]

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

        # Fix date parsing
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        # Filter for last 2 years
        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            skip_reasons["Insufficient data"] += 1
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # === Primary Strategy Checks ===
            if not (35 <= row["RSI"] <= 70):
                skip_reasons["RSI fail"] += 1
                entry_type = "Fallback"
            else:
                indicator_pass_counts["RSI"] += 1
                entry_type = "Primary"

            if entry_type == "Primary" and row["EMA10"] <= row["EMA21"]:
                skip_reasons["EMA fail"] += 1
                entry_type = "Fallback"
            else:
                if entry_type == "Primary":
                    indicator_pass_counts["EMA"] += 1

            if entry_type == "Primary" and row["Supertrend"] != 1:
                skip_reasons["Supertrend fail"] += 1
                entry_type = "Fallback"
            else:
                if entry_type == "Primary":
                    indicator_pass_counts["Supertrend"] += 1

            if entry_type == "Primary":
                features_df = pd.DataFrame([[row["RSI"], row["ATR"], row["ADX"] if "ADX" in df.columns else 0,
                                             row["EMA10"], row["EMA21"], row["VolumeChange"]]],
                                           columns=["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"])
                ai_score = model.predict_proba(features_df)[0][1]
                if ai_score < 0.25:
                    skip_reasons["AI score fail"] += 1
                    entry_type = "Fallback"
                else:
                    indicator_pass_counts["AI_Score"] += 1

            # === If primary failed, try BB fallback ===
            if entry_type == "Fallback":
                if row["close"] < row["BB_Lower"]:  # bounce from lower band
                    entry_price = row["close"]
                    stop_loss_price = entry_price * (1 - BB_SL_PCT)
                    target_price = entry_price * (1 + BB_TARGET_PCT)
                    for j in range(i + 1, len(df)):
                        future_price = df.iloc[j]["close"]
                        if future_price <= stop_loss_price:
                            exit_reasons["BB SL"] += 1
                            total_trades += 1
                            fallback_trades["total"] += 1
                            if future_price > entry_price:
                                profitable_trades += 1
                                fallback_trades["wins"] += 1
                            trade_log.append([symbol, entry_price, future_price, future_price-entry_price, "BB SL", "Fallback_BB"])
                            break
                        if future_price >= target_price:
                            exit_reasons["BB Target"] += 1
                            total_trades += 1
                            profitable_trades += 1
                            fallback_trades["total"] += 1
                            fallback_trades["wins"] += 1
                            trade_log.append([symbol, entry_price, future_price, future_price-entry_price, "BB Target", "Fallback_BB"])
                            break
                continue

            # === Primary Entry ===
            entry_price = row["close"]
            atr_value = row["ATR"]
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PROFIT_PCT)
            trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

            for j in range(i + 1, len(df)):
                ltp = df.iloc[j]["close"]

                if ltp <= stop_loss_price:
                    exit_reasons["Fixed SL breach (-3%)"] += 1
                    total_trades += 1
                    primary_trades["total"] += 1
                    if ltp > entry_price:
                        profitable_trades += 1
                        primary_trades["wins"] += 1
                    trade_log.append([symbol, entry_price, ltp, ltp-entry_price, "Fixed SL breach (-3%)", "Primary"])
                    break

                if ltp >= target_price:
                    exit_reasons["Profit >=8% hit"] += 1
                    total_trades += 1
                    profitable_trades += 1
                    primary_trades["total"] += 1
                    primary_trades["wins"] += 1
                    trade_log.append([symbol, entry_price, ltp, ltp-entry_price, "Profit >=8% hit", "Primary"])
                    break

                trailing_sl = max(trailing_sl, ltp - TRAILING_SL_MULTIPLIER * atr_value)
                if ltp <= trailing_sl:
                    exit_reasons["Trailing SL breached"] += 1
                    total_trades += 1
                    primary_trades["total"] += 1
                    if ltp > entry_price:
                        profitable_trades += 1
                        primary_trades["wins"] += 1
                    trade_log.append([symbol, entry_price, ltp, ltp-entry_price, "Trailing SL breached", "Primary"])
                    break

    # === SUMMARY ===
    print(f"Period: Last {PERIOD_YEARS} years")
    print(f"Total Trades: {total_trades}")
    print(f"Primary Trades: {primary_trades['total']} | Wins: {primary_trades['wins']} ({(primary_trades['wins']/primary_trades['total']*100 if primary_trades['total'] else 0):.2f}%)")
    print(f"Fallback BB Trades: {fallback_trades['total']} | Wins: {fallback_trades['wins']} ({(fallback_trades['wins']/fallback_trades['total']*100 if fallback_trades['total'] else 0):.2f}%)")
    print(f"Overall Win %: {(profitable_trades/total_trades*100 if total_trades else 0):.2f}%")
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

    pd.DataFrame(trade_log, columns=["Symbol", "Entry Price", "Exit Price", "PnL", "Exit Reason", "EntryType"]).to_csv(CSV_TRADE_LOG, index=False)
    print(f"\n✅ {CSV_TRADE_LOG} saved.")

if __name__ == "__main__":
    run_backtest()
