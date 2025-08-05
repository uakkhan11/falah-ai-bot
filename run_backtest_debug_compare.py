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

# Detect feature names from model
if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES = list(model.feature_names_in_)
else:
    MODEL_FEATURES = ["RSI", "EMA10", "EMA21", "ATR", "VolumeChange", "MACD_Hist"]

# Summary stats
total_trades = 0
profitable_trades = 0
total_pnl = 0
indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {
    "AI score fail": 0, "EMA fail": 0, "RSI fail": 0,
    "Supertrend fail": 0, "Insufficient data": 0, "NaN in features": 0
}
exit_reasons = {
    "Fixed SL breach (-3%)": 0,
    "Profit >=8% hit": 0,
    "Trailing SL breached": 0
}
trade_log = []


def calculate_indicators(df):
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema10"] = ta.ema(df["close"], length=10)
    df["ema21"] = ta.ema(df["close"], length=21)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["supertrend"] = st["SUPERTd_10_3.0"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    df["volume_change"] = df["volume"].pct_change().fillna(0)
    macd = ta.macd(df["close"])
    df["macd_hist"] = macd["MACDh_12_26_9"]
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

        # Fix date parsing with tz-safe handling
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        # Filter last N years
        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            skip_reasons["Insufficient data"] += 1
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # RSI filter
            if not (35 <= row["rsi"] <= 70):
                skip_reasons["RSI fail"] += 1
                continue
            indicator_pass_counts["RSI"] += 1

            # EMA filter
            if row["ema10"] <= row["ema21"]:
                skip_reasons["EMA fail"] += 1
                continue
            indicator_pass_counts["EMA"] += 1

            # Supertrend filter
            if row["supertrend"] != 1:
                skip_reasons["Supertrend fail"] += 1
                continue
            indicator_pass_counts["Supertrend"] += 1

            # AI features — map to model's expected names
            feature_map = {
                "RSI": row["rsi"],
                "EMA10": row["ema10"],
                "EMA21": row["ema21"],
                "ATR": row["atr"],
                "VolumeChange": row["volume_change"],
                "MACD_Hist": row["macd_hist"]
            }

            try:
                features_df = pd.DataFrame([[feature_map[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
            except KeyError as e:
                skip_reasons["NaN in features"] += 1
                continue

            if features_df.isnull().any(axis=1).iloc[0]:
                skip_reasons["NaN in features"] += 1
                continue

            ai_score = model.predict_proba(features_df)[0][1]
            if ai_score < 0.25:
                skip_reasons["AI score fail"] += 1
                continue
            indicator_pass_counts["AI_Score"] += 1

            # Entry trade
            entry_price = row["close"]
            atr_value = row["atr"]
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PROFIT_PCT)
            trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

            for j in range(i + 1, len(df)):
                future_row = df.iloc[j]
                ltp = future_row["close"]

                # Fixed SL
                if ltp <= stop_loss_price:
                    pnl = (ltp - entry_price)
                    exit_reasons["Fixed SL breach (-3%)"] += 1
                    total_trades += 1
                    total_pnl += pnl
                    if pnl > 0:
                        profitable_trades += 1
                    trade_log.append([symbol, entry_price, ltp, pnl, "Fixed SL breach (-3%)"])
                    break

                # Profit target
                if ltp >= target_price:
                    pnl = (ltp - entry_price)
                    exit_reasons["Profit >=8% hit"] += 1
                    total_trades += 1
                    total_pnl += pnl
                    profitable_trades += 1
                    trade_log.append([symbol, entry_price, ltp, pnl, "Profit >=8% hit"])
                    break

                # Trailing SL
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
        print("Profitable Trades: 0 (0.00%)")
    print(f"Total PnL: ₹{total_pnl:,.2f}")
    print(f"Final Capital: ₹{(total_pnl):,.2f}\n")

    print("Indicator Pass Counts:")
    for k, v in indicator_pass_counts.items():
        print(f"{k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reasons.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reasons.items():
        print(f"{k}: {v}")

    # Save trades
    pd.DataFrame(trade_log, columns=["Symbol", "Entry Price", "Exit Price", "PnL", "Exit Reason"]).to_csv(CSV_TRADE_LOG, index=False)
    print(f"\n✅ {CSV_TRADE_LOG} saved.")


if __name__ == "__main__":
    run_backtest()
