import os
import pandas as pd
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta
import pytz

# === CONFIG ===
HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
START_CAPITAL = 1000000
RISK_PER_TRADE = 0.02
FIXED_SL_PCT = -0.03  # -3%
PROFIT_TARGET_PCT = 0.08  # 8%
TRAILING_ATR_MULTIPLIER = 1.2
TRAILING_LOOKBACK = 5
PERIOD_YEARS = 2

# === STATS TRACKERS ===
indicator_pass_count = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
exit_reason_count = {}
skip_reason_count = {}

# === MODEL ===
model = joblib.load(MODEL_PATH)
FEATURES = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]

def record_skip(reason):
    skip_reason_count[reason] = skip_reason_count.get(reason, 0) + 1

def track_exit_reason(reason):
    exit_reason_count[reason] = exit_reason_count.get(reason, 0) + 1

def calculate_features(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["VolumeChange"] = df["volume"].pct_change()

    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = st["SUPERT_10_3.0"]

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

def apply_ai_score(df):
    X = df[FEATURES].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA).dropna()
    df = df.loc[X.index].copy()
    X = X.astype("float32")
    df["ai_score"] = model.predict_proba(X)[:, 1]
    return df

def run_backtest():
    global START_CAPITAL
    capital = START_CAPITAL
    trades = []
    cutoff_date = datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(days=PERIOD_YEARS * 365)

    files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".csv")]
    for file in files:
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns:
            record_skip("No date col")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize("Asia/Kolkata", ambiguous='NaT', nonexistent='NaT')
        df = df.dropna(subset=["date"])
        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            record_skip("Insufficient data")
            continue

        df = calculate_features(df)
        df = apply_ai_score(df)

        for i in range(len(df) - 1):
            row = df.iloc[i]

            rsi_pass = 30 < row["RSI"] < 70
            ema_pass = row["EMA10"] > row["EMA21"]
            st_pass = row["close"] > row["Supertrend"]
            ai_pass = row["ai_score"] > 0.20

            if not rsi_pass: record_skip("RSI fail")
            if not ema_pass: record_skip("EMA fail")
            if not st_pass: record_skip("Supertrend fail")
            if not ai_pass: record_skip("AI score fail")

            if rsi_pass: indicator_pass_count["RSI"] += 1
            if ema_pass: indicator_pass_count["EMA"] += 1
            if st_pass: indicator_pass_count["Supertrend"] += 1
            if ai_pass: indicator_pass_count["AI_Score"] += 1

            if rsi_pass and ema_pass and st_pass and ai_pass:
                entry_price = row["close"]
                atr_value = row["ATR"]
                qty = int((capital * RISK_PER_TRADE) / atr_value)
                if qty <= 0:
                    record_skip("Zero qty")
                    continue

                peak_price = entry_price
                for j in range(i + 1, len(df)):
                    exit_row = df.iloc[j]
                    ltp = exit_row["close"]

                    if ltp > peak_price:
                        peak_price = ltp

                    trailing_sl_atr = peak_price - TRAILING_ATR_MULTIPLIER * atr_value
                    trailing_sl_recent = df["low"].iloc[max(j - TRAILING_LOOKBACK, 0): j+1].min()
                    trailing_sl = max(trailing_sl_atr, trailing_sl_recent)

                    reason = None
                    if ltp <= entry_price * (1 + FIXED_SL_PCT):
                        reason = f"Fixed SL breach ({FIXED_SL_PCT*100:.0f}%)"
                    elif ltp <= trailing_sl:
                        reason = "Trailing SL breached"
                    elif ltp >= entry_price * (1 + PROFIT_TARGET_PCT):
                        reason = f"Profit >={PROFIT_TARGET_PCT*100:.0f}% hit"

                    if reason:
                        pnl = (ltp - entry_price) * qty
                        capital += pnl
                        trades.append({
                            "symbol": symbol,
                            "entry_date": df["date"].iloc[i],
                            "exit_date": df["date"].iloc[j],
                            "entry_price": entry_price,
                            "exit_price": ltp,
                            "qty": qty,
                            "pnl": pnl,
                            "reason": reason
                        })
                        track_exit_reason(reason)
                        break

    total_trades = len(trades)
    wins = len([t for t in trades if t["pnl"] > 0])
    pnl_total = sum(t["pnl"] for t in trades)

    print(f"\nPeriod: Last {PERIOD_YEARS} years")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {wins} ({wins / total_trades * 100:.2f}%)")
    print(f"Total PnL: ₹{pnl_total:,.2f}")
    print(f"Final Capital: ₹{capital:,.2f}")

    print("\nIndicator Pass Counts:")
    for k, v in indicator_pass_count.items():
        print(f"{k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reason_count.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reason_count.items():
        print(f"{k}: {v}")

    pd.DataFrame(trades).to_csv("backtest_trades.csv", index=False)
    print("\n✅ backtest_trades.csv saved.")

if __name__ == "__main__":
    run_backtest()
