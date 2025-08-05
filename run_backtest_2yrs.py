import os
import pandas as pd
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta, timezone

# ==== CONFIG ====
HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
START_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.02
SLIPPAGE_PCT = 0.002   # 0.2% slippage
BROKERAGE = 20         # flat per trade

# Load model
model = joblib.load(MODEL_PATH)
FEATURES = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]

# Stats tracking
indicator_pass_count = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {}
exit_reason_count = {}

# Helpers
def track_skip(reason):
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

def track_exit(reason):
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
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.dropna()
    df = df.loc[X.index].copy()
    X = X.astype("float32")
    df["ai_score"] = model.predict_proba(X)[:, 1]
    return df

def run_backtest():
    capital = START_CAPITAL
    trades = []

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=730)  # 2 years
    files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".csv")]

    for file in files:
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns:
            track_skip("Missing date"); continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            track_skip("Insufficient data"); continue

        df = calculate_features(df)
        df = apply_ai_score(df)

        for i in range(len(df) - 1):
            row = df.iloc[i]

            # Indicator passes
            rsi_pass = 35 < row["RSI"] < 65
            ema_pass = row["EMA10"] > row["EMA21"]
            st_pass = row["close"] > row["Supertrend"]
            ai_pass = row["ai_score"] > 0.25

            if rsi_pass: indicator_pass_count["RSI"] += 1
            if ema_pass: indicator_pass_count["EMA"] += 1
            if st_pass: indicator_pass_count["Supertrend"] += 1
            if ai_pass: indicator_pass_count["AI_Score"] += 1

            if not (rsi_pass and ema_pass and st_pass and ai_pass):
                if not rsi_pass: track_skip("RSI fail")
                if not ema_pass: track_skip("EMA fail")
                if not st_pass: track_skip("Supertrend fail")
                if not ai_pass: track_skip("AI score fail")
                continue

            # Entry
            entry_price = row["close"] * (1 + SLIPPAGE_PCT)
            atr = row["ATR"]
            qty = int((capital * RISK_PER_TRADE) / atr)
            if qty <= 0: track_skip("Zero qty"); continue

            # Hold until exit
            for j in range(i + 1, len(df)):
                exit_row = df.iloc[j]
                ltp = exit_row["close"]
                reason = None
                if ltp <= entry_price * 0.98:
                    reason = "Fixed SL breach (-2%)"
                elif ltp <= max(ltp - 1.5 * atr, df["low"].iloc[max(0, j - 7):j + 1].min()):
                    reason = "Trailing SL breached"
                elif ltp >= entry_price * 1.12:
                    reason = "Profit >=12% hit"

                if reason:
                    sell_price = ltp * (1 - SLIPPAGE_PCT)
                    pnl = (sell_price - entry_price) * qty - BROKERAGE
                    capital += pnl
                    trades.append({
                        "symbol": symbol,
                        "entry_date": row["date"],
                        "exit_date": exit_row["date"],
                        "entry_price": entry_price,
                        "exit_price": sell_price,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": reason
                    })
                    track_exit(reason)
                    break

    # Summary
    total_trades = len(trades)
    wins = len([t for t in trades if t["pnl"] > 0])
    pnl_total = sum(t["pnl"] for t in trades)

    print(f"\nPeriod: Last 2 years")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {wins} ({wins / total_trades * 100:.2f}%)")
    print(f"Total PnL: ₹{pnl_total:,.2f}")
    print(f"Final Capital: ₹{capital:,.2f}")

    print("\nIndicator Pass Counts:")
    for k, v in indicator_pass_count.items():
        print(f"{k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reasons.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reason_count.items():
        print(f"{k}: {v}")

    pd.DataFrame(trades).to_csv("backtest_trades.csv", index=False)
    print("\n✅ backtest_trades.csv saved.")

if __name__ == "__main__":
    run_backtest()
