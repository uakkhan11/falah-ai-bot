import os
import pandas as pd
import pandas_ta as ta
import joblib
from datetime import datetime

# ================= CONFIG =================
HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
START_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.02  # 2% per trade
FEATURES = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]

# ================= TRACKERS =================
indicator_pass_count = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
exit_reason_count = {}
trades = []

# Load AI model
model = joblib.load(MODEL_PATH)


def calculate_features(df):
    """Calculate indicators & clean data."""
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

    # Precompute trailing low for last 7 days
    df["rolling_low_7"] = df["low"].rolling(window=7).min()

    # Clean infinities & NaNs
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df


def apply_ai_score(df):
    """Predict AI score from model."""
    X = df[FEATURES].astype("float32")
    df["ai_score"] = model.predict_proba(X)[:, 1]
    return df


def track_exit_reason(reason):
    """Count exit reasons."""
    exit_reason_count[reason] = exit_reason_count.get(reason, 0) + 1


def run_backtest():
    global trades
    capital = START_CAPITAL

    files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".csv")]

    for file in files:
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 200:
            continue

        df = calculate_features(df)
        df = apply_ai_score(df)

        position_open = False
        entry_price = 0
        qty = 0
        entry_date = None
        atr_value = 0

        for i in range(len(df)):
            row = df.iloc[i]

            # If no position, check entry
            if not position_open:
                rsi_pass = 35 < row["RSI"] < 65
                ema_pass = row["EMA10"] > row["EMA21"]
                st_pass = row["close"] > row["Supertrend"]
                ai_pass = row["ai_score"] > 0.25

                if rsi_pass: indicator_pass_count["RSI"] += 1
                if ema_pass: indicator_pass_count["EMA"] += 1
                if st_pass: indicator_pass_count["Supertrend"] += 1
                if ai_pass: indicator_pass_count["AI_Score"] += 1

                if rsi_pass and ema_pass and st_pass and ai_pass:
                    entry_price = row["close"]
                    atr_value = row["ATR"]
                    qty = int((capital * RISK_PER_TRADE) / atr_value)
                    entry_date = row.get("date", i)
                    position_open = True

            # If position open, check exit
            else:
                ltp = row["close"]
                reason = None

                if ltp < entry_price * 0.98:
                    reason = "Fixed SL breach (-2%)"
                elif ltp < max(entry_price - 1.5 * atr_value, row["rolling_low_7"]):
                    reason = "Trailing SL breached"
                elif ltp >= entry_price * 1.12:
                    reason = "Profit >=12% hit"

                if reason:
                    pnl = (ltp - entry_price) * qty
                    capital += pnl
                    trades.append({
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "exit_date": row.get("date", i),
                        "entry_price": entry_price,
                        "exit_price": ltp,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": reason
                    })
                    track_exit_reason(reason)
                    position_open = False

    # Summary
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)

    print("\n===== BACKTEST SUMMARY =====")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {wins} ({(wins / total_trades) * 100:.2f}%)")
    print(f"Total PnL: ₹{total_pnl:,.2f}")
    print(f"Final Capital: ₹{capital:,.2f}")

    print("\nIndicator Pass Counts:")
    for k, v in indicator_pass_count.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reason_count.items():
        print(f"{k}: {v}")

    pd.DataFrame(trades).to_csv("backtest_trades.csv", index=False)
    print("\n✅ backtest_trades.csv saved.")


if __name__ == "__main__":
    run_backtest()
