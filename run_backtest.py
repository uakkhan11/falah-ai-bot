import os
import pandas as pd
import pandas_ta as ta
import joblib
from datetime import datetime

HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
START_CAPITAL = 1000000
RISK_PER_TRADE = 0.02  # 2% of capital risked per trade

# Track stats
indicator_pass_count = {
    "RSI": 0,
    "EMA": 0,
    "Supertrend": 0,
    "AI_Score": 0
}
exit_reason_count = {}

# Load model
model = joblib.load(MODEL_PATH)
FEATURES = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]


def calculate_features(df):
    # Ensure numeric OHLCV
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

    return df


def apply_ai_score(df):
    X = df[FEATURES].copy()

    # Replace inf/-inf with NaN, then drop them
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.dropna()

    # Align df to cleaned X
    df = df.loc[X.index].copy()

    # Convert all features to float32 for safety
    X = X.astype("float32")

    # Predict AI score
    df.loc[:, "ai_score"] = model.predict_proba(X)[:, 1]
    return df


def track_exit_reason(reason):
    exit_reason_count[reason] = exit_reason_count.get(reason, 0) + 1


def run_backtest():
    capital = START_CAPITAL
    trades = []
    files = [f for f in os.listdir(HISTORICAL_DIR) if f.endswith(".csv")]

    for file in files:
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        df.columns = [c.lower() for c in df.columns]

        # Clean numeric data
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

        if len(df) < 200:
            print(f"⚠️ Skipping {symbol}: insufficient clean data")
            continue

        try:
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

    # Clean infinities
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(inplace=True)

    return df

        df = apply_ai_score(df)

        for i in range(len(df) - 1):
            row = df.iloc[i]

            # Track indicator passes
            rsi_pass = 35 < row["RSI"] < 65
            ema_pass = row["EMA10"] > row["EMA21"]
            st_pass = row["close"] > row["Supertrend"]
            ai_pass = row["ai_score"] > 0.25

            if rsi_pass: indicator_pass_count["RSI"] += 1
            if ema_pass: indicator_pass_count["EMA"] += 1
            if st_pass: indicator_pass_count["Supertrend"] += 1
            if ai_pass: indicator_pass_count["AI_Score"] += 1

            # Entry condition
            if rsi_pass and ema_pass and st_pass and ai_pass:
                entry_price = row["close"]
                atr = row["ATR"]
                qty = int((capital * RISK_PER_TRADE) / atr)

                # Simulate trade until exit
                for j in range(i + 1, len(df)):
                    exit_row = df.iloc[j]
                    ltp = exit_row["close"]

                    # Exit logic (mirrors monitor.py)
                    reason = None
                    if ltp < entry_price * 0.98:
                        reason = "Fixed SL breach (-2%)"
                    elif ltp < ltp - 1.5 * atr:
                        reason = "Trailing SL breached"
                    elif ltp >= entry_price * 1.12:
                        reason = "Profit >=12% hit"

                    if reason:
                        pnl = (ltp - entry_price) * qty
                        capital += pnl
                        trades.append({
                            "symbol": symbol,
                            "entry_date": df.index[i],
                            "exit_date": df.index[j],
                            "entry_price": entry_price,
                            "exit_price": ltp,
                            "qty": qty,
                            "pnl": pnl,
                            "reason": reason
                        })
                        track_exit_reason(reason)
                        break

    # Summary
    total_trades = len(trades)
    wins = len([t for t in trades if t["pnl"] > 0])
    pnl_total = sum(t["pnl"] for t in trades)

    print("\n===== BACKTEST SUMMARY =====")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {wins} ({wins / total_trades * 100:.2f}%)")
    print(f"Total PnL: ₹{pnl_total:,.2f}")
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
